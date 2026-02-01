import time
import asyncio
import logging
import math
import pandas as pd
from typing import List, Dict, Optional
from bot.config import AppSettings
from bot.state import BotState, TradeRecord
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy, build_ml_signals
from bot.strategy import Action, Signal, Bias
from bot.notification_manager import NotificationManager, NotificationLevel

logger = logging.getLogger(__name__)

class TradingLoop:
    def __init__(self, settings: AppSettings, state: BotState, bybit: BybitClient, tg_bot=None):
        self.settings = settings
        self.state = state
        self.bybit = bybit
        self.tg_bot = tg_bot
        self.notifier = NotificationManager(tg_bot, settings)
        self.strategies: Dict[str, MLStrategy] = {}

    async def run(self):
        logger.info("Starting Trading Loop...")
        
        # Синхронизируем позиции с биржей при старте
        await self.sync_positions_with_exchange()
        
        # Запускаем оба цикла параллельно
        await asyncio.gather(
            self._signal_processing_loop(),
            self._position_monitoring_loop()
        )
    
    async def _signal_processing_loop(self):
        """Основной цикл обработки сигналов"""
        logger.info("Starting Signal Processing Loop...")
        while True:
            try:
                if not self.state.is_running:
                    await asyncio.sleep(10)
                    continue

                for symbol in self.state.active_symbols:
                    await self.process_symbol(symbol)
                
                # Пауза между циклами (из настроек)
                await asyncio.sleep(self.settings.live_poll_seconds)
            except Exception as e:
                logger.error(f"[trading_loop] Error in signal processing loop: {e}")
                await asyncio.sleep(30)
    
    async def _position_monitoring_loop(self):
        """Цикл мониторинга открытых позиций для breakeven и trailing stop"""
        logger.info("Starting Position Monitoring Loop...")
        await asyncio.sleep(10)  # Даем время запуститься основному циклу
        
        while True:
            try:
                if not self.state.is_running:
                    await asyncio.sleep(10)
                    continue
                
                for symbol in self.state.active_symbols:
                    try:
                        # Получаем информацию о позиции
                        pos_info = await asyncio.to_thread(
                            self.bybit.get_position_info,
                            symbol=symbol
                        )
                        
                        if pos_info.get("retCode") == 0:
                            list_data = pos_info.get("result", {}).get("list", [])
                            if list_data:
                                position = list_data[0]
                                size = float(position.get("size", 0))
                                
                                if size > 0:
                                    # Проверяем частичное закрытие
                                    await self.check_partial_close(symbol, position)
                                    
                                    # Обновляем breakeven stop
                                    await self.update_breakeven_stop(symbol, position)
                                    
                                    # Обновляем trailing stop
                                    await self.update_trailing_stop(symbol, position)
                    
                    except Exception as e:
                        logger.error(f"Error monitoring position for {symbol}: {e}")
                
                # Проверяем позиции каждые 15 секунд
                await asyncio.sleep(15)
            
            except Exception as e:
                logger.error(f"[trading_loop] Error in position monitoring loop: {e}")
                await asyncio.sleep(30)

    async def process_symbol(self, symbol: str):
        try:
            # 0. Проверяем cooldown
            if self.state.is_symbol_in_cooldown(symbol):
                logger.debug(f"Symbol {symbol} is in cooldown, skipping...")
                return
            
            # 1. Получаем данные
            df = self.bybit.get_kline_df(symbol, self.settings.timeframe, limit=200)
            if df.empty: return

            # 2. Инициализируем стратегию если нужно
            if symbol not in self.strategies:
                model_path = self.state.symbol_models.get(symbol)
                # Если путь не задан, используем автопоиск из конфига (реализован в _auto_find_ml_model)
                if not model_path:
                    # Пытаемся найти модель в папке ml_models
                    from pathlib import Path
                    models = list(Path("ml_models").glob(f"*_{symbol}_*.pkl"))
                    if models:
                        model_path = str(models[0])
                        self.state.symbol_models[symbol] = model_path
                
                if model_path:
                    self.strategies[symbol] = MLStrategy(
                        model_path=model_path,
                        confidence_threshold=self.settings.ml_strategy.confidence_threshold,
                        min_signal_strength=self.settings.ml_strategy.min_signal_strength
                    )
                else:
                    logger.warning(f"No model found for {symbol}, skipping...")
                    return

            # 3. Генерируем сигнал
            strategy = self.strategies[symbol]
            # row - последний бар
            row = df.iloc[-1]
            
            # Проверяем позицию
            pos_info = self.bybit.get_position_info(symbol=symbol)
            has_pos = None
            size = 0.0
            entry_price = 0.0
            
            if pos_info.get("retCode") == 0:
                list_data = pos_info.get("result", {}).get("list", [])
                if list_data:
                    p = list_data[0]
                    size = float(p.get("size", 0))
                    if size > 0:
                        side = p.get("side")
                        has_pos = Bias.LONG if side == "Buy" else Bias.SHORT
                        entry_price = float(p.get("avgPrice", 0))

            # Генерация сигнала
            signal = strategy.generate_signal(
                row=row,
                df=df,
                has_position=has_pos,
                current_price=row["close"],
                leverage=self.settings.leverage
            )

            # 4. Логируем сигнал в историю
            if signal.action != Action.HOLD:
                self.state.add_signal(
                    symbol=symbol,
                    action=signal.action.value,
                    price=signal.price,
                    confidence=signal.indicators_info.get("confidence", 0.0),
                    reason=signal.reason,
                    indicators=signal.indicators_info
                )
                
                # Уведомление о сигнале высокой уверенности
                if signal.indicators_info.get("confidence", 0) > 0.7:
                    await self.notifier.medium(f"🔔 СИГНАЛ {signal.action.value} по {symbol}\nУверенность: {int(signal.indicators_info['confidence']*100)}%\nЦена: {signal.price}")

            # 5. Исполнение сделок (упрощенно)
            if signal.action == Action.LONG and has_pos != Bias.LONG:
                # Открываем LONG
                await self.execute_trade(symbol, "Buy", signal)
            elif signal.action == Action.SHORT and has_pos != Bias.SHORT:
                # Открываем SHORT
                await self.execute_trade(symbol, "Sell", signal)

        except Exception as e:
            logger.error(f"[trading_loop] Error processing {symbol}: {e}")

    async def execute_trade(self, symbol: str, side: str, signal: Signal):
        try:
            # Получаем qtyStep для символа
            qty_step = self.bybit.get_qty_step(symbol)
            
            if qty_step <= 0:
                logger.error(f"Invalid qtyStep for {symbol}: {qty_step}")
                return
            
            # Определяем precision из qtyStep
            qty_step_str = str(qty_step)
            if '.' in qty_step_str:
                precision = len(qty_step_str.split('.')[1])
            else:
                precision = 0
            
            # Выбираем режим расчета размера позиции
            if self.settings.risk.position_size_mode == "percentage":
                # РЕЖИМ: Процент от баланса
                # Получаем баланс
                balance_info = await asyncio.to_thread(self.bybit.get_wallet_balance)
                balance = 0.0
                
                if balance_info.get("retCode") == 0:
                    result = balance_info.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        wallet = list_data[0].get("coin", [])
                        usdt_coin = next((c for c in wallet if c.get("coin") == "USDT"), None)
                        if usdt_coin:
                            balance_str = usdt_coin.get("walletBalance", "0")
                            balance = float(balance_str) if balance_str and balance_str != "" else 0.0
                
                if balance <= 0:
                    logger.error(f"Cannot get balance or balance is zero for {symbol}")
                    return
                
                # РАСЧЕТ: margin_pct_balance% от баланса с использованием плеча
                # Маржа = баланс * margin_pct_balance
                # Количество = (маржа * leverage) / цена
                margin = balance * self.settings.risk.margin_pct_balance
                total_qty = (margin * self.settings.leverage) / signal.price
                
                logger.info(f"Position size (percentage mode) for {symbol}: balance=${balance:.2f}, margin=${margin:.2f} ({self.settings.risk.margin_pct_balance*100}%), leverage={self.settings.leverage}x")
            else:
                # РЕЖИМ: Фиксированная сумма
                # РАСЧЕТ: base_order_usd / цена
                total_qty = self.settings.risk.base_order_usd / signal.price
                
                logger.info(f"Position size (fixed mode) for {symbol}: ${self.settings.risk.base_order_usd:.2f} at price ${signal.price:.2f}")
            
            # Округляем вниз до ближайшего кратного qtyStep (как в примере кода)
            # Округляем вниз: Math.floor(totalQty / qtyStep) * qtyStep
            rounded_qty = math.floor(total_qty / qty_step) * qty_step
            
            # Если получилось меньше qtyStep, используем минимальный шаг
            if rounded_qty < qty_step:
                qty = qty_step
            else:
                qty = rounded_qty
            
            # Форматируем до нужной точности
            qty = float(f"{qty:.{precision}f}")
            
            resp = self.bybit.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss
            )
            
            if resp.get("retCode") == 0:
                logger.info(f"Successfully opened {side} for {symbol}")
                await self.notifier.high(f"🚀 ОТКРЫТА ПОЗИЦИЯ {side} {symbol}\nЦена: {signal.price}\nTP: {signal.take_profit}\nSL: {signal.stop_loss}")
                
                # Добавляем в историю (пока как открытую)
                trade = TradeRecord(
                    symbol=symbol,
                    side=side,
                    entry_price=signal.price,
                    qty=qty,
                    status="open",
                    model_name=self.state.symbol_models.get(symbol, "")
                )
                self.state.add_trade(trade)
            else:
                logger.error(f"Failed to place order: {resp}")
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def update_breakeven_stop(self, symbol: str, position_info: dict):
        """Перемещает SL в безубыток при достижении порога прибыли"""
        try:
            if not position_info or not position_info.get("size"):
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            current_sl = position_info.get("stopLoss")
            
            if not entry_price or not mark_price:
                return
            
            # Рассчитываем текущий PnL в процентах
            if side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price) * 100
            else:  # Sell
                pnl_pct = ((entry_price - mark_price) / entry_price) * 100
            
            # Проверяем, нужно ли активировать безубыток
            breakeven_activation = self.settings.risk.breakeven_activation_pct * 100  # Конвертируем в %
            
            if pnl_pct >= breakeven_activation:
                # Рассчитываем новый SL
                if pnl_pct >= 1.0:
                    # При прибыли >= 1% ставим SL на entry + 0.5%
                    if side == "Buy":
                        new_sl = entry_price * 1.005
                    else:
                        new_sl = entry_price * 0.995
                else:
                    # При прибыли >= 0.5% ставим SL на уровень входа
                    new_sl = entry_price
                
                # Округляем до tick size
                new_sl = self.bybit.round_price(new_sl, symbol)
                
                # Проверяем, нужно ли обновлять SL
                should_update = False
                if current_sl:
                    current_sl_float = float(current_sl)
                    if side == "Buy" and new_sl > current_sl_float:
                        should_update = True
                    elif side == "Sell" and new_sl < current_sl_float:
                        should_update = True
                else:
                    should_update = True
                
                if should_update:
                    logger.info(f"Moving {symbol} SL to breakeven: {new_sl} (PnL: {pnl_pct:.2f}%)")
                    resp = await asyncio.to_thread(
                        self.bybit.set_trading_stop,
                        symbol=symbol,
                        stop_loss=new_sl
                    )
                    
                    if resp.get("retCode") == 0:
                        await self.notifier.medium(
                            f"🛡️ БЕЗУБЫТОК АКТИВИРОВАН\n{symbol} SL → ${new_sl:.2f}\nТекущий PnL: +{pnl_pct:.2f}%"
                        )
        
        except Exception as e:
            logger.error(f"Error updating breakeven stop for {symbol}: {e}")
    
    async def update_trailing_stop(self, symbol: str, position_info: dict):
        """Активирует трейлинг стоп при достижении порога прибыли"""
        try:
            if not self.settings.risk.enable_trailing_stop:
                return
            
            if not position_info or not position_info.get("size"):
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            trailing_stop = position_info.get("trailingStop")
            
            if not entry_price or not mark_price:
                return
            
            # Рассчитываем текущий PnL в процентах
            if side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price)
            else:  # Sell
                pnl_pct = ((entry_price - mark_price) / entry_price)
            
            # Проверяем, нужно ли активировать трейлинг стоп
            if pnl_pct >= self.settings.risk.trailing_stop_activation_pct and not trailing_stop:
                # Активируем трейлинг стоп
                trailing_pct = self.settings.risk.trailing_stop_distance_pct * 100  # Bybit принимает в %
                
                logger.info(f"Activating trailing stop for {symbol}: {trailing_pct}% (PnL: {pnl_pct*100:.2f}%)")
                resp = await asyncio.to_thread(
                    self.bybit.set_trading_stop,
                    symbol=symbol,
                    trailing_stop=trailing_pct
                )
                
                if resp.get("retCode") == 0:
                    await self.notifier.medium(
                        f"📊 ТРЕЙЛИНГ СТОП АКТИВИРОВАН\n{symbol} | {trailing_pct}%\nТекущий PnL: +{pnl_pct*100:.2f}%"
                    )
        
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")
    
    async def check_partial_close(self, symbol: str, position_info: dict):
        """Проверяет и выполняет частичное закрытие позиции"""
        try:
            if not self.settings.risk.enable_partial_close:
                return
            
            if not position_info or not position_info.get("size"):
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            take_profit = position_info.get("takeProfit")
            
            if not entry_price or not mark_price or not take_profit:
                return
            
            take_profit_price = float(take_profit)
            
            # Рассчитываем прогресс к TP
            if side == "Buy":
                distance_to_tp = take_profit_price - entry_price
                current_progress = mark_price - entry_price
            else:  # Sell
                distance_to_tp = entry_price - take_profit_price
                current_progress = entry_price - mark_price
            
            if distance_to_tp <= 0:
                return
            
            progress_pct = current_progress / distance_to_tp
            
            # Проверяем уровни частичного закрытия
            for level_progress, close_pct in self.settings.risk.partial_close_levels:
                if progress_pct >= level_progress:
                    # Проверяем, не закрывали ли мы уже на этом уровне
                    # (это можно отслеживать через метаданные в state)
                    
                    # Рассчитываем количество для закрытия
                    close_qty = size * close_pct
                    
                    # Округляем
                    qty_step = self.bybit.get_qty_step(symbol)
                    close_qty = round(close_qty / qty_step) * qty_step
                    
                    if close_qty > 0:
                        logger.info(f"Partial close {symbol}: {close_pct*100}% at {progress_pct*100:.1f}% to TP")
                        
                        # Закрываем частично (reduce_only ордер)
                        close_side = "Sell" if side == "Buy" else "Buy"
                        resp = await asyncio.to_thread(
                            self.bybit.place_order,
                            symbol=symbol,
                            side=close_side,
                            qty=close_qty,
                            order_type="Market",
                            reduce_only=True
                        )
                        
                        if resp.get("retCode") == 0:
                            await self.notifier.high(
                                f"💰 ЧАСТИЧНОЕ ЗАКРЫТИЕ\n{symbol} | {close_pct*100}%\nПрогресс к TP: {progress_pct*100:.1f}%"
                            )
                        
                        break  # Закрываем только на одном уровне за раз
        
        except Exception as e:
            logger.error(f"Error checking partial close for {symbol}: {e}")
    
    async def sync_positions_with_exchange(self):
        """Синхронизирует локальное состояние с позициями на бирже при старте"""
        logger.info("Syncing positions with exchange...")
        
        try:
            for symbol in self.state.active_symbols:
                try:
                    # Получаем позицию с биржи
                    pos_info = await asyncio.to_thread(
                        self.bybit.get_position_info,
                        symbol=symbol
                    )
                    
                    if pos_info.get("retCode") == 0:
                        list_data = pos_info.get("result", {}).get("list", [])
                        if list_data:
                            position = list_data[0]
                            size = float(position.get("size", 0))
                            
                            if size > 0:
                                # Есть открытая позиция на бирже
                                side = position.get("side")
                                entry_price = float(position.get("avgPrice", 0))
                                
                                # Проверяем, есть ли она в локальном состоянии
                                local_pos = self.state.get_open_position(symbol)
                                
                                if not local_pos:
                                    # Позиции нет в локальном состоянии, добавляем
                                    logger.info(f"Found open position on exchange for {symbol}, adding to state")
                                    
                                    trade = TradeRecord(
                                        symbol=symbol,
                                        side=side,
                                        entry_price=entry_price,
                                        qty=size,
                                        status="open",
                                        model_name=self.state.symbol_models.get(symbol, "")
                                    )
                                    self.state.add_trade(trade)
                                    
                                    await self.notifier.medium(
                                        f"🔄 СИНХРОНИЗАЦИЯ\nНайдена открытая позиция:\n{symbol} {side} | Размер: {size}"
                                    )
                                else:
                                    # Позиция есть, обновляем данные если нужно
                                    if abs(local_pos.qty - size) > 0.0001 or abs(local_pos.entry_price - entry_price) > 0.01:
                                        logger.info(f"Updating position data for {symbol}")
                                        self.state.update_position(symbol, size, entry_price)
                            else:
                                # Позиции нет на бирже, но может быть в локальном состоянии
                                local_pos = self.state.get_open_position(symbol)
                                if local_pos:
                                    # Закрываем локальную позицию
                                    logger.warning(f"Position {symbol} closed on exchange but open locally, closing in state")
                                    self.state.update_trade_on_close(symbol, 0, 0, 0)
                
                except Exception as e:
                    logger.error(f"Error syncing position for {symbol}: {e}")
            
            logger.info("Position sync completed")
        
        except Exception as e:
            logger.error(f"Error during position sync: {e}")
