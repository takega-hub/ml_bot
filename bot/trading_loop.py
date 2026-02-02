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
        # Отслеживаем последнюю обработанную свечу для каждого символа
        self.last_processed_candle: Dict[str, Optional[pd.Timestamp]] = {}

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
                    # Добавляем задержку между символами для снижения нагрузки на API
                    if len(self.state.active_symbols) > 1:
                        await asyncio.sleep(2)
                
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
                
                # ОПТИМИЗАЦИЯ: получаем ВСЕ позиции одним запросом вместо отдельных для каждого символа
                # Это значительно снижает количество API запросов и предотвращает rate limit ошибки
                try:
                    all_positions = await asyncio.to_thread(
                        self.bybit.get_position_info,
                        settle_coin="USDT"  # Получаем все USDT позиции одним запросом
                    )
                    
                    if all_positions and all_positions.get("retCode") == 0:
                        result = all_positions.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            
                            # Создаем словарь позиций по символам для быстрого доступа
                            positions_by_symbol = {}
                            for pos in list_data:
                                if pos and isinstance(pos, dict):
                                    symbol = pos.get("symbol")
                                    if symbol in self.state.active_symbols:
                                        positions_by_symbol[symbol] = pos
                            
                            # Обрабатываем позиции для каждого активного символа
                            for symbol in self.state.active_symbols:
                                try:
                                    position = positions_by_symbol.get(symbol)
                                    
                                    if position:
                                        size = float(position.get("size", 0))
                                        
                                        # Проверяем, закрылась ли позиция на бирже
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos and size == 0:
                                            # Позиция закрылась на бирже, но в state еще открыта
                                            await self.handle_position_closed(symbol, local_pos)
                                        elif size > 0:
                                            # Позиция открыта, проверяем частичное закрытие и обновляем стопы
                                            await self.check_partial_close(symbol, position)
                                            
                                            # Обновляем breakeven stop
                                            await self.update_breakeven_stop(symbol, position)
                                            
                                            # Обновляем trailing stop
                                            await self.update_trailing_stop(symbol, position)
                                    else:
                                        # Позиции нет в списке, проверяем локальное состояние
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos:
                                            # Позиция закрылась на бирже
                                            await self.handle_position_closed(symbol, local_pos)
                                
                                except Exception as e:
                                    logger.error(f"Error processing position for {symbol}: {e}")
                    else:
                        logger.warning(f"Failed to get positions: retCode={all_positions.get('retCode') if all_positions else 'None'}")
                
                except Exception as e:
                    logger.error(f"Error getting all positions: {e}")
                
                # Проверяем позиции каждые 25 секунд (увеличено с 15 для снижения нагрузки на API)
                await asyncio.sleep(25)
            
            except Exception as e:
                logger.error(f"[trading_loop] Error in position monitoring loop: {e}")
                await asyncio.sleep(30)

    async def process_symbol(self, symbol: str):
        try:
            logger.debug(f"[{symbol}] Processing symbol...")
            
            # 0. Проверяем cooldown
            if self.state.is_symbol_in_cooldown(symbol):
                logger.debug(f"Symbol {symbol} is in cooldown, skipping...")
                return
            
            # 1. Получаем данные
            df = self.bybit.get_kline_df(symbol, self.settings.timeframe, limit=200)
            if df.empty:
                logger.warning(f"[{symbol}] No data received from exchange")
                return
            logger.debug(f"[{symbol}] Received {len(df)} candles, last close: {df['close'].iloc[-1]:.2f}")

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
                    logger.info(f"[{symbol}] Loading model: {model_path}")
                    logger.info(f"[{symbol}] Confidence threshold: {self.settings.ml_strategy.confidence_threshold}, Min signal strength: {self.settings.ml_strategy.min_signal_strength}")
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
            # ВАЖНО: Используем предпоследнюю закрытую свечу для предсказания
            # Последняя свеча может быть незакрытой и меняться, что приводит к одинаковым предсказаниям
            if len(df) >= 2:
                row = df.iloc[-2]  # Предпоследняя закрытая свеча
                current_price = df.iloc[-1]['close']  # Текущая цена из последней свечи
                # Получаем timestamp из колонки timestamp (индекс сброшен в get_kline_df)
                candle_timestamp = row.get('timestamp') if 'timestamp' in row else df.iloc[-2].get('timestamp', None)
                if candle_timestamp is None:
                    # Если timestamp не в колонке, пытаемся получить из индекса
                    candle_timestamp = df.index[-2] if len(df.index) > 1 else None
            else:
                row = df.iloc[-1]
                current_price = row['close']
                candle_timestamp = row.get('timestamp') if 'timestamp' in row else df.iloc[-1].get('timestamp', None)
                if candle_timestamp is None:
                    candle_timestamp = df.index[-1] if len(df.index) > 0 else None
            
            # Проверяем, не обрабатывали ли мы уже эту свечу
            # ВАЖНО: Проверяем только если timestamp валиден
            # Это предотвращает генерацию одинаковых сигналов для одной и той же закрытой свечи
            if candle_timestamp is not None:
                if symbol in self.last_processed_candle:
                    last_timestamp = self.last_processed_candle[symbol]
                    if last_timestamp is not None and last_timestamp == candle_timestamp:
                        # Эта свеча уже была обработана, пропускаем
                        logger.debug(f"[{symbol}] Candle {candle_timestamp} already processed, skipping signal generation...")
                        return
                
                # Сохраняем timestamp обработанной свечи
                self.last_processed_candle[symbol] = candle_timestamp
                logger.debug(f"[{symbol}] Processing new candle: {candle_timestamp}")
            else:
                logger.warning(f"[{symbol}] Warning: candle_timestamp is None, proceeding anyway...")
                # Если timestamp None, не сохраняем его, чтобы не блокировать следующие проверки
            
            # Проверяем позицию
            try:
                pos_info = self.bybit.get_position_info(symbol=symbol)
            except Exception as e:
                logger.error(f"Error getting position info for {symbol}: {e}")
                pos_info = None
            
            has_pos = None
            size = 0.0
            entry_price = 0.0
            
            if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                result = pos_info.get("result")
                if result and isinstance(result, dict):
                    list_data = result.get("list", [])
                    if list_data and len(list_data) > 0:
                        p = list_data[0]
                        if p and isinstance(p, dict):
                            size = float(p.get("size", 0))
                            if size > 0:
                                side = p.get("side")
                                has_pos = Bias.LONG if side == "Buy" else Bias.SHORT
                                entry_price = float(p.get("avgPrice", 0))
            elif pos_info is None:
                logger.warning(f"Position info is None for {symbol}")

            # Генерация сигнала
            try:
                signal = strategy.generate_signal(
                    row=row,
                    df=df.iloc[:-1] if len(df) >= 2 else df,  # Используем все данные кроме последней незакрытой свечи
                    has_position=has_pos,
                    current_price=current_price,  # Используем текущую цену из последней свечи
                    leverage=self.settings.leverage
                )
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                return
            
            if not signal:
                logger.warning(f"No signal generated for {symbol}")
                return
            
            # Логируем каждый сигнал (для отладки)
            indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
            confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
            logger.info(f"[{symbol}] Signal: {signal.action.value} | Reason: {signal.reason} | Price: {current_price:.2f} | Confidence: {confidence:.2%} | Candle: {candle_timestamp}")

            # 4. Логируем сигнал в историю
            if signal.action != Action.HOLD:
                self.state.add_signal(
                    symbol=symbol,
                    action=signal.action.value,
                    price=signal.price,
                    confidence=confidence,
                    reason=signal.reason,
                    indicators=indicators_info
                )
                
                # Уведомление о сигнале высокой уверенности
                if confidence > 0.7:
                    await self.notifier.medium(f"🔔 СИГНАЛ {signal.action.value} по {symbol}\nУверенность: {int(confidence*100)}%\nЦена: {signal.price}")

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
            
            # Вычисляем размер позиции: используем минимум из двух вариантов
            # 1. Процент от баланса
            # Получаем баланс
            balance_info = await asyncio.to_thread(self.bybit.get_wallet_balance)
            balance = 0.0
            
            if balance_info and balance_info.get("retCode") == 0:
                result = balance_info.get("result")
                if result and isinstance(result, dict):
                    list_data = result.get("list", [])
                    if list_data and len(list_data) > 0:
                        wallet_item = list_data[0]
                        if wallet_item and isinstance(wallet_item, dict):
                            wallet = wallet_item.get("coin", [])
                            if wallet and isinstance(wallet, list):
                                usdt_coin = next((c for c in wallet if isinstance(c, dict) and c.get("coin") == "USDT"), None)
                                if usdt_coin:
                                    balance_str = usdt_coin.get("walletBalance", "0")
                                    balance = float(balance_str) if balance_str and balance_str != "" else 0.0
            
            if balance <= 0:
                logger.error(f"Cannot get balance or balance is zero for {symbol}")
                return
            
            # РАСЧЕТ 1: margin_pct_balance% от баланса с использованием плеча
            # Маржа = баланс * margin_pct_balance
            # Количество = (маржа * leverage) / цена
            margin_from_percentage = balance * self.settings.risk.margin_pct_balance
            qty_from_percentage = (margin_from_percentage * self.settings.leverage) / signal.price
            
            # РАСЧЕТ 2: Фиксированная сумма
            # Количество = base_order_usd / цена
            qty_from_fixed = self.settings.risk.base_order_usd / signal.price
            
            # Используем минимум из двух вариантов
            total_qty = min(qty_from_percentage, qty_from_fixed)
            used_method = "percentage" if qty_from_percentage < qty_from_fixed else "fixed"
            
            logger.info(
                f"Position size for {symbol}: "
                f"balance=${balance:.2f}, "
                f"percentage_margin=${margin_from_percentage:.2f} ({self.settings.risk.margin_pct_balance*100}%) -> qty={qty_from_percentage:.6f}, "
                f"fixed=${self.settings.risk.base_order_usd:.2f} -> qty={qty_from_fixed:.6f}, "
                f"selected={used_method}, final_qty={total_qty:.6f}, leverage={self.settings.leverage}x"
            )
            
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
            
            if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
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
            if not position_info or not isinstance(position_info, dict):
                return
            
            if not position_info.get("size"):
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
                    
                    if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
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
            
            if not position_info or not isinstance(position_info, dict):
                return
            
            if not position_info.get("size"):
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
                
                if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
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
            
            if not position_info or not isinstance(position_info, dict):
                return
            
            if not position_info.get("size"):
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
                        
                        if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                            await self.notifier.high(
                                f"💰 ЧАСТИЧНОЕ ЗАКРЫТИЕ\n{symbol} | {close_pct*100}%\nПрогресс к TP: {progress_pct*100:.1f}%"
                            )
                        
                        break  # Закрываем только на одном уровне за раз
        
        except Exception as e:
            logger.error(f"Error checking partial close for {symbol}: {e}")
    
    async def handle_position_closed(self, symbol: str, local_pos: TradeRecord):
        """Обрабатывает закрытие позиции, которая была открыта локально, но закрылась на бирже"""
        try:
            logger.info(f"Position {symbol} closed on exchange, updating state...")
            
            # Пытаемся получить информацию о закрытии из истории исполнений
            # Увеличиваем временной диапазон до 1 часа, чтобы найти закрытие
            import time
            from datetime import datetime, timedelta
            
            end_time = int(time.time() * 1000)
            start_time = int((time.time() - 3600) * 1000)  # 1 час назад (было 5 минут)
            
            exit_price = None
            pnl_usd = 0.0
            pnl_pct = 0.0
            
            # Метод 1: Пытаемся получить из закрытых позиций (closed PnL) - самый точный источник
            try:
                closed_pnl = await asyncio.to_thread(
                    self.bybit.get_closed_pnl,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    limit=10
                )
                
                if closed_pnl and isinstance(closed_pnl, dict) and closed_pnl.get("retCode") == 0:
                    result = closed_pnl.get("result")
                    if result and isinstance(result, dict):
                        pnl_list = result.get("list", [])
                        if pnl_list and len(pnl_list) > 0:
                            # Ищем последнюю закрытую позицию для этого символа
                            for pnl_item in pnl_list:
                                if pnl_item and isinstance(pnl_item, dict):
                                    pnl_symbol = pnl_item.get("symbol", "")
                                    pnl_side = pnl_item.get("side", "")
                                    # Проверяем, что это наша позиция (тот же символ и сторона)
                                    if pnl_symbol == symbol and pnl_side == local_pos.side:
                                        # Получаем точные данные из API
                                        avg_exit_price = float(pnl_item.get("avgExitPrice", 0))
                                        closed_pnl_value = float(pnl_item.get("closedPnl", 0))
                                        
                                        if avg_exit_price > 0:
                                            exit_price = avg_exit_price
                                            # Используем closedPnl из API, если доступен
                                            if closed_pnl_value != 0:
                                                pnl_usd = closed_pnl_value
                                                # Рассчитываем процент PnL на основе closedPnl
                                                margin = (local_pos.entry_price * local_pos.qty) / self.settings.leverage
                                                if margin > 0:
                                                    pnl_pct = (pnl_usd / margin) * 100
                                            logger.info(f"Found closed PnL data: exit_price={exit_price:.2f}, pnl_usd={pnl_usd:.2f}, pnl_pct={pnl_pct:.2f}%")
                                            break
            except Exception as e:
                logger.warning(f"Error getting closed PnL for {symbol}: {e}")
            
            # Метод 2: Если не нашли в closed PnL, пытаемся получить из истории исполнений
            try:
                executions = await asyncio.to_thread(
                    self.bybit.get_execution_list,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    limit=50  # Увеличиваем лимит
                )
                
                if executions and isinstance(executions, dict) and executions.get("retCode") == 0:
                    result = executions.get("result")
                    if result and isinstance(result, dict):
                        exec_list = result.get("list", [])
                        if exec_list and len(exec_list) > 0:
                            # Ищем закрывающий ордер (reduceOnly или противоположный side)
                            close_side = "Sell" if local_pos.side == "Buy" else "Buy"
                            for exec_item in exec_list:
                                if exec_item and isinstance(exec_item, dict):
                                    exec_side = exec_item.get("side", "")
                                    # Ищем исполнение противоположного направления или reduceOnly
                                    if exec_side == close_side or exec_item.get("reduceOnly", False):
                                        exit_price = float(exec_item.get("execPrice", 0))
                                        if exit_price > 0:
                                            logger.info(f"Found exit price from execution list: {exit_price}")
                                            break
            except Exception as e:
                logger.warning(f"Error getting execution list for {symbol}: {e}")
            
            # Метод 3: Если не нашли в closed PnL и execution list, пытаемся получить из текущей позиции
            if exit_price is None or exit_price == 0:
                try:
                    # Получаем информацию о текущей позиции (может быть закрыта недавно)
                    pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
                    if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data and len(list_data) > 0:
                                position = list_data[0]
                                if position and isinstance(position, dict):
                                    # Если позиция закрыта (size == 0), используем markPrice
                                    size = float(position.get("size", 0))
                                    if size == 0:
                                        mark_price = float(position.get("markPrice", 0))
                                        if mark_price > 0:
                                            exit_price = mark_price
                                            logger.info(f"Using markPrice as exit price: {exit_price}")
                except Exception as e:
                    logger.warning(f"Error getting position info for closed position {symbol}: {e}")
            
            # Метод 4: Если все еще не нашли, используем текущую цену из свечей
            if exit_price is None or exit_price == 0:
                try:
                    df = self.bybit.get_kline_df(symbol, self.settings.timeframe, limit=1)
                    if not df.empty:
                        exit_price = float(df['close'].iloc[-1])
                        logger.info(f"Using current price from candles as exit price: {exit_price}")
                except Exception as e:
                    logger.warning(f"Error getting current price for {symbol}: {e}")
            
            # Если все методы не сработали, используем entry_price (но это плохо)
            if exit_price is None or exit_price == 0:
                exit_price = local_pos.entry_price
                logger.warning(f"Could not determine exit price for {symbol}, using entry_price: {exit_price}")
            
            # Рассчитываем PnL
            # Используем правильную формулу с учетом плеча
            # PnL% = ((exit_price - entry_price) / entry_price) * leverage * 100 для LONG
            # PnL% = ((entry_price - exit_price) / entry_price) * leverage * 100 для SHORT
            leverage = self.settings.leverage
            
            if local_pos.side == "Buy":
                price_diff_pct = ((exit_price - local_pos.entry_price) / local_pos.entry_price)
                pnl_pct = price_diff_pct * leverage * 100
            else:  # Sell
                price_diff_pct = ((local_pos.entry_price - exit_price) / local_pos.entry_price)
                pnl_pct = price_diff_pct * leverage * 100
            
            # PnL в USD = (процент PnL / 100) * маржа
            # Маржа = entry_price * qty / leverage
            margin = (local_pos.entry_price * local_pos.qty) / leverage
            pnl_usd = (pnl_pct / 100) * margin
            
            logger.info(f"Calculated PnL for {symbol}: exit_price={exit_price:.2f}, pnl_pct={pnl_pct:.2f}%, pnl_usd={pnl_usd:.2f}")
            
            # Обновляем статус сделки
            self.state.update_trade_on_close(symbol, exit_price, pnl_usd, pnl_pct)
            
            # Отправляем уведомление
            pnl_emoji = "✅" if pnl_usd > 0 else "❌"
            reason = "TP" if pnl_usd > 0 else "SL"
            await self.notifier.high(
                f"{pnl_emoji} ПОЗИЦИЯ ЗАКРЫТА ({reason})\n"
                f"{symbol} {local_pos.side}\n"
                f"Вход: ${local_pos.entry_price:.2f}\n"
                f"Выход: ${exit_price:.2f}\n"
                f"PnL: {pnl_usd:+.2f} USD ({pnl_pct:+.2f}%)"
            )
            
            logger.info(f"Position {symbol} closed: PnL={pnl_usd:.2f} USD ({pnl_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error handling closed position for {symbol}: {e}")
            # В случае ошибки пытаемся получить текущую цену и закрыть позицию
            try:
                # Пытаемся получить текущую цену из свечей
                df = self.bybit.get_kline_df(symbol, self.settings.timeframe, limit=1)
                if not df.empty:
                    exit_price = float(df['close'].iloc[-1])
                    # Рассчитываем PnL даже при ошибке
                    if local_pos.side == "Buy":
                        pnl_pct = ((exit_price - local_pos.entry_price) / local_pos.entry_price) * 100
                    else:
                        pnl_pct = ((local_pos.entry_price - exit_price) / local_pos.entry_price) * 100
                    pnl_usd = (pnl_pct / 100) * (local_pos.entry_price * local_pos.qty)
                    self.state.update_trade_on_close(symbol, exit_price, pnl_usd, pnl_pct)
                else:
                    # Если не удалось получить цену, используем entry_price с нулевым PnL
                    self.state.update_trade_on_close(symbol, local_pos.entry_price, 0.0, 0.0)
            except Exception as e2:
                logger.error(f"Error in fallback close handling for {symbol}: {e2}")
                # Последняя попытка - закрываем с entry_price
                try:
                    self.state.update_trade_on_close(symbol, local_pos.entry_price, 0.0, 0.0)
                except:
                    pass
    
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
                    
                    if pos_info and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data and len(list_data) > 0:
                                position = list_data[0]
                                if position and isinstance(position, dict):
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
                                        # Позиции нет на бирже (size == 0), но может быть в локальном состоянии
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos:
                                            # Закрываем локальную позицию
                                            logger.warning(f"Position {symbol} closed on exchange but open locally, closing in state")
                                            await self.handle_position_closed(symbol, local_pos)
                
                except Exception as e:
                    logger.error(f"Error syncing position for {symbol}: {e}")
            
            logger.info("Position sync completed")
        
        except Exception as e:
            logger.error(f"Error during position sync: {e}")
