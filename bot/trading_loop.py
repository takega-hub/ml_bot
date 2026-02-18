import time
import asyncio
import logging
import math
import pandas as pd
from typing import List, Dict, Optional, Union, TYPE_CHECKING
from bot.config import AppSettings
from bot.state import BotState, TradeRecord
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy, build_ml_signals
from bot.strategy import Action, Signal, Bias
from bot.notification_manager import NotificationManager, NotificationLevel

if TYPE_CHECKING:
    from bot.ml.mtf_strategy import MultiTimeframeMLStrategy

# Импортируем исключение для обработки ошибки недостатка средств
try:
    from pybit.exceptions import InvalidRequestError
except ImportError:
    InvalidRequestError = Exception  # Fallback если pybit не установлен

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")
signal_logger = logging.getLogger("signals")

class TradingLoop:
    def __init__(self, settings: AppSettings, state: BotState, bybit: BybitClient, tg_bot=None):
        self.settings = settings
        self.state = state
        self.bybit = bybit
        self.tg_bot = tg_bot
        self.notifier = NotificationManager(tg_bot, settings)
        self.strategies: Dict[str, Union[MLStrategy, 'MultiTimeframeMLStrategy']] = {}
        # Отслеживаем последнюю обработанную свечу для каждого символа
        self.last_processed_candle: Dict[str, Optional[pd.Timestamp]] = {}
        # Кэш сигнала BTCUSDT для проверки направления других пар (обновляется каждые 5 минут)
        self._btc_signal_cache: Optional[Dict] = None
        self._btc_signal_cache_time: Optional[float] = None
        
        # Валидация моделей при старте
        if self.settings.ml_strategy.use_mtf_strategy:
            self._validate_mtf_models()
    
    def _validate_mtf_models(self):
        """Проверяет наличие MTF моделей для активных символов при старте"""
        from bot.ml.model_selector import select_best_models
        
        logger.info("🔍 Валидация MTF моделей для активных символов...")
        missing_models = []
        
        for symbol in self.state.active_symbols:
            model_1h, model_15m, model_info = select_best_models(symbol=symbol)
            
            if not model_1h or not model_15m:
                missing_models.append(symbol)
                logger.warning(f"[{symbol}] ⚠️ MTF модели не найдены (1h: {model_1h is not None}, 15m: {model_15m is not None})")
            else:
                logger.info(f"[{symbol}] ✅ MTF модели найдены (source: {model_info.get('source', 'unknown')})")
        
        if missing_models:
            logger.warning(f"⚠️ MTF стратегия включена, но модели не найдены для: {', '.join(missing_models)}")
            logger.warning("Бот будет использовать обычную стратегию для этих символов")
        else:
            logger.info("✅ Все активные символы имеют MTF модели")

    async def run(self):
        logger.info("Starting Trading Loop...")
        
        # Устанавливаем is_running = True при запуске (если еще не установлено)
        if not self.state.is_running:
            logger.info("Setting bot state to running...")
            self.state.set_running(True)
        
        # Синхронизируем позиции с биржей при старте
        await self.sync_positions_with_exchange()
        
        # Запускаем оба цикла параллельно с обработкой ошибок
        logger.info("Trading Loop: About to start both loops in parallel...")
        try:
            logger.info("Trading Loop: Starting asyncio.gather...")
            results = await asyncio.gather(
                self._signal_processing_loop(),
                self._position_monitoring_loop(),
                return_exceptions=True  # Не останавливаемся при ошибке в одном из циклов
            )
            logger.info(f"Trading Loop: asyncio.gather completed with results: {results}")
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
            raise
    
    def _get_seconds_until_next_candle_close(self, timeframe: str) -> float:
        """
        Вычисляет количество секунд до закрытия следующей свечи.
        
        Args:
            timeframe: Таймфрейм ('15m', '1h', '4h', и т.д.)
        
        Returns:
            Количество секунд до закрытия следующей свечи
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # Парсим таймфрейм
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 24 * 60
        else:
            # Пытаемся распарсить как число (минуты)
            try:
                minutes = int(timeframe)
            except:
                minutes = 15  # По умолчанию 15 минут
        
        # Вычисляем время закрытия следующей свечи
        # Для 15m: закрытие в :00, :15, :30, :45
        # Для 1h: закрытие в :00 каждого часа
        # Для 4h: закрытие в 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        
        if minutes < 60:
            # Минутные свечи: округляем до ближайшего кратного minutes
            current_minute = now.minute
            next_close_minute = ((current_minute // minutes) + 1) * minutes
            if next_close_minute >= 60:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = now.replace(minute=next_close_minute, second=0, microsecond=0)
        elif minutes == 60:
            # Часовые свечи: закрытие в :00 каждого часа
            if now.minute == 0 and now.second < 5:
                # Свеча только что закрылась, следующая через час
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            # Многочасовые свечи (4h, 1d и т.д.)
            hours = minutes // 60
            current_hour = now.hour
            next_close_hour = ((current_hour // hours) + 1) * hours
            if next_close_hour >= 24:
                next_close = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                next_close = now.replace(hour=next_close_hour, minute=0, second=0, microsecond=0)
        
        seconds_until_close = (next_close - now).total_seconds()
        return max(0, seconds_until_close)
    
    def _get_seconds_since_last_candle_close(self, timeframe: str) -> float:
        """
        Вычисляет количество секунд с момента закрытия последней свечи.
        
        Args:
            timeframe: Таймфрейм ('15m', '1h', '4h', и т.д.)
        
        Returns:
            Количество секунд с момента закрытия последней свечи
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # Парсим таймфрейм
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 24 * 60
        else:
            try:
                minutes = int(timeframe)
            except:
                minutes = 15
        
        # Вычисляем время закрытия последней свечи
        if minutes < 60:
            current_minute = now.minute
            last_close_minute = (current_minute // minutes) * minutes
            last_close = now.replace(minute=last_close_minute, second=0, microsecond=0)
        elif minutes == 60:
            last_close = now.replace(minute=0, second=0, microsecond=0)
        else:
            hours = minutes // 60
            current_hour = now.hour
            last_close_hour = (current_hour // hours) * hours
            last_close = now.replace(hour=last_close_hour, minute=0, second=0, microsecond=0)
        
        seconds_since_close = (now - last_close).total_seconds()
        return max(0, seconds_since_close)

    async def _signal_processing_loop(self):
        """Основной цикл обработки сигналов с оптимизацией для немедленной обработки после закрытия свечи"""
        logger.info("Starting Signal Processing Loop...")
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.debug(f"Signal Processing Loop: Iteration {iteration}, is_running={self.state.is_running}")
                
                if not self.state.is_running:
                    logger.debug("Signal Processing Loop: Bot not running, sleeping...")
                    await asyncio.sleep(10)
                    continue

                logger.info(f"🔄 Signal Processing Loop: Processing {len(self.state.active_symbols)} symbols...")
                for symbol in self.state.active_symbols:
                    logger.info(f"🎯 Signal Processing Loop: Starting to process {symbol}")
                    await self.process_symbol(symbol)
                    logger.info(f"✅ Signal Processing Loop: Completed processing {symbol}")
                    # Добавляем задержку между символами для снижения нагрузки на API
                    if len(self.state.active_symbols) > 1:
                        await asyncio.sleep(2)
                
                # УМНАЯ ПАУЗА: проверяем, когда закроется следующая свеча
                # Если свеча только что закрылась (в пределах последних 30 секунд), проверяем снова через короткое время
                seconds_since_close = self._get_seconds_since_last_candle_close(self.settings.timeframe)
                
                if seconds_since_close <= 30:
                    # Свеча только что закрылась, проверяем снова через 10 секунд для надежности
                    sleep_time = 10
                    logger.info(f"✅ Signal Processing Loop: Candle closed {seconds_since_close:.1f}s ago, checking again in {sleep_time}s...")
                else:
                    # Обычная пауза, но не больше времени до следующего закрытия
                    seconds_until_close = self._get_seconds_until_next_candle_close(self.settings.timeframe)
                    # Используем минимум из обычной паузы и времени до закрытия (но не меньше 10 секунд)
                    sleep_time = min(self.settings.live_poll_seconds, max(10, seconds_until_close - 5))
                    logger.info(f"✅ Signal Processing Loop: Completed iteration {iteration}, sleeping for {sleep_time}s (next candle closes in {seconds_until_close:.1f}s)...")
                
                await asyncio.sleep(sleep_time)
                logger.debug(f"Signal Processing Loop: Woke up from sleep, starting next iteration...")
            except Exception as e:
                logger.error(f"[trading_loop] Error in signal processing loop: {e}")
                await asyncio.sleep(30)
    
    async def _position_monitoring_loop(self):
        """Цикл мониторинга открытых позиций для breakeven и trailing stop"""
        logger.info("Starting Position Monitoring Loop...")
        try:
            logger.info("Position Monitoring Loop: About to sleep for 10 seconds...")
            await asyncio.sleep(10)  # Даем время запуститься основному циклу
            logger.info("Position Monitoring Loop: Sleep completed, continuing...")
        except Exception as e:
            logger.error(f"Error in position monitoring loop initial sleep: {e}", exc_info=True)
            raise
        logger.info("Position Monitoring Loop: Initial delay completed, starting main loop...")
        
        cycle_count = 0
        while True:
            try:
                if not self.state.is_running:
                    logger.debug("Bot is not running, waiting...")
                    await asyncio.sleep(10)
                    continue
                
                cycle_count += 1
                # Логируем каждые 10 циклов (примерно каждые 4 минуты), чтобы видеть, что цикл работает
                if cycle_count % 10 == 0:
                    logger.info(f"📊 Position Monitoring Loop: Cycle {cycle_count} (checking positions every 25s)")
                
                # ОПТИМИЗАЦИЯ: получаем ВСЕ позиции одним запросом вместо отдельных для каждого символа
                # Это значительно снижает количество API запросов и предотвращает rate limit ошибки
                try:
                    logger.debug("Fetching all positions from exchange...")
                    # Добавляем таймаут для предотвращения зависания
                    all_positions = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.bybit.get_position_info,
                            settle_coin="USDT"  # Получаем все USDT позиции одним запросом
                        ),
                        timeout=30.0  # Таймаут 30 секунд
                    )
                    logger.debug(f"Received positions response: retCode={all_positions.get('retCode') if all_positions else 'None'}")
                    
                    if all_positions and all_positions.get("retCode") == 0:
                        result = all_positions.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            
                            # Логируем начало цикла мониторинга позиций
                            open_count = sum(1 for pos in list_data if pos and isinstance(pos, dict) and float(pos.get("size", 0)) > 0)
                            if open_count > 0:
                                logger.info(f"📊 Position Monitoring: Checking {open_count} open position(s)...")
                            
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
                                            logger.debug(f"[{symbol}] Calling update_breakeven_stop for position size={size}")
                                            await self.update_breakeven_stop(symbol, position)
                                            
                                            # Обновляем trailing stop
                                            await self.update_trailing_stop(symbol, position)

                                            # Проверяем Time Stop
                                            await self.check_time_stop(symbol, position)

                                            # Проверяем Early Exit
                                            await self.check_early_exit(symbol, position)
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
                
                except asyncio.TimeoutError:
                    logger.error("Timeout while fetching positions from exchange (30s)")
                except Exception as e:
                    logger.error(f"Error getting all positions: {e}", exc_info=True)
                
                # Проверяем позиции каждые 25 секунд (увеличено с 15 для снижения нагрузки на API)
                logger.debug("Position monitoring cycle completed, sleeping for 25 seconds...")
                await asyncio.sleep(25)
                logger.debug("Position Monitoring Loop: Woke up from sleep, starting next cycle...")
            
            except Exception as e:
                logger.error(f"[trading_loop] Error in position monitoring loop: {e}")
                await asyncio.sleep(30)

    async def process_symbol(self, symbol: str):
        try:
            logger.info(f"[{symbol}] 🚀 START process_symbol()")
            
            # 0. Проверяем cooldown
            # КРИТИЧНО: is_symbol_in_cooldown() может вызывать save() (запись в файл)
            # Оборачиваем в to_thread() чтобы не блокировать event loop
            # Добавляем таймаут, чтобы избежать зависания
            logger.info(f"[{symbol}] Checking cooldown...")
            try:
                in_cooldown = await asyncio.wait_for(
                    asyncio.to_thread(self.state.is_symbol_in_cooldown, symbol),
                    timeout=5.0  # Таймаут 5 секунд
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{symbol}] Cooldown check timed out, assuming no cooldown")
                in_cooldown = False
            
            if in_cooldown:
                logger.info(f"[{symbol}] In cooldown, returning")
                return
            logger.info(f"[{symbol}] No cooldown, continuing...")
            
            # 1. Получаем данные (асинхронно, чтобы не блокировать event loop)
            # Используем кэширование для 15m данных, чтобы не загружать все 500 свечей каждый раз
            use_mtf = self.settings.ml_strategy.use_mtf_strategy
            required_limit = 500 if use_mtf else 200  # Для MTF запрашиваем больше данных
            
            # Загружаем кэшированные 15m данные
            df = await asyncio.to_thread(self._load_cached_15m_data, symbol)
            
            # Проверяем актуальность данных
            needs_update = False
            if df is None or df.empty:
                logger.info(f"[{symbol}] ⚠️ No cached 15m data found, fetching from exchange...")
                needs_update = True
            else:
                # Проверяем, актуальны ли данные
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                    last_candle_time = df.index[-1]
                    current_time = pd.Timestamp.now()
                    
                    # Для 15m свечей вычисляем, когда должна была закрыться следующая свеча после последней в кэше
                    # 15-минутные свечи закрываются в :00, :15, :30, :45
                    last_minute = last_candle_time.minute
                    next_close_minute = ((last_minute // 15) + 1) * 15
                    if next_close_minute >= 60:
                        next_close_time = last_candle_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
                    else:
                        next_close_time = last_candle_time.replace(minute=next_close_minute, second=0, microsecond=0)
                    
                    # Если время закрытия следующей свечи уже прошло, значит должна быть новая свеча
                    should_have_new_candle = current_time >= next_close_time
                    minutes_since_last = (current_time - last_candle_time).total_seconds() / 60
                    
                    if should_have_new_candle or minutes_since_last > 20 or len(df) < required_limit:
                        if should_have_new_candle:
                            logger.info(f"[{symbol}] ⚠️ Cached 15m data is outdated: next candle should have closed at {next_close_time}, but it's {current_time} (last candle: {last_candle_time}), updating...")
                        else:
                            logger.info(f"[{symbol}] ⚠️ Cached 15m data is outdated or insufficient (last candle: {last_candle_time}, {minutes_since_last:.1f}min ago, have {len(df)} candles, need {required_limit}), updating...")
                        needs_update = True
                    else:
                        logger.debug(f"[{symbol}] ✅ Cached 15m data is fresh (last candle: {last_candle_time}, {minutes_since_last:.1f}min ago, next close: {next_close_time}, {len(df)} candles)")
                else:
                    logger.warning(f"[{symbol}] ⚠️ Could not check cache freshness, updating...")
                    needs_update = True
            
            # Обновляем кэш если нужно
            if needs_update:
                # Подгружаем только новые данные
                df = await asyncio.to_thread(self._fetch_and_cache_15m_data, symbol, df, required_limit)
                if df is not None and not df.empty:
                    logger.info(f"[{symbol}] ✅ Updated cache with {len(df)} 15m candles from exchange")
                else:
                    logger.warning(f"[{symbol}] ⚠️ Failed to fetch 15m data from exchange")
                    return
            else:
                logger.info(f"[{symbol}] ✅ Using cached 15m data ({len(df)} candles)")
            
            if df.empty:
                logger.warning(f"[{symbol}] ⚠️ No data available")
                return
            
            # Для MTF стратегии загружаем кэшированные 1h данные из ml_data
            # Проверяем актуальность и обновляем при необходимости
            df_1h_cached = None
            if use_mtf:
                df_1h_cached = await asyncio.to_thread(self._load_cached_1h_data, symbol)
                
                # Проверяем актуальность данных
                needs_update = False
                if df_1h_cached is None or df_1h_cached.empty:
                    logger.info(f"[{symbol}] ⚠️ No cached 1h data found, fetching from exchange...")
                    needs_update = True
                else:
                    # Проверяем, актуальны ли данные (последняя свеча не старше 2 часов для 1h данных)
                    if isinstance(df_1h_cached.index, pd.DatetimeIndex) and len(df_1h_cached) > 0:
                        last_candle_time = df_1h_cached.index[-1]
                        current_time = pd.Timestamp.now()
                        hours_since_last = (current_time - last_candle_time).total_seconds() / 3600
                        
                        # Если последняя свеча старше 2 часов, обновляем кэш
                        if hours_since_last > 2:
                            logger.info(f"[{symbol}] ⚠️ Cached 1h data is outdated (last candle: {last_candle_time}, {hours_since_last:.1f}h ago), updating...")
                            needs_update = True
                        else:
                            logger.debug(f"[{symbol}] ✅ Cached 1h data is fresh (last candle: {last_candle_time}, {hours_since_last:.1f}h ago)")
                    else:
                        logger.warning(f"[{symbol}] ⚠️ Could not check cache freshness, updating...")
                        needs_update = True
                
                # Обновляем кэш если нужно
                if needs_update:
                    # Передаем существующий кэш для подгрузки только новых данных
                    df_1h_cached = await asyncio.to_thread(self._fetch_and_cache_1h_data, symbol, df_1h_cached)
                    if df_1h_cached is not None and not df_1h_cached.empty:
                        logger.info(f"[{symbol}] ✅ Updated cache with {len(df_1h_cached)} 1h candles from exchange")
                    else:
                        logger.warning(f"[{symbol}] Failed to fetch 1h data, will aggregate from 15m")
                else:
                    logger.info(f"[{symbol}] ✅ Using cached 1h data ({len(df_1h_cached)} candles)")

            # 2. Инициализируем стратегию если нужно
            # Проверяем, нужно ли переинициализировать стратегию из-за изменения настроек MTF
            current_strategy = self.strategies.get(symbol)
            use_mtf = self.settings.ml_strategy.use_mtf_strategy
            
            if current_strategy is not None:
                # Проверяем, соответствует ли текущая стратегия настройкам
                is_mtf_strategy = hasattr(current_strategy, 'predict_combined')
                if is_mtf_strategy != use_mtf:
                    logger.info(f"[{symbol}] Strategy type mismatch: current={'MTF' if is_mtf_strategy else 'Single'}, required={'MTF' if use_mtf else 'Single'}")
                    logger.info(f"[{symbol}] Removing old strategy to reinitialize with new settings")
                    del self.strategies[symbol]
                    current_strategy = None  # Помечаем, что нужно переинициализировать
            
            if symbol not in self.strategies:
                from pathlib import Path
                
                # Проверяем, включена ли MTF стратегия и есть ли обе модели
                use_mtf = self.settings.ml_strategy.use_mtf_strategy
                logger.info(f"[{symbol}] MTF strategy setting: use_mtf_strategy={use_mtf}")
                
                if use_mtf:
                    # Используем комбинированную MTF стратегию
                    from bot.ml.mtf_strategy import MultiTimeframeMLStrategy
                    from bot.ml.model_selector import select_best_models
                    
                    logger.info(f"[{symbol}] Attempting to load MTF strategy...")
                    # Выбираем лучшие модели автоматически
                    model_1h, model_15m, model_info = select_best_models(
                        symbol=symbol,
                        use_best_from_comparison=True,
                    )
                    
                    logger.info(f"[{symbol}] MTF model selection result: model_1h={model_1h}, model_15m={model_15m}, source={model_info.get('source', 'unknown')}")
                    
                    if model_1h and model_15m:
                        # Используем параметры из best_strategies.json, если доступны
                        # Если параметр None, используем значение из настроек, если и оно None - используем значения по умолчанию
                        confidence_threshold_1h = model_info.get('confidence_threshold_1h')
                        if confidence_threshold_1h is None:
                            confidence_threshold_1h = self.settings.ml_strategy.mtf_confidence_threshold_1h
                        if confidence_threshold_1h is None:
                            confidence_threshold_1h = 0.50  # Значение по умолчанию
                        
                        confidence_threshold_15m = model_info.get('confidence_threshold_15m')
                        if confidence_threshold_15m is None:
                            confidence_threshold_15m = self.settings.ml_strategy.mtf_confidence_threshold_15m
                        if confidence_threshold_15m is None:
                            confidence_threshold_15m = 0.35  # Значение по умолчанию
                        
                        alignment_mode = model_info.get('alignment_mode')
                        if alignment_mode is None:
                            alignment_mode = self.settings.ml_strategy.mtf_alignment_mode
                        if alignment_mode is None:
                            alignment_mode = "strict"  # Значение по умолчанию
                        
                        require_alignment = model_info.get('require_alignment')
                        if require_alignment is None:
                            require_alignment = self.settings.ml_strategy.mtf_require_alignment
                        if require_alignment is None:
                            require_alignment = True  # Значение по умолчанию
                        
                        logger.info(f"[{symbol}] 🔄 Loading MTF strategy:")
                        logger.info(f"  Source: {model_info.get('source', 'unknown')}")
                        logger.info(f"  1h model: {Path(model_1h).name}")
                        logger.info(f"  15m model: {Path(model_15m).name}")
                        logger.info(f"  Parameters: 1h_threshold={confidence_threshold_1h}, 15m_threshold={confidence_threshold_15m}, alignment_mode={alignment_mode}, require_alignment={require_alignment}")
                        if model_info.get('metrics'):
                            metrics = model_info['metrics']
                            logger.info(f"  Expected metrics: PnL={metrics.get('total_pnl_pct', 0):.2f}%, "
                                      f"WR={metrics.get('win_rate', 0):.1f}%, "
                                      f"PF={metrics.get('profit_factor', 0):.2f}")
                        
                        self.strategies[symbol] = MultiTimeframeMLStrategy(
                            model_1h_path=model_1h,
                            model_15m_path=model_15m,
                            confidence_threshold_1h=confidence_threshold_1h,
                            confidence_threshold_15m=confidence_threshold_15m,
                            require_alignment=require_alignment,
                            alignment_mode=alignment_mode,
                        )
                        logger.info(f"[{symbol}] ✅ MTF strategy loaded successfully")
                    else:
                        # Нет обеих моделей - используем обычную стратегию
                        logger.warning(f"[{symbol}] MTF strategy enabled but models not found:")
                        logger.warning(f"  1h model: {model_1h}, 15m model: {model_15m}")
                        logger.warning(f"[{symbol}] Falling back to single timeframe strategy")
                        use_mtf = False
                
                if not use_mtf:
                    # Используем обычную стратегию (15m или 1h)
                    model_path = self.state.symbol_models.get(symbol)
                    # Если путь не задан, используем автопоиск из конфига (реализован в _auto_find_ml_model)
                    if not model_path:
                        # Пытаемся найти модель в папке ml_models
                        models = list(Path("ml_models").glob(f"*_{symbol}_*.pkl"))
                        if models:
                            model_path = str(models[0])
                            self.state.symbol_models[symbol] = model_path
                    
                    if model_path:
                        logger.info(f"[{symbol}] 🔄 Loading model: {model_path}")
                        self.strategies[symbol] = MLStrategy(
                            model_path=model_path,
                            confidence_threshold=self.settings.ml_strategy.confidence_threshold,
                            min_signal_strength=self.settings.ml_strategy.min_signal_strength
                        )
                        logger.info(f"[{symbol}] ✅ Model loaded successfully (threshold: {self.settings.ml_strategy.confidence_threshold}, min_strength: {self.settings.ml_strategy.min_signal_strength})")
                    else:
                        logger.warning(f"No model found for {symbol}, skipping...")
                        return

            # 3. Генерируем сигнал
            strategy = self.strategies[symbol]
            # Определяем, какая свеча закрыта и может быть использована для предсказания
            # ВАЖНО: Используем последнюю закрытую свечу (как в тесте), а не предпоследнюю
            # Последняя свеча считается закрытой, если прошло достаточно времени с момента её закрытия
            # Для 15m свечей: если прошло > 1 минуты с момента закрытия, свеча считается закрытой
            
            # Получаем timestamp последней свечи
            last_row = df.iloc[-1]
            last_timestamp = last_row.get('timestamp') if 'timestamp' in last_row else None
            if last_timestamp is None:
                last_timestamp = df.index[-1] if len(df.index) > 0 else None
            
            # ВАЖНО: В тесте всегда используется последняя свеча (которая уже закрыта в исторических данных)
            # В реальном боте мы должны использовать последнюю закрытую свечу
            # Для максимального соответствия тесту, используем последнюю свечу из данных
            # (она считается закрытой, так как мы обрабатываем её после закрытия)
            
            # В тесте: row = df_with_features.iloc[idx] - текущая свеча
            # В реальном боте: row = df.iloc[-1] - последняя свеча (аналог текущей в тесте)
            if len(df) >= 1:
                # Используем последнюю свечу (как в тесте используется текущая свеча)
                row = df.iloc[-1]
                current_price = row['close']
                candle_timestamp = last_timestamp
                use_last_candle = True  # Всегда используем последнюю свечу (как в тесте)
                logger.debug(f"[{symbol}] Using last candle for prediction (as in backtest)")
            else:
                row = df.iloc[-1]
                current_price = row['close']
                candle_timestamp = last_timestamp
                use_last_candle = True
                logger.debug(f"[{symbol}] Using only available candle for prediction")
            
            # Логируем время закрытия свечи и задержку обработки
            if candle_timestamp is not None:
                try:
                    from datetime import datetime
                    if isinstance(candle_timestamp, pd.Timestamp):
                        candle_close_time = candle_timestamp
                    elif isinstance(candle_timestamp, (int, float)):
                        candle_close_time = pd.Timestamp(candle_timestamp, unit='ms')
                    else:
                        candle_close_time = pd.Timestamp(candle_timestamp)
                    
                    now = pd.Timestamp.now()
                    delay_seconds = (now - candle_close_time).total_seconds()
                    delay_minutes = delay_seconds / 60
                    
                    logger.info(
                        f"[{symbol}] 📊 Candle info: closed at {candle_close_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"processing delay: {delay_seconds:.1f}s ({delay_minutes:.2f} min)"
                    )
                except Exception as e:
                    logger.debug(f"[{symbol}] Could not calculate candle delay: {e}")
            
            # Проверяем, не обрабатывали ли мы уже эту свечу
            # ВАЖНО: Проверяем только если timestamp валиден
            # Это предотвращает генерацию одинаковых сигналов для одной и той же закрытой свечи
            if candle_timestamp is not None:
                if symbol in self.last_processed_candle:
                    last_timestamp = self.last_processed_candle[symbol]
                    if last_timestamp is not None and last_timestamp == candle_timestamp:
                        # Эта свеча уже была обработана, пропускаем
                        logger.info(f"[{symbol}] ⏭️ Candle already processed: {candle_timestamp}, skipping signal generation")
                        logger.debug(f"[{symbol}] Last processed: {last_timestamp}, Current: {candle_timestamp}")
                        # Логируем, была ли свеча обработана с сигналом или без
                        logger.debug(f"[{symbol}] This candle was already processed in a previous iteration")
                        return
                
                # ВАЖНО: НЕ сохраняем timestamp здесь, а только после успешной обработки сигнала
                # Это позволит повторить обработку при ошибке
                logger.debug(f"[{symbol}] 📝 New candle detected: {candle_timestamp} (will save after successful processing)")
            else:
                logger.warning(f"[{symbol}] ⚠️ Warning: candle_timestamp is None, proceeding anyway...")
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

            local_pos = self.state.get_open_position(symbol)

            # Генерация сигнала
            # КРИТИЧНО: generate_signal() выполняет долгие синхронные операции (feature engineering, model.predict)
            # Оборачиваем в to_thread() чтобы не блокировать event loop
            try:
                logger.info(f"[{symbol}] 🔄 Calling strategy.generate_signal() in thread...")
                
                # Подготавливаем данные для стратегии
                # ВАЖНО: В тесте используется df_window = df_with_features.iloc[:idx+1] - ВСЕ данные до текущего момента ВКЛЮЧИТЕЛЬНО
                # В реальном боте: df_for_strategy = df - ВСЕ данные, включая последнюю свечу (аналог [:idx+1] в тесте)
                df_for_strategy = df  # Используем все данные, включая последнюю свечу (как в тесте)
                logger.debug(f"[{symbol}] Using all data including last candle for strategy (as in backtest)")
                
                # Для MTF стратегии передаем df_15m (текущие данные) и df_1h (из кэша или None)
                # Для обычной стратегии передаем df как обычно
                if hasattr(strategy, 'predict_combined'):
                    # Это MTF стратегия - передаем df_15m и df_1h
                    # Логируем информацию о данных для диагностики
                    if df_1h_cached is not None and not df_1h_cached.empty:
                        logger.debug(f"[{symbol}] MTF: Using cached 1h data ({len(df_1h_cached)} candles)")
                        if isinstance(df_1h_cached.index, pd.DatetimeIndex) and len(df_1h_cached) > 0:
                            logger.debug(f"[{symbol}] MTF: 1h data range: {df_1h_cached.index[0]} to {df_1h_cached.index[-1]}")
                    else:
                        logger.debug(f"[{symbol}] MTF: No cached 1h data, will aggregate from 15m")
                    
                    logger.debug(f"[{symbol}] MTF: 15m data: {len(df_for_strategy)} candles")
                    
                    signal = await asyncio.to_thread(
                        strategy.generate_signal,
                        row=row,
                        df_15m=df_for_strategy,  # 15m данные
                        df_1h=df_1h_cached,  # 1h данные из кэша (если есть) или None (будет агрегировано)
                        has_position=has_pos,
                        current_price=current_price,
                        leverage=self.settings.leverage,
                        target_profit_pct_margin=self.settings.ml_strategy.target_profit_pct_margin,
                        max_loss_pct_margin=self.settings.ml_strategy.max_loss_pct_margin,
                        stop_loss_pct=self.settings.risk.stop_loss_pct,
                        take_profit_pct=self.settings.risk.take_profit_pct,
                    )
                else:
                    # Обычная стратегия - передаем df как обычно
                    signal = await asyncio.to_thread(
                        strategy.generate_signal,
                        row=row,
                        df=df_for_strategy,
                        has_position=has_pos,
                        current_price=current_price,
                        leverage=self.settings.leverage,
                        stop_loss_pct=self.settings.risk.stop_loss_pct,
                        take_profit_pct=self.settings.risk.take_profit_pct,
                    )
                logger.info(f"[{symbol}] ✅ strategy.generate_signal() completed")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
            
            if not signal:
                logger.warning(f"No signal generated for {symbol}")
                return
            
            # Сохраняем время получения сигнала для проверки "свежести"
            signal_received_time = pd.Timestamp.now()
            
            # Логируем каждый сигнал (для отладки)
            indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
            
            # Сохраняем время получения сигнала в indicators_info для использования в execute_trade
            if indicators_info is None:
                indicators_info = {}
            indicators_info['signal_received_time'] = signal_received_time.isoformat()
            signal.indicators_info = indicators_info
            
            confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
            # Для MTF стратегии показываем детальную информацию
            if isinstance(indicators_info, dict) and indicators_info.get('strategy') == 'MTF_ML':
                mtf_reason = indicators_info.get('reason', 'unknown')
                conf_1h = indicators_info.get('1h_conf', 0)
                conf_15m = indicators_info.get('15m_conf', 0)
                pred_1h = indicators_info.get('1h_pred', 0)
                pred_15m = indicators_info.get('15m_pred', 0)
                logger.info(
                    f"[{symbol}] Signal: {signal.action.value} | Reason: {signal.reason} | Price: {current_price:.2f} | "
                    f"Confidence: {confidence:.2%} | "
                    f"1h: {pred_1h}({conf_1h:.2%}) | 15m: {pred_15m}({conf_15m:.2%}) | "
                    f"Candle: {candle_timestamp}"
                )
            else:
                logger.info(f"[{symbol}] Signal: {signal.action.value} | Reason: {signal.reason} | Price: {current_price:.2f} | Confidence: {confidence:.2%} | Candle: {candle_timestamp}")
            
            # Log ALL non-HOLD signals to signals.log
            if signal.action != Action.HOLD:
                signal_logger.info(f"SIGNAL GEN: {symbol} {signal.action.value} Conf={confidence:.2f} Price={current_price:.2f} Reason={signal.reason}")
            
            logger.info(f"[{symbol}] ⏭️ Signal generated at {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, continuing processing...")

            # 4. Логируем сигнал в историю (только если уверенность >= reverse_min_confidence)
            # Это гарантирует, что в истории отображаются только сигналы с достаточной уверенностью
            min_confidence_for_history = self.settings.risk.reverse_min_confidence
            if signal.action != Action.HOLD:
                if confidence >= min_confidence_for_history:
                    logger.info(f"[{symbol}] 📝 Adding signal to history (confidence {confidence:.2%} >= {min_confidence_for_history:.2%})...")
                    self.state.add_signal(
                        symbol=symbol,
                        action=signal.action.value,
                        price=signal.price,
                        confidence=confidence,
                        reason=signal.reason,
                        indicators=indicators_info
                    )
                    logger.info(f"[{symbol}] ✅ Signal added to history, checking notification...")
                    
                    # Уведомление о сигнале высокой уверенности
                    if confidence > 0.7:
                        logger.info(f"[{symbol}] 📢 Sending notification...")
                        await self.notifier.medium(f"🔔 СИГНАЛ {signal.action.value} по {symbol}\nУверенность: {int(confidence*100)}%\nЦена: {signal.price}")
                        logger.info(f"[{symbol}] ✅ Notification sent")
                else:
                    logger.debug(f"[{symbol}] ⏭️ Signal skipped from history: confidence {confidence:.2%} < {min_confidence_for_history:.2%}")
            
            logger.info(f"[{symbol}] ✅ Signal processing completed, returning from process_symbol")

            # 5. Исполнение сделок
            # ВАЖНО: Проверяем уверенность перед открытием позиции
            # Для MTF стратегии используем специфичные пороги, для обычной - общий порог
            strategy = self.strategies.get(symbol)
            is_mtf_strategy = strategy and hasattr(strategy, 'predict_combined')
            
            if is_mtf_strategy:
                # Для MTF стратегии используем настройку min_confidence_for_trade
                # MTF уже проверила уверенность внутри, но для открытия сделки используем настройку
                mtf_threshold_1h = getattr(strategy, 'confidence_threshold_1h', 0.50)
                mtf_threshold_15m = getattr(strategy, 'confidence_threshold_15m', 0.35)
                # Используем настройку из конфигурации
                min_confidence_for_trade = self.settings.ml_strategy.min_confidence_for_trade
                logger.debug(f"[{symbol}] MTF strategy: using threshold {min_confidence_for_trade:.2%} (1h: {mtf_threshold_1h:.2%}, 15m: {mtf_threshold_15m:.2%})")
            else:
                # Для обычной стратегии используем настройку min_confidence_for_trade
                min_confidence_for_trade = self.settings.ml_strategy.min_confidence_for_trade
                logger.debug(f"[{symbol}] Single strategy: using threshold {min_confidence_for_trade:.2%}")
            
            if signal.action in (Action.LONG, Action.SHORT):
                # Проверяем уверенность перед открытием позиции
                if confidence < min_confidence_for_trade:
                    logger.info(
                        f"[{symbol}] ⏭️ Signal rejected for trade: confidence {confidence:.2%} < "
                        f"threshold {min_confidence_for_trade:.2%}"
                    )
                    # Сохраняем timestamp даже при отклонении сигнала, чтобы не обрабатывать эту свечу повторно
                    if candle_timestamp is not None:
                        self.last_processed_candle[symbol] = candle_timestamp
                        logger.debug(f"[{symbol}] ✅ Candle timestamp saved after signal rejection: {candle_timestamp}")
                    return  # Не открываем позицию, если уверенность ниже порога
                
                # КРИТИЧНО: Проверяем "свежесть" сигнала - открываем сделки только по свежим сигналам (не старше 15 минут)
                signal_age_seconds = (pd.Timestamp.now() - signal_received_time).total_seconds()
                signal_age_minutes = signal_age_seconds / 60
                max_signal_age_minutes = 15  # Максимальный возраст сигнала для открытия сделки
                
                if signal_age_minutes > max_signal_age_minutes:
                    logger.warning(
                        f"[{symbol}] ⏭️ Signal rejected: too old ({signal_age_minutes:.1f} minutes > {max_signal_age_minutes} minutes). "
                        f"Signal received at {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    return  # Не открываем позицию по устаревшему сигналу
                
                logger.info(
                    f"[{symbol}] ✅ Signal is fresh: {signal_age_minutes:.1f} minutes old (max: {max_signal_age_minutes} minutes)"
                )
                
                signal_side = Bias.LONG if signal.action == Action.LONG else Bias.SHORT
                
                # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ для диагностики
                indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
                signal_tp = signal.take_profit or indicators_info.get('take_profit')
                signal_sl = signal.stop_loss or indicators_info.get('stop_loss')
                tp_str = f"{signal_tp:.2f}" if signal_tp else "None"
                sl_str = f"{signal_sl:.2f}" if signal_sl else "None"
                logger.info(
                    f"[{symbol}] 🔍 TRADE DECISION: action={signal.action.value}, "
                    f"has_pos={has_pos}, local_pos={local_pos is not None}, "
                    f"signal_side={signal_side}, confidence={confidence:.2%} (>= {min_confidence_for_trade:.2%}), "
                    f"TP={tp_str}, SL={sl_str}, "
                    f"price={current_price:.2f}"
                )

                # Если позиция уже есть, решаем: игнорировать реверс или усреднять
                if has_pos is not None and local_pos:
                    # Проверяем, нужно ли реверсировать позицию по сильному сигналу
                    if has_pos != signal_side and self._is_strong_reverse_signal(signal, confidence):
                        logger.info(f"[{symbol}] Strong reverse signal detected, closing & reversing.")
                        if size > 0:
                            await self._close_position_market(symbol, has_pos, size)
                        await self.execute_trade(
                            symbol,
                            "Buy" if signal_side == Bias.LONG else "Sell",
                            signal,
                            position_horizon=self._classify_position_horizon(signal),
                        )
                        return

                    # Не закрываем средне/долгосрочные позиции по противоположному сигналу
                    if (
                        has_pos != signal_side
                        and local_pos.horizon in ("mid_term", "long_term")
                        and self.settings.risk.long_term_ignore_reverse
                    ):
                        logger.info(
                            f"[{symbol}] Opposite signal ignored for {local_pos.horizon} position."
                        )
                        return

                    # Усреднение при сигнале в ту же сторону и в минусе
                    if has_pos == signal_side:
                        if self._should_dca(local_pos, signal, current_price, confidence):
                            logger.info(f"[{symbol}] DCA conditions met, adding to position.")
                            await self.execute_trade(
                                symbol,
                                "Buy" if signal_side == Bias.LONG else "Sell",
                                signal,
                                is_add=True,
                                position_horizon=local_pos.horizon,
                            )
                        else:
                            logger.info(f"[{symbol}] ⏭️ Skipping trade: position already exists in same direction (DCA conditions not met)")
                        return

                # Проверка сигнала BTCUSDT для других пар (альткоины следуют за BTC)
                if symbol != "BTCUSDT":
                    btc_signal = await self._get_btc_signal()
                    if btc_signal and btc_signal.get("action") in (Action.LONG, Action.SHORT):
                        btc_action = btc_signal["action"]
                        # Если сигнал BTC противоположен сигналу текущего символа - игнорируем
                        if (btc_action == Action.LONG and signal.action == Action.SHORT) or \
                           (btc_action == Action.SHORT and signal.action == Action.LONG):
                            logger.info(
                                f"[{symbol}] ⏭️ Signal ignored: BTCUSDT={btc_action.value}, "
                                f"{symbol}={signal.action.value} (opposite direction, following BTC)"
                            )
                            return
                
                # Открываем позицию, если ее нет или она в другую сторону (для short_term)
                if signal.action == Action.LONG and has_pos != Bias.LONG:
                    logger.info(f"[{symbol}] ✅ Opening LONG position (no position or opposite)")
                    await self.execute_trade(symbol, "Buy", signal)
                elif signal.action == Action.SHORT and has_pos != Bias.SHORT:
                    logger.info(f"[{symbol}] ✅ Opening SHORT position (no position or opposite)")
                    await self.execute_trade(symbol, "Sell", signal)
                else:
                    logger.info(f"[{symbol}] ⏭️ Skipping trade: action={signal.action.value}, has_pos={has_pos}")
            
            # Сохраняем timestamp обработанной свечи ТОЛЬКО после успешной обработки сигнала
            # Это позволяет повторить обработку, если произошла ошибка при открытии позиции
            if candle_timestamp is not None:
                self.last_processed_candle[symbol] = candle_timestamp
                logger.debug(f"[{symbol}] ✅ Candle timestamp saved after successful processing: {candle_timestamp}")

        except Exception as e:
            logger.error(f"[trading_loop] Error processing {symbol}: {e}")
            # При ошибке НЕ сохраняем timestamp, чтобы можно было повторить обработку

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        signal: Signal,
        is_add: bool = False,
        position_horizon: Optional[str] = None,
    ):
        try:
            logger.info(f"[{symbol}] 🚀 execute_trade() called: side={side}, is_add={is_add}, price={signal.price:.2f}")
            
            # Проверяем наличие TP/SL в сигнале (критично для открытия позиции)
            indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: проверяем возраст сигнала (для защиты от устаревших сигналов)
            # Используем signal_received_time из indicators_info, если он есть, иначе используем timestamp свечи
            signal_received_time = None
            if indicators_info and 'signal_received_time' in indicators_info:
                signal_received_time = pd.Timestamp(indicators_info['signal_received_time'])
            elif signal.timestamp:
                # Используем timestamp свечи как приблизительное время получения сигнала
                signal_received_time = signal.timestamp
            
            if signal_received_time and not is_add:  # Проверяем только для новых позиций, не для DCA
                signal_age_seconds = (pd.Timestamp.now() - signal_received_time).total_seconds()
                signal_age_minutes = signal_age_seconds / 60
                max_signal_age_minutes = 15  # Максимальный возраст сигнала для открытия сделки
                
                if signal_age_minutes > max_signal_age_minutes:
                    logger.warning(
                        f"[{symbol}] ❌ Cannot open position: signal is too old ({signal_age_minutes:.1f} minutes > {max_signal_age_minutes} minutes). "
                        f"Signal timestamp: {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    return  # Не открываем позицию по устаревшему сигналу
            
            signal_tp = signal.take_profit or indicators_info.get('take_profit')
            signal_sl = signal.stop_loss or indicators_info.get('stop_loss')
            
            if not is_add and (not signal_tp or not signal_sl):
                logger.warning(
                    f"[{symbol}] ❌ Cannot open position: missing TP/SL! "
                    f"TP={signal_tp}, SL={signal_sl}, signal.take_profit={signal.take_profit}, "
                    f"signal.stop_loss={signal.stop_loss}, indicators_info={indicators_info}"
                )
                return
            
            tp_str = f"{signal_tp:.2f}" if signal_tp else "None"
            sl_str = f"{signal_sl:.2f}" if signal_sl else "None"
            logger.info(f"[{symbol}] ✅ TP/SL check passed: TP={tp_str}, SL={sl_str}")
            
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
                logger.error(f"[{symbol}] ❌ Cannot get balance or balance is zero: {balance}")
                return
            
            logger.info(f"[{symbol}] ✅ Balance check passed: ${balance:.2f}")
            
            # РАСЧЕТ: Фиксированная сумма маржи с учетом плеча
            # base_order_usd - это маржа в USD
            # Размер позиции в USD = маржа * leverage
            # Количество = (маржа * leverage) / цена
            # При добавлении к позиции используем половину от размера позиции (notional)
            if is_add:
                # Сначала считаем размер позиции для новой позиции
                base_position_size_usd = self.settings.risk.base_order_usd * self.settings.leverage
                # При добавлении используем половину от размера позиции
                position_size_usd = base_position_size_usd / 2.0
                # Маржа = размер позиции / leverage
                fixed_margin_usd = position_size_usd / self.settings.leverage
            else:
                fixed_margin_usd = self.settings.risk.base_order_usd
                # Размер позиции в USD = маржа * leverage
                position_size_usd = fixed_margin_usd * self.settings.leverage
            
            # Проверяем, что маржа не превышает баланс
            if fixed_margin_usd > balance:
                logger.warning(
                    f"[{symbol}] ⚠️ Fixed margin ${fixed_margin_usd:.2f} exceeds balance ${balance:.2f}, "
                    f"using available balance"
                )
                fixed_margin_usd = balance
                # Пересчитываем размер позиции с учетом ограничения по балансу
                position_size_usd = fixed_margin_usd * self.settings.leverage
            
            # Количество монет = размер позиции / цена
            total_qty = position_size_usd / signal.price
            
            logger.info(
                f"Position size for {symbol}: "
                f"balance=${balance:.2f}, "
                f"margin=${fixed_margin_usd:.2f}, "
                f"position_size_usd=${position_size_usd:.2f}, "
                f"qty={total_qty:.6f}, leverage={self.settings.leverage}x"
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
            
            if qty <= 0:
                logger.error(f"[{symbol}] ❌ Calculated qty is zero or negative: {qty}")
                return
            
            logger.info(f"[{symbol}] ✅ Position size calculated: qty={qty:.6f}, placing order...")
            
            try:
                resp = self.bybit.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Market",
                    take_profit=None if is_add else signal.take_profit,
                    stop_loss=None if is_add else signal.stop_loss,
                )
            except InvalidRequestError as e:
                # Обрабатываем ошибку недостатка средств (код 110007)
                error_code = getattr(e, 'status_code', None) or getattr(e, 'ret_code', None)
                error_msg = str(e)
                
                # Проверяем, что это ошибка недостатка средств
                if error_code == 110007 or "not enough" in error_msg.lower() or "ab not enough" in error_msg.lower():
                    # Рассчитываем недостающую сумму
                    required_margin = fixed_margin_usd
                    shortfall = max(0, required_margin - balance)
                    
                    # Формируем детальное сообщение
                    message = (
                        f"⚠️ НЕДОСТАТОЧНО СРЕДСТВ ДЛЯ ОТКРЫТИЯ ПОЗИЦИИ\n\n"
                        f"📊 Параметры сделки:\n"
                        f"• Символ: {symbol}\n"
                        f"• Направление: {side}\n"
                        f"• Цена входа: ${signal.price:.6f}\n"
                        f"• Количество: {qty:.6f}\n"
                        f"• Размер позиции: ${position_size_usd:.2f}\n"
                        f"• Требуемая маржа: ${required_margin:.2f}\n"
                        f"• Плечо: {self.settings.leverage}x\n"
                    )
                    
                    if signal.take_profit and signal.stop_loss:
                        message += (
                            f"• TP: ${signal.take_profit:.6f}\n"
                            f"• SL: ${signal.stop_loss:.6f}\n"
                        )
                    
                    message += (
                        f"\n💰 Баланс:\n"
                        f"• Доступно: ${balance:.2f}\n"
                        f"• Не хватает: ${shortfall:.2f}\n"
                        f"• Нужно всего: ${required_margin:.2f}"
                    )
                    
                    # Отправляем уведомление
                    await self.notifier.critical(message)
                    logger.error(
                        f"[{symbol}] ❌ Insufficient balance: required=${required_margin:.2f}, "
                        f"available=${balance:.2f}, shortfall=${shortfall:.2f}"
                    )
                    return
                else:
                    # Другая ошибка InvalidRequestError - пробрасываем дальше
                    raise
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ответа от биржи
            if resp:
                ret_code = resp.get("retCode") if isinstance(resp, dict) else None
                ret_msg = resp.get("retMsg", "") if isinstance(resp, dict) else ""
                logger.info(f"[{symbol}] 📡 Order response: retCode={ret_code}, retMsg={ret_msg}, full_response={resp}")
            else:
                logger.error(f"[{symbol}] ❌ Order response is None or empty!")
            
            if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                if is_add:
                    # Log to trades.log
                    trade_logger.info(f"ORDER PLACED (ADD): {symbol} {side} Qty={qty} Price={signal.price}")
                    
                    logger.info(f"Successfully added to {side} for {symbol}")
                    await self.notifier.medium(
                        f"➕ ДОБАВЛЕНИЕ К ПОЗИЦИИ {side} {symbol}\n"
                        f"Цена: {signal.price}\n"
                        f"Объем: {qty}"
                    )
                    self.state.increment_dca(symbol)
                    # Обновляем среднюю цену и размер по бирже
                    pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
                    if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data:
                                position = list_data[0]
                                if position and isinstance(position, dict):
                                    size = float(position.get("size", 0))
                                    avg_price = float(position.get("avgPrice", 0))
                                    if size > 0 and avg_price > 0:
                                        self.state.update_position(symbol, size, avg_price)
                else:
                    # Log to trades.log
                    trade_logger.info(f"ORDER PLACED (OPEN): {symbol} {side} Qty={qty} Price={signal.price} TP={signal.take_profit} SL={signal.stop_loss}")
                    
                    logger.info(f"Successfully opened {side} for {symbol}")
                    await self.notifier.high(
                        f"🚀 ОТКРЫТА ПОЗИЦИЯ {side} {symbol}\n"
                        f"Цена: {signal.price}\nTP: {signal.take_profit}\nSL: {signal.stop_loss}"
                    )
                    
                    # Добавляем в историю (пока как открытую)
                    indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
                    confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
                    signal_strength = indicators_info.get('strength', '') if isinstance(indicators_info, dict) else ''
                    
                    # Извлекаем TP/SL проценты из сигнала
                    signal_tp = signal.take_profit or indicators_info.get('take_profit')
                    signal_sl = signal.stop_loss or indicators_info.get('stop_loss')
                    tp_pct = None
                    sl_pct = None
                    if signal_tp and signal.price:
                        tp_pct = abs(signal_tp - signal.price) / signal.price
                    if signal_sl and signal.price:
                        sl_pct = abs(signal.price - signal_sl) / signal.price
                    
                    # Вычисляем маржу
                    margin_usd = fixed_margin_usd
                    
                    # Параметры сигнала
                    signal_parameters = {
                        'take_profit_pct': tp_pct,
                        'stop_loss_pct': sl_pct,
                        'risk_reward_ratio': (tp_pct / sl_pct) if (tp_pct and sl_pct and sl_pct > 0) else None,
                    }
                    
                    trade = TradeRecord(
                        symbol=symbol,
                        side=side,
                        entry_price=signal.price,
                        qty=qty,
                        status="open",
                        model_name=self.state.symbol_models.get(symbol, ""),
                        horizon=position_horizon or self._classify_position_horizon(signal),
                        entry_reason=signal.reason or "",
                        confidence=confidence,
                        take_profit=signal_tp,
                        stop_loss=signal_sl,
                        leverage=self.settings.leverage,
                        margin_usd=margin_usd,
                        signal_strength=signal_strength,
                        signal_parameters=signal_parameters,
                    )
                    self.state.add_trade(trade)
            else:
                ret_code = resp.get("retCode") if resp and isinstance(resp, dict) else "unknown"
                ret_msg = resp.get("retMsg", "") if resp and isinstance(resp, dict) else ""
                
                # Обрабатываем ошибку недостатка средств (код 110007)
                if ret_code == 110007 or (ret_msg and ("not enough" in ret_msg.lower() or "ab not enough" in ret_msg.lower())):
                    # Рассчитываем недостающую сумму
                    required_margin = fixed_margin_usd
                    shortfall = max(0, required_margin - balance)
                    
                    # Формируем детальное сообщение
                    message = (
                        f"⚠️ НЕДОСТАТОЧНО СРЕДСТВ ДЛЯ ОТКРЫТИЯ ПОЗИЦИИ\n\n"
                        f"📊 Параметры сделки:\n"
                        f"• Символ: {symbol}\n"
                        f"• Направление: {side}\n"
                        f"• Цена входа: ${signal.price:.6f}\n"
                        f"• Количество: {qty:.6f}\n"
                        f"• Размер позиции: ${position_size_usd:.2f}\n"
                        f"• Требуемая маржа: ${required_margin:.2f}\n"
                        f"• Плечо: {self.settings.leverage}x\n"
                    )
                    
                    if signal.take_profit and signal.stop_loss:
                        message += (
                            f"• TP: ${signal.take_profit:.6f}\n"
                            f"• SL: ${signal.stop_loss:.6f}\n"
                        )
                    
                    message += (
                        f"\n💰 Баланс:\n"
                        f"• Доступно: ${balance:.2f}\n"
                        f"• Не хватает: ${shortfall:.2f}\n"
                        f"• Нужно всего: ${required_margin:.2f}"
                    )
                    
                    # Отправляем уведомление
                    await self.notifier.critical(message)
                    logger.error(
                        f"[{symbol}] ❌ Insufficient balance (retCode={ret_code}): required=${required_margin:.2f}, "
                        f"available=${balance:.2f}, shortfall=${shortfall:.2f}"
                    )
                    return
                
                # Другие ошибки - просто логируем
                logger.error(
                    f"[{symbol}] ❌ Failed to open {side} position: "
                    f"retCode={ret_code}, retMsg={ret_msg}, "
                    f"qty={qty:.6f}, price={signal.price:.2f}, "
                    f"TP={signal.take_profit if not is_add else 'N/A'}, "
                    f"SL={signal.stop_loss if not is_add else 'N/A'}, "
                    f"full_response={resp}"
                )
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Exception in execute_trade: {e}", exc_info=True)
    
    async def update_breakeven_stop(self, symbol: str, position_info: dict):
        """Перемещает SL в безубыток при достижении порога прибыли"""
        try:
            # Проверяем, включен ли безубыток
            if not self.settings.risk.enable_breakeven:
                logger.debug(f"[{symbol}] Безубыток выключен, пропускаем обновление SL")
                return
            
            if not position_info or not isinstance(position_info, dict):
                logger.debug(f"[{symbol}] update_breakeven_stop: position_info is None or not dict")
                return
            
            if not position_info.get("size"):
                logger.debug(f"[{symbol}] update_breakeven_stop: position size is empty")
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                logger.debug(f"[{symbol}] update_breakeven_stop: position size is 0")
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            current_sl = position_info.get("stopLoss")
            
            if not entry_price or not mark_price:
                logger.debug(f"[{symbol}] update_breakeven_stop: entry_price={entry_price}, mark_price={mark_price}")
                return
            
            # Рассчитываем текущий PnL в процентах
            if side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price) * 100
            else:  # Sell
                pnl_pct = ((entry_price - mark_price) / entry_price) * 100
            
            logger.debug(f"[{symbol}] update_breakeven_stop: side={side}, entry={entry_price:.6f}, mark={mark_price:.6f}, pnl_pct={pnl_pct:.2f}%, current_sl={current_sl}")
            
            # Проверяем, нужно ли активировать безубыток (многоуровневый)
            level1_activation = self.settings.risk.breakeven_level1_activation_pct * 100  # Конвертируем в %
            level2_activation = self.settings.risk.breakeven_level2_activation_pct * 100  # Конвертируем в %
            level1_sl_pct = self.settings.risk.breakeven_level1_sl_pct
            level2_sl_pct = self.settings.risk.breakeven_level2_sl_pct
            
            logger.debug(f"[{symbol}] Breakeven thresholds: level1_activation={level1_activation:.2f}%, level2_activation={level2_activation:.2f}%, level1_sl_pct={level1_sl_pct:.4f}, level2_sl_pct={level2_sl_pct:.4f}")
            
            # Определяем, какой уровень активировать
            new_sl = None
            level = None
            
            if pnl_pct >= level2_activation:
                # 2-я ступень: при прибыли >= level2_activation ставим SL на level2_sl_pct от входа
                level = "2-я ступень"
                if side == "Buy":
                    new_sl = entry_price * (1 + level2_sl_pct)
                else:
                    new_sl = entry_price * (1 - level2_sl_pct)
            elif pnl_pct >= level1_activation:
                # 1-я ступень: при прибыли >= level1_activation ставим SL на level1_sl_pct от входа
                level = "1-я ступень"
                if side == "Buy":
                    new_sl = entry_price * (1 + level1_sl_pct)
                else:
                    new_sl = entry_price * (1 - level1_sl_pct)
            else:
                # Прибыль недостаточна для активации безубытка
                logger.debug(f"[{symbol}] PnL {pnl_pct:.2f}% < level1_activation {level1_activation:.2f}%, безубыток не активируется")
                return
            
            # Округляем до tick size
            new_sl = self.bybit.round_price(new_sl, symbol)
            tick_size = self.bybit.get_price_step(symbol)
            
            # Проверяем, нужно ли обновлять SL
            should_update = False
            if current_sl:
                current_sl_float = float(current_sl)
                # Если новый SL совпадает с текущим (с учетом шага цены), не обновляем
                if tick_size > 0 and abs(new_sl - current_sl_float) < (tick_size / 2):
                    should_update = False
                elif side == "Buy" and new_sl > current_sl_float:
                    should_update = True
                elif side == "Sell" and new_sl < current_sl_float:
                    should_update = True
            else:
                should_update = True
            
            if should_update:
                logger.info(f"Moving {symbol} SL to breakeven ({level}): {new_sl} (PnL: {pnl_pct:.2f}%)")
                resp = await asyncio.to_thread(
                    self.bybit.set_trading_stop,
                    symbol=symbol,
                    stop_loss=new_sl
                )
                
                if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                    await self.notifier.medium(
                        f"🛡️ БЕЗУБЫТОК АКТИВИРОВАН ({level})\n{symbol} SL → ${new_sl:.2f}\nТекущий PnL: +{pnl_pct:.2f}%"
                    )
        
        except Exception as e:
            # Bybit возвращает "not modified" если стоп-лосс уже равен текущему
            if "ErrCode: 34040" in str(e) or "not modified" in str(e).lower():
                logger.info(f"{symbol} breakeven SL not modified (already set): {e}")
                return
            logger.error(f"Error updating breakeven stop for {symbol}: {e}")

    def _calculate_fees_usd(self, entry_price: float, exit_price: float, qty: float) -> float:
        """Считает комиссию биржи в USD (per side) по notional на входе и выходе."""
        fee_rate = self.settings.risk.fee_rate
        if fee_rate <= 0:
            return 0.0
        notional = (entry_price + exit_price) * qty
        return notional * fee_rate

    def _classify_position_horizon(self, signal: Signal) -> str:
        """Категоризирует позицию по расстоянию до TP/SL."""
        if not signal.take_profit or not signal.stop_loss or not signal.price:
            return "unknown"
        
        tp_dist = abs(signal.take_profit - signal.price) / signal.price
        sl_dist = abs(signal.stop_loss - signal.price) / signal.price
        
        if tp_dist >= self.settings.risk.long_term_tp_pct or sl_dist >= self.settings.risk.long_term_sl_pct:
            return "long_term"
        elif tp_dist >= self.settings.risk.mid_term_tp_pct:
            return "mid_term"
        else:
            return "short_term"

    async def close_position(self, symbol: str, reason: str):
        try:
            logger.info(f"[{symbol}] 🚨 Initiating close position: {reason}")
            # Get current position info to be sure about size
            pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
            
            if not pos_info or pos_info.get("retCode") != 0:
                logger.error(f"[{symbol}] Failed to get position info for closing: {pos_info}")
                return

            result = pos_info.get("result", {}).get("list", [])
            if not result:
                logger.warning(f"[{symbol}] No position found to close")
                return

            position = result[0]
            size = float(position.get("size", 0))
            side = position.get("side")
            
            if size <= 0:
                logger.warning(f"[{symbol}] Position size is 0, nothing to close")
                return

            close_side = "Sell" if side == "Buy" else "Buy"
            
            logger.info(f"[{symbol}] Closing {side} position size {size} via Market order...")
            
            resp = await asyncio.to_thread(
                self.bybit.place_order,
                symbol=symbol,
                side=close_side,
                qty=size,
                order_type="Market",
                reduce_only=True
            )
            
            if resp and resp.get("retCode") == 0:
                logger.info(f"[{symbol}] ✅ Position closed successfully: {reason}")
                
                # Log to trades.log
                trade_logger.info(f"POSITION CLOSED: {symbol} {close_side} Size={size} Reason={reason}")
                
                await self.notifier.medium(f"🚫 ПОЗИЦИЯ ЗАКРЫТА ({reason})\n{symbol}\nSize: {size}")
            else:
                logger.error(f"[{symbol}] Failed to close position: {resp}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def check_time_stop(self, symbol: str, position: dict):
        try:
            # createdTime is usually when the position was opened (or updatedTime if added to)
            # Bybit V5: createdTime is string ms timestamp
            created_time = float(position.get("createdTime", 0))
            if created_time == 0:
                created_time = float(position.get("updatedTime", 0))
            
            if created_time == 0:
                return

            open_time = pd.Timestamp(created_time, unit='ms')
            now = pd.Timestamp.now()
            duration_minutes = (now - open_time).total_seconds() / 60
            
            max_minutes = self.settings.risk.time_stop_minutes
            
            if duration_minutes > max_minutes:
                logger.info(f"[{symbol}] ⏰ Time Stop triggered! Duration: {duration_minutes:.1f} min > {max_minutes} min")
                await self.close_position(symbol, f"Time Stop > {max_minutes} min")
                
        except Exception as e:
            logger.error(f"[{symbol}] Error in check_time_stop: {e}")

    async def check_early_exit(self, symbol: str, position: dict):
        try:
            created_time = float(position.get("createdTime", 0))
            if created_time == 0:
                created_time = float(position.get("updatedTime", 0))
            
            if created_time == 0:
                return

            open_time = pd.Timestamp(created_time, unit='ms')
            now = pd.Timestamp.now()
            duration_minutes = (now - open_time).total_seconds() / 60
            
            early_exit_minutes = self.settings.risk.early_exit_minutes
            min_profit_pct = self.settings.risk.early_exit_min_profit_pct
            
            # Если прошло время раннего выхода, а прибыль все еще маленькая
            if duration_minutes > early_exit_minutes:
                entry_price = float(position.get("avgPrice", 0))
                mark_price = float(position.get("markPrice", entry_price))
                side = position.get("side")
                
                if entry_price == 0:
                    return

                if side == "Buy":
                    pnl_pct = (mark_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - mark_price) / entry_price
                
                if pnl_pct < min_profit_pct:
                    logger.info(f"[{symbol}] 📉 Early Exit triggered! Duration: {duration_minutes:.1f} min > {early_exit_minutes} min, PnL: {pnl_pct*100:.2f}% < {min_profit_pct*100:.2f}%")
                    await self.close_position(symbol, f"Early Exit (Low PnL < {min_profit_pct*100:.1f}%)")
                    
        except Exception as e:
            logger.error(f"[{symbol}] Error in check_early_exit: {e}")

    def _should_dca(self, local_pos: TradeRecord, signal: Signal, current_price: float, confidence: float) -> bool:
        """Проверяет условия для усреднения позиции."""
        if not self.settings.risk.dca_enabled:
            return False
        if local_pos.horizon not in ("mid_term", "long_term"):
            return False
        if local_pos.dca_count >= self.settings.risk.dca_max_adds:
            return False
        if confidence < self.settings.risk.dca_min_confidence:
            return False
        if not current_price or not local_pos.entry_price:
            return False

        if local_pos.side == "Buy":
            drawdown_pct = (local_pos.entry_price - current_price) / local_pos.entry_price
        else:
            drawdown_pct = (current_price - local_pos.entry_price) / local_pos.entry_price

        return drawdown_pct >= self.settings.risk.dca_drawdown_pct

    def _is_strong_reverse_signal(self, signal: Signal, confidence: float) -> bool:
        """Определяет, является ли обратный сигнал сильным для реверса."""
        if not self.settings.risk.reverse_on_strong_signal:
            return False
        if confidence < self.settings.risk.reverse_min_confidence:
            return False
        # Проверяем силу сигнала, если доступна
        strength = None
        if signal.indicators_info and isinstance(signal.indicators_info, dict):
            strength = signal.indicators_info.get("strength")
        if strength is None and signal.reason:
            # Пытаемся вытащить силу из текста причины (ml_..._сила_сильное_..)
            parts = str(signal.reason).split("_сила_")
            if len(parts) == 2:
                strength = parts[1].split("_")[0]
        if strength:
            order = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
            try:
                if order.index(strength) < order.index(self.settings.risk.reverse_min_strength):
                    return False
            except ValueError:
                # неизвестная сила — не блокируем, но логируем
                logger.warning(f"Unknown signal strength '{strength}', allowing reverse by confidence only.")
        return True
    
    def _load_cached_1h_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загружает кэшированные 1h данные из ml_data/{symbol}_60_cache.csv
        
        Returns:
            DataFrame с 1h данными или None если файл не найден
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            cache_file = ml_data_dir / f"{symbol}_60_cache.csv"
            
            if not cache_file.exists():
                return None
            
            df = pd.read_csv(cache_file)
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"[{symbol}] Cached 1h data missing required columns")
                return None
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps in cache, removing them")
                    df = df[~invalid_dates].copy()
                
                if len(df) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing")
                    return None
                
                # Сортируем по времени и удаляем дубликаты
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            
            # Берем последние 500 свечей (достаточно для 1h модели)
            if len(df) > 500:
                df = df.tail(500).reset_index(drop=True)
            
            # Устанавливаем timestamp как индекс для совместимости с MTF стратегией
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                # Проверяем, что индекс корректный
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from timestamp column")
                    # Удаляем некорректный кэш
                    try:
                        cache_file.unlink()
                        logger.info(f"[{symbol}] Removed invalid cache file (no DatetimeIndex)")
                    except Exception as e:
                        logger.debug(f"[{symbol}] Could not remove invalid cache: {e}")
                    return None
                
                # Проверяем, что даты разумные (не 1970 год)
                if len(df) > 0:
                    first_date = df.index[0]
                    last_date = df.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates in cache: first={first_date}, last={last_date}")
                        # Удаляем некорректный кэш и возвращаем None для пересоздания
                        try:
                            cache_file.unlink()
                            logger.info(f"[{symbol}] Removed invalid cache file (invalid dates), will recreate")
                        except Exception as e:
                            logger.warning(f"[{symbol}] Failed to remove invalid cache: {e}")
                        return None
            
            logger.debug(f"[{symbol}] Loaded {len(df)} cached 1h candles from {cache_file}")
            if len(df) > 0:
                logger.debug(f"[{symbol}] Cache date range: {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load cached 1h data: {e}")
            return None
    
    def _fetch_and_cache_1h_data(self, symbol: str, existing_cache: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Запрашивает 1h данные с биржи и сохраняет в кэш.
        Если есть существующий кэш, подгружает только новые данные.
        
        Args:
            symbol: Торговая пара
            existing_cache: Существующий кэш (если есть) для подгрузки только новых данных
        
        Returns:
            DataFrame с 1h данными или None при ошибке
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            ml_data_dir.mkdir(exist_ok=True)
            cache_file = ml_data_dir / f"{symbol}_60_cache.csv"
            
            # Определяем, с какого времени нужно подгружать данные
            start_from = None
            if existing_cache is not None and not existing_cache.empty:
                if isinstance(existing_cache.index, pd.DatetimeIndex) and len(existing_cache) > 0:
                    # Берем последнюю свечу из кэша и запрашиваем данные начиная с неё
                    last_candle_time = existing_cache.index[-1]
                    # Запрашиваем немного больше, чтобы перекрыть последнюю свечу (она может быть незакрыта)
                    start_from = last_candle_time - pd.Timedelta(hours=1)
                    logger.debug(f"[{symbol}] Updating cache from {last_candle_time} (will fetch from {start_from})")
            
            # Запрашиваем 500 свечей 1h с биржи
            logger.info(f"[{symbol}] Fetching 1h data from exchange (limit=500)...")
            df_new = self.bybit.get_kline_df(symbol, "60", 500)
            
            if df_new.empty:
                logger.warning(f"[{symbol}] No 1h data received from exchange")
                return existing_cache  # Возвращаем существующий кэш, если новый запрос пуст
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df_new.columns for col in required_cols):
                logger.warning(f"[{symbol}] 1h data from exchange missing required columns")
                return existing_cache
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df_new.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df_new["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df_new["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps from exchange, removing them")
                    df_new = df_new[~invalid_dates].copy()
                
                if len(df_new) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing from exchange")
                    return existing_cache
                
                df_new = df_new.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                df_new = df_new.set_index("timestamp")
                
                # Проверяем, что индекс корректный и даты разумные
                if not isinstance(df_new.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from exchange data")
                    return existing_cache
                
                if len(df_new) > 0:
                    first_date = df_new.index[0]
                    last_date = df_new.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates from exchange: first={first_date}, last={last_date}")
                        return existing_cache
                    logger.debug(f"[{symbol}] Exchange data date range: {first_date} to {last_date}")
            elif not isinstance(df_new.index, pd.DatetimeIndex):
                logger.warning(f"[{symbol}] Could not set timestamp index for new 1h data")
                return existing_cache
            
            # Объединяем с существующим кэшем, если есть
            if existing_cache is not None and not existing_cache.empty:
                # Объединяем данные, удаляя дубликаты
                df_combined = pd.concat([existing_cache, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # Оставляем последние значения для дубликатов
                df_combined = df_combined.sort_index()
                
                # Берем последние 500 свечей
                if len(df_combined) > 500:
                    df_combined = df_combined.tail(500)
                
                df_final = df_combined
                logger.info(f"[{symbol}] Merged cache: {len(existing_cache)} old + {len(df_new)} new = {len(df_final)} total candles")
            else:
                df_final = df_new
                logger.info(f"[{symbol}] Created new cache with {len(df_final)} candles")
            
            # Сохраняем в кэш
            try:
                df_to_save = df_final.reset_index() if isinstance(df_final.index, pd.DatetimeIndex) else df_final.copy()
                
                # Убеждаемся, что timestamp в правильном формате для сохранения
                if "timestamp" in df_to_save.columns:
                    # Проверяем, что timestamp - это datetime, а не что-то другое
                    if not pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        # Если это не datetime, пытаемся преобразовать
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        # Удаляем некорректные даты
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                    
                    # Сохраняем timestamp в формате строки для читаемости
                    # Проверяем, что timestamp - это datetime перед форматированием
                    if pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Если это не datetime, пытаемся преобразовать сначала
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                        if len(df_to_save) > 0:
                            df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"[{symbol}] ✅ Saved {len(df_final)} 1h candles to cache: {cache_file}")
                if len(df_to_save) > 0:
                    logger.debug(f"[{symbol}] Saved cache date range: {df_to_save['timestamp'].iloc[0]} to {df_to_save['timestamp'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to save 1h cache: {e}", exc_info=True)
            
            return df_final
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch and cache 1h data: {e}", exc_info=True)
            return existing_cache
    
    def _load_cached_15m_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загружает кэшированные 15m данные из ml_data/{symbol}_15_cache.csv
        
        Returns:
            DataFrame с 15m данными или None если файл не найден
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            cache_file = ml_data_dir / f"{symbol}_15_cache.csv"
            
            if not cache_file.exists():
                return None
            
            df = pd.read_csv(cache_file)
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"[{symbol}] Cached 15m data missing required columns")
                return None
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps in cache, removing them")
                    df = df[~invalid_dates].copy()
                
                if len(df) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing")
                    return None
                
                # Сортируем по времени и удаляем дубликаты
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            
            # Устанавливаем timestamp как индекс
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                # Проверяем, что индекс корректный
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from timestamp column")
                    # Удаляем некорректный кэш
                    try:
                        cache_file.unlink()
                        logger.info(f"[{symbol}] Removed invalid cache file (no DatetimeIndex)")
                    except Exception as e:
                        logger.debug(f"[{symbol}] Could not remove invalid cache: {e}")
                    return None
                
                # Проверяем, что даты разумные (не 1970 год)
                if len(df) > 0:
                    first_date = df.index[0]
                    last_date = df.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates in cache: first={first_date}, last={last_date}")
                        # Удаляем некорректный кэш и возвращаем None для пересоздания
                        try:
                            cache_file.unlink()
                            logger.info(f"[{symbol}] Removed invalid cache file (invalid dates), will recreate")
                        except Exception as e:
                            logger.warning(f"[{symbol}] Failed to remove invalid cache: {e}")
                        return None
            
            logger.debug(f"[{symbol}] Loaded {len(df)} cached 15m candles from {cache_file}")
            if len(df) > 0:
                logger.debug(f"[{symbol}] Cache date range: {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load cached 15m data: {e}")
            return None
    
    def _fetch_and_cache_15m_data(self, symbol: str, existing_cache: Optional[pd.DataFrame] = None, required_limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Запрашивает 15m данные с биржи и сохраняет в кэш.
        Если есть существующий кэш, подгружает только новые данные.
        
        Args:
            symbol: Торговая пара
            existing_cache: Существующий кэш (если есть) для подгрузки только новых данных
            required_limit: Минимальное количество свечей, которое нужно иметь
        
        Returns:
            DataFrame с 15m данными или None при ошибке
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            ml_data_dir.mkdir(exist_ok=True)
            cache_file = ml_data_dir / f"{symbol}_15_cache.csv"
            
            # Определяем, сколько свечей нужно запросить
            if existing_cache is not None and not existing_cache.empty:
                if isinstance(existing_cache.index, pd.DatetimeIndex) and len(existing_cache) > 0:
                    # Если есть кэш, запрашиваем последние свечи для обновления
                    # Запрашиваем достаточно, чтобы гарантированно получить самую новую свечу
                    # Для 15m свечей запрашиваем последние 20 свечей (5 часов данных) - этого достаточно
                    # чтобы покрыть возможные пропуски и получить самую новую свечу
                    limit = 20  # Запрашиваем последние 20 свечей для обновления
                    logger.debug(f"[{symbol}] Updating cache: have {len(existing_cache)} candles, fetching last {limit} candles from exchange")
                else:
                    limit = required_limit
            else:
                # Если кэша нет, запрашиваем полное количество
                limit = required_limit
            
            # Запрашиваем 15m данные с биржи
            # Используем "15" для 15-минутных свечей (независимо от основного timeframe)
            logger.info(f"[{symbol}] Fetching 15m data from exchange (limit={limit})...")
            df_new = self.bybit.get_kline_df(symbol, "15", limit)
            
            if df_new.empty:
                logger.warning(f"[{symbol}] No 15m data received from exchange")
                return existing_cache  # Возвращаем существующий кэш, если новый запрос пуст
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df_new.columns for col in required_cols):
                logger.warning(f"[{symbol}] 15m data from exchange missing required columns")
                return existing_cache
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df_new.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df_new["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df_new["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps from exchange, removing them")
                    df_new = df_new[~invalid_dates].copy()
                
                if len(df_new) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing from exchange")
                    return existing_cache
                
                df_new = df_new.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                df_new = df_new.set_index("timestamp")
                
                # Проверяем, что индекс корректный и даты разумные
                if not isinstance(df_new.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from exchange data")
                    return existing_cache
                
                if len(df_new) > 0:
                    first_date = df_new.index[0]
                    last_date = df_new.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates from exchange: first={first_date}, last={last_date}")
                        return existing_cache
                    logger.debug(f"[{symbol}] Exchange data date range: {first_date} to {last_date}")
            elif not isinstance(df_new.index, pd.DatetimeIndex):
                logger.warning(f"[{symbol}] Could not set timestamp index for new 15m data")
                return existing_cache
            
            # Объединяем с существующим кэшем, если есть
            if existing_cache is not None and not existing_cache.empty:
                # Объединяем данные, удаляя дубликаты
                df_combined = pd.concat([existing_cache, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # Оставляем последние значения для дубликатов
                df_combined = df_combined.sort_index()
                
                # Берем последние required_limit свечей (или больше, если нужно)
                if len(df_combined) > required_limit:
                    df_combined = df_combined.tail(required_limit)
                
                df_final = df_combined
                logger.info(f"[{symbol}] Merged cache: {len(existing_cache)} old + {len(df_new)} new = {len(df_final)} total candles")
            else:
                df_final = df_new
                logger.info(f"[{symbol}] Created new cache with {len(df_final)} candles")
            
            # Сохраняем в кэш
            try:
                df_to_save = df_final.reset_index() if isinstance(df_final.index, pd.DatetimeIndex) else df_final.copy()
                
                # Убеждаемся, что timestamp в правильном формате для сохранения
                if "timestamp" in df_to_save.columns:
                    # Проверяем, что timestamp - это datetime, а не что-то другое
                    if not pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        # Если это не datetime, пытаемся преобразовать
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        # Удаляем некорректные даты
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                    
                    # Сохраняем timestamp в формате строки для читаемости
                    # Проверяем, что timestamp - это datetime перед форматированием
                    if pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Если это не datetime, пытаемся преобразовать сначала
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                        if len(df_to_save) > 0:
                            df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"[{symbol}] ✅ Saved {len(df_final)} 15m candles to cache: {cache_file}")
                if len(df_to_save) > 0:
                    logger.debug(f"[{symbol}] Saved cache date range: {df_to_save['timestamp'].iloc[0]} to {df_to_save['timestamp'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to save 15m cache: {e}", exc_info=True)
            
            return df_final
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch and cache 15m data: {e}", exc_info=True)
            return existing_cache  # Возвращаем существующий кэш при ошибке

    async def _get_btc_signal(self) -> Optional[Dict]:
        """
        Получает сигнал BTCUSDT для проверки направления других пар.
        Использует кэш на 5 минут, чтобы не делать лишние запросы.
        
        Returns:
            Dict с ключами 'action' (Action) и 'confidence' (float) или None
        """
        import time
        
        # Проверяем кэш (актуален 5 минут)
        current_time = time.time()
        if (self._btc_signal_cache is not None and 
            self._btc_signal_cache_time is not None and 
            current_time - self._btc_signal_cache_time < 300):  # 5 минут
            return self._btc_signal_cache
        
        # Если BTCUSDT не в активных символах, возвращаем None
        if "BTCUSDT" not in self.state.active_symbols:
            return None
        
        try:
            # Получаем данные BTCUSDT
            btc_df = await asyncio.to_thread(
                self.bybit.get_kline_df,
                "BTCUSDT",
                self.settings.timeframe,
                200
            )
            
            if btc_df.empty or len(btc_df) < 2:
                return None
            
            # Инициализируем стратегию BTCUSDT если нужно
            if "BTCUSDT" not in self.strategies:
                model_path = self.state.symbol_models.get("BTCUSDT")
                if not model_path:
                    from pathlib import Path
                    models = list(Path("ml_models").glob("*_BTCUSDT_*.pkl"))
                    if models:
                        model_path = str(models[0])
                        self.state.symbol_models["BTCUSDT"] = model_path
                
                if model_path:
                    self.strategies["BTCUSDT"] = MLStrategy(
                        model_path=model_path,
                        confidence_threshold=self.settings.ml_strategy.confidence_threshold,
                        min_signal_strength=self.settings.ml_strategy.min_signal_strength
                    )
                else:
                    return None
            
            # Получаем позицию BTCUSDT
            try:
                btc_pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol="BTCUSDT")
                btc_has_pos = None
                if btc_pos_info and isinstance(btc_pos_info, dict) and btc_pos_info.get("retCode") == 0:
                    result = btc_pos_info.get("result")
                    if result and isinstance(result, dict):
                        list_data = result.get("list", [])
                        if list_data and len(list_data) > 0:
                            p = list_data[0]
                            if p and isinstance(p, dict):
                                btc_size = float(p.get("size", 0))
                                if btc_size > 0:
                                    btc_side = p.get("side")
                                    btc_has_pos = Bias.LONG if btc_side == "Buy" else Bias.SHORT
            except Exception as e:
                logger.debug(f"Error getting BTCUSDT position: {e}")
                btc_has_pos = None
            
            # Генерируем сигнал BTCUSDT
            btc_strategy = self.strategies["BTCUSDT"]
            btc_row = btc_df.iloc[-2] if len(btc_df) >= 2 else btc_df.iloc[-1]
            btc_current_price = btc_df.iloc[-1]['close']
            
            btc_signal = await asyncio.to_thread(
                btc_strategy.generate_signal,
                row=btc_row,
                df=btc_df.iloc[:-1] if len(btc_df) >= 2 else btc_df,
                has_position=btc_has_pos,
                current_price=btc_current_price,
                leverage=self.settings.leverage
            )
            
            if btc_signal:
                # Сохраняем в кэш
                indicators_info = btc_signal.indicators_info if btc_signal.indicators_info and isinstance(btc_signal.indicators_info, dict) else {}
                btc_confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
                
                self._btc_signal_cache = {
                    'action': btc_signal.action,
                    'confidence': btc_confidence
                }
                self._btc_signal_cache_time = current_time
                
                return self._btc_signal_cache
            
        except Exception as e:
            logger.debug(f"Error getting BTCUSDT signal: {e}")
        
        return None

    async def _close_position_market(self, symbol: str, side: Bias, size: float):
        """Закрывает позицию по рынку (reduce_only)."""
        if size <= 0:
            return
        close_side = "Sell" if side == Bias.LONG else "Buy"
        logger.info(f"[{symbol}] Closing position by market for reverse: {size} {close_side}")
        resp = await asyncio.to_thread(
            self.bybit.place_order,
            symbol=symbol,
            side=close_side,
            qty=size,
            order_type="Market",
            reduce_only=True,
        )
        if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
            await self.notifier.high(f"🔁 РЕВЕРС: позиция {symbol} закрыта и будет открыта в обратную сторону")
        else:
            logger.error(f"[{symbol}] Failed to close position for reverse: {resp}")
    
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
                    df = await asyncio.to_thread(
                        self.bybit.get_kline_df,
                        symbol,
                        self.settings.timeframe,
                        1
                    )
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

            # Учитываем комиссию биржи
            fee_usd = self._calculate_fees_usd(local_pos.entry_price, exit_price, local_pos.qty)
            if fee_usd > 0:
                pnl_usd -= fee_usd
                if margin > 0:
                    pnl_pct = (pnl_usd / margin) * 100
                logger.info(
                    f"Applied fees for {symbol}: fee_usd={fee_usd:.4f}, pnl_usd={pnl_usd:.2f}, pnl_pct={pnl_pct:.2f}%"
                )
            
            logger.info(f"Calculated PnL for {symbol}: exit_price={exit_price:.2f}, pnl_pct={pnl_pct:.2f}%, pnl_usd={pnl_usd:.2f}")
            
            # Определяем причину закрытия
            exit_reason = "TP" if pnl_usd > 0 else "SL"
            # Можно добавить более детальную причину, если доступна информация о trailing stop и т.д.
            
            # Обновляем статус сделки (может установить кулдаун от убытков)
            self.state.update_trade_on_close(symbol, exit_price, pnl_usd, pnl_pct, exit_reason)
            
            # Устанавливаем кулдаун до закрытия следующей свечи (15 минут)
            # Этот кулдаун не перезапишет более длительный кулдаун от убытков
            self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
            
            # Отправляем уведомление
            pnl_emoji = "✅" if pnl_usd > 0 else "❌"
            await self.notifier.high(
                f"{pnl_emoji} ПОЗИЦИЯ ЗАКРЫТА ({exit_reason})\n"
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
                df = await asyncio.to_thread(
                    self.bybit.get_kline_df,
                    symbol,
                    self.settings.timeframe,
                    1
                )
                if not df.empty:
                    exit_price = float(df['close'].iloc[-1])
                    # Рассчитываем PnL даже при ошибке
                    if local_pos.side == "Buy":
                        pnl_pct = ((exit_price - local_pos.entry_price) / local_pos.entry_price) * 100
                    else:
                        pnl_pct = ((local_pos.entry_price - exit_price) / local_pos.entry_price) * 100
                    margin = (local_pos.entry_price * local_pos.qty) / self.settings.leverage
                    pnl_usd = (pnl_pct / 100) * margin
                    fee_usd = self._calculate_fees_usd(local_pos.entry_price, exit_price, local_pos.qty)
                    if fee_usd > 0:
                        pnl_usd -= fee_usd
                        if margin > 0:
                            pnl_pct = (pnl_usd / margin) * 100
                    self.state.update_trade_on_close(symbol, exit_price, pnl_usd, pnl_pct, "MANUAL_CLOSE")
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
                else:
                    # Если не удалось получить цену, используем entry_price с нулевым PnL
                    self.state.update_trade_on_close(symbol, local_pos.entry_price, 0.0, 0.0, "ERROR_CLOSE")
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
            except Exception as e2:
                logger.error(f"Error in fallback close handling for {symbol}: {e2}")
                # Последняя попытка - закрываем с entry_price
                try:
                    self.state.update_trade_on_close(symbol, local_pos.entry_price, 0.0, 0.0, "ERROR_CLOSE")
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
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
