import logging
import asyncio
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
    from telegram.error import BadRequest, Conflict
except ImportError as e:
    raise ImportError(
        "python-telegram-bot не установлен. Установите его командой: pip install python-telegram-bot\n"
        "Или установите все зависимости: pip install -r requirements.txt"
    ) from e
from bot.config import AppSettings
from bot.state import BotState
from bot.model_manager import ModelManager
from pathlib import Path
from typing import Optional

# Логирование уже настроено в run_bot.py, не нужно настраивать здесь
# logging.basicConfig() добавляет обработчик к root logger, что вызывает дублирование логов
logger = logging.getLogger(__name__)


def _find_script_path(script_name: str) -> Optional[Path]:
    """Ищет скрипт в корне проекта или в текущей рабочей директории (для деплоя)."""
    candidates = [
        Path(__file__).resolve().parent.parent / script_name,
        Path.cwd() / script_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def safe_float(value, default=0.0):
    """Безопасное преобразование в float, обрабатывает пустые строки и None"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

class TelegramBot:
    def __init__(self, settings: AppSettings, state: BotState, model_manager: ModelManager, bybit_client=None):
        self.settings = settings
        self.state = state
        self.model_manager = model_manager
        self.bybit = bybit_client
        self.app = None
        self.trading_loop = None  # Будет установлен из run_bot.py
        self.waiting_for_symbol = {}  # user_id -> True если ждем ввод символа
        self.waiting_for_risk_setting = {}  # user_id -> setting_name для редактирования настроек риска
        self.waiting_for_ml_setting = {}  # user_id -> setting_name для редактирования ML настроек
        self.waiting_for_mtf_selection = {}  # user_id -> {"symbol": str, "step": "1h"|"15m"} для выбора MTF моделей
        
        # Инициализируем файл настроек при старте (добавляем недостающие поля)
        self._ensure_ml_settings_file()

    async def start(self):
        if not self.settings.telegram_token:
            logger.error("No Telegram token found in settings!")
            return

        self.app = Application.builder().token(self.settings.telegram_token).build()

        # Handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        logger.info("Starting Telegram bot...")
        try:
            await self.app.initialize()
            await self.app.start()
            
            # Try to start polling with conflict error handling
            try:
                await self.app.updater.start_polling()
            except Conflict as e:
                logger.error(
                    f"Telegram bot conflict error: {e}\n"
                    "Another bot instance is already running. "
                    "Please stop the other instance before starting this one."
                )
                # Clean up before raising
                await self.shutdown()
                raise
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}", exc_info=True)
            # Ensure cleanup on any error
            if self.app:
                await self.shutdown()
            raise

    async def shutdown(self):
        """Properly shutdown the Telegram bot"""
        if not self.app:
            return
        
        try:
            logger.info("Shutting down Telegram bot...")
            # Stop polling first
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            # Then shutdown the application
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot shutdown complete")
        except Exception as e:
            logger.error(f"Error during Telegram bot shutdown: {e}", exc_info=True)

    async def check_auth(self, update: Update) -> bool:
        user_id = update.effective_user.id
        if self.settings.allowed_user_id and user_id != self.settings.allowed_user_id:
            await update.message.reply_text("⛔ Доступ запрещен. Ваш ID не в вайтлисте.")
            return False
        return True
    
    async def safe_edit_message(self, query, text: str, reply_markup=None):
        """Безопасное редактирование сообщения с обработкой ошибки 'Message is not modified'"""
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except BadRequest as e:
            # Игнорируем ошибку "Message is not modified" - это нормально, если содержимое не изменилось
            if "Message is not modified" in str(e):
                logger.debug(f"Message not modified (non-critical): {e}")
            else:
                raise

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_auth(update): return
        
        keyboard = [
            [InlineKeyboardButton("🟢 СТАРТ", callback_data="bot_start"),
             InlineKeyboardButton("🔴 СТОП", callback_data="bot_stop")],
            [InlineKeyboardButton("📊 СТАТУС", callback_data="status_info"),
             InlineKeyboardButton("📈 СТАТИСТИКА", callback_data="stats")],
            [InlineKeyboardButton("⚙️ НАСТРОЙКИ ПАР", callback_data="settings_pairs"),
             InlineKeyboardButton("🤖 МОДЕЛИ", callback_data="settings_models")],
            [InlineKeyboardButton("📝 ИСТОРИЯ", callback_data="history_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("🤖 ML Trading Bot Terminal", reply_markup=reply_markup)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_auth(update): return
        await self.show_status(update)
    
    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_auth(update): return
        # Создаем фейковый query для использования show_dashboard
        class FakeQuery:
            def __init__(self, message):
                self.message = message
            async def edit_message_text(self, text, reply_markup=None):
                await self.message.reply_text(text, reply_markup=reply_markup)
        await self.show_dashboard(FakeQuery(update.message))

    async def show_status(self, update_or_query):
        status_text = f"🤖 СТАТУС ТЕРМИНАЛА: {'🟢 РАБОТАЕТ' if self.state.is_running else '🔴 ОСТАНОВЛЕН'}\n\n"
        
        # Account Info и Open Positions (если есть доступ к bybit)
        wallet_balance = 0.0
        open_positions = []
        total_margin = 0.0
        
        if self.bybit:
            try:
                balance_info = self.bybit.get_wallet_balance()
                if balance_info.get("retCode") == 0:
                    result = balance_info.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        wallet = list_data[0].get("coin", [])
                        usdt_coin = next((c for c in wallet if c.get("coin") == "USDT"), None)
                        if usdt_coin:
                            wallet_balance = safe_float(usdt_coin.get("walletBalance"), 0)
            
            except Exception as e:
                logger.error(f"Error getting balance: {e}")
            
            # Open Positions
            try:
                for symbol in self.state.active_symbols:
                    pos_info = self.bybit.get_position_info(symbol=symbol)
                    if pos_info.get("retCode") == 0:
                        list_data = pos_info.get("result", {}).get("list", [])
                        for p in list_data:
                            size = safe_float(p.get("size"), 0)
                            if size > 0:
                                side = p.get("side")
                                entry_price = safe_float(p.get("avgPrice"), 0)
                                
                                # Получаем текущую цену (пробуем разные поля)
                                mark_price = safe_float(p.get("markPrice"), 0)
                                if mark_price == 0:
                                    mark_price = safe_float(p.get("lastPrice"), entry_price)
                                if mark_price == 0:
                                    mark_price = entry_price
                                
                                unrealised_pnl = safe_float(p.get("unrealisedPnl"), 0)
                                symbol_for_lev = p.get("symbol", "")
                                default_lev = self.settings.get_leverage_for_symbol(symbol_for_lev)
                                leverage_str = p.get("leverage", str(default_lev))
                                leverage = safe_float(leverage_str, default_lev)
                                
                                # Получаем маржу (пробуем разные поля)
                                margin = safe_float(p.get("positionMargin"), 0)
                                if margin == 0:
                                    margin = safe_float(p.get("positionIM"), 0)  # Initial Margin
                                if margin == 0:
                                    # Рассчитываем маржу из стоимости позиции и плеча
                                    position_value = safe_float(p.get("positionValue"), 0)
                                    if position_value > 0 and leverage > 0:
                                        margin = position_value / leverage
                                
                                tp = p.get("takeProfit")
                                sl = p.get("stopLoss")
                                
                                # Логируем для отладки если данные неполные
                                if margin == 0 or mark_price == 0:
                                    logger.debug(f"Position data for {symbol}: size={size}, margin={margin}, markPrice={mark_price}, raw_data={p}")
                                
                                pnl_pct = ((mark_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - mark_price) / entry_price * 100)
                                
                                open_positions.append({
                                    "symbol": symbol,
                                    "side": side,
                                    "size": size,
                                    "entry": entry_price,
                                    "current": mark_price,
                                    "pnl": unrealised_pnl,
                                    "pnl_pct": pnl_pct,
                                    "leverage": leverage,
                                    "margin": margin,
                                    "tp": float(tp) if tp else None,
                                    "sl": float(sl) if sl else None
                                })
                                # Суммируем маржу для расчета доступного баланса
                                total_margin += margin
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
        
        # Вычисляем доступный баланс: баланс минус сумма маржи всех позиций
        available = wallet_balance - total_margin
        if available < 0:
            available = 0.0  # Не показываем отрицательные значения
        
        # Показываем Account Info
        if wallet_balance > 0:
            status_text += f"💰 ACCOUNT INFO:\n"
            status_text += f"Баланс: ${wallet_balance:.2f} | Доступно: ${available:.2f}\n\n"
        
        if open_positions:
            status_text += "📊 OPEN POSITIONS:\n"
            for pos in open_positions:
                side_emoji = "📈" if pos["side"] == "Buy" else "📉"
                pnl_sign = "+" if pos["pnl"] >= 0 else ""
                status_text += f"{side_emoji} {pos['symbol']} ({pos['leverage']}x) | {pos['side']}\n"
                status_text += f"   Размер: {pos['size']:.4f} | Маржа: ${pos['margin']:.2f}\n"
                status_text += f"   Вход: ${pos['entry']:.2f} | Тек: ${pos['current']:.2f}\n"
                status_text += f"   PnL: {pnl_sign}${pos['pnl']:.2f} ({pnl_sign}{pos['pnl_pct']:.2f}%)\n"
                if pos["tp"]:
                    status_text += f"   TP: ${pos['tp']:.2f}"
                if pos["sl"]:
                    status_text += f" | SL: ${pos['sl']:.2f}"
                status_text += "\n\n"
        else:
            status_text += "📊 OPEN POSITIONS:\n(нет открытых позиций)\n\n"
        
        # Active Strategy
        status_text += "📈 ACTIVE STRATEGY:\n"
        if not self.state.active_symbols:
            status_text += "  (нет активных пар)\n"
        else:
            for symbol in self.state.active_symbols:
                # Проверяем, есть ли стратегия в trading_loop (это актуальная информация)
                strategy = None
                if hasattr(self, 'trading_loop') and self.trading_loop:
                    strategy = self.trading_loop.strategies.get(symbol)
                
                if strategy and hasattr(strategy, 'predict_combined'):
                    # Это MTF стратегия
                    model_1h_path = getattr(strategy, 'model_1h_path', None)
                    model_15m_path = getattr(strategy, 'model_15m_path', None)
                    
                    if model_1h_path and model_15m_path:
                        # Извлекаем имя модели из Path объекта
                        if isinstance(model_1h_path, Path):
                            model_1h_name = model_1h_path.stem
                        elif isinstance(model_1h_path, str):
                            model_1h_name = Path(model_1h_path).stem
                        else:
                            model_1h_name = str(model_1h_path)
                        
                        if isinstance(model_15m_path, Path):
                            model_15m_name = model_15m_path.stem
                        elif isinstance(model_15m_path, str):
                            model_15m_name = Path(model_15m_path).stem
                        else:
                            model_15m_name = str(model_15m_path)
                        
                        status_text += f"Пара: {symbol} | 🔄 MTF стратегия:\n"
                        status_text += f"   1h: {model_1h_name}\n"
                        status_text += f"   15m: {model_15m_name}\n"
                        
                        # Безопасное получение порогов уверенности
                        conf_1h = getattr(strategy, 'confidence_threshold_1h', None)
                        conf_15m = getattr(strategy, 'confidence_threshold_15m', None)
                        if conf_1h is not None and conf_15m is not None:
                            status_text += f"   🎯 Уверенность 1h: ≥{conf_1h*100:.0f}% | 15m: ≥{conf_15m*100:.0f}%\n"
                        else:
                            # Используем значения из настроек, если параметры стратегии не установлены
                            conf_1h = self.settings.ml_strategy.mtf_confidence_threshold_1h
                            conf_15m = self.settings.ml_strategy.mtf_confidence_threshold_15m
                            status_text += f"   🎯 Уверенность 1h: ≥{conf_1h*100:.0f}% | 15m: ≥{conf_15m*100:.0f}%\n"
                    else:
                        status_text += f"Пара: {symbol} | 🔄 MTF стратегия (модели загружаются...)\n"
                elif strategy:
                    # Обычная стратегия
                    model_path = getattr(strategy, 'model_path', None)
                    if model_path and Path(model_path).exists():
                        model_name = Path(model_path).stem
                        
                        # Определяем тип модели
                        is_ensemble = "ensemble" in model_name.lower()
                        min_strength = 0.3 if is_ensemble else 60.0
                        
                        status_text += f"Пара: {symbol} | Модель: {model_name}\n"
                        status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                    else:
                        # Fallback к старому способу
                        model_path = self.state.symbol_models.get(symbol)
                        if model_path and Path(model_path).exists():
                            model_name = Path(model_path).stem
                            is_ensemble = "ensemble" in model_name.lower()
                            min_strength = 0.3 if is_ensemble else 60.0
                            status_text += f"Пара: {symbol} | Модель: {model_name}\n"
                            status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                        else:
                            # Пытаемся найти модель автоматически
                            models = self.model_manager.find_models_for_symbol(symbol)
                            if models:
                                model_path = str(models[0])
                                self.model_manager.apply_model(symbol, model_path)
                                model_name = models[0].stem
                                is_ensemble = "ensemble" in model_name.lower()
                                min_strength = 0.3 if is_ensemble else 60.0
                                status_text += f"Пара: {symbol} | Модель: {model_name} (авто)\n"
                                status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                            else:
                                status_text += f"Пара: {symbol} | Модель: ❌ Не найдена\n"
                else:
                    # Стратегия еще не инициализирована
                    model_path = self.state.symbol_models.get(symbol)
                    if model_path and Path(model_path).exists():
                        model_name = Path(model_path).stem
                        is_ensemble = "ensemble" in model_name.lower()
                        min_strength = 0.3 if is_ensemble else 60.0
                        status_text += f"Пара: {symbol} | Модель: {model_name} (ожидание инициализации)\n"
                        status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                    else:
                        status_text += f"Пара: {symbol} | Модель: ❌ Не найдена\n"
                
                # Проверяем cooldown для пары
                cooldown_info = self.state.get_cooldown_info(symbol)
                if cooldown_info and cooldown_info["active"]:
                    hours_left = cooldown_info["hours_left"]
                    if hours_left < 1:
                        minutes_left = int(hours_left * 60)
                        status_text += f"   ❄️ Cooldown: {cooldown_info['reason']} | Разморозка через {minutes_left} мин\n"
                    else:
                        status_text += f"   ❄️ Cooldown: {cooldown_info['reason']} | Разморозка через {hours_left:.1f} ч\n"
        
        # Overall Stats
        stats = self.state.get_stats()
        status_text += f"\n💰 ОБЩИЙ PnL: {stats['total_pnl']:.2f} USD ({stats['win_rate']:.1f}% WR, {stats['total_trades']} сделок)"
        
        if hasattr(update_or_query, 'message'):
            await update_or_query.message.reply_text(status_text, reply_markup=self.get_main_keyboard())
        else:
            await update_or_query.edit_message_text(status_text, reply_markup=self.get_main_keyboard())

    def get_main_keyboard(self):
        keyboard = [
            [InlineKeyboardButton("🟢 СТАРТ", callback_data="bot_start"),
             InlineKeyboardButton("🔴 СТОП", callback_data="bot_stop")],
            [InlineKeyboardButton("📊 СТАТУС", callback_data="status_info"),
             InlineKeyboardButton("📈 СТАТИСТИКА", callback_data="stats")],
            [InlineKeyboardButton("⚙️ НАСТРОЙКИ ПАР", callback_data="settings_pairs"),
             InlineKeyboardButton("🤖 МОДЕЛИ", callback_data="settings_models")],
            [InlineKeyboardButton("⚙️ НАСТРОЙКИ РИСКА", callback_data="settings_risk"),
             InlineKeyboardButton("🧠 ML НАСТРОЙКИ", callback_data="settings_ml")],
            [InlineKeyboardButton("📝 ИСТОРИЯ", callback_data="history_menu"),
             InlineKeyboardButton("🚨 ЭКСТРЕННЫЕ", callback_data="emergency_menu")]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        # Отвечаем на callback query сразу, чтобы избежать таймаута Telegram
        # Если ответ не успел - не критично, пользователь все равно получит обновленное сообщение
        try:
            await query.answer()
        except Exception as e:
            # Игнорируем ошибки "Query is too old" - это не критично
            logger.debug(f"Could not answer callback query (non-critical): {e}")

        if query.data == "bot_start":
            self.state.set_running(True)
            await query.edit_message_text("✅ Бот запущен!", reply_markup=self.get_main_keyboard())
        elif query.data == "bot_stop":
            self.state.set_running(False)
            await query.edit_message_text("🛑 Бот остановлен!", reply_markup=self.get_main_keyboard())
        elif query.data == "status_info":
            await self.show_status(query)
        elif query.data == "settings_pairs":
            await self.show_pairs_settings(query)
        elif query.data.startswith("toggle_risk_"):
            setting_name = query.data.replace("toggle_risk_", "")
            await self.toggle_risk_setting(query, setting_name)
        elif query.data.startswith("remove_cooldown_"):
            symbol = query.data.replace("remove_cooldown_", "")
            # Защита от конфликтов
            if not symbol.endswith("USDT"):
                await query.answer("⚠️ Некорректный символ", show_alert=True)
                return
            
            logger.info(f"[telegram_bot] Removing cooldown for {symbol}")
            try:
                # Выполняем в отдельном потоке с таймаутом, чтобы не блокировать event loop
                await asyncio.wait_for(
                    asyncio.to_thread(self.state.remove_cooldown, symbol),
                    timeout=3.0  # Таймаут 3 секунды
                )
                logger.info(f"[telegram_bot] Cooldown removed for {symbol}")
            except asyncio.TimeoutError:
                logger.warning(f"[telegram_bot] Timeout removing cooldown for {symbol}")
                await query.answer("⚠️ Таймаут при снятии разморозки, попробуйте еще раз", show_alert=True)
                return
            except Exception as e:
                logger.error(f"[telegram_bot] Error removing cooldown for {symbol}: {e}", exc_info=True)
                await query.answer(f"❌ Ошибка при снятии разморозки: {str(e)}", show_alert=True)
                return
            
            await query.answer(f"✅ Разморозка снята для {symbol}", show_alert=True)
            
            # Обновляем меню с таймаутом
            try:
                await asyncio.wait_for(
                    self.show_pairs_settings(query),
                    timeout=5.0  # Таймаут 5 секунд для обновления меню
                )
            except asyncio.TimeoutError:
                logger.warning(f"[telegram_bot] Timeout showing pairs settings after removing cooldown")
                await query.answer("⚠️ Меню обновляется...", show_alert=False)
        elif query.data.startswith("toggle_ml_"):
            # Обрабатываем переключение ML настроек ПЕРЕД общим toggle_
            setting_name = query.data.replace("toggle_ml_", "")
            logger.info(f"Handling toggle_ml callback: query.data={query.data}, setting_name={setting_name}")
            try:
                await self.toggle_ml_setting(query, setting_name)
            except Exception as e:
                logger.error(f"Error in toggle_ml_setting: {e}", exc_info=True)
                await query.answer(f"❌ Ошибка при переключении настройки: {str(e)}", show_alert=True)
        elif query.data.startswith("toggle_risk_"):
            setting_name = query.data.replace("toggle_risk_", "")
            await self.toggle_risk_setting(query, setting_name)
        elif query.data.startswith("toggle_"):
            symbol = query.data.split("_", 1)[1]
            # Защита от конфликтов с другими callback_data
            if not symbol.endswith("USDT"):
                await query.answer("⚠️ Некорректный символ", show_alert=True)
                return
            res = self.state.toggle_symbol(symbol)
            if res is None:
                await query.answer("⚠️ Достигнут лимит в 5 пар!", show_alert=True)
            await self.show_pairs_settings(query)
        elif query.data.startswith("edit_pair_leverage_"):
            symbol = query.data.replace("edit_pair_leverage_", "")
            user_id = query.from_user.id
            if not hasattr(self, 'waiting_for_pair_leverage'):
                self.waiting_for_pair_leverage = {}
            self.waiting_for_pair_leverage[user_id] = symbol
            
            pair_leverage = self.settings.get_leverage_for_symbol(symbol)
            text = (
                f"⚙️ <b>Настройка плеча для {symbol}</b>\n\n"
                f"Текущее значение: <b>{pair_leverage}x</b>\n\n"
                f"Введите новое значение плеча (от 1 до 100):\n"
                f"<i>(При изменении плеча будут автоматически пересчитаны уровни TP/SL для новых сделок)</i>"
            )
            keyboard = [[InlineKeyboardButton("🔙 Отмена", callback_data="settings_pairs")]]
            await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
            await query.answer()
        elif query.data == "history_menu":
            await self.show_history_menu(query)
        elif query.data == "history_signals":
            await self.show_signals(query)
        elif query.data == "history_trades":
            await self.show_trades(query)
        elif query.data == "logs_menu":
            await self.show_logs_menu(query)
        elif query.data == "logs_bot":
            await self.show_bot_logs(query)
        elif query.data == "logs_trades":
            await self.show_trades_logs(query)
        elif query.data == "logs_signals":
            await self.show_signals_logs(query)
        elif query.data == "logs_errors":
            await self.show_errors_logs(query)
        elif query.data == "stats":
            await self.show_stats(query)
        elif query.data == "settings_models":
            await self.show_models_settings(query)
        elif query.data == "add_pair":
            user_id = query.from_user.id
            self.waiting_for_symbol[user_id] = True
            await query.edit_message_text(
                "➕ ДОБАВЛЕНИЕ НОВОЙ ПАРЫ\n\n"
                "Введите символ торговой пары (например: XRPUSDT, ADAUSDT, DOGEUSDT)\n\n"
                "Символ должен быть в формате: BASEUSDT\n"
                "Например: BTCUSDT, ETHUSDT, SOLUSDT",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отмена", callback_data="cancel_add_pair")]])
            )
        elif query.data == "cancel_add_pair":
            user_id = query.from_user.id
            self.waiting_for_symbol.pop(user_id, None)
            await self.show_pairs_settings(query)
        elif query.data.startswith("select_model_"):
            symbol = query.data.replace("select_model_", "")
            await self.show_model_selection(query, symbol)
        elif query.data.startswith("select_mtf_1h_"):
            # Проверяем более специфичный префикс первым
            symbol = query.data.replace("select_mtf_1h_", "").upper()
            logger.debug(f"MTF 1h selection: callback_data={query.data}, extracted symbol={symbol}")
            await self.show_mtf_timeframe_selection(query, symbol, "1h")
        elif query.data.startswith("select_mtf_15m_"):
            # Проверяем более специфичный префикс первым
            symbol = query.data.replace("select_mtf_15m_", "").upper()
            logger.debug(f"MTF 15m selection: callback_data={query.data}, extracted symbol={symbol}")
            await self.show_mtf_timeframe_selection(query, symbol, "15m")
        elif query.data.startswith("select_mtf_"):
            symbol = query.data.replace("select_mtf_", "").upper()
            logger.debug(f"MTF model selection: callback_data={query.data}, extracted symbol={symbol}")
            await self.show_mtf_model_selection(query, symbol)
        elif query.data.startswith("apply_mtf_model_"):
            # Формат: apply_mtf_model_{symbol}_{timeframe}_{model_index}
            # Символы обычно не содержат подчеркиваний, так что split должен работать
            remaining = query.data.replace("apply_mtf_model_", "")
            parts = remaining.split("_", 2)  # Разбиваем максимум на 3 части
            if len(parts) == 3:
                symbol = parts[0].upper()
                timeframe = parts[1]  # "1h" или "15m"
                try:
                    model_index = int(parts[2])
                    logger.debug(f"apply_mtf_model: symbol={symbol}, timeframe={timeframe}, index={model_index}")
                    await self.apply_mtf_model_selection(query, symbol, timeframe, model_index)
                except ValueError:
                    logger.error(f"Invalid model_index in callback_data: {query.data}, parts={parts}")
                    await query.answer("❌ Ошибка: некорректный индекс модели", show_alert=True)
            else:
                logger.error(f"Invalid callback_data format: {query.data}, parts={parts}")
                await query.answer("❌ Ошибка: некорректный формат данных", show_alert=True)
        elif query.data.startswith("apply_mtf_strategy_"):
            symbol = query.data.replace("apply_mtf_strategy_", "")
            await self.apply_mtf_strategy(query, symbol)
        elif query.data.startswith("retrain_all_models_for_symbol_"):
            symbol = query.data.replace("retrain_all_models_for_symbol_", "").upper()
            await query.edit_message_text(
                f"🔄 Запускаю переобучение всех моделей для {symbol}...\n\n"
                "Это включает:\n"
                "• 15m модели (без MTF и с MTF)\n"
                "• 1h модели (без MTF и с MTF)\n\n"
                "Это может занять 30-60 минут.\n"
                "Вы будете получать уведомления о прогрессе.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⏳ Ожидание...", callback_data="waiting")]])
            )
            asyncio.create_task(self.retrain_all_models_for_symbol_async(symbol, query.from_user.id))
        elif query.data.startswith("test_all_mtf_combinations_"):
            symbol = query.data.replace("test_all_mtf_combinations_", "").upper()
            await query.edit_message_text(
                f"🧪 Запускаю тестирование всех MTF комбинаций для {symbol}...\n\n"
                "Это включает:\n"
                "• Тестирование всех комбинаций 1h × 15m моделей\n"
                "• Сравнение результатов всех комбинаций\n"
                "• Выбор лучшей комбинации\n\n"
                "Это может занять 1-3 часа в зависимости от количества моделей.\n"
                "Вы будете получать уведомления о прогрессе.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⏳ Ожидание...", callback_data="waiting")]])
            )
            asyncio.create_task(self.test_all_mtf_combinations_async(symbol, query.from_user.id))
        elif query.data.startswith("apply_model_"):
            # Формат: apply_model_{symbol}_{model_index}
            parts = query.data.replace("apply_model_", "").split("_", 1)
            if len(parts) == 2:
                symbol = parts[0]
                model_index = int(parts[1])
                await self.apply_selected_model(query, symbol, model_index)
        elif query.data.startswith("test_all_"):
            symbol = query.data.replace("test_all_", "")
            await query.edit_message_text(
                f"🧪 Запускаю тестирование всех моделей для {symbol}...\n"
                "Это может занять несколько минут.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⏳ Ожидание...", callback_data="waiting")]])
            )
            asyncio.create_task(self.test_all_models_async(symbol, query.from_user.id))
        elif query.data == "retrain_all":
            await query.edit_message_text("🔄 Запускаю переобучение всех моделей...\nЭто может занять время.", reply_markup=self.get_main_keyboard())
            # Запускаем в фоне
            asyncio.create_task(self.retrain_all_models_async(query.from_user.id))
        elif query.data.startswith("retrain_"):
            symbol = query.data.replace("retrain_", "")
            await query.edit_message_text(
                f"🎓 Запускаю обучение всех моделей для {symbol}...\n"
                "Это может занять 10-30 минут в зависимости от количества моделей.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⏳ Ожидание...", callback_data="waiting")]])
            )
            asyncio.create_task(self.retrain_symbol_models_async(symbol, query.from_user.id))
        elif query.data == "main_menu":
            await query.edit_message_text("🤖 ML Trading Bot Terminal", reply_markup=self.get_main_keyboard())
        elif query.data == "settings_risk":
            await self.show_risk_settings(query)
        elif query.data == "settings_ml":
            await self.show_ml_settings(query)
        elif query.data.startswith("edit_ml_"):
            setting_name = query.data.replace("edit_ml_", "")
            await self.start_edit_ml_setting(query, setting_name)
        elif query.data == "optimize_mtf_strategies":
            await query.edit_message_text(
                "🚀 ЗАПУСК ОПТИМИЗАЦИИ MTF СТРАТЕГИЙ\n\n"
                "Процесс включает:\n"
                "1. 📚 Обучение моделей (1h и 15m)\n"
                "2. 🔮 Предсказание лучших комбинаций\n"
                "3. 🧪 Реальное тестирование топ-15\n"
                "4. ✅ Выбор и применение лучших\n\n"
                "Это может занять 1-3 часа в зависимости от количества символов.\n"
                "Вы получите уведомления о прогрессе.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⏳ Ожидание...", callback_data="waiting")]])
            )
            # Используем run_in_executor для запуска тяжелой задачи
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, lambda: asyncio.run(self.optimize_mtf_strategies_async(query.from_user.id)))

        elif query.data.startswith("edit_risk_"):
            setting_name = query.data.replace("edit_risk_", "")
            await self.start_edit_risk_setting(query, setting_name)
        elif query.data == "reset_risk_defaults":
            await self.reset_risk_defaults(query)
        elif query.data == "risk_info":
            await self.show_risk_info(query)
        elif query.data == "emergency_menu":
            await self.show_emergency_menu(query)
        elif query.data == "emergency_stop_all":
            await self.emergency_stop_all(query)
        elif query.data == "dashboard":
            await self.show_dashboard(query)

    async def show_pairs_settings(self, query):
        # Получаем все известные символы (из state) - выполняем в отдельном потоке
        def get_symbols_data():
            all_possible = [
                s for s in self.state.known_symbols
                if isinstance(s, str) and s.endswith("USDT")
            ]
            # Гарантируем присутствие активных пар
            for s in self.state.active_symbols:
                if s not in all_possible:
                    all_possible.append(s)
            all_possible = sorted(set(all_possible))
            return all_possible, self.state.active_symbols
        
        all_possible, active_symbols = await asyncio.to_thread(get_symbols_data)
        
        keyboard = []
        # Собираем информацию о cooldown для всех символов параллельно с таймаутом
        async def get_cooldown_with_timeout(symbol):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.state.get_cooldown_info, symbol),
                    timeout=2.0  # Таймаут 2 секунды на каждый символ
                )
            except asyncio.TimeoutError:
                logger.warning(f"[telegram_bot] Cooldown info timeout for {symbol}")
                return None
        
        cooldown_tasks = [get_cooldown_with_timeout(s) for s in all_possible]
        cooldown_infos = await asyncio.gather(*cooldown_tasks)
        
        for s, cooldown_info in zip(all_possible, cooldown_infos):
            status = "✅" if s in active_symbols else "❌"
            button_text = f"{status} {s}"
            
            # Проверяем, есть ли cooldown для этой пары
            if cooldown_info and cooldown_info.get("active"):
                # Добавляем индикатор cooldown
                hours_left = cooldown_info.get("hours_left", 0)
                if hours_left < 1:
                    minutes_left = int(hours_left * 60)
                    button_text += f" ❄️({minutes_left}м)"
                else:
                    button_text += f" ❄️({hours_left:.1f}ч)"
            
            pair_leverage = self.settings.get_leverage_for_symbol(s)
            
            keyboard.append([
                InlineKeyboardButton(button_text, callback_data=f"toggle_{s}"),
                InlineKeyboardButton(f"⚙️ Плечо {pair_leverage}x", callback_data=f"edit_pair_leverage_{s}")
            ])
            
            # Если пара в cooldown, добавляем кнопку для снятия разморозки
            if cooldown_info and cooldown_info.get("active"):
                keyboard.append([InlineKeyboardButton(
                    f"🔥 Снять разморозку {s}", 
                    callback_data=f"remove_cooldown_{s}"
                )])
        
        keyboard.append([InlineKeyboardButton("➕ Добавить новую пару", callback_data="add_pair")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="status_info")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        await self.safe_edit_message(query, "⚙️ Настройка активных пар (макс 5):", reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_history_menu(self, query):
        keyboard = [
            [InlineKeyboardButton("🔍 ИСТОРИЯ СИГНАЛОВ", callback_data="history_signals")],
            [InlineKeyboardButton("📈 ИСТОРИЯ СДЕЛОК", callback_data="history_trades")],
            [InlineKeyboardButton("📋 ЛОГИ", callback_data="logs_menu")],
            [InlineKeyboardButton("🔙 Назад", callback_data="status_info")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await query.edit_message_text("📝 Меню истории:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_signals(self, query):
        signals = self.state.signals[-10:]
        if not signals:
            text = "История сигналов пуста."
        else:
            text = "🔍 ПОСЛЕДНИЕ СИГНАЛЫ:\n\n"
            for s in reversed(signals):
                text += f"🕒 {s.timestamp[11:19]} | {s.symbol} | {s.action} ({int(s.confidence*100)}%)\n"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="history_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_stats(self, query):
        stats = self.state.get_stats()
        all_trades = self.state.trades
        closed_trades = [t for t in all_trades if t.status == "closed"]
        open_trades = [t for t in all_trades if t.status == "open"]
        
        text = "📈 СТАТИСТИКА ТОРГОВЛИ:\n\n"
        text += f"💰 Общий PnL: {stats['total_pnl']:.2f} USD\n"
        text += f"📊 Винрейт: {stats['win_rate']:.1f}%\n"
        text += f"🔢 Всего сделок: {len(all_trades)}\n"
        text += f"   • Закрыто: {len(closed_trades)}\n"
        text += f"   • Открыто: {len(open_trades)}\n\n"
        
        if closed_trades:
            wins = [t for t in closed_trades if t.pnl_usd > 0]
            losses = [t for t in closed_trades if t.pnl_usd < 0]
            text += f"✅ Прибыльных: {len(wins)}\n"
            text += f"❌ Убыточных: {len(losses)}\n"
            if wins:
                avg_win = sum(t.pnl_usd for t in wins) / len(wins)
                text += f"📈 Средний выигрыш: ${avg_win:.2f}\n"
            if losses:
                avg_loss = sum(t.pnl_usd for t in losses) / len(losses)
                text += f"📉 Средний проигрыш: ${avg_loss:.2f}\n"
        else:
            text += "ℹ️ Нет закрытых сделок для расчета статистики.\n"
            if open_trades:
                text += f"\n⚠️ Есть {len(open_trades)} открытая(ых) позиция(ий), которая(ые) не учитывается(ются) в статистике до закрытия.\n"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="status_info")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_trades(self, query):
        closed_trades = [t for t in self.state.trades if t.status == "closed"][-10:]
        if not closed_trades:
            text = "История сделок пуста."
        else:
            text = "📈 ПОСЛЕДНИЕ СДЕЛКИ:\n\n"
            for idx, t in enumerate(reversed(closed_trades)):
                pnl_sign = "+" if t.pnl_usd >= 0 else ""
                trade_idx = len(self.state.trades) - len(closed_trades) + idx
                
                # Форматируем время выхода
                exit_time_str = "N/A"
                if t.exit_time:
                    try:
                        exit_time_str = t.exit_time[11:19] if len(t.exit_time) > 19 else t.exit_time
                    except:
                        exit_time_str = str(t.exit_time)[:8]
                
                # Форматируем время входа
                entry_time_str = "N/A"
                if t.entry_time:
                    try:
                        entry_time_str = t.entry_time[11:19] if len(t.entry_time) > 19 else t.entry_time
                    except:
                        entry_time_str = str(t.entry_time)[:8]
                
                # Рассчитываем длительность
                duration_str = "N/A"
                if t.entry_time and t.exit_time:
                    try:
                        from datetime import datetime
                        entry_dt = datetime.fromisoformat(t.entry_time.replace('Z', '+00:00'))
                        exit_dt = datetime.fromisoformat(t.exit_time.replace('Z', '+00:00'))
                        duration = exit_dt - entry_dt
                        hours = duration.total_seconds() / 3600
                        if hours < 1:
                            duration_str = f"{int(duration.total_seconds() / 60)}м"
                        elif hours < 24:
                            duration_str = f"{hours:.1f}ч"
                        else:
                            duration_str = f"{hours/24:.1f}д"
                    except:
                        pass
                
                # Форматируем цену выхода
                exit_price = t.exit_price if t.exit_price and t.exit_price > 0 else None
                
                # Форматируем количество
                qty_str = f"{t.qty:.4f}" if t.qty > 0 else "N/A"
                
                # Эмодзи для PnL
                pnl_emoji = "✅" if t.pnl_usd > 0 else "❌" if t.pnl_usd < 0 else "➖"
                
                text += f"#{trade_idx} {pnl_emoji} {t.symbol} {t.side}\n"
                text += f"   📅 Вход: {entry_time_str} → Выход: {exit_time_str} ({duration_str})\n"
                text += f"   💰 Вход: ${t.entry_price:.2f}"
                if exit_price:
                    text += f" | Выход: ${exit_price:.2f}\n"
                else:
                    text += f" | Выход: N/A\n"
                text += f"   📊 Количество: {qty_str}\n"
                text += f"   💵 PnL: {pnl_sign}${t.pnl_usd:.2f} ({pnl_sign}{t.pnl_pct:.2f}%)\n\n"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="history_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    def _read_log_file(self, log_path: Path, max_lines: int = 50) -> list:
        """Читает последние N строк из лог-файла"""
        try:
            if not log_path.exists():
                return []
            
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Возвращаем последние max_lines строк
                return lines[-max_lines:] if len(lines) > max_lines else lines
        except Exception as e:
            logger.error(f"Error reading log file {log_path}: {e}", exc_info=True)
            return []

    async def show_logs_menu(self, query):
        """Показывает меню выбора типа логов"""
        keyboard = [
            [InlineKeyboardButton("📋 Основной лог", callback_data="logs_bot")],
            [InlineKeyboardButton("📈 Лог сделок", callback_data="logs_trades")],
            [InlineKeyboardButton("🔍 Лог сигналов", callback_data="logs_signals")],
            [InlineKeyboardButton("🚨 Лог ошибок", callback_data="logs_errors")],
            [InlineKeyboardButton("🔙 Назад", callback_data="history_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, "📋 Выберите тип логов для просмотра:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_bot_logs(self, query):
        """Показывает последние записи из основного лога"""
        log_path = Path("logs/bot.log")
        lines = self._read_log_file(log_path, max_lines=50)
        
        if not lines:
            text = "📋 ОСНОВНОЙ ЛОГ\n\nЛог-файл пуст или не найден."
        else:
            text = "📋 ПОСЛЕДНИЕ ЗАПИСИ ИЗ ОСНОВНОГО ЛОГА:\n\n"
            # Показываем последние 30 строк (чтобы поместилось в сообщение)
            for line in lines[-30:]:
                # Ограничиваем длину строки для Telegram (макс 4096 символов на сообщение)
                if len(line) > 200:
                    line = line[:197] + "..."
                text += line
                if len(text) > 3500:  # Оставляем запас для заголовка и кнопок
                    text += "\n\n... (показаны последние записи)"
                    break
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить", callback_data="logs_bot")],
            [InlineKeyboardButton("🔙 Назад", callback_data="logs_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_trades_logs(self, query):
        """Показывает последние записи из лога сделок"""
        log_path = Path("logs/trades.log")
        lines = self._read_log_file(log_path, max_lines=50)
        
        if not lines:
            text = "📈 ЛОГ СДЕЛОК\n\nЛог-файл пуст или не найден."
        else:
            text = "📈 ПОСЛЕДНИЕ ЗАПИСИ ИЗ ЛОГА СДЕЛОК:\n\n"
            for line in lines[-30:]:
                if len(line) > 200:
                    line = line[:197] + "..."
                text += line
                if len(text) > 3500:
                    text += "\n\n... (показаны последние записи)"
                    break
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить", callback_data="logs_trades")],
            [InlineKeyboardButton("🔙 Назад", callback_data="logs_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_signals_logs(self, query):
        """Показывает последние записи из лога сигналов"""
        log_path = Path("logs/signals.log")
        lines = self._read_log_file(log_path, max_lines=50)
        
        if not lines:
            text = "🔍 ЛОГ СИГНАЛОВ\n\nЛог-файл пуст или не найден."
        else:
            text = "🔍 ПОСЛЕДНИЕ ЗАПИСИ ИЗ ЛОГА СИГНАЛОВ:\n\n"
            for line in lines[-30:]:
                if len(line) > 200:
                    line = line[:197] + "..."
                text += line
                if len(text) > 3500:
                    text += "\n\n... (показаны последние записи)"
                    break
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить", callback_data="logs_signals")],
            [InlineKeyboardButton("🔙 Назад", callback_data="logs_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_errors_logs(self, query):
        """Показывает последние записи из лога ошибок"""
        log_path = Path("logs/errors.log")
        lines = self._read_log_file(log_path, max_lines=50)
        
        if not lines:
            text = "🚨 ЛОГ ОШИБОК\n\nЛог-файл пуст или не найден."
        else:
            text = "🚨 ПОСЛЕДНИЕ ЗАПИСИ ИЗ ЛОГА ОШИБОК:\n\n"
            for line in lines[-30:]:
                if len(line) > 200:
                    line = line[:197] + "..."
                text += line
                if len(text) > 3500:
                    text += "\n\n... (показаны последние записи)"
                    break
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить", callback_data="logs_errors")],
            [InlineKeyboardButton("🔙 Назад", callback_data="logs_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_models_settings(self, query):
        text = "🤖 УПРАВЛЕНИЕ МОДЕЛЯМИ:\n\n"
        
        if not self.state.active_symbols:
            text += "Нет активных пар. Добавьте пары в настройках."
        else:
            for symbol in self.state.active_symbols:
                model_path = self.state.symbol_models.get(symbol)
                if model_path and Path(model_path).exists():
                    model_name = Path(model_path).stem
                    text += f"✅ {symbol}: {model_name}\n"
                else:
                    text += f"❌ {symbol}: Авто-поиск\n"
        
        keyboard = []
        # Кнопки для выбора модели для каждой пары
        for symbol in self.state.active_symbols:
            keyboard.append([InlineKeyboardButton(f"📌 Выбрать модель для {symbol}", callback_data=f"select_model_{symbol}")])
            # Добавляем кнопку для выбора MTF моделей
            keyboard.append([InlineKeyboardButton(f"🔄 Выбрать MTF модели для {symbol}", callback_data=f"select_mtf_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("🔄 Переобучить все модели", callback_data="retrain_all")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="status_info")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_auth(update): return
        
        user_id = update.effective_user.id
        text = update.message.text.strip()
        
        # Проверяем, ждем ли мы ввод настройки риска
        if user_id in self.waiting_for_risk_setting:
            setting_name = self.waiting_for_risk_setting.pop(user_id)
            await self.process_risk_setting_input(update, setting_name, text)
            return
        
        if user_id in getattr(self, 'waiting_for_pair_leverage', {}):
            symbol = self.waiting_for_pair_leverage.pop(user_id)
            await self.process_pair_leverage_input(update, symbol, text)
            return
        
        if user_id in self.waiting_for_ml_setting:
            setting_name = self.waiting_for_ml_setting.pop(user_id)
            await self.process_ml_setting_input(update, setting_name, text)
            return
        
        # Проверяем, ждем ли мы ввод символа
        if self.waiting_for_symbol.get(user_id, False):
            self.waiting_for_symbol.pop(user_id, None)
            
            # Валидация формата символа
            if not text.endswith("USDT"):
                await update.message.reply_text(
                    "❌ Неверный формат! Символ должен заканчиваться на USDT.\n"
                    "Примеры: XRPUSDT, ADAUSDT, DOGEUSDT",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            # Проверяем, не добавлена ли уже эта пара
            if text in self.state.active_symbols:
                await update.message.reply_text(
                    f"ℹ️ Пара {text} уже активна.",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            # Валидируем символ через Bybit API
            await update.message.reply_text(f"🔍 Проверка символа {text} на бирже...")
            
            try:
                # Пытаемся получить информацию об инструменте
                instrument_info = self.bybit.get_instrument_info(text)
                if not instrument_info or not instrument_info.get("symbol"):
                    await update.message.reply_text(
                        f"❌ Символ {text} не найден на бирже Bybit.\n"
                        "Проверьте правильность написания.",
                        reply_markup=self.get_main_keyboard()
                    )
                    return
                
                was_known = text in self.state.known_symbols
                # Символ валиден, добавляем в список известных
                self.state.add_known_symbol(text)
                
                # Включаем пару (если лимит)
                enable_result = self.state.enable_symbol(text)
                if enable_result is None:
                    await update.message.reply_text(
                        f"⚠️ Пара {text} сохранена, но лимит активных пар достигнут.\n"
                        "Отключите одну из активных пар и включите эту из списка.",
                        reply_markup=self.get_main_keyboard()
                    )
                    return
                
                # Проверяем, есть ли уже модели для пары
                has_models = False
                model_path = self.state.symbol_models.get(text)
                if model_path and Path(model_path).exists():
                    has_models = True
                if not has_models:
                    has_models = bool(self.model_manager.find_models_for_symbol(text))
                
                if has_models:
                    await update.message.reply_text(
                        f"✅ Пара {text} включена.\n"
                        "Модели уже существуют — обучение не требуется.",
                        reply_markup=self.get_main_keyboard()
                    )
                    return
                
                # Запускаем процесс обучения модели в фоне
                await update.message.reply_text(
                    f"✅ Пара {text} добавлена!\n\n"
                    "🔄 Запускаю автоматическое обучение модели...\n"
                    "Это может занять несколько минут. Вы получите уведомление по завершении.",
                    reply_markup=self.get_main_keyboard()
                )
                
                # Запускаем обучение в фоне (не блокируем бота)
                asyncio.create_task(self.train_new_pair_async(text, user_id))
                
            except Exception as e:
                logger.error(f"Error validating/adding symbol {text}: {e}")
                await update.message.reply_text(
                    f"❌ Ошибка при добавлении пары {text}:\n{str(e)}",
                    reply_markup=self.get_main_keyboard()
                )
            return
        
        # Если не ждем ввод, просто игнорируем текст
        pass
    
    async def show_model_selection(self, query, symbol: str):
        """Показывает список доступных моделей для выбора с результатами тестов"""
        models = self.model_manager.find_models_for_symbol(symbol)
        
        if not models:
            await query.edit_message_text(
                f"❌ Для {symbol} не найдено моделей.\n\n"
                "Используйте кнопку 'Переобучить модель' для создания модели.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Переобучить", callback_data=f"retrain_{symbol}")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="settings_models")]
                ])
            )
            return
        
        text = f"📌 ВЫБОР МОДЕЛИ ДЛЯ {symbol}:\n\n"
        keyboard = []
        
        # Загружаем результаты тестов
        test_results = self.model_manager.get_model_test_results(symbol)
        
        # Проверяем, есть ли хотя бы одна протестированная модель
        has_tested = any(str(m) in test_results for m in models)
        
        for idx, model_path in enumerate(models):
            model_name = model_path.stem
            is_current = self.state.symbol_models.get(symbol) == str(model_path)
            prefix = "✅ " if is_current else ""
            
            # Получаем результаты теста для этой модели
            model_results = test_results.get(str(model_path), {})
            
            if model_results:
                pnl = model_results.get("total_pnl_pct", 0)
                winrate = model_results.get("win_rate", 0)
                trades = model_results.get("total_trades", 0)
                trades_per_day = model_results.get("trades_per_day", 0)
                profit_factor = model_results.get("profit_factor", 0)
                
                pnl_sign = "+" if pnl >= 0 else ""
                pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                text += f"{prefix}{pnl_color} {model_name}\n"
                text += f"   PnL: {pnl_sign}{pnl:.2f}% | WR: {winrate:.1f}% | PF: {profit_factor:.2f}\n"
                text += f"   Сделок: {trades} ({trades_per_day:.1f}/день)\n\n"
            else:
                text += f"{prefix}⚪ {model_name} (не тестирована)\n\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{'✅ ' if is_current else ''}{model_name}",
                callback_data=f"apply_model_{symbol}_{idx}"
            )])
        
        if not has_tested:
            keyboard.append([InlineKeyboardButton("🧪 Тестировать все модели (14 дней)", callback_data=f"test_all_{symbol}")])
        else:
            keyboard.append([InlineKeyboardButton("🔄 Обновить тесты", callback_data=f"test_all_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("🎓 Обучить все модели", callback_data=f"retrain_{symbol}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="settings_models")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def send_model_selection_menu(self, symbol: str, user_id: int):
        """Отправляет новое сообщение с меню выбора моделей для символа"""
        if not self.app or not self.settings.allowed_user_id:
            return
        
        models = self.model_manager.find_models_for_symbol(symbol)
        
        if not models:
            await self.app.bot.send_message(
                chat_id=user_id,
                text=f"❌ Для {symbol} не найдено моделей.\n\n"
                     "Используйте кнопку 'Переобучить модель' для создания модели.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Переобучить", callback_data=f"retrain_{symbol}")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="settings_models")]
                ])
            )
            return
        
        text = f"📌 ВЫБОР МОДЕЛИ ДЛЯ {symbol}:\n\n"
        keyboard = []
        
        # Загружаем результаты тестов
        test_results = self.model_manager.get_model_test_results(symbol)
        
        # Проверяем, есть ли хотя бы одна протестированная модель
        has_tested = any(str(m) in test_results for m in models)
        
        for idx, model_path in enumerate(models):
            model_name = model_path.stem
            is_current = self.state.symbol_models.get(symbol) == str(model_path)
            prefix = "✅ " if is_current else ""
            
            # Получаем результаты теста для этой модели
            model_results = test_results.get(str(model_path), {})
            
            if model_results:
                pnl = model_results.get("total_pnl_pct", 0)
                winrate = model_results.get("win_rate", 0)
                trades = model_results.get("total_trades", 0)
                trades_per_day = model_results.get("trades_per_day", 0)
                profit_factor = model_results.get("profit_factor", 0)
                
                pnl_sign = "+" if pnl >= 0 else ""
                pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                text += f"{prefix}{pnl_color} {model_name}\n"
                text += f"   PnL: {pnl_sign}{pnl:.2f}% | WR: {winrate:.1f}% | PF: {profit_factor:.2f}\n"
                text += f"   Сделок: {trades} ({trades_per_day:.1f}/день)\n\n"
            else:
                text += f"{prefix}⚪ {model_name} (не тестирована)\n\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{'✅ ' if is_current else ''}{model_name}",
                callback_data=f"apply_model_{symbol}_{idx}"
            )])
        
        if not has_tested:
            keyboard.append([InlineKeyboardButton("🧪 Тестировать все модели (14 дней)", callback_data=f"test_all_{symbol}")])
        else:
            keyboard.append([InlineKeyboardButton("🔄 Обновить тесты", callback_data=f"test_all_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("🎓 Обучить все модели", callback_data=f"retrain_{symbol}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="settings_models")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        
        try:
            await self.app.bot.send_message(
                chat_id=user_id,
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"Error sending model selection menu: {e}")
    
    async def apply_selected_model(self, query, symbol: str, model_index: int):
        """Применяет выбранную модель для символа"""
        models = self.model_manager.find_models_for_symbol(symbol)
        
        if model_index >= len(models):
            await query.answer("Ошибка: модель не найдена", show_alert=True)
            return
        
        model_path = models[model_index]
        self.model_manager.apply_model(symbol, str(model_path))
        
        await query.answer(f"✅ Модель применена для {symbol}!", show_alert=True)
        await self.show_models_settings(query)
    
    def find_models_for_timeframe(self, symbol: str, timeframe: str) -> list:
        """Находит модели для указанного таймфрейма (1h или 15m)"""
        models_dir = Path("ml_models")
        symbol_upper = symbol.upper()
        
        if timeframe == "1h":
            # Ищем модели 1h: *_{SYMBOL}_60_*.pkl или *_{SYMBOL}_*1h*.pkl
            patterns = [
                f"*_{symbol_upper}_60_*.pkl",
                f"*_{symbol_upper}_*1h*.pkl"
            ]
        elif timeframe == "15m":
            # Ищем модели 15m: *_{SYMBOL}_15_*.pkl или *_{SYMBOL}_*15m*.pkl
            patterns = [
                f"*_{symbol_upper}_15_*.pkl",
                f"*_{symbol_upper}_*15m*.pkl"
            ]
        else:
            return []
        
        models = []
        for pattern in patterns:
            for model_file in models_dir.glob(pattern):
                if model_file.is_file() and model_file not in models:
                    models.append(model_file)
        
        # Сортируем по времени изменения (новые первыми)
        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return models
    
    async def show_mtf_model_selection(self, query, symbol: str):
        """Показывает меню выбора MTF моделей (1h и 15m)"""
        # Нормализуем символ
        symbol = symbol.upper()
        logger.debug(f"show_mtf_model_selection called with symbol={symbol}")
        
        # Загружаем сохраненные MTF модели для символа
        mtf_models = self.load_mtf_models_for_symbol(symbol)
        
        text = f"🔄 ВЫБОР MTF МОДЕЛЕЙ ДЛЯ {symbol}:\n\n"
        
        if mtf_models:
            model_1h_name = mtf_models.get("model_1h", "Не выбрана")
            model_15m_name = mtf_models.get("model_15m", "Не выбрана")
            text += f"📊 Текущие модели:\n"
            text += f"   1h: {model_1h_name}\n"
            text += f"   15m: {model_15m_name}\n\n"
        else:
            text += "📊 Модели не выбраны\n\n"
        
        text += "Выберите таймфрейм для выбора модели:"
        
        keyboard = [
            [InlineKeyboardButton("⏰ Выбрать 1h модель", callback_data=f"select_mtf_1h_{symbol}")],
            [InlineKeyboardButton("⏱ Выбрать 15m модель", callback_data=f"select_mtf_15m_{symbol}")],
            [InlineKeyboardButton("✅ Применить MTF стратегию", callback_data=f"apply_mtf_strategy_{symbol}")],
            [InlineKeyboardButton("🧪 Тестировать все MTF комбинации", callback_data=f"test_all_mtf_combinations_{symbol}")],
            [InlineKeyboardButton("🔄 Переобучить все модели", callback_data=f"retrain_all_models_for_symbol_{symbol}")],
            [InlineKeyboardButton("🔙 Назад", callback_data="settings_models")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def _show_mtf_model_selection_with_status(self, query, symbol: str, status_message: str):
        """Показывает меню выбора MTF моделей с дополнительным статусом"""
        symbol = symbol.upper()
        
        # Загружаем сохраненные MTF модели для символа
        mtf_models = self.load_mtf_models_for_symbol(symbol)
        
        text = f"🔄 ВЫБОР MTF МОДЕЛЕЙ ДЛЯ {symbol}:\n\n"
        text += f"{status_message}\n\n"
        
        if mtf_models:
            model_1h_name = mtf_models.get("model_1h", "Не выбрана")
            model_15m_name = mtf_models.get("model_15m", "Не выбрана")
            text += f"📊 Текущие модели:\n"
            text += f"   1h: {model_1h_name}\n"
            text += f"   15m: {model_15m_name}\n\n"
        else:
            text += "📊 Модели не выбраны\n\n"
        
        text += "Выберите таймфрейм для выбора модели:"
        
        keyboard = [
            [InlineKeyboardButton("⏰ Выбрать 1h модель", callback_data=f"select_mtf_1h_{symbol}")],
            [InlineKeyboardButton("⏱ Выбрать 15m модель", callback_data=f"select_mtf_15m_{symbol}")],
            [InlineKeyboardButton("✅ Применить MTF стратегию", callback_data=f"apply_mtf_strategy_{symbol}")],
            [InlineKeyboardButton("🧪 Тестировать все MTF комбинации", callback_data=f"test_all_mtf_combinations_{symbol}")],
            [InlineKeyboardButton("🔄 Переобучить все модели", callback_data=f"retrain_all_models_for_symbol_{symbol}")],
            [InlineKeyboardButton("🔙 Назад", callback_data="settings_models")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def show_mtf_timeframe_selection(self, query, symbol: str, timeframe: str):
        """Показывает список моделей для выбранного таймфрейма"""
        # Нормализуем символ
        symbol = symbol.upper()
        logger.debug(f"show_mtf_timeframe_selection called with symbol={symbol}, timeframe={timeframe}")
        
        models = self.find_models_for_timeframe(symbol, timeframe)
        
        if not models:
            await self.safe_edit_message(
                query,
                f"❌ Для {symbol} не найдено {timeframe} моделей.\n\n"
                "Используйте кнопку 'Переобучить модель' для создания модели.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Переобучить", callback_data=f"retrain_{symbol}")],
                    [InlineKeyboardButton("🔙 Назад", callback_data=f"select_mtf_{symbol}")]
                ])
            )
            return
        
        # Загружаем сохраненные MTF модели
        mtf_models = self.load_mtf_models_for_symbol(symbol)
        current_model_name = None
        if timeframe == "1h":
            current_model_name = mtf_models.get("model_1h") if mtf_models else None
        else:
            current_model_name = mtf_models.get("model_15m") if mtf_models else None
        
        text = f"📌 ВЫБОР {timeframe.upper()} МОДЕЛИ ДЛЯ {symbol}:\n\n"
        keyboard = []
        
        # Загружаем результаты тестов
        test_results = self.model_manager.get_model_test_results(symbol)
        
        for idx, model_path in enumerate(models):
            model_name = model_path.stem
            is_current = current_model_name and model_name == current_model_name
            prefix = "✅ " if is_current else ""
            
            # Получаем результаты теста для этой модели
            model_results = test_results.get(str(model_path), {})
            
            if model_results:
                pnl = model_results.get("total_pnl_pct", 0)
                winrate = model_results.get("win_rate", 0)
                trades = model_results.get("total_trades", 0)
                profit_factor = model_results.get("profit_factor", 0)
                
                pnl_sign = "+" if pnl >= 0 else ""
                pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                text += f"{prefix}{pnl_color} {model_name}\n"
                text += f"   PnL: {pnl_sign}{pnl:.2f}% | WR: {winrate:.1f}% | PF: {profit_factor:.2f}\n"
                text += f"   Сделок: {trades}\n\n"
            else:
                text += f"{prefix}⚪ {model_name} (не тестирована)\n\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{'✅ ' if is_current else ''}{model_name}",
                callback_data=f"apply_mtf_model_{symbol}_{timeframe}_{idx}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data=f"select_mtf_{symbol}")])
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def apply_mtf_model_selection(self, query, symbol: str, timeframe: str, model_index: int):
        """Применяет выбранную модель для MTF стратегии"""
        # Нормализуем символ
        symbol = symbol.upper()
        logger.debug(f"apply_mtf_model_selection called with symbol={symbol}, timeframe={timeframe}, model_index={model_index}")
        
        models = self.find_models_for_timeframe(symbol, timeframe)
        
        if model_index >= len(models):
            await query.answer("Ошибка: модель не найдена", show_alert=True)
            return
        
        model_path = models[model_index]
        model_name = model_path.stem
        
        # Сохраняем выбранную модель
        self.save_mtf_model_for_symbol(symbol, timeframe, model_name)
        
        await query.answer(f"✅ {timeframe} модель {model_name} выбрана!", show_alert=True)
        
        # Показываем меню выбора MTF моделей снова
        await self.show_mtf_model_selection(query, symbol)
    
    def load_mtf_models_for_symbol(self, symbol: str) -> dict:
        """Загружает сохраненные MTF модели для символа из ml_settings.json"""
        try:
            from pathlib import Path
            import json
            
            project_root = Path(__file__).parent.parent
            config_file = project_root / "ml_settings.json"
            
            if not config_file.exists():
                return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверяем наличие секции mtf_models
            mtf_models = data.get("mtf_models", {})
            return mtf_models.get(symbol.upper(), {})
        except Exception as e:
            logger.error(f"Error loading MTF models for {symbol}: {e}")
            return {}
    
    def save_mtf_model_for_symbol(self, symbol: str, timeframe: str, model_name: str):
        """Сохраняет выбранную MTF модель для символа в ml_settings.json"""
        try:
            from pathlib import Path
            import json
            
            project_root = Path(__file__).parent.parent
            config_file = project_root / "ml_settings.json"
            
            # Загружаем существующие настройки
            data = {}
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read ml_settings.json: {e}")
            
            # Инициализируем секцию mtf_models если её нет
            if "mtf_models" not in data:
                data["mtf_models"] = {}
            
            symbol_upper = symbol.upper()
            if symbol_upper not in data["mtf_models"]:
                data["mtf_models"][symbol_upper] = {}
            
            # Сохраняем выбранную модель
            if timeframe == "1h":
                data["mtf_models"][symbol_upper]["model_1h"] = model_name
            elif timeframe == "15m":
                data["mtf_models"][symbol_upper]["model_15m"] = model_name
            
            # Сохраняем обратно
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved MTF model for {symbol}: {timeframe}={model_name}")
        except Exception as e:
            logger.error(f"Error saving MTF model for {symbol}: {e}", exc_info=True)
    
    async def apply_mtf_strategy(self, query, symbol: str):
        """Применяет выбранные MTF модели и перезапускает стратегию"""
        mtf_models = self.load_mtf_models_for_symbol(symbol)
        
        if not mtf_models or not mtf_models.get("model_1h") or not mtf_models.get("model_15m"):
            await query.answer(
                "❌ Необходимо выбрать обе модели (1h и 15m) перед применением MTF стратегии!",
                show_alert=True
            )
            await self.show_mtf_model_selection(query, symbol)
            return
        
        # Проверяем, что модели существуют
        models_dir = Path("ml_models")
        model_1h_path = models_dir / f"{mtf_models['model_1h']}.pkl"
        model_15m_path = models_dir / f"{mtf_models['model_15m']}.pkl"
        
        if not model_1h_path.exists() or not model_15m_path.exists():
            await query.answer(
                "❌ Одна из выбранных моделей не найдена! Проверьте файлы моделей.",
                show_alert=True
            )
            await self.show_mtf_model_selection(query, symbol)
            return
        
        # Убеждаемся, что MTF стратегия включена
        if not self.settings.ml_strategy.use_mtf_strategy:
            await query.answer(
                "⚠️ MTF стратегия не включена в настройках ML. Включите её сначала.",
                show_alert=True
            )
            return
        
        # Перезапускаем стратегию в trading_loop
        if hasattr(self, 'trading_loop') and self.trading_loop:
            try:
                # Очищаем существующую стратегию для символа
                if symbol in self.trading_loop.strategies:
                    del self.trading_loop.strategies[symbol]
                    logger.info(f"Cleared existing strategy for {symbol} to apply new MTF models")
                
                await query.answer(
                    f"✅ MTF стратегия применена для {symbol}!\n"
                    f"1h: {mtf_models['model_1h']}\n"
                    f"15m: {mtf_models['model_15m']}\n\n"
                    "Стратегия будет перезагружена при следующем цикле торговли.",
                    show_alert=True
                )
                # Обновляем сообщение с информацией о применении
                await self._show_mtf_model_selection_with_status(query, symbol, "✅ Стратегия применена!")
            except Exception as e:
                logger.error(f"Error applying MTF strategy for {symbol}: {e}", exc_info=True)
                await query.answer("❌ Ошибка при применении стратегии. Проверьте логи.", show_alert=True)
        else:
            await query.answer(
                f"✅ MTF модели сохранены для {symbol}!\n"
                f"1h: {mtf_models['model_1h']}\n"
                f"15m: {mtf_models['model_15m']}\n\n"
                "Стратегия будет загружена при следующем запуске бота.",
                show_alert=True
            )
            # Обновляем сообщение с информацией о сохранении
            await self._show_mtf_model_selection_with_status(query, symbol, "✅ Модели сохранены!")
    
    async def test_all_models_async(self, symbol: str, user_id: int):
        """Тестирует все модели для символа"""
        try:
            models = self.model_manager.find_models_for_symbol(symbol)
            if not models:
                await self.send_notification(f"❌ Для {symbol} не найдено моделей для тестирования.")
                return
            
            await self.send_notification(f"🧪 Начато тестирование {len(models)} моделей для {symbol}...")
            
            tested = 0
            for model_path in models:
                model_name = model_path.stem
                await self.send_notification(f"🧪 Тестирую {model_name}...")
                
                try:
                    results = self.model_manager.test_model(model_path, symbol, days=14)
                    
                    if results:
                        self.model_manager.save_model_test_result(symbol, str(model_path), results)
                        tested += 1
                        await self.send_notification(
                            f"✅ {model_name}:\n"
                            f"PnL: {results['total_pnl_pct']:+.2f}% | "
                            f"WR: {results['win_rate']:.1f}% | "
                            f"Сделок: {results['total_trades']} ({results['trades_per_day']:.1f}/день)"
                        )
                    else:
                        await self.send_notification(f"❌ Ошибка при тестировании {model_name}\n(проверьте логи для деталей)")
                except Exception as e:
                    logger.error(f"Error testing {model_name}: {e}", exc_info=True)
                    await self.send_notification(f"❌ Ошибка при тестировании {model_name}:\n{str(e)[:200]}")
            
            await self.send_notification(
                f"✅ Тестирование завершено!\n"
                f"Протестировано: {tested}/{len(models)} моделей"
            )
            
            # Автоматически открываем меню с моделями
            await self.send_model_selection_menu(symbol, user_id)
        except Exception as e:
            logger.error(f"Error testing models for {symbol}: {e}")
            await self.send_notification(f"❌ Ошибка при тестировании моделей: {str(e)}")
    
    async def retrain_all_models_async(self, user_id: int):
        """Переобучает все модели для активных пар"""
        try:
            await self.send_notification("🔄 Начато переобучение всех моделей...")
            
            for symbol in self.state.active_symbols:
                await self.send_notification(f"🔄 Обучение модели для {symbol}...")
                # Обучаем ТОЛЬКО модели БЕЗ MTF фичей (MTF фичи отключены)
                comparison = self.model_manager.train_and_compare(symbol, use_mtf=False)
                
                if comparison:
                    best_model = comparison.get("new_model", {})
                    model_path = best_model.get("model_path")
                    if model_path:
                        self.model_manager.apply_model(symbol, model_path)
                        await self.send_notification(f"✅ {symbol}: модель обновлена")
            
            await self.send_notification("✅ Переобучение всех моделей завершено!")
        except Exception as e:
            logger.error(f"Error retraining all models: {e}")
            await self.send_notification(f"❌ Ошибка при переобучении: {str(e)}")
    
    async def optimize_mtf_strategies_async(self, user_id: int):
        """Запускает оптимизацию MTF стратегий в фоне"""
        import subprocess
        import sys
        from pathlib import Path
        
        try:
            await self.send_notification("🚀 Начало оптимизации MTF стратегий...")
            
            # Получаем активные символы
            active_symbols = self.state.get_active_symbols()
            if not active_symbols:
                await self.send_notification("❌ Нет активных символов для оптимизации")
                return
            
            symbols_str = ",".join(active_symbols)
            
            # Запускаем скрипт оптимизации
            cmd = [
                sys.executable,
                "optimize_mtf_strategies.py",
                "--symbols", symbols_str,
                "--days", "30",
                "--top-n", "15"
            ]
            
            logger.info(f"Запуск оптимизации MTF: {' '.join(cmd)}")
            
            # Запускаем процесс
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Отправляем уведомление о начале
            await self.send_notification(
                f"📚 Этап 1/4: Обучение моделей для {len(active_symbols)} символов...\n"
                f"Символы: {symbols_str}"
            )
            
            # Читаем вывод процесса
            stdout_lines = []
            stderr_lines = []
            
            async def read_stdout():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        stdout_lines.append(line_str)
                        logger.info(f"[OPTIMIZE] {line_str}")
            
            async def read_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        stderr_lines.append(line_str)
                        logger.error(f"[OPTIMIZE ERROR] {line_str}")
            
            # Запускаем чтение вывода
            await asyncio.gather(read_stdout(), read_stderr())
            
            # Ждем завершения процесса
            return_code = await process.wait()
            
            if return_code == 0:
                # Ищем результаты в выводе
                results_text = "\n".join(stdout_lines[-20:])  # Последние 20 строк
                
                await self.send_notification(
                    f"✅ Оптимизация завершена успешно!\n\n"
                    f"Результаты:\n{results_text[-500:]}"  # Последние 500 символов
                )
                
                # Перезагружаем настройки, если они были обновлены
                if self.trading_loop:
                    try:
                        # Перезагружаем стратегии в trading_loop
                        await self.send_notification("🔄 Обновление стратегий в боте...")
                        # Trading loop автоматически перезагрузит стратегии при следующей итерации
                    except Exception as e:
                        logger.error(f"Ошибка обновления стратегий: {e}")
            else:
                error_text = "\n".join(stderr_lines[-10:])
                await self.send_notification(
                    f"❌ Ошибка оптимизации (код: {return_code})\n\n"
                    f"Ошибки:\n{error_text[-500:]}"
                )
        
        except Exception as e:
            logger.error(f"Ошибка при оптимизации MTF стратегий: {e}", exc_info=True)
            await self.send_notification(f"❌ Критическая ошибка оптимизации: {str(e)}")
    
    async def optimize_mtf_for_symbol_async(self, symbol: str):
        import subprocess
        import sys
        from pathlib import Path
        
        try:
            await self.send_notification(
                f"🚀 Начинаю оптимизацию MTF стратегий для {symbol}..."
            )
            
            cmd = [
                sys.executable,
                "optimize_mtf_strategies.py",
                "--symbols", symbol,
                "--days", "30",
                "--top-n", "15"
            ]
            
            logger.info(f"Запуск оптимизации MTF для {symbol}: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout_lines = []
            stderr_lines = []
            
            async def read_stdout():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        stdout_lines.append(line_str)
                        logger.info(f"[OPTIMIZE {symbol}] {line_str}")
            
            async def read_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        stderr_lines.append(line_str)
                        logger.error(f"[OPTIMIZE {symbol} ERROR] {line_str}")
            
            await asyncio.gather(read_stdout(), read_stderr())
            
            return_code = await process.wait()
            
            if return_code == 0:
                results_text = "\n".join(stdout_lines[-20:])
                await self.send_notification(
                    f"✅ Оптимизация MTF стратегий для {symbol} завершена!\n\n"
                    f"Результаты:\n{results_text[-500:]}"
                )
            else:
                error_text = "\n".join(stderr_lines[-10:])
                await self.send_notification(
                    f"❌ Ошибка оптимизации MTF стратегий для {symbol} (код: {return_code})\n\n"
                    f"Ошибки:\n{error_text[-500:]}"
                )
        except Exception as e:
            logger.error(f"Ошибка при оптимизации MTF стратегий для {symbol}: {e}", exc_info=True)
            await self.send_notification(f"❌ Критическая ошибка оптимизации MTF для {symbol}: {str(e)}")
    
    async def retrain_symbol_models_async(self, symbol: str, user_id: int):
        """Обучает все модели для конкретной торговой пары"""
        import subprocess
        from pathlib import Path
        
        try:
            await self.send_notification(
                f"🎓 Начато обучение всех моделей для {symbol}...\n"
                "Это может занять 10-30 минут.\n"
                "Вы будете получать уведомления о прогрессе."
            )
            
            # Путь к скрипту обучения (поддержка деплоя: ищем в корне проекта и в cwd)
            script_path = _find_script_path("retrain_all_models.py")
            if not script_path:
                await self.send_notification(f"❌ Скрипт обучения не найден: retrain_all_models.py")
                return
            
            # Обучаем ТОЛЬКО модели БЕЗ MTF фичей (MTF фичи отключены)
            cmd_args = ["python3", str(script_path), "--symbol", symbol, "--no-mtf"]
            
            # Запускаем обучение в отдельном процессе
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_path.parent)
            )
            
            # Отслеживаем вывод
            trained_models = []
            current_model = None
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_text = line.decode('utf-8', errors='ignore').strip()
                
                # Парсим вывод для уведомлений
                if "Обучение:" in line_text and symbol in line_text:
                    # Извлекаем название модели
                    parts = line_text.split("Обучение:")
                    if len(parts) > 1:
                        model_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        if model_name:
                            current_model = model_name
                            await self.send_notification(f"🔄 Обучение модели: {model_name} для {symbol}...")
                
                if "✅ Успешно завершено" in line_text and current_model:
                    trained_models.append(current_model)
                    await self.send_notification(f"✅ {current_model} обучена для {symbol}")
                    current_model = None
                
                if "❌ Ошибка" in line_text and current_model:
                    await self.send_notification(f"❌ Ошибка при обучении {current_model} для {symbol}")
                    current_model = None
            
            # Ждем завершения процесса
            await process.wait()
            
            if process.returncode == 0:
                await self.send_notification(
                    f"✅ Обучение всех моделей для {symbol} завершено!\n"
                    f"Обучено моделей: {len(trained_models)}\n\n"
                    "Обновите список моделей для просмотра результатов."
                )
                
                # Автоматически открываем меню с моделями
                await self.send_model_selection_menu(symbol, user_id)
            else:
                # Читаем ошибки
                stderr = await process.stderr.read()
                error_msg = stderr.decode('utf-8', errors='ignore')[:500]
                await self.send_notification(
                    f"❌ Ошибка при обучении моделей для {symbol}:\n{error_msg}"
                )
                
        except Exception as e:
            logger.error(f"Error retraining models for {symbol}: {e}", exc_info=True)
            await self.send_notification(f"❌ Ошибка при обучении моделей для {symbol}: {str(e)}")
    
    async def retrain_all_models_for_symbol_async(self, symbol: str, user_id: int):
        """Переобучает все модели по всем таймфреймам (15m и 1h) с MTF и без MTF для символа"""
        import subprocess
        import sys
        from pathlib import Path
        
        try:
            await self.send_notification(
                f"🔄 Начато переобучение всех моделей для {symbol}...\n\n"
                "Конфигурации:\n"
                "• 15m без MTF\n"
                "• 1h без MTF\n\n"
                "MTF фичи отключены.\n"
                "Это может занять 15-30 минут.\n"
                "Вы будете получать уведомления о прогрессе."
            )
            
            # Путь к скрипту обучения (поддержка деплоя: ищем в корне проекта и в cwd)
            script_path = _find_script_path("train_all_models_for_symbol.py")
            if not script_path:
                await self.send_notification(f"❌ Скрипт обучения не найден: train_all_models_for_symbol.py")
                return
            
            # Используем sys.executable для запуска с правильным Python
            python_exe = sys.executable
            cmd_args = [python_exe, str(script_path), "--symbol", symbol]
            
            # Запускаем обучение в отдельном процессе
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_path.parent)
            )
            
            # Отслеживаем вывод
            completed_configs = []
            current_config = None
            config_patterns = {
                "15m БЕЗ MTF": ["15m", "БЕЗ MTF"],
                "1h БЕЗ MTF": ["1h", "БЕЗ MTF"],
            }
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_text = line.decode('utf-8', errors='ignore').strip()
                
                # Парсим вывод для уведомлений
                for config_name, patterns in config_patterns.items():
                    if all(pattern in line_text for pattern in patterns):
                        if "Обучение моделей:" in line_text:
                            current_config = config_name
                            await self.send_notification(f"🔄 Обучение: {config_name} для {symbol}...")
                            break
                
                if "✅ Успешно:" in line_text and current_config:
                    completed_configs.append(current_config)
                    await self.send_notification(f"✅ {current_config} завершено для {symbol}")
                    current_config = None
                
                if "❌ Ошибка:" in line_text and current_config:
                    await self.send_notification(f"❌ Ошибка при обучении {current_config} для {symbol}")
                    current_config = None
                
                # Проверяем итоговую сводку
                if "ИТОГОВАЯ СВОДКА" in line_text:
                    await self.send_notification(f"📊 Итоговая сводка для {symbol}:\n{line_text[:200]}")
            
            # Ждем завершения процесса
            await process.wait()
            
            if process.returncode == 0:
                await self.send_notification(
                    f"✅ Переобучение всех моделей для {symbol} завершено!\n\n"
                    f"Завершено конфигураций: {len(completed_configs)}/2\n\n"
                    "Обновите список моделей для просмотра результатов."
                )
            else:
                # Читаем ошибки
                stderr = await process.stderr.read()
                error_msg = stderr.decode('utf-8', errors='ignore')[:500]
                await self.send_notification(
                    f"❌ Ошибка при переобучении моделей для {symbol}:\n{error_msg}"
                )
                
        except Exception as e:
            logger.error(f"Error retraining all models for {symbol}: {e}", exc_info=True)
            await self.send_notification(f"❌ Ошибка при переобучении моделей для {symbol}: {str(e)}")
    
    async def test_all_mtf_combinations_async(self, symbol: str, user_id: int):
        """Тестирует все MTF комбинации моделей для символа"""
        import subprocess
        import sys
        from pathlib import Path
        
        try:
            await self.send_notification(
                f"🧪 Начато тестирование всех MTF комбинаций для {symbol}...\n\n"
                "Это включает:\n"
                "• Тестирование всех комбинаций 1h × 15m моделей\n"
                "• Сравнение результатов всех комбинаций\n"
                "• Выбор лучшей комбинации\n\n"
                "Это может занять 1-3 часа в зависимости от количества моделей.\n"
                "Вы будете получать уведомления о прогрессе."
            )
            
            # Путь к скрипту тестирования (поддержка деплоя: ищем в корне проекта и в cwd)
            script_path = _find_script_path("test_all_mtf_combinations.py")
            if not script_path:
                await self.send_notification(f"❌ Скрипт тестирования не найден: test_all_mtf_combinations.py")
                return
            
            # Используем sys.executable для запуска с правильным Python
            python_exe = sys.executable
            cmd_args = [python_exe, str(script_path), "--symbol", symbol, "--days", "30"]
            
            # Запускаем тестирование в отдельном процессе
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_path.parent)
            )
            
            # Отслеживаем вывод
            total_combinations = 0
            completed_combinations = 0
            current_combo = None
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_text = line.decode('utf-8', errors='ignore').strip()
                
                # Парсим вывод для уведомлений
                if "Всего комбинаций:" in line_text:
                    try:
                        # Извлекаем количество комбинаций
                        parts = line_text.split(":")
                        if len(parts) > 1:
                            total_combinations = int(parts[1].strip())
                            await self.send_notification(
                                f"📊 Найдено {total_combinations} комбинаций для тестирования"
                            )
                    except:
                        pass
                
                if "Комбинация" in line_text and "/" in line_text:
                    try:
                        # Извлекаем номер комбинации
                        parts = line_text.split("Комбинация")[1].split("/")[0].strip()
                        combo_num = int(parts)
                        completed_combinations = combo_num
                        
                        # Отправляем уведомление каждые 10 комбинаций или при важных событиях
                        if combo_num % 10 == 0 or combo_num == 1:
                            progress_pct = (combo_num / total_combinations * 100) if total_combinations > 0 else 0
                            await self.send_notification(
                                f"🔄 Прогресс: {combo_num}/{total_combinations} комбинаций ({progress_pct:.1f}%)"
                            )
                    except:
                        pass
                
                if "✅ Результат:" in line_text:
                    # Извлекаем информацию о результате
                    if "PnL:" in line_text:
                        try:
                            pnl_part = line_text.split("PnL:")[1].split(",")[0].strip()
                            await self.send_notification(
                                f"✅ Комбинация {completed_combinations}: {line_text}"
                            )
                        except:
                            pass
                
                # Проверяем лучшие комбинации
                if "🏆 ЛУЧШИЕ КОМБИНАЦИИ" in line_text:
                    await self.send_notification("🏆 Начинаю анализ лучших комбинаций...")
                
                # Проверяем завершение
                if "✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" in line_text:
                    await self.send_notification("✅ Тестирование завершено! Анализирую результаты...")
            
            # Ждем завершения процесса
            await process.wait()
            
            if process.returncode == 0:
                # Ищем файл с результатами
                results_files = sorted(
                    Path(".").glob(f"mtf_combinations_{symbol}_*.csv"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True
                )
                
                if results_files:
                    results_file = results_files[0]
                    await self.send_notification(
                        f"✅ Тестирование всех MTF комбинаций для {symbol} завершено!\n\n"
                        f"📊 Результаты сохранены в:\n{results_file.name}\n\n"
                        "Откройте файл для просмотра всех комбинаций и выбора лучшей."
                    )
                else:
                    await self.send_notification(
                        f"✅ Тестирование завершено, но файл результатов не найден"
                    )
            else:
                # Читаем ошибки
                stderr = await process.stderr.read()
                error_msg = stderr.decode('utf-8', errors='ignore')[:500]
                await self.send_notification(
                    f"❌ Ошибка при тестировании MTF комбинаций для {symbol}:\n{error_msg}"
                )
                
        except Exception as e:
            logger.error(f"Error testing all MTF combinations for {symbol}: {e}", exc_info=True)
            await self.send_notification(f"❌ Ошибка при тестировании MTF комбинаций для {symbol}: {str(e)}")
    
    async def train_new_pair_async(self, symbol: str, user_id: int):
        """Асинхронная функция для обучения модели новой пары"""
        try:
            await self.send_notification(f"🔄 Начато обучение модели для {symbol}...")
            
            # Запускаем обучение (это синхронная операция, но мы в отдельной задаче)
            comparison = self.model_manager.train_and_compare(symbol, use_mtf=False)
            
            if comparison:
                best_model = comparison.get("new_model", {})
                model_name = best_model.get("model_filename", "unknown")
                pnl_pct = best_model.get("total_pnl_pct", 0)
                win_rate = best_model.get("win_rate_pct", 0)
                
                # Автоматически применяем лучшую модель
                if model_name and "model_path" in best_model:
                    self.model_manager.apply_model(symbol, best_model["model_path"])
                
                await self.send_notification(
                    f"✅ Обучение завершено для {symbol}!\n\n"
                    f"Модель: {model_name}\n"
                    f"PnL (14 дней): {pnl_pct:.2f}%\n"
                    f"Winrate: {win_rate:.1f}%\n\n"
                    f"Модель автоматически применена и готова к торговле.\n\n"
                    f"Дополнительно запущена оптимизация MTF стратегий для {symbol}."
                )
                
                asyncio.create_task(self.optimize_mtf_for_symbol_async(symbol))
            else:
                await self.send_notification(
                    f"⚠️ Обучение для {symbol} завершено, но не удалось выбрать лучшую модель.\n"
                    "Проверьте логи для деталей."
                )
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            await self.send_notification(
                f"❌ Ошибка при обучении модели для {symbol}:\n{str(e)}"
            )
    
    async def start_edit_risk_setting(self, query, setting_name: str):
        """Начинает редактирование настройки риска"""
        user_id = query.from_user.id
        
        # Определяем описание и примеры для разных настроек
        descriptions = {
            "margin_pct_balance": ("Маржа от баланса (в %)", "20", "Пример: 20 означает 20% от баланса"),
            "base_order_usd": ("Фиксированная сумма (в USD)", "50", "Пример: 50 означает $50 на позицию"),
            "stop_loss_pct": ("Stop Loss (в %)", "1.0", "Пример: 1.0 означает 1%"),
            "take_profit_pct": ("Take Profit (в %)", "2.5", "Пример: 2.5 означает 2.5%"),
            "fee_rate": ("Комиссия биржи (per side, в %)", "0.06", "Пример: 0.06 означает 0.06% за вход/выход"),
            "mid_term_tp_pct": ("Порог mid-term TP (в %)", "2.5", "Пример: 2.5 означает 2.5% от цены"),
            "long_term_tp_pct": ("Порог long-term TP (в %)", "4.0", "Пример: 4.0 означает 4% от цены"),
            "long_term_sl_pct": ("Порог long-term SL (в %)", "2.0", "Пример: 2.0 означает 2% от цены"),
            "dca_drawdown_pct": ("Просадка для DCA (в %)", "0.3", "Пример: 0.3 означает 0.3% от цены"),
            "dca_max_adds": ("Максимум DCA добавлений", "2", "Пример: 2 означает максимум 2 усреднения"),
            "dca_min_confidence": ("Мин. уверенность для DCA (в %)", "60", "Пример: 60 означает 60%"),
            "reverse_min_confidence": ("Мин. уверенность для реверса (в %)", "75", "Пример: 75 означает 75%"),
            "reverse_min_strength": ("Мин. сила для реверса", "сильное", "Пример: сильное или очень_сильное"),
            "trailing_stop_activation_pct": ("Активация трейлинг стопа (в %)", "0.3", "Пример: 0.3 означает 0.3%"),
            "trailing_stop_distance_pct": ("Расстояние трейлинг стопа (в %)", "0.2", "Пример: 0.2 означает 0.2%"),
            "breakeven_level1_activation_pct": ("Активация 1-й ступени безубытка (в %)", "0.5", "Пример: 0.5 означает 0.5%"),
            "breakeven_level1_sl_pct": ("SL для 1-й ступени безубытка (в %)", "0.2", "Пример: 0.2 означает 0.2%"),
            "breakeven_level2_activation_pct": ("Активация 2-й ступени безубытка (в %)", "1.0", "Пример: 1.0 означает 1.0%"),
            "breakeven_level2_sl_pct": ("SL для 2-й ступени безубытка (в %)", "0.5", "Пример: 0.5 означает 0.5%"),
        }
        
        if setting_name not in descriptions:
            await query.answer("Неизвестная настройка", show_alert=True)
            return
        
        desc, example, hint = descriptions[setting_name]
        current_value = getattr(self.settings.risk, setting_name, 0)
        
        # Для процентов показываем в процентах
        if setting_name.endswith("_pct"):
            current_display = current_value * 100
        elif setting_name == "base_order_usd":
            # Для USD показываем как есть
            current_display = current_value
        else:
            current_display = current_value
        
        self.waiting_for_risk_setting[user_id] = setting_name
        
        await query.edit_message_text(
            f"✏️ РЕДАКТИРОВАНИЕ: {desc}\n\n"
            f"Текущее значение: {current_display:.2f}\n"
            f"{hint}\n\n"
            f"Введите новое значение (только число):",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отмена", callback_data="settings_risk")]
            ])
        )
    
    async def process_pair_leverage_input(self, update: Update, symbol: str, text: str):
        try:
            value = int(text.strip())
            if not (1 <= value <= 100):
                await update.message.reply_text("❌ Значение должно быть от 1 до 100")
                return
            
            ml_settings = self.settings.get_ml_settings_for_symbol(symbol)
            ml_settings.leverage = value
            self.settings.set_ml_settings_for_symbol(symbol, ml_settings)
            
            from bot.config import save_symbol_ml_settings
            save_symbol_ml_settings(self.settings)
            
            if hasattr(self, 'trading_loop') and self.trading_loop:
                asyncio.create_task(self.trading_loop.update_leverage_for_symbol(symbol, value))
            
            text_resp = (
                f"✅ <b>Плечо для {symbol} обновлено на {value}x!</b>\n\n"
                f"Новые сделки будут открываться с этим плечом.\n"
                f"Для существующих позиций отправлен запрос на обновление плеча и пересчет TP/SL."
            )
            
            keyboard = [[InlineKeyboardButton("🔙 К настройкам пар", callback_data="settings_pairs")]]
            await update.message.reply_text(text_resp, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
            
        except ValueError:
            await update.message.reply_text("❌ Неверный формат. Введите целое число (например: 10)")
        except Exception as e:
            logger.error(f"Error processing pair leverage input: {e}")
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def process_ml_setting_input(self, update: Update, setting_name: str, text: str):
        """Обрабатывает ввод значения ML настройки"""
        try:
            # Парсим число
            value = float(text.replace(",", "."))
            
            # Валидация и применение
            ml_settings = self.settings.ml_strategy
            
            if setting_name == "confidence_threshold":
                if 1.0 <= value <= 100.0:  # 1% - 100%
                    ml_settings.confidence_threshold = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 100%")
                    return
            elif setting_name == "min_confidence_for_trade":
                if 1.0 <= value <= 100.0:  # 1% - 100%
                    ml_settings.min_confidence_for_trade = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 100%")
                    return
            
            # Сохраняем настройки
            self.save_ml_settings()
            
            # Обновляем стратегии в trading_loop (если он существует)
            if hasattr(self, 'trading_loop') and self.trading_loop:
                # Обновляем confidence_threshold в существующих стратегиях
                for symbol, strategy in self.trading_loop.strategies.items():
                    if hasattr(strategy, 'confidence_threshold'):
                        strategy.confidence_threshold = ml_settings.confidence_threshold
                        logger.info(f"Updated confidence_threshold for {symbol} strategy to {ml_settings.confidence_threshold}")
            
            # Показываем обновленные настройки
            ml_settings = self.settings.ml_strategy
            
            text = "🧠 НАСТРОЙКИ ML СТРАТЕГИИ\n\n"
            text += f"🎯 Минимальная уверенность: {ml_settings.confidence_threshold*100:.0f}%\n"
            text += f"💪 Минимальная сила сигнала:\n"
            text += f"   • Ансамбли: 0.3% (фиксировано)\n"
            text += f"   • Одиночные модели: 60% (фиксировано)\n\n"
            
            text += f"✅ Настройка обновлена!\n\n"
            text += f"ℹ️ Уверенность модели — это вероятность правильного предсказания.\n"
            text += f"Чем выше порог, тем меньше сигналов, но качественнее.\n\n"
            text += f"🔹 Рекомендуемые значения:\n"
            text += f"   • Консервативно: 70-80%\n"
            text += f"   • Сбалансированно: 50-70%\n"
            text += f"   • Агрессивно: 30-50%\n"
            
            keyboard = [
                [InlineKeyboardButton(f"🎯 Уверенность: {ml_settings.confidence_threshold*100:.0f}%", callback_data="edit_ml_confidence_threshold")],
                [InlineKeyboardButton("🔙 Главное меню", callback_data="main_menu")]
            ]
            
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
            
        except ValueError:
            await update.message.reply_text("❌ Неверный формат. Введите число (например: 50)")
        except Exception as e:
            logger.error(f"Error processing ML setting input: {e}")
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")
    
    async def process_risk_setting_input(self, update: Update, setting_name: str, text: str):
        """Обрабатывает ввод значения настройки риска"""
        try:
            # Парсим число
            value = float(text.replace(",", "."))
            
            # Валидация и применение
            risk = self.settings.risk
            
            if setting_name == "margin_pct_balance":
                if 1.0 <= value <= 100.0:  # 1% - 100%
                    risk.margin_pct_balance = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 100%")
                    return
            
            elif setting_name == "stop_loss_pct":
                if 0.1 <= value <= 10.0:
                    risk.stop_loss_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.1 до 10%")
                    return
            
            elif setting_name == "take_profit_pct":
                if 0.5 <= value <= 20.0:
                    risk.take_profit_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.5 до 20%")
                    return
            
            elif setting_name == "fee_rate":
                if 0.0 <= value <= 5.0:
                    risk.fee_rate = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0 до 5%")
                    return
            
            elif setting_name == "mid_term_tp_pct":
                if 0.5 <= value <= 10.0:
                    risk.mid_term_tp_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.5 до 10%")
                    return
            
            elif setting_name == "long_term_tp_pct":
                if 1.0 <= value <= 20.0:
                    risk.long_term_tp_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 20%")
                    return
            
            elif setting_name == "long_term_sl_pct":
                if 0.5 <= value <= 10.0:
                    risk.long_term_sl_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.5 до 10%")
                    return
            
            elif setting_name == "dca_drawdown_pct":
                if 0.05 <= value <= 5.0:
                    risk.dca_drawdown_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.05 до 5%")
                    return
            
            elif setting_name == "dca_max_adds":
                if 0 <= value <= 10:
                    risk.dca_max_adds = int(value)
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0 до 10")
                    return
            
            elif setting_name == "dca_min_confidence":
                if 1.0 <= value <= 100.0:
                    risk.dca_min_confidence = value / 100.0
                elif 0.0 <= value <= 1.0:
                    risk.dca_min_confidence = value
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 100%")
                    return
            
            elif setting_name == "reverse_min_confidence":
                if 1.0 <= value <= 100.0:
                    risk.reverse_min_confidence = value / 100.0
                elif 0.0 <= value <= 1.0:
                    risk.reverse_min_confidence = value
                else:
                    await update.message.reply_text("❌ Значение должно быть от 1 до 100%")
                    return
            
            elif setting_name == "reverse_min_strength":
                normalized = text.strip().lower().replace(" ", "_")
                valid_strengths = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
                if normalized in valid_strengths:
                    risk.reverse_min_strength = normalized
                else:
                    await update.message.reply_text(
                        "❌ Неверное значение. Используйте: слабое, умеренное, среднее, сильное, очень_сильное"
                    )
                    return
            
            elif setting_name == "trailing_stop_activation_pct":
                if 0.1 <= value <= 5.0:
                    risk.trailing_stop_activation_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.1 до 5%")
                    return
            
            elif setting_name == "trailing_stop_distance_pct":
                if 0.05 <= value <= 2.0:
                    risk.trailing_stop_distance_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.05 до 2%")
                    return
            
            elif setting_name == "breakeven_level1_activation_pct":
                if 0.1 <= value <= 5.0:
                    risk.breakeven_level1_activation_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.1 до 5%")
                    return
            elif setting_name == "breakeven_level1_sl_pct":
                if 0.05 <= value <= 2.0:
                    risk.breakeven_level1_sl_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.05 до 2%")
                    return
            elif setting_name == "breakeven_level2_activation_pct":
                if 0.1 <= value <= 5.0:
                    risk.breakeven_level2_activation_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.1 до 5%")
                    return
            elif setting_name == "breakeven_level2_sl_pct":
                if 0.05 <= value <= 2.0:
                    risk.breakeven_level2_sl_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.05 до 2%")
                    return
            
            elif setting_name == "base_order_usd":
                if 1.0 <= value <= 10000.0:
                    risk.base_order_usd = value
                else:
                    await update.message.reply_text("❌ Значение должно быть от $1 до $10000")
                    return
            
            # Сохраняем настройки
            self.save_risk_settings()
            
            # Показываем обновленные настройки
            risk = self.settings.risk
            
            text = "⚙️ НАСТРОЙКИ РИСКА\n\n"
            
            # Форматируем значение для отображения
            if setting_name.endswith("_pct"):
                display_value = f"{value:.2f}%"
            elif setting_name in ("fee_rate", "dca_min_confidence"):
                display_value = f"{value:.4f}%" if setting_name == "fee_rate" else f"{value:.2f}%"
            elif setting_name == "reverse_min_strength":
                display_value = text.strip()
            elif setting_name == "base_order_usd":
                display_value = f"${value:.2f}"
            else:
                display_value = f"{value:.2f}"
            
            text += f"✅ Настройка обновлена: {setting_name} = {display_value}\n\n"
            
            # Режим расчета размера позиции
            text += f"💰 Маржа от баланса: {risk.margin_pct_balance*100:.0f}%\n"
            text += f"💰 Фиксированная сумма: ${risk.base_order_usd:.2f}\n"
            text += f"ℹ️ Используется меньшее значение\n"
            text += f"📉 Stop Loss: {risk.stop_loss_pct*100:.2f}%\n"
            tp_suffix = "(ставим ордер TP на бирже)" if getattr(risk, "use_take_profit", True) else "(не ставим TP на бирже; выход по трейлингу/SL/логике)"
            text += f"📈 Take Profit: {risk.take_profit_pct*100:.2f}% {tp_suffix}\n"
            text += f"💸 Комиссия (per side): {risk.fee_rate*100:.4f}%\n\n"
            text += (
                f"🧭 Горизонт: mid TP≥{risk.mid_term_tp_pct*100:.2f}% | "
                f"long TP≥{risk.long_term_tp_pct*100:.2f}% или SL≥{risk.long_term_sl_pct*100:.2f}%\n"
            )
            text += f"↪️ Игнорировать реверс (mid/long): {'✅' if risk.long_term_ignore_reverse else '❌'}\n\n"
            text += (
                f"➕ DCA: {'✅' if risk.dca_enabled else '❌'} | "
                f"Просадка: {risk.dca_drawdown_pct*100:.2f}% | "
                f"Макс: {risk.dca_max_adds} | "
                f"Мин. уверенность: {risk.dca_min_confidence*100:.0f}%\n\n"
            )
            text += (
                f"🔁 Реверс по сильному сигналу: {'✅' if risk.reverse_on_strong_signal else '❌'} | "
                f"Мин. уверенность: {risk.reverse_min_confidence*100:.0f}% | "
                f"Мин. сила: {risk.reverse_min_strength}\n\n"
            )
            text += f"🔄 Трейлинг стоп: {'✅ Включен' if risk.enable_trailing_stop else '❌ Выключен'}\n"
            text += f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%\n"
            text += f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%\n\n"
            text += f"💎 Частичное закрытие: {'✅ Включено' if risk.enable_partial_close else '❌ Выключено'}\n"
            text += f"🛡️ Безубыток: {'✅ Включен' if risk.enable_breakeven else '❌ Выключен'}\n"
            text += f"   1-я ступень: при {risk.breakeven_level1_activation_pct*100:.2f}% → SL {risk.breakeven_level1_sl_pct*100:.2f}%\n"
            text += f"   2-я ступень: при {risk.breakeven_level2_activation_pct*100:.2f}% → SL {risk.breakeven_level2_sl_pct*100:.2f}%\n\n"
            text += f"❄️ Cooldown после убытков: {'✅ Включен' if risk.enable_loss_cooldown else '❌ Выключен'}\n"
            
            keyboard = [
                [InlineKeyboardButton(f"💰 Маржа: {risk.margin_pct_balance*100:.0f}%", callback_data="edit_risk_margin_pct_balance")],
                [InlineKeyboardButton(f"💰 Сумма: ${risk.base_order_usd:.2f}", callback_data="edit_risk_base_order_usd")],
            ]
            
            keyboard.extend([
                [InlineKeyboardButton(f"📉 SL: {risk.stop_loss_pct*100:.2f}%", callback_data="edit_risk_stop_loss_pct")],
                [InlineKeyboardButton(f"📈 TP: {risk.take_profit_pct*100:.2f}%", callback_data="edit_risk_take_profit_pct")],
                [InlineKeyboardButton(f"🎯 TP ордер: {'✅' if getattr(risk, 'use_take_profit', True) else '❌'}", callback_data="toggle_risk_use_take_profit")],
                [InlineKeyboardButton(f"💸 Комиссия: {risk.fee_rate*100:.4f}%", callback_data="edit_risk_fee_rate")],
                [InlineKeyboardButton(f"🧭 Mid TP: {risk.mid_term_tp_pct*100:.2f}%", callback_data="edit_risk_mid_term_tp_pct")],
                [InlineKeyboardButton(f"🧭 Long TP: {risk.long_term_tp_pct*100:.2f}%", callback_data="edit_risk_long_term_tp_pct")],
                [InlineKeyboardButton(f"🧭 Long SL: {risk.long_term_sl_pct*100:.2f}%", callback_data="edit_risk_long_term_sl_pct")],
                [InlineKeyboardButton(f"↪️ Игнор. реверс: {'✅' if risk.long_term_ignore_reverse else '❌'}", callback_data="toggle_risk_long_term_ignore_reverse")],
                [InlineKeyboardButton(f"➕ DCA: {'✅' if risk.dca_enabled else '❌'}", callback_data="toggle_risk_dca_enabled")],
                [InlineKeyboardButton(f"   Просадка: {risk.dca_drawdown_pct*100:.2f}%", callback_data="edit_risk_dca_drawdown_pct")],
                [InlineKeyboardButton(f"   Макс: {risk.dca_max_adds}", callback_data="edit_risk_dca_max_adds")],
                [InlineKeyboardButton(f"   Мин. уверенность: {risk.dca_min_confidence*100:.0f}%", callback_data="edit_risk_dca_min_confidence")],
                [InlineKeyboardButton(f"🔁 Реверс: {'✅' if risk.reverse_on_strong_signal else '❌'}", callback_data="toggle_risk_reverse_on_strong_signal")],
                [InlineKeyboardButton(f"   Мин. уверенность: {risk.reverse_min_confidence*100:.0f}%", callback_data="edit_risk_reverse_min_confidence")],
                [InlineKeyboardButton(f"   Мин. сила: {risk.reverse_min_strength}", callback_data="edit_risk_reverse_min_strength")],
                [InlineKeyboardButton(f"🔄 Трейлинг: {'✅' if risk.enable_trailing_stop else '❌'}", callback_data="toggle_risk_enable_trailing_stop")],
                [InlineKeyboardButton(f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_activation_pct")],
                [InlineKeyboardButton(f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_distance_pct")],
                [InlineKeyboardButton(f"💎 Частичное закрытие: {'✅' if risk.enable_partial_close else '❌'}", callback_data="toggle_risk_enable_partial_close")],
                [InlineKeyboardButton(f"🛡️ Безубыток: {'✅' if risk.enable_breakeven else '❌'}", callback_data="toggle_risk_enable_breakeven")],
                [InlineKeyboardButton(f"   1-я активация: {risk.breakeven_level1_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_level1_activation_pct")],
                [InlineKeyboardButton(f"   1-я SL: {risk.breakeven_level1_sl_pct*100:.2f}%", callback_data="edit_risk_breakeven_level1_sl_pct")],
                [InlineKeyboardButton(f"   2-я активация: {risk.breakeven_level2_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_level2_activation_pct")],
                [InlineKeyboardButton(f"   2-я SL: {risk.breakeven_level2_sl_pct*100:.2f}%", callback_data="edit_risk_breakeven_level2_sl_pct")],
                [InlineKeyboardButton(f"❄️ Cooldown: {'✅' if risk.enable_loss_cooldown else '❌'}", callback_data="toggle_risk_enable_loss_cooldown")],
                [InlineKeyboardButton("🔄 Сбросить на стандартные", callback_data="reset_risk_defaults")],
                [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ])
            
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        
        except ValueError:
            await update.message.reply_text("❌ Неверный формат! Введите число (например: 1.5)")
        except Exception as e:
            logger.error(f"Error processing risk setting input: {e}")
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")
    
    async def toggle_ml_setting(self, query, setting_name: str):
        """Переключает булеву настройку ML стратегии"""
        logger.info(f"toggle_ml_setting called with setting_name: {setting_name}")
        ml_settings = self.settings.ml_strategy
        
        if setting_name == "use_mtf_strategy":
            # Переключаем MTF стратегию
            old_value = ml_settings.use_mtf_strategy
            ml_settings.use_mtf_strategy = not ml_settings.use_mtf_strategy
            new_value = ml_settings.use_mtf_strategy
            logger.info(f"MTF strategy toggled: {old_value} -> {new_value}")
            logger.info(f"Current ml_settings.use_mtf_strategy value: {ml_settings.use_mtf_strategy}")
            
            # Сохраняем настройки
            self.save_ml_settings()
            
            # Проверяем, что настройка действительно сохранилась
            logger.info(f"After save_ml_settings: ml_settings.use_mtf_strategy={ml_settings.use_mtf_strategy}")
            
            # Сбрасываем стратегии для всех активных символов, чтобы они переинициализировались
            if self.trading_loop:
                for symbol in self.settings.active_symbols:
                    if symbol in self.trading_loop.strategies:
                        del self.trading_loop.strategies[symbol]
                        logger.info(f"[{symbol}] Strategy reset due to MTF setting change")
            
            status = "включена" if ml_settings.use_mtf_strategy else "выключена"
            await query.answer(f"✅ MTF стратегия {status}", show_alert=True)
            
            # Показываем обновленные настройки
            await self.show_ml_settings(query)
        elif setting_name == "auto_optimize_strategies":
            # Переключаем автообновление стратегий
            old_value = ml_settings.auto_optimize_strategies
            ml_settings.auto_optimize_strategies = not ml_settings.auto_optimize_strategies
            logger.info(f"Auto optimize toggled: {old_value} -> {ml_settings.auto_optimize_strategies}")
            
            # Сохраняем настройки
            self.save_ml_settings()
            
            status = "включено" if ml_settings.auto_optimize_strategies else "выключено"
            message = f"✅ Автообновление стратегий {status}"
            if ml_settings.auto_optimize_strategies:
                day_names = {
                    "monday": "понедельник",
                    "tuesday": "вторник",
                    "wednesday": "среда",
                    "thursday": "четверг",
                    "friday": "пятница",
                    "saturday": "суббота",
                    "sunday": "воскресенье"
                }
                day_name = day_names.get(ml_settings.auto_optimize_day, ml_settings.auto_optimize_day)
                message += f"\nРасписание: {day_name}, {ml_settings.auto_optimize_hour:02d}:00"
                message += "\n\n⚠️ Убедитесь, что планировщик запущен:\npython schedule_strategy_optimizer.py"
            
            await query.answer(message, show_alert=True)
            
            # Показываем обновленные настройки
            await self.show_ml_settings(query)
        elif setting_name == "atr_filter_enabled":
            # Переключаем фильтр волатильности (ATR 1h)
            old_value = ml_settings.atr_filter_enabled
            ml_settings.atr_filter_enabled = not ml_settings.atr_filter_enabled
            logger.info(f"ATR filter toggled: {old_value} -> {ml_settings.atr_filter_enabled}")
            self.save_ml_settings()
            status = "включен" if ml_settings.atr_filter_enabled else "выключен"
            await query.answer(f"✅ Фильтр волатильности (ATR 1h) {status}", show_alert=True)
            await self.show_ml_settings(query)
        elif setting_name == "follow_btc_filter_enabled":
            old_value = getattr(ml_settings, "follow_btc_filter_enabled", True)
            ml_settings.follow_btc_filter_enabled = not bool(old_value)
            logger.info(f"Follow BTC filter toggled: {old_value} -> {ml_settings.follow_btc_filter_enabled}")
            self.save_ml_settings()
            status = "включено" if ml_settings.follow_btc_filter_enabled else "выключено"
            await query.answer(f"✅ Следование за BTC {status}", show_alert=True)
            await self.show_ml_settings(query)
        elif setting_name == "use_fixed_sl_from_risk":
            use_fixed = getattr(ml_settings, "use_fixed_sl_from_risk", False)
            ml_settings.use_fixed_sl_from_risk = not use_fixed
            self.save_ml_settings()
            status = "фиксированный из риска" if ml_settings.use_fixed_sl_from_risk else "от модели/ATR"
            await query.answer(f"✅ SL в сигнале: {status}", show_alert=True)
            if self.trading_loop:
                for symbol in self.settings.active_symbols:
                    if symbol in self.trading_loop.strategies:
                        del self.trading_loop.strategies[symbol]
                        logger.info(f"[{symbol}] Strategy reset due to use_fixed_sl_from_risk change")
            await self.show_ml_settings(query)
        else:
            logger.warning(f"Unknown ML setting: {setting_name}")
            await query.answer("⚠️ Неизвестная настройка", show_alert=True)
    
    async def toggle_risk_setting(self, query, setting_name: str):
        """Переключает булеву настройку риска"""
        risk = self.settings.risk
        
        if setting_name == "use_take_profit":
            risk.use_take_profit = not bool(getattr(risk, "use_take_profit", True))
        elif setting_name == "enable_trailing_stop":
            risk.enable_trailing_stop = not risk.enable_trailing_stop
        elif setting_name == "enable_partial_close":
            risk.enable_partial_close = not risk.enable_partial_close
        elif setting_name == "enable_breakeven":
            risk.enable_breakeven = not risk.enable_breakeven
        elif setting_name == "enable_loss_cooldown":
            risk.enable_loss_cooldown = not risk.enable_loss_cooldown
        elif setting_name == "long_term_ignore_reverse":
            risk.long_term_ignore_reverse = not risk.long_term_ignore_reverse
        elif setting_name == "dca_enabled":
            risk.dca_enabled = not risk.dca_enabled
        elif setting_name == "reverse_on_strong_signal":
            risk.reverse_on_strong_signal = not risk.reverse_on_strong_signal
        else:
            await query.answer("Неизвестная настройка", show_alert=True)
            return
        
        # Сохраняем настройки
        self.save_risk_settings()
        
        await query.answer("✅ Настройка обновлена!")
        await self.show_risk_settings(query)
    
    async def reset_risk_defaults(self, query):
        """Сбрасывает настройки риска на стандартные"""
        from bot.config import RiskParams
        
        # Создаем новые стандартные настройки
        self.settings.risk = RiskParams()
        
        # Сохраняем
        self.save_risk_settings()
        
        await query.answer("✅ Настройки сброшены на стандартные!", show_alert=True)
        await self.show_risk_settings(query)
    
    def save_ml_settings(self):
        """Сохраняет ML настройки в файл"""
        try:
            from pathlib import Path
            import json
            
            # Используем абсолютный путь относительно корня проекта
            project_root = Path(__file__).parent.parent
            config_file = project_root / "ml_settings.json"
            
            # Загружаем существующие настройки, если файл есть (для сохранения других параметров)
            existing_dict = {}
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        existing_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing ml_settings.json: {e}")
            
            # Создаем полный словарь настроек
            ml_dict = {
                "confidence_threshold": self.settings.ml_strategy.confidence_threshold,
                "min_confidence_for_trade": self.settings.ml_strategy.min_confidence_for_trade,
                "use_mtf_strategy": self.settings.ml_strategy.use_mtf_strategy,
                "mtf_confidence_threshold_1h": self.settings.ml_strategy.mtf_confidence_threshold_1h,
                "mtf_confidence_threshold_15m": self.settings.ml_strategy.mtf_confidence_threshold_15m,
                "mtf_alignment_mode": self.settings.ml_strategy.mtf_alignment_mode,
                "mtf_require_alignment": self.settings.ml_strategy.mtf_require_alignment,
                "auto_optimize_strategies": self.settings.ml_strategy.auto_optimize_strategies,
                "auto_optimize_day": self.settings.ml_strategy.auto_optimize_day,
                "auto_optimize_hour": self.settings.ml_strategy.auto_optimize_hour,
                "atr_filter_enabled": self.settings.ml_strategy.atr_filter_enabled,
                "atr_min_pct": self.settings.ml_strategy.atr_min_pct,
                "atr_max_pct": self.settings.ml_strategy.atr_max_pct,
                "follow_btc_filter_enabled": getattr(self.settings.ml_strategy, "follow_btc_filter_enabled", True),
                "follow_btc_override_confidence": getattr(self.settings.ml_strategy, "follow_btc_override_confidence", 0.80),
                "use_dynamic_ensemble_weights": getattr(self.settings.ml_strategy, "use_dynamic_ensemble_weights", False),
                "adx_trend_threshold": getattr(self.settings.ml_strategy, "adx_trend_threshold", 25.0),
                "adx_flat_threshold": getattr(self.settings.ml_strategy, "adx_flat_threshold", 20.0),
                "trend_weights": getattr(self.settings.ml_strategy, "trend_weights", None),
                "flat_weights": getattr(self.settings.ml_strategy, "flat_weights", None),
                "use_fixed_sl_from_risk": getattr(self.settings.ml_strategy, "use_fixed_sl_from_risk", False),
                "ai_entry_confirmation_enabled": getattr(self.settings.ml_strategy, "ai_entry_confirmation_enabled", False),
                "ai_fallback_force_enabled": getattr(self.settings.ml_strategy, "ai_fallback_force_enabled", False),
                "ai_fallback_spread_reduce_pct": getattr(self.settings.ml_strategy, "ai_fallback_spread_reduce_pct", 0.10),
                "ai_fallback_spread_veto_pct": getattr(self.settings.ml_strategy, "ai_fallback_spread_veto_pct", 0.25),
                "ai_fallback_min_depth_usd_5": getattr(self.settings.ml_strategy, "ai_fallback_min_depth_usd_5", 0.0),
                "ai_fallback_imbalance_abs_reduce": getattr(self.settings.ml_strategy, "ai_fallback_imbalance_abs_reduce", 0.60),
                "ai_fallback_orderflow_ratio_low": getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_low", 0.40),
                "ai_fallback_orderflow_ratio_high": getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_high", 2.50),
            }
            
            # Сохраняем все настройки
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(ml_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ML settings saved to {config_file}: use_mtf_strategy={ml_dict['use_mtf_strategy']}, auto_optimize_strategies={ml_dict['auto_optimize_strategies']}")
            logger.debug(f"Full ML settings dict: {ml_dict}")
        
        except Exception as e:
            logger.error(f"Error saving ML settings: {e}", exc_info=True)
    
    def _ensure_ml_settings_file(self):
        """Проверяет и обновляет файл ml_settings.json, добавляя недостающие поля"""
        try:
            from pathlib import Path
            import json
            
            project_root = Path(__file__).parent.parent
            config_file = project_root / "ml_settings.json"
            
            # Если файл существует, загружаем его и проверяем наличие всех полей
            existing_dict = {}
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        existing_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read ml_settings.json: {e}")
                    existing_dict = {}
            
            # Список обязательных полей
            required_fields = {
                "confidence_threshold": self.settings.ml_strategy.confidence_threshold,
                "min_confidence_for_trade": self.settings.ml_strategy.min_confidence_for_trade,
                "use_mtf_strategy": self.settings.ml_strategy.use_mtf_strategy,
                "mtf_confidence_threshold_1h": self.settings.ml_strategy.mtf_confidence_threshold_1h,
                "mtf_confidence_threshold_15m": self.settings.ml_strategy.mtf_confidence_threshold_15m,
                "mtf_alignment_mode": self.settings.ml_strategy.mtf_alignment_mode,
                "mtf_require_alignment": self.settings.ml_strategy.mtf_require_alignment,
                "auto_optimize_strategies": self.settings.ml_strategy.auto_optimize_strategies,
                "auto_optimize_day": self.settings.ml_strategy.auto_optimize_day,
                "auto_optimize_hour": self.settings.ml_strategy.auto_optimize_hour,
                "atr_filter_enabled": self.settings.ml_strategy.atr_filter_enabled,
                "atr_min_pct": self.settings.ml_strategy.atr_min_pct,
                "atr_max_pct": self.settings.ml_strategy.atr_max_pct,
                "follow_btc_filter_enabled": getattr(self.settings.ml_strategy, "follow_btc_filter_enabled", True),
                "follow_btc_override_confidence": getattr(self.settings.ml_strategy, "follow_btc_override_confidence", 0.80),
                "use_dynamic_ensemble_weights": getattr(self.settings.ml_strategy, "use_dynamic_ensemble_weights", False),
                "adx_trend_threshold": getattr(self.settings.ml_strategy, "adx_trend_threshold", 25.0),
                "adx_flat_threshold": getattr(self.settings.ml_strategy, "adx_flat_threshold", 20.0),
                "trend_weights": getattr(self.settings.ml_strategy, "trend_weights", None),
                "flat_weights": getattr(self.settings.ml_strategy, "flat_weights", None),
                "use_fixed_sl_from_risk": getattr(self.settings.ml_strategy, "use_fixed_sl_from_risk", False),
                "ai_entry_confirmation_enabled": getattr(self.settings.ml_strategy, "ai_entry_confirmation_enabled", False),
                "ai_fallback_force_enabled": getattr(self.settings.ml_strategy, "ai_fallback_force_enabled", False),
                "ai_fallback_spread_reduce_pct": getattr(self.settings.ml_strategy, "ai_fallback_spread_reduce_pct", 0.10),
                "ai_fallback_spread_veto_pct": getattr(self.settings.ml_strategy, "ai_fallback_spread_veto_pct", 0.25),
                "ai_fallback_min_depth_usd_5": getattr(self.settings.ml_strategy, "ai_fallback_min_depth_usd_5", 0.0),
                "ai_fallback_imbalance_abs_reduce": getattr(self.settings.ml_strategy, "ai_fallback_imbalance_abs_reduce", 0.60),
                "ai_fallback_orderflow_ratio_low": getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_low", 0.40),
                "ai_fallback_orderflow_ratio_high": getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_high", 2.50),
            }
            
            # Проверяем, нужно ли обновить файл
            needs_update = False
            for field, default_value in required_fields.items():
                if field not in existing_dict:
                    existing_dict[field] = default_value
                    needs_update = True
                    logger.info(f"Adding missing field to ml_settings.json: {field}={default_value}")
            
            # Сохраняем обновленный файл, если нужно
            if needs_update:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated ml_settings.json with missing fields")
        
        except Exception as e:
            logger.error(f"Error ensuring ml_settings.json: {e}", exc_info=True)
    
    def save_risk_settings(self):
        """Сохраняет настройки риска в файл"""
        try:
            from pathlib import Path
            import json
            
            config_file = Path("risk_settings.json")
            
            # Преобразуем настройки в словарь
            risk_dict = {
                "margin_pct_balance": self.settings.risk.margin_pct_balance,
                "base_order_usd": self.settings.risk.base_order_usd,
                "stop_loss_pct": self.settings.risk.stop_loss_pct,
                "take_profit_pct": self.settings.risk.take_profit_pct,
                "use_take_profit": bool(getattr(self.settings.risk, "use_take_profit", True)),
                "enable_trailing_stop": self.settings.risk.enable_trailing_stop,
                "trailing_stop_activation_pct": self.settings.risk.trailing_stop_activation_pct,
                "trailing_stop_distance_pct": self.settings.risk.trailing_stop_distance_pct,
                "enable_partial_close": self.settings.risk.enable_partial_close,
                "enable_breakeven": self.settings.risk.enable_breakeven,
                "breakeven_level1_activation_pct": self.settings.risk.breakeven_level1_activation_pct,
                "breakeven_level1_sl_pct": self.settings.risk.breakeven_level1_sl_pct,
                "breakeven_level2_activation_pct": self.settings.risk.breakeven_level2_activation_pct,
                "breakeven_level2_sl_pct": self.settings.risk.breakeven_level2_sl_pct,
                "enable_loss_cooldown": self.settings.risk.enable_loss_cooldown,
                "fee_rate": self.settings.risk.fee_rate,
                "mid_term_tp_pct": self.settings.risk.mid_term_tp_pct,
                "long_term_tp_pct": self.settings.risk.long_term_tp_pct,
                "long_term_sl_pct": self.settings.risk.long_term_sl_pct,
                "long_term_ignore_reverse": self.settings.risk.long_term_ignore_reverse,
                "dca_enabled": self.settings.risk.dca_enabled,
                "dca_drawdown_pct": self.settings.risk.dca_drawdown_pct,
                "dca_max_adds": self.settings.risk.dca_max_adds,
                "dca_min_confidence": self.settings.risk.dca_min_confidence,
            "reverse_on_strong_signal": self.settings.risk.reverse_on_strong_signal,
            "reverse_min_confidence": self.settings.risk.reverse_min_confidence,
            "reverse_min_strength": self.settings.risk.reverse_min_strength,
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(risk_dict, f, indent=2, ensure_ascii=False)
            
            logger.info("Risk settings saved to risk_settings.json")
        
        except Exception as e:
            logger.error(f"Error saving risk settings: {e}")
    
    async def show_risk_info(self, query):
        """Показывает информацию о настройках риска"""
        text = "ℹ️ ИНФОРМАЦИЯ О НАСТРОЙКАХ РИСКА\n\n"
        text += "💰 Маржа от баланса:\n"
        text += "Процент от баланса, используемый для маржи позиции.\n"
        text += "Пример: 20% при балансе $1000 = $200 маржи.\n\n"
        
        text += "📉 Stop Loss:\n"
        text += "Процент убытка от цены входа для закрытия позиции.\n"
        text += "Пример: 1% при входе $100 = закрытие при $99.\n\n"
        
        text += "📈 Take Profit:\n"
        text += "Процент прибыли от цены входа для закрытия позиции.\n"
        text += "Пример: 2.5% при входе $100 = закрытие при $102.50.\n\n"
        
        text += "🔄 Трейлинг стоп:\n"
        text += "Автоматически перемещает SL вслед за ценой.\n"
        text += "Активация: при какой прибыли включить.\n"
        text += "Расстояние: на сколько % от максимума держать SL.\n\n"
        
        text += "💎 Частичное закрытие:\n"
        text += "Закрывает часть позиции при достижении % пути к TP.\n"
        text += "Пример: 50% позиции при 50% пути к TP.\n\n"
        
        text += "🛡️ Безубыток:\n"
        text += "Перемещает SL на уровень входа при достижении прибыли.\n"
        text += "Активация: при какой прибыли включить.\n\n"
        
        text += "❄️ Cooldown:\n"
        text += "Пауза после убыточных сделок:\n"
        text += "• 1 убыток: 30 минут\n"
        text += "• 2 убытка: 2 часа\n"
        text += "• 3+ убытков: 24 часа\n"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад к настройкам", callback_data="settings_risk")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def show_ml_settings(self, query):
        """Показывает настройки ML стратегии"""
        ml_settings = self.settings.ml_strategy
        
        # Логируем текущие значения для отладки
        logger.info(f"show_ml_settings: use_mtf_strategy={ml_settings.use_mtf_strategy}, auto_optimize_strategies={ml_settings.auto_optimize_strategies}")
        
        text = "🧠 НАСТРОЙКИ ML СТРАТЕГИИ\n\n"
        text += f"🔄 MTF стратегия (1h + 15m): {'✅ Включена' if ml_settings.use_mtf_strategy else '❌ Выключена'}\n"
        if ml_settings.use_mtf_strategy:
            text += f"   • Порог 1h: {ml_settings.mtf_confidence_threshold_1h*100:.0f}%\n"
            text += f"   • Порог 15m: {ml_settings.mtf_confidence_threshold_15m*100:.0f}%\n"
            text += f"   • Режим: {ml_settings.mtf_alignment_mode}\n\n"
        text += f"📊 Фильтр волатильности (ATR 1h): {'✅ Включен' if ml_settings.atr_filter_enabled else '❌ Выключен'}\n"
        if ml_settings.atr_filter_enabled:
            text += f"   • Диапазон ATR: {ml_settings.atr_min_pct}% – {ml_settings.atr_max_pct}%\n\n"
        text += f"₿ Следование за BTC: {'✅ Включено' if getattr(ml_settings, 'follow_btc_filter_enabled', True) else '❌ Выключено'}\n"
        text += f"   • Пропуск при сильном сигнале: {getattr(ml_settings, 'follow_btc_override_confidence', 0.80)*100:.0f}%+\n\n"
        text += f"🤖 Автообновление стратегий: {'✅ Включено' if ml_settings.auto_optimize_strategies else '❌ Выключено'}\n"
        if ml_settings.auto_optimize_strategies:
            day_names = {
                "monday": "Понедельник",
                "tuesday": "Вторник",
                "wednesday": "Среда",
                "thursday": "Четверг",
                "friday": "Пятница",
                "saturday": "Суббота",
                "sunday": "Воскресенье"
            }
            day_name = day_names.get(ml_settings.auto_optimize_day, ml_settings.auto_optimize_day)
            text += f"   • Расписание: {day_name}, {ml_settings.auto_optimize_hour:02d}:00\n\n"
        text += f"📉 SL в сигнале: {'фиксированный из риска' if getattr(ml_settings, 'use_fixed_sl_from_risk', False) else 'от модели/ATR'}\n"
        text += f"🎯 Минимальная уверенность модели: {ml_settings.confidence_threshold*100:.0f}%\n"
        text += f"💰 Минимальная уверенность для сделки: {ml_settings.min_confidence_for_trade*100:.0f}%\n"
        text += f"💪 Минимальная сила сигнала:\n"
        text += f"   • Ансамбли: 0.3% (фиксировано)\n"
        text += f"   • Одиночные модели: 60% (фиксировано)\n\n"
        
        text += f"ℹ️ Уверенность модели — это вероятность правильного предсказания.\n"
        text += f"Минимальная уверенность для сделки — порог для открытия позиции.\n"
        text += f"Чем выше порог, тем меньше сигналов, но качественнее.\n\n"
        text += f"🔹 Рекомендуемые значения:\n"
        text += f"   • Консервативно: 70-80%\n"
        text += f"   • Сбалансированно: 50-70%\n"
        text += f"   • Агрессивно: 30-50%\n"
        
        keyboard = [
            [InlineKeyboardButton(
                f"🔄 MTF стратегия: {'✅ Вкл' if ml_settings.use_mtf_strategy else '❌ Выкл'}", 
                callback_data="toggle_ml_use_mtf_strategy"
            )],
            [InlineKeyboardButton(
                f"📊 Фильтр волатильности: {'✅ Вкл' if ml_settings.atr_filter_enabled else '❌ Выкл'}", 
                callback_data="toggle_ml_atr_filter_enabled"
            )],
            [InlineKeyboardButton(
                f"₿ Следование за BTC: {'✅ Вкл' if getattr(ml_settings, 'follow_btc_filter_enabled', True) else '❌ Выкл'}",
                callback_data="toggle_ml_follow_btc_filter_enabled"
            )],
            [InlineKeyboardButton(
                f"🤖 Автообновление: {'✅ Вкл' if ml_settings.auto_optimize_strategies else '❌ Выкл'}", 
                callback_data="toggle_ml_auto_optimize_strategies"
            )],
            [InlineKeyboardButton(
                f"📉 SL: {'фикс. из риска' if getattr(ml_settings, 'use_fixed_sl_from_risk', False) else 'от модели/ATR'}", 
                callback_data="toggle_ml_use_fixed_sl_from_risk"
            )],
            [InlineKeyboardButton(f"🎯 Уверенность модели: {ml_settings.confidence_threshold*100:.0f}%", callback_data="edit_ml_confidence_threshold")],
            [InlineKeyboardButton(f"💰 Уверенность для сделки: {ml_settings.min_confidence_for_trade*100:.0f}%", callback_data="edit_ml_min_confidence_for_trade")],
            [InlineKeyboardButton("🚀 Оптимизировать MTF стратегии", callback_data="optimize_mtf_strategies")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def start_edit_ml_setting(self, query, setting_name: str):
        """Начинает редактирование ML настройки"""
        user_id = query.from_user.id
        
        if setting_name == "confidence_threshold":
            current_value = self.settings.ml_strategy.confidence_threshold * 100
            self.waiting_for_ml_setting[user_id] = setting_name
            
            await query.edit_message_text(
                f"✏️ РЕДАКТИРОВАНИЕ: Минимальная уверенность модели\n\n"
                f"Текущее значение: {current_value:.0f}%\n\n"
                f"Введите новое значение от 1 до 100 (в процентах):\n"
                f"Пример: 50 означает 50%",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("❌ Отмена", callback_data="settings_ml")]
                ])
            )
        elif setting_name == "min_confidence_for_trade":
            current_value = self.settings.ml_strategy.min_confidence_for_trade * 100
            self.waiting_for_ml_setting[user_id] = setting_name
            
            await query.edit_message_text(
                f"✏️ РЕДАКТИРОВАНИЕ: Минимальная уверенность для сделки\n\n"
                f"Текущее значение: {current_value:.0f}%\n\n"
                f"Введите новое значение от 1 до 100 (в процентах):\n"
                f"Пример: 50 означает 50%",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("❌ Отмена", callback_data="settings_ml")]
                ])
            )
        else:
            await query.answer("Неизвестная настройка", show_alert=True)

    async def send_notification(self, text: str):
        if self.app and self.settings.allowed_user_id:
            try:
                await self.app.bot.send_message(chat_id=self.settings.allowed_user_id, text=text)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    async def show_risk_settings(self, query):
        """Показывает настройки риска"""
        risk = self.settings.risk
        
        text = "⚙️ НАСТРОЙКИ РИСКА\n\n"
        
        # Размер позиции (используется меньшее значение)
        text += f"💰 Маржа от баланса: {risk.margin_pct_balance*100:.0f}%\n"
        text += f"💰 Фиксированная сумма: ${risk.base_order_usd:.2f}\n"
        text += f"ℹ️ Используется меньшее значение\n"
        
        text += f"\n📉 Stop Loss: {risk.stop_loss_pct*100:.2f}%\n"
        tp_suffix = "(ставим ордер TP на бирже)" if getattr(risk, "use_take_profit", True) else "(не ставим TP на бирже; выход по трейлингу/SL/логике)"
        text += f"📈 Take Profit: {risk.take_profit_pct*100:.2f}% {tp_suffix}\n\n"
        text += f"💸 Комиссия (per side): {risk.fee_rate*100:.4f}%\n\n"
        text += (
            f"🧭 Горизонт: mid TP≥{risk.mid_term_tp_pct*100:.2f}% | "
            f"long TP≥{risk.long_term_tp_pct*100:.2f}% или SL≥{risk.long_term_sl_pct*100:.2f}%\n"
        )
        text += f"↪️ Игнорировать реверс (mid/long): {'✅' if risk.long_term_ignore_reverse else '❌'}\n\n"
        text += (
            f"➕ DCA: {'✅' if risk.dca_enabled else '❌'} | "
            f"Просадка: {risk.dca_drawdown_pct*100:.2f}% | "
            f"Макс: {risk.dca_max_adds} | "
            f"Мин. уверенность: {risk.dca_min_confidence*100:.0f}%\n\n"
        )
        text += (
            f"🔁 Реверс по сильному сигналу: {'✅' if risk.reverse_on_strong_signal else '❌'} | "
            f"Мин. уверенность: {risk.reverse_min_confidence*100:.0f}%\n\n"
        )
        text += f"🔄 Трейлинг стоп: {'✅ Включен' if risk.enable_trailing_stop else '❌ Выключен'}\n"
        text += f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%\n"
        text += f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%\n\n"
        text += f"💎 Частичное закрытие: {'✅ Включено' if risk.enable_partial_close else '❌ Выключено'}\n"
        text += f"🛡️ Безубыток: {'✅ Включен' if risk.enable_breakeven else '❌ Выключен'}\n"
        text += f"   1-я ступень: при {risk.breakeven_level1_activation_pct*100:.2f}% → SL {risk.breakeven_level1_sl_pct*100:.2f}%\n"
        text += f"   2-я ступень: при {risk.breakeven_level2_activation_pct*100:.2f}% → SL {risk.breakeven_level2_sl_pct*100:.2f}%\n\n"
        text += f"❄️ Cooldown после убытков: {'✅ Включен' if risk.enable_loss_cooldown else '❌ Выключен'}\n"
        
        keyboard = [
            [InlineKeyboardButton(f"💰 Маржа: {risk.margin_pct_balance*100:.0f}%", callback_data="edit_risk_margin_pct_balance")],
            [InlineKeyboardButton(f"💰 Сумма: ${risk.base_order_usd:.2f}", callback_data="edit_risk_base_order_usd")],
        ]
        
        keyboard.extend([
            [InlineKeyboardButton(f"📉 SL: {risk.stop_loss_pct*100:.2f}%", callback_data="edit_risk_stop_loss_pct")],
            [InlineKeyboardButton(f"📈 TP: {risk.take_profit_pct*100:.2f}%", callback_data="edit_risk_take_profit_pct")],
            [InlineKeyboardButton(f"🎯 TP ордер: {'✅' if getattr(risk, 'use_take_profit', True) else '❌'}", callback_data="toggle_risk_use_take_profit")],
            [InlineKeyboardButton(f"💸 Комиссия: {risk.fee_rate*100:.4f}%", callback_data="edit_risk_fee_rate")],
            [InlineKeyboardButton(f"🧭 Mid TP: {risk.mid_term_tp_pct*100:.2f}%", callback_data="edit_risk_mid_term_tp_pct")],
            [InlineKeyboardButton(f"🧭 Long TP: {risk.long_term_tp_pct*100:.2f}%", callback_data="edit_risk_long_term_tp_pct")],
            [InlineKeyboardButton(f"🧭 Long SL: {risk.long_term_sl_pct*100:.2f}%", callback_data="edit_risk_long_term_sl_pct")],
            [InlineKeyboardButton(f"↪️ Игнор. реверс: {'✅' if risk.long_term_ignore_reverse else '❌'}", callback_data="toggle_risk_long_term_ignore_reverse")],
            [InlineKeyboardButton(f"➕ DCA: {'✅' if risk.dca_enabled else '❌'}", callback_data="toggle_risk_dca_enabled")],
            [InlineKeyboardButton(f"   Просадка: {risk.dca_drawdown_pct*100:.2f}%", callback_data="edit_risk_dca_drawdown_pct")],
            [InlineKeyboardButton(f"   Макс: {risk.dca_max_adds}", callback_data="edit_risk_dca_max_adds")],
            [InlineKeyboardButton(f"   Мин. уверенность: {risk.dca_min_confidence*100:.0f}%", callback_data="edit_risk_dca_min_confidence")],
            [InlineKeyboardButton(f"🔁 Реверс: {'✅' if risk.reverse_on_strong_signal else '❌'}", callback_data="toggle_risk_reverse_on_strong_signal")],
            [InlineKeyboardButton(f"   Мин. уверенность: {risk.reverse_min_confidence*100:.0f}%", callback_data="edit_risk_reverse_min_confidence")],
            [InlineKeyboardButton(f"🔄 Трейлинг: {'✅' if risk.enable_trailing_stop else '❌'}", callback_data="toggle_risk_enable_trailing_stop")],
            [InlineKeyboardButton(f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_activation_pct")],
            [InlineKeyboardButton(f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_distance_pct")],
            [InlineKeyboardButton(f"💎 Частичное закрытие: {'✅' if risk.enable_partial_close else '❌'}", callback_data="toggle_risk_enable_partial_close")],
            [InlineKeyboardButton(f"🛡️ Безубыток: {'✅' if risk.enable_breakeven else '❌'}", callback_data="toggle_risk_enable_breakeven")],
            [InlineKeyboardButton(f"   1-я активация: {risk.breakeven_level1_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_level1_activation_pct")],
            [InlineKeyboardButton(f"   1-я SL: {risk.breakeven_level1_sl_pct*100:.2f}%", callback_data="edit_risk_breakeven_level1_sl_pct")],
            [InlineKeyboardButton(f"   2-я активация: {risk.breakeven_level2_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_level2_activation_pct")],
            [InlineKeyboardButton(f"   2-я SL: {risk.breakeven_level2_sl_pct*100:.2f}%", callback_data="edit_risk_breakeven_level2_sl_pct")],
            [InlineKeyboardButton(f"❄️ Cooldown: {'✅' if risk.enable_loss_cooldown else '❌'}", callback_data="toggle_risk_enable_loss_cooldown")],
            [InlineKeyboardButton("🔄 Сбросить на стандартные", callback_data="reset_risk_defaults")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ])
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def show_emergency_menu(self, query):
        """Показывает меню экстренных действий"""
        text = "🚨 ЭКСТРЕННЫЕ ДЕЙСТВИЯ\n\n"
        text += "Внимание! Эти действия необратимы.\n"
        text += "Используйте только в случае необходимости.\n"
        
        keyboard = [
            [InlineKeyboardButton("🛑 СТОП И ЗАКРЫТЬ ВСЕ ПОЗИЦИИ", callback_data="emergency_stop_all")],
            [InlineKeyboardButton("⏸️ ПАУЗА (остановить торговлю)", callback_data="bot_stop")],
            [InlineKeyboardButton("❌ Отмена", callback_data="main_menu")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def emergency_stop_all(self, query):
        """Экстренная остановка с закрытием всех позиций"""
        await query.answer("⚠️ Выполняю экстренную остановку...", show_alert=True)
        
        try:
            # Останавливаем бота
            self.state.set_running(False)
            
            # Закрываем все открытые позиции
            closed_positions = []
            for symbol in self.state.active_symbols:
                try:
                    pos_info = self.bybit.get_position_info(symbol=symbol)
                    if pos_info.get("retCode") == 0:
                        list_data = pos_info.get("result", {}).get("list", [])
                        if list_data:
                            position = list_data[0]
                            size = safe_float(position.get("size"), 0)
                            
                            if size > 0:
                                side = position.get("side")
                                close_side = "Sell" if side == "Buy" else "Buy"
                                
                                # Закрываем позицию
                                resp = self.bybit.place_order(
                                    symbol=symbol,
                                    side=close_side,
                                    qty=size,
                                    order_type="Market",
                                    reduce_only=True
                                )
                                
                                if resp.get("retCode") == 0:
                                    closed_positions.append(symbol)
                                    logger.info(f"Emergency closed position for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error closing position for {symbol}: {e}")
            
            message = "🚨 ЭКСТРЕННАЯ ОСТАНОВКА ВЫПОЛНЕНА\n\n"
            message += f"Бот остановлен: ✅\n"
            message += f"Закрыто позиций: {len(closed_positions)}\n"
            if closed_positions:
                message += f"Символы: {', '.join(closed_positions)}"
            
            await query.edit_message_text(message, reply_markup=self.get_main_keyboard())
        
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            await query.edit_message_text(
                f"❌ Ошибка при экстренной остановке:\n{str(e)}",
                reply_markup=self.get_main_keyboard()
            )
    
    async def show_dashboard(self, query):
        """Показывает dashboard с ключевыми метриками"""
        from datetime import datetime, timedelta
        
        text = "📊 DASHBOARD\n\n"
        text += f"🕐 Обновлено: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        # Баланс
        if self.bybit:
            try:
                balance_info = self.bybit.get_wallet_balance()
                if balance_info.get("retCode") == 0:
                    result = balance_info.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        wallet = list_data[0].get("coin", [])
                        usdt_coin = next((c for c in wallet if c.get("coin") == "USDT"), None)
                        if usdt_coin:
                            wallet_balance = safe_float(usdt_coin.get("walletBalance"), 0)
            except Exception as e:
                logger.error(f"Error getting balance: {e}")
        
        # Открытые позиции (для расчета маржи)
        open_count = 0
        total_pnl = 0
        total_margin = 0.0
        if self.bybit:
            try:
                for symbol in self.state.active_symbols:
                    pos_info = self.bybit.get_position_info(symbol=symbol)
                    if pos_info.get("retCode") == 0:
                        list_data = pos_info.get("result", {}).get("list", [])
                        for p in list_data:
                            size = safe_float(p.get("size"), 0)
                            if size > 0:
                                open_count += 1
                                unrealised_pnl = safe_float(p.get("unrealisedPnl"), 0)
                                total_pnl += unrealised_pnl
                                
                                # Получаем маржу позиции для расчета доступного баланса
                                margin = safe_float(p.get("positionMargin"), 0)
                                if margin == 0:
                                    margin = safe_float(p.get("positionIM"), 0)  # Initial Margin
                                if margin == 0:
                                    # Рассчитываем маржу из стоимости позиции и плеча
                                    position_value = safe_float(p.get("positionValue"), 0)
                                    symbol_for_lev = p.get("symbol", "")
                                    default_lev = self.settings.get_leverage_for_symbol(symbol_for_lev)
                                    leverage_str = p.get("leverage", str(default_lev))
                                    leverage = safe_float(leverage_str, default_lev)
                                    if position_value > 0 and leverage > 0:
                                        margin = position_value / leverage
                                
                                total_margin += margin
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
        
        # Вычисляем доступный баланс: баланс минус сумма маржи всех позиций
        available = wallet_balance - total_margin
        if available < 0:
            available = 0.0  # Не показываем отрицательные значения
        
        # Показываем баланс
        if wallet_balance > 0:
            stats = self.state.get_stats()
            total_pnl_pct = (stats['total_pnl'] / wallet_balance * 100) if wallet_balance > 0 else 0
            
            text += "💰 БАЛАНС\n"
            text += f"Текущий: ${wallet_balance:.2f} "
            text += f"({total_pnl_pct:+.2f}%)\n"
            text += f"Доступно: ${available:.2f}\n"
            text += f"В позициях: ${total_margin:.2f}\n\n"
        
        text += f"📈 ОТКРЫТЫЕ ПОЗИЦИИ ({open_count})\n"
        if open_count > 0:
            text += f"Текущий PnL: ${total_pnl:+.2f}\n\n"
        else:
            text += "(нет открытых позиций)\n\n"
        
        # Статистика за сегодня
        today = datetime.now().date()
        today_trades = [t for t in self.state.trades 
                       if t.status == "closed" and 
                       datetime.fromisoformat(t.exit_time).date() == today if t.exit_time]
        
        if today_trades:
            today_pnl = sum(t.pnl_usd for t in today_trades)
            today_wins = len([t for t in today_trades if t.pnl_usd > 0])
            
            text += "📊 СЕГОДНЯ\n"
            text += f"Сделок: {len(today_trades)} ({today_wins} прибыльных)\n"
            text += f"PnL: ${today_pnl:+.2f}\n"
            
            if today_trades:
                best_trade = max(today_trades, key=lambda t: t.pnl_usd)
                text += f"Лучшая: {best_trade.symbol} ${best_trade.pnl_usd:+.2f}\n\n"
        else:
            text += "📊 СЕГОДНЯ\n(нет завершенных сделок)\n\n"
        
        # Статистика за неделю
        week_ago = datetime.now() - timedelta(days=7)
        week_trades = [t for t in self.state.trades 
                      if t.status == "closed" and 
                      datetime.fromisoformat(t.exit_time) >= week_ago if t.exit_time]
        
        if week_trades:
            week_pnl = sum(t.pnl_usd for t in week_trades)
            week_wins = len([t for t in week_trades if t.pnl_usd > 0])
            week_winrate = (week_wins / len(week_trades) * 100) if week_trades else 0
            
            text += "🎯 НЕДЕЛЯ\n"
            text += f"PnL: ${week_pnl:+.2f}\n"
            text += f"Винрейт: {week_winrate:.1f}% ({week_wins}/{len(week_trades)})\n\n"
        
        # Статус системы
        text += "⚡ СИСТЕМА\n"
        text += f"Статус: {'🟢 Работает' if self.state.is_running else '🔴 Остановлен'}\n"
        text += f"Активных пар: {len(self.state.active_symbols)}\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Обновить", callback_data="dashboard")],
            [InlineKeyboardButton("📊 Подробная статистика", callback_data="stats")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await self.safe_edit_message(query, text, reply_markup=InlineKeyboardMarkup(keyboard))
