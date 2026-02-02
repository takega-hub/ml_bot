import logging
import asyncio
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
except ImportError as e:
    raise ImportError(
        "python-telegram-bot не установлен. Установите его командой: pip install python-telegram-bot\n"
        "Или установите все зависимости: pip install -r requirements.txt"
    ) from e
from bot.config import AppSettings
from bot.state import BotState
from bot.model_manager import ModelManager
from pathlib import Path

# Логирование уже настроено в run_bot.py, не нужно настраивать здесь
# logging.basicConfig() добавляет обработчик к root logger, что вызывает дублирование логов
logger = logging.getLogger(__name__)

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
        self.waiting_for_symbol = {}  # user_id -> True если ждем ввод символа
        self.waiting_for_risk_setting = {}  # user_id -> setting_name для редактирования настроек риска
        self.waiting_for_ml_setting = {}  # user_id -> setting_name для редактирования ML настроек

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
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def check_auth(self, update: Update) -> bool:
        user_id = update.effective_user.id
        if self.settings.allowed_user_id and user_id != self.settings.allowed_user_id:
            await update.message.reply_text("⛔ Доступ запрещен. Ваш ID не в вайтлисте.")
            return False
        return True

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
                                leverage_str = p.get("leverage", str(self.settings.leverage))
                                leverage = safe_float(leverage_str, self.settings.leverage)
                                
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
                model_path = self.state.symbol_models.get(symbol)
                if model_path and Path(model_path).exists():
                    model_name = Path(model_path).stem
                    
                    # Определяем тип модели
                    is_ensemble = "ensemble" in model_name.lower()
                    min_strength = 0.3 if is_ensemble else 60.0
                    
                    status_text += f"Пара: {symbol} | Модель: {model_name}\n"
                    status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                else:
                    # Пытаемся найти модель автоматически
                    models = self.model_manager.find_models_for_symbol(symbol)
                    if models:
                        # Берем самую новую
                        model_path = str(models[0])
                        self.model_manager.apply_model(symbol, model_path)
                        model_name = models[0].stem
                        
                        # Определяем тип модели
                        is_ensemble = "ensemble" in model_name.lower()
                        min_strength = 0.3 if is_ensemble else 60.0
                        
                        status_text += f"Пара: {symbol} | Модель: {model_name} (авто)\n"
                        status_text += f"   🎯 Уверенность: ≥{self.settings.ml_strategy.confidence_threshold*100:.0f}% | Сила: ≥{min_strength:.1f}%\n"
                    else:
                        status_text += f"Пара: {symbol} | Модель: ❌ Не найдена\n"
        
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
        elif query.data == "history_menu":
            await self.show_history_menu(query)
        elif query.data == "history_signals":
            await self.show_signals(query)
        elif query.data == "history_trades":
            await self.show_trades(query)
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
        # Получаем все известные символы (из state и предопределенные)
        all_possible = list(
            set(
                [s for s in (["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"] + self.state.active_symbols)
                 if isinstance(s, str) and s.endswith("USDT")]
            )
        )
        all_possible.sort()
        
        keyboard = []
        for s in all_possible:
            status = "✅" if s in self.state.active_symbols else "❌"
            keyboard.append([InlineKeyboardButton(f"{status} {s}", callback_data=f"toggle_{s}")])
        
        keyboard.append([InlineKeyboardButton("➕ Добавить новую пару", callback_data="add_pair")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="status_info")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        await query.edit_message_text("⚙️ Настройка активных пар (макс 5):", reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_history_menu(self, query):
        keyboard = [
            [InlineKeyboardButton("🔍 ИСТОРИЯ СИГНАЛОВ", callback_data="history_signals")],
            [InlineKeyboardButton("📈 ИСТОРИЯ СДЕЛОК", callback_data="history_trades")],
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
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

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
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

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
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

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
        
        keyboard.append([InlineKeyboardButton("🔄 Переобучить все модели", callback_data="retrain_all")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="status_info")])
        keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_auth(update): return
        
        user_id = update.effective_user.id
        text = update.message.text.strip()
        
        # Проверяем, ждем ли мы ввод настройки риска
        if user_id in self.waiting_for_risk_setting:
            setting_name = self.waiting_for_risk_setting.pop(user_id)
            await self.process_risk_setting_input(update, setting_name, text)
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
            
            # Проверяем, не превышен ли лимит
            if len(self.state.active_symbols) >= self.state.max_active_symbols:
                await update.message.reply_text(
                    f"⚠️ Достигнут лимит в {self.state.max_active_symbols} активных пар!\n"
                    "Сначала отключите одну из текущих пар.",
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
                
                # Символ валиден, добавляем в список
                self.state.toggle_symbol(text)
                
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
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
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
                    await self.send_notification(f"❌ Ошибка при тестировании {model_name}")
            
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
                comparison = self.model_manager.train_and_compare(symbol)
                
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
            
            # Путь к скрипту обучения
            script_path = Path(__file__).parent.parent / "retrain_all_models.py"
            
            if not script_path.exists():
                await self.send_notification(f"❌ Скрипт обучения не найден: {script_path}")
                return
            
            # Запускаем обучение в отдельном процессе
            process = await asyncio.create_subprocess_exec(
                "python3", str(script_path), "--symbol", symbol,
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
    
    async def train_new_pair_async(self, symbol: str, user_id: int):
        """Асинхронная функция для обучения модели новой пары"""
        try:
            await self.send_notification(f"🔄 Начато обучение модели для {symbol}...")
            
            # Запускаем обучение (это синхронная операция, но мы в отдельной задаче)
            comparison = self.model_manager.train_and_compare(symbol)
            
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
                    f"Модель автоматически применена и готова к торговле."
                )
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
            "trailing_stop_activation_pct": ("Активация трейлинг стопа (в %)", "0.3", "Пример: 0.3 означает 0.3%"),
            "trailing_stop_distance_pct": ("Расстояние трейлинг стопа (в %)", "0.2", "Пример: 0.2 означает 0.2%"),
            "breakeven_activation_pct": ("Активация безубытка (в %)", "0.5", "Пример: 0.5 означает 0.5%"),
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
            
            # Сохраняем настройки
            self.save_ml_settings()
            
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
            
            elif setting_name == "breakeven_activation_pct":
                if 0.1 <= value <= 5.0:
                    risk.breakeven_activation_pct = value / 100.0
                else:
                    await update.message.reply_text("❌ Значение должно быть от 0.1 до 5%")
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
            text += f"📈 Take Profit: {risk.take_profit_pct*100:.2f}%\n"
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
            text += f"🔄 Трейлинг стоп: {'✅ Включен' if risk.enable_trailing_stop else '❌ Выключен'}\n"
            text += f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%\n"
            text += f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%\n\n"
            text += f"💎 Частичное закрытие: {'✅ Включено' if risk.enable_partial_close else '❌ Выключено'}\n"
            text += f"🛡️ Безубыток: {'✅ Включен' if risk.enable_breakeven else '❌ Выключен'}\n"
            text += f"   Активация при: {risk.breakeven_activation_pct*100:.2f}%\n\n"
            text += f"❄️ Cooldown после убытков: {'✅ Включен' if risk.enable_loss_cooldown else '❌ Выключен'}\n"
            
            keyboard = [
                [InlineKeyboardButton(f"💰 Маржа: {risk.margin_pct_balance*100:.0f}%", callback_data="edit_risk_margin_pct_balance")],
                [InlineKeyboardButton(f"💰 Сумма: ${risk.base_order_usd:.2f}", callback_data="edit_risk_base_order_usd")],
            ]
            
            keyboard.extend([
                [InlineKeyboardButton(f"📉 SL: {risk.stop_loss_pct*100:.2f}%", callback_data="edit_risk_stop_loss_pct")],
                [InlineKeyboardButton(f"📈 TP: {risk.take_profit_pct*100:.2f}%", callback_data="edit_risk_take_profit_pct")],
                [InlineKeyboardButton(f"💸 Комиссия: {risk.fee_rate*100:.4f}%", callback_data="edit_risk_fee_rate")],
                [InlineKeyboardButton(f"🧭 Mid TP: {risk.mid_term_tp_pct*100:.2f}%", callback_data="edit_risk_mid_term_tp_pct")],
                [InlineKeyboardButton(f"🧭 Long TP: {risk.long_term_tp_pct*100:.2f}%", callback_data="edit_risk_long_term_tp_pct")],
                [InlineKeyboardButton(f"🧭 Long SL: {risk.long_term_sl_pct*100:.2f}%", callback_data="edit_risk_long_term_sl_pct")],
                [InlineKeyboardButton(f"↪️ Игнор. реверс: {'✅' if risk.long_term_ignore_reverse else '❌'}", callback_data="toggle_risk_long_term_ignore_reverse")],
                [InlineKeyboardButton(f"➕ DCA: {'✅' if risk.dca_enabled else '❌'}", callback_data="toggle_risk_dca_enabled")],
                [InlineKeyboardButton(f"   Просадка: {risk.dca_drawdown_pct*100:.2f}%", callback_data="edit_risk_dca_drawdown_pct")],
                [InlineKeyboardButton(f"   Макс: {risk.dca_max_adds}", callback_data="edit_risk_dca_max_adds")],
                [InlineKeyboardButton(f"   Мин. уверенность: {risk.dca_min_confidence*100:.0f}%", callback_data="edit_risk_dca_min_confidence")],
                [InlineKeyboardButton(f"🔄 Трейлинг: {'✅' if risk.enable_trailing_stop else '❌'}", callback_data="toggle_risk_enable_trailing_stop")],
                [InlineKeyboardButton(f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_activation_pct")],
                [InlineKeyboardButton(f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_distance_pct")],
                [InlineKeyboardButton(f"💎 Частичное закрытие: {'✅' if risk.enable_partial_close else '❌'}", callback_data="toggle_risk_enable_partial_close")],
                [InlineKeyboardButton(f"🛡️ Безубыток: {'✅' if risk.enable_breakeven else '❌'}", callback_data="toggle_risk_enable_breakeven")],
                [InlineKeyboardButton(f"   Активация: {risk.breakeven_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_activation_pct")],
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
    
    async def toggle_risk_setting(self, query, setting_name: str):
        """Переключает булеву настройку риска"""
        risk = self.settings.risk
        
        if setting_name == "enable_trailing_stop":
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
            
            config_file = Path("ml_settings.json")
            
            ml_dict = {
                "confidence_threshold": self.settings.ml_strategy.confidence_threshold,
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(ml_dict, f, indent=2, ensure_ascii=False)
            
            logger.info("ML settings saved to ml_settings.json")
        
        except Exception as e:
            logger.error(f"Error saving ML settings: {e}")
    
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
                "enable_trailing_stop": self.settings.risk.enable_trailing_stop,
                "trailing_stop_activation_pct": self.settings.risk.trailing_stop_activation_pct,
                "trailing_stop_distance_pct": self.settings.risk.trailing_stop_distance_pct,
                "enable_partial_close": self.settings.risk.enable_partial_close,
                "enable_breakeven": self.settings.risk.enable_breakeven,
                "breakeven_activation_pct": self.settings.risk.breakeven_activation_pct,
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
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    async def show_ml_settings(self, query):
        """Показывает настройки ML стратегии"""
        ml_settings = self.settings.ml_strategy
        
        text = "🧠 НАСТРОЙКИ ML СТРАТЕГИИ\n\n"
        text += f"🎯 Минимальная уверенность: {ml_settings.confidence_threshold*100:.0f}%\n"
        text += f"💪 Минимальная сила сигнала:\n"
        text += f"   • Ансамбли: 0.3% (фиксировано)\n"
        text += f"   • Одиночные модели: 60% (фиксировано)\n\n"
        
        text += f"ℹ️ Уверенность модели — это вероятность правильного предсказания.\n"
        text += f"Чем выше порог, тем меньше сигналов, но качественнее.\n\n"
        text += f"🔹 Рекомендуемые значения:\n"
        text += f"   • Консервативно: 70-80%\n"
        text += f"   • Сбалансированно: 50-70%\n"
        text += f"   • Агрессивно: 30-50%\n"
        
        keyboard = [
            [InlineKeyboardButton(f"🎯 Уверенность: {ml_settings.confidence_threshold*100:.0f}%", callback_data="edit_ml_confidence_threshold")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
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
        text += f"📈 Take Profit: {risk.take_profit_pct*100:.2f}%\n\n"
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
        text += f"🔄 Трейлинг стоп: {'✅ Включен' if risk.enable_trailing_stop else '❌ Выключен'}\n"
        text += f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%\n"
        text += f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%\n\n"
        text += f"💎 Частичное закрытие: {'✅ Включено' if risk.enable_partial_close else '❌ Выключено'}\n"
        text += f"🛡️ Безубыток: {'✅ Включен' if risk.enable_breakeven else '❌ Выключен'}\n"
        text += f"   Активация при: {risk.breakeven_activation_pct*100:.2f}%\n\n"
        text += f"❄️ Cooldown после убытков: {'✅ Включен' if risk.enable_loss_cooldown else '❌ Выключен'}\n"
        
        keyboard = [
            [InlineKeyboardButton(f"💰 Маржа: {risk.margin_pct_balance*100:.0f}%", callback_data="edit_risk_margin_pct_balance")],
            [InlineKeyboardButton(f"💰 Сумма: ${risk.base_order_usd:.2f}", callback_data="edit_risk_base_order_usd")],
        ]
        
        keyboard.extend([
            [InlineKeyboardButton(f"📉 SL: {risk.stop_loss_pct*100:.2f}%", callback_data="edit_risk_stop_loss_pct")],
            [InlineKeyboardButton(f"📈 TP: {risk.take_profit_pct*100:.2f}%", callback_data="edit_risk_take_profit_pct")],
            [InlineKeyboardButton(f"💸 Комиссия: {risk.fee_rate*100:.4f}%", callback_data="edit_risk_fee_rate")],
            [InlineKeyboardButton(f"🧭 Mid TP: {risk.mid_term_tp_pct*100:.2f}%", callback_data="edit_risk_mid_term_tp_pct")],
            [InlineKeyboardButton(f"🧭 Long TP: {risk.long_term_tp_pct*100:.2f}%", callback_data="edit_risk_long_term_tp_pct")],
            [InlineKeyboardButton(f"🧭 Long SL: {risk.long_term_sl_pct*100:.2f}%", callback_data="edit_risk_long_term_sl_pct")],
            [InlineKeyboardButton(f"↪️ Игнор. реверс: {'✅' if risk.long_term_ignore_reverse else '❌'}", callback_data="toggle_risk_long_term_ignore_reverse")],
            [InlineKeyboardButton(f"➕ DCA: {'✅' if risk.dca_enabled else '❌'}", callback_data="toggle_risk_dca_enabled")],
            [InlineKeyboardButton(f"   Просадка: {risk.dca_drawdown_pct*100:.2f}%", callback_data="edit_risk_dca_drawdown_pct")],
            [InlineKeyboardButton(f"   Макс: {risk.dca_max_adds}", callback_data="edit_risk_dca_max_adds")],
            [InlineKeyboardButton(f"   Мин. уверенность: {risk.dca_min_confidence*100:.0f}%", callback_data="edit_risk_dca_min_confidence")],
            [InlineKeyboardButton(f"🔄 Трейлинг: {'✅' if risk.enable_trailing_stop else '❌'}", callback_data="toggle_risk_enable_trailing_stop")],
            [InlineKeyboardButton(f"   Активация: {risk.trailing_stop_activation_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_activation_pct")],
            [InlineKeyboardButton(f"   Расстояние: {risk.trailing_stop_distance_pct*100:.2f}%", callback_data="edit_risk_trailing_stop_distance_pct")],
            [InlineKeyboardButton(f"💎 Частичное закрытие: {'✅' if risk.enable_partial_close else '❌'}", callback_data="toggle_risk_enable_partial_close")],
            [InlineKeyboardButton(f"🛡️ Безубыток: {'✅' if risk.enable_breakeven else '❌'}", callback_data="toggle_risk_enable_breakeven")],
            [InlineKeyboardButton(f"   Активация: {risk.breakeven_activation_pct*100:.2f}%", callback_data="edit_risk_breakeven_activation_pct")],
            [InlineKeyboardButton(f"❄️ Cooldown: {'✅' if risk.enable_loss_cooldown else '❌'}", callback_data="toggle_risk_enable_loss_cooldown")],
            [InlineKeyboardButton("🔄 Сбросить на стандартные", callback_data="reset_risk_defaults")],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ])
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
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
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
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
                                    leverage_str = p.get("leverage", str(self.settings.leverage))
                                    leverage = safe_float(leverage_str, self.settings.leverage)
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
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
