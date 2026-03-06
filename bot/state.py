import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")
signal_logger = logging.getLogger("signals")

@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    qty: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    entry_time: str = field(default_factory=lambda: datetime.now().isoformat())
    exit_time: Optional[str] = None
    status: str = "open"  # open, closed
    model_name: str = ""
    horizon: str = "short_term"  # short_term, mid_term, long_term
    dca_count: int = 0
    # Дополнительная информация для анализа
    entry_reason: str = ""  # Причина входа (reason из сигнала)
    exit_reason: str = ""  # Причина выхода (TP, SL, trailing, manual и т.д.)
    confidence: float = 0.0  # Уверенность модели при входе
    take_profit: Optional[float] = None  # TP цена
    stop_loss: Optional[float] = None  # SL цена
    leverage: int = 1  # Плечо
    margin_usd: float = 0.0  # Маржа в USD
    commission: float = 0.0  # Комиссия за сделку (открытие + закрытие)
    signal_strength: str = ""  # Сила сигнала (слабое, умеренное, сильное и т.д.)
    signal_parameters: Dict[str, Any] = field(default_factory=dict)  # Дополнительные параметры сигнала

@dataclass
class SignalRecord:
    timestamp: str
    symbol: str
    action: str
    price: float
    confidence: float
    reason: str
    indicators: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationRecord:
    id: str
    timestamp: str
    message: str
    type: str = "info" # info, success, warning, error
    read: bool = False

@dataclass
class SymbolCooldown:
    """Cooldown для символа после убыточных сделок"""
    symbol: str
    cooldown_until: str  # ISO format datetime
    consecutive_losses: int
    reason: str

@dataclass
class TPReentryGuard:
    symbol: str
    side: str
    exit_time_utc: str
    exit_price: float
    wait_until_utc: str
    wait_candles: int
    skipped_signals: int = 0
    skipped_wait: int = 0
    skipped_criteria: int = 0
    allowed_reentries: int = 0
    last_decision: str = ""

class BotState:
    def __init__(self, state_file: str = "runtime_state.json"):
        self.state_file = Path(state_file)
        self.lock = threading.Lock()
        self._save_lock = threading.Lock()  # Отдельный lock для сохранения, чтобы избежать блокировки
        import time
        self._last_save_time = time.time()  # Время последнего сохранения
        self._save_cooldown = 0.1  # Минимальный интервал между сохранениями (100ms)
        
        # Default state
        self.is_running: bool = False
        self.active_symbols: List[str] = ["BTCUSDT"]
        self.known_symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        self.symbol_models: Dict[str, str] = {}  # symbol -> model_path
        self.symbol_strategies: Dict[str, Dict[str, Any]] = {}  # symbol -> strategy config (mode, models, etc)
        self.max_active_symbols: int = 5
        
        # History
        self.trades: List[TradeRecord] = []
        self.signals: List[SignalRecord] = []
        self.notifications: List[NotificationRecord] = []
        
        # Cooldowns для защиты от убытков
        self.cooldowns: Dict[str, SymbolCooldown] = {}  # symbol -> cooldown

        self.tp_reentry: Dict[str, TPReentryGuard] = {}
        
        self.load()

    def load(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.is_running = data.get("is_running", False)
                self.active_symbols = data.get("active_symbols", ["BTCUSDT"])
                self.known_symbols = data.get(
                    "known_symbols",
                    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
                )
                # Убираем некорректные символы (например, от старых callback конфликтов)
                self.active_symbols = [
                    s for s in self.active_symbols
                    if isinstance(s, str) and s.endswith("USDT")
                ]
                self.known_symbols = [
                    s for s in self.known_symbols
                    if isinstance(s, str) and s.endswith("USDT")
                ]
                # Гарантируем, что активные пары присутствуют в списке известных
                for s in self.active_symbols:
                    if s not in self.known_symbols:
                        self.known_symbols.append(s)
                self.symbol_models = data.get("symbol_models", {})
                self.symbol_strategies = data.get("symbol_strategies", {})
                
                # Load trades (с поддержкой обратной совместимости для старых записей)
                for t in data.get("trades", []):
                    # Устанавливаем значения по умолчанию для новых полей, если их нет
                    trade_dict = dict(t)
                    trade_dict.setdefault("entry_reason", "")
                    trade_dict.setdefault("exit_reason", "")
                    trade_dict.setdefault("confidence", 0.0)
                    trade_dict.setdefault("take_profit", None)
                    trade_dict.setdefault("stop_loss", None)
                    trade_dict.setdefault("leverage", 1)
                    trade_dict.setdefault("margin_usd", 0.0)
                    trade_dict.setdefault("commission", 0.0)
                    trade_dict.setdefault("signal_strength", "")
                    trade_dict.setdefault("signal_parameters", {})
                    self.trades.append(TradeRecord(**trade_dict))
                
                # Load signals
                for s in data.get("signals", []):
                    self.signals.append(SignalRecord(**s))
                
                # Load cooldowns
                for symbol, cooldown_data in data.get("cooldowns", {}).items():
                    self.cooldowns[symbol] = SymbolCooldown(**cooldown_data)

                for symbol, guard_data in data.get("tp_reentry", {}).items():
                    self.tp_reentry[symbol] = TPReentryGuard(**guard_data)
        except Exception as e:
            print(f"[state] Error loading state: {e}")

    def save(self):
        # Защита от частых вызовов - пропускаем сохранение, если прошло меньше _save_cooldown секунд
        import time
        current_time = time.time()
        if current_time - self._last_save_time < self._save_cooldown:
            return  # Пропускаем сохранение, если оно было недавно
        
        # Используем отдельный lock для сохранения, чтобы не блокировать основной lock
        if not self._save_lock.acquire(blocking=False):
            return  # Если сохранение уже идет, пропускаем
        
        try:
            # Минимизируем время удержания основного lock - готовим данные под lock, пишем без lock
            try:
                # Быстро копируем данные под lock
                with self.lock:
                    data = {
                        "is_running": self.is_running,
                        "active_symbols": list(self.active_symbols),  # Копируем список
                        "known_symbols": list(self.known_symbols),  # Копируем список
                        "symbol_models": dict(self.symbol_models),  # Копируем словарь
                        "symbol_strategies": dict(self.symbol_strategies),
                        "trades": [asdict(t) for t in self.trades[-500:]],  # Keep last 500
                        "signals": [asdict(s) for s in self.signals[-1000:]],  # Keep last 1000
                        "cooldowns": {symbol: asdict(cooldown) for symbol, cooldown in self.cooldowns.items()},
                        "tp_reentry": {symbol: asdict(guard) for symbol, guard in self.tp_reentry.items()},
                    }
                
                # Записываем в файл БЕЗ основного lock, чтобы не блокировать другие операции
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                self._last_save_time = current_time
            except Exception as e:
                logger.error(f"[state] Error saving state: {e}")
        finally:
            self._save_lock.release()

    def set_running(self, status: bool):
        self.is_running = status
        self.save()

    def set_tp_reentry_guard(
        self,
        symbol: str,
        side: str,
        exit_time_utc: str,
        exit_price: float,
        wait_until_utc: str,
        wait_candles: int,
    ) -> None:
        symbol = symbol.upper()
        with self.lock:
            self.tp_reentry[symbol] = TPReentryGuard(
                symbol=symbol,
                side=side,
                exit_time_utc=exit_time_utc,
                exit_price=float(exit_price),
                wait_until_utc=wait_until_utc,
                wait_candles=int(wait_candles),
            )
        self.save()

    def get_tp_reentry_guard(self, symbol: str) -> Optional[TPReentryGuard]:
        symbol = symbol.upper()
        with self.lock:
            return self.tp_reentry.get(symbol)

    def clear_tp_reentry_guard(self, symbol: str) -> None:
        symbol = symbol.upper()
        with self.lock:
            if symbol in self.tp_reentry:
                del self.tp_reentry[symbol]
                should_save = True
            else:
                should_save = False
        if should_save:
            self.save()

    def tp_reentry_record_skip(self, symbol: str, reason: str, is_wait: bool) -> None:
        symbol = symbol.upper()
        with self.lock:
            g = self.tp_reentry.get(symbol)
            if not g:
                return
            g.skipped_signals += 1
            if is_wait:
                g.skipped_wait += 1
            else:
                g.skipped_criteria += 1
            g.last_decision = reason[:200]
        self.save()

    def tp_reentry_record_allow(self, symbol: str, reason: str) -> None:
        symbol = symbol.upper()
        with self.lock:
            g = self.tp_reentry.get(symbol)
            if not g:
                return
            g.allowed_reentries += 1
            g.last_decision = reason[:200]
        self.save()

    def toggle_symbol(self, symbol: str) -> bool:
        """Returns True if enabled, False if disabled, None if limit reached"""
        symbol = symbol.upper()
        with self.lock:
            if symbol in self.active_symbols:
                self.active_symbols.remove(symbol)
                res = False
            else:
                if len(self.active_symbols) >= self.max_active_symbols:
                    return None
                self.active_symbols.append(symbol)
                res = True
        self.save()
        return res

    def ensure_known_symbols(self, symbols: List[str]) -> None:
        """Объединяет известные пары с указанными и сохраняет."""
        normalized = [s for s in symbols if isinstance(s, str) and s.endswith("USDT")]
        changed = False
        with self.lock:
            for s in normalized:
                if s not in self.known_symbols:
                    self.known_symbols.append(s)
                    changed = True
            for s in self.active_symbols:
                if s not in self.known_symbols:
                    self.known_symbols.append(s)
                    changed = True
        if changed:
            self.save()

    def add_known_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        with self.lock:
            if symbol not in self.known_symbols:
                self.known_symbols.append(symbol)
                should_save = True
            else:
                should_save = False
        if should_save:
            self.save()

    def enable_symbol(self, symbol: str) -> bool:
        """Включает пару, если возможно. True=включена, False=уже включена, None=лимит."""
        symbol = symbol.upper()
        with self.lock:
            if symbol in self.active_symbols:
                return False
            if len(self.active_symbols) >= self.max_active_symbols:
                return None
            self.active_symbols.append(symbol)
        self.save()
        return True

    def add_signal(self, symbol: str, action: str, price: float, confidence: float, reason: str, indicators: Dict[str, Any] = None):
        signal = SignalRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=action,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators=indicators or {}
        )
        
        # Log to signals.log
        signal_logger.info(f"SIGNAL: {symbol} | Action: {action} | Price: {price} | Conf: {confidence:.2f} | Reason: {reason}")
        
        with self.lock:
            self.signals.append(signal)
            if len(self.signals) > 1000:
                self.signals.pop(0)
        self.save()

    def add_trade(self, trade: TradeRecord):
        # Log to trades.log
        trade_logger.info(f"TRADE OPEN: {trade.symbol} | Side: {trade.side} | Entry: {trade.entry_price} | Size: {trade.qty} | Reason: {trade.entry_reason}")
        
        with self.lock:
            self.trades.append(trade)
            if len(self.trades) > 500:
                self.trades.pop(0)
        self.save()

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            closed_trades = [t for t in self.trades if t.status == "closed"]
            if not closed_trades:
                return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0}
            
            total_pnl = sum(t.pnl_usd for t in closed_trades)
            wins = len([t for t in closed_trades if t.pnl_usd > 0])
            win_rate = (wins / len(closed_trades)) * 100
            
            return {
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "total_trades": len(closed_trades)
            }
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Проверяет, находится ли символ в cooldown"""
        should_save = False
        try:
            # Используем timeout для lock, чтобы избежать зависания
            if not self.lock.acquire(timeout=2.0):
                logger.warning(f"[state] Failed to acquire lock for is_symbol_in_cooldown({symbol}), assuming no cooldown")
                return False
            
            try:
                if symbol not in self.cooldowns:
                    return False
                
                cooldown = self.cooldowns[symbol]
                cooldown_until = datetime.fromisoformat(cooldown.cooldown_until)
                
                if datetime.now() < cooldown_until:
                    return True
                else:
                    # Cooldown истек, удаляем
                    del self.cooldowns[symbol]
                    should_save = True
            finally:
                self.lock.release()
        except Exception as e:
            logger.error(f"[state] Error in is_symbol_in_cooldown({symbol}): {e}")
            return False
        
        if should_save:
            # Сохраняем вне lock, чтобы избежать дедлока
            # Используем try-except, чтобы не блокировать, если save() зависнет
            try:
                self.save()
            except Exception as e:
                logger.error(f"[state] Error saving after cooldown removal: {e}")
        return False
    
    def get_cooldown_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о cooldown для символа (если есть)"""
        should_save = False
        try:
            # Используем timeout для lock, чтобы избежать зависания
            if not self.lock.acquire(timeout=2.0):
                logger.warning(f"[state] Failed to acquire lock for get_cooldown_info({symbol})")
                return None
            
            try:
                if symbol not in self.cooldowns:
                    return None
                
                cooldown = self.cooldowns[symbol]
                cooldown_until = datetime.fromisoformat(cooldown.cooldown_until)
                now = datetime.now()
                
                if now < cooldown_until:
                    # Cooldown активен
                    time_left = cooldown_until - now
                    hours_left = time_left.total_seconds() / 3600
                    return {
                        "active": True,
                        "cooldown_until": cooldown_until,
                        "hours_left": hours_left,
                        "consecutive_losses": cooldown.consecutive_losses,
                        "reason": cooldown.reason
                    }
                else:
                    # Cooldown истек, удаляем
                    del self.cooldowns[symbol]
                    should_save = True
            finally:
                self.lock.release()
        except Exception as e:
            logger.error(f"[state] Error in get_cooldown_info({symbol}): {e}")
            return None
        
        if should_save:
            # Сохраняем вне lock, чтобы избежать дедлока
            try:
                self.save()
            except Exception as e:
                logger.error(f"[state] Error saving after cooldown removal in get_cooldown_info: {e}")
        return None
    
    def set_cooldown(self, symbol: str, consecutive_losses: int, reason: str):
        """Устанавливает cooldown для символа на основе количества убытков подряд"""
        # Определяем длительность cooldown
        if consecutive_losses == 1:
            cooldown_duration = timedelta(minutes=30)
        elif consecutive_losses == 2:
            cooldown_duration = timedelta(hours=2)
        else:  # 3 и больше
            cooldown_duration = timedelta(hours=24)
        
        cooldown_until = datetime.now() + cooldown_duration
        
        with self.lock:
            self.cooldowns[symbol] = SymbolCooldown(
                symbol=symbol,
                cooldown_until=cooldown_until.isoformat(),
                consecutive_losses=consecutive_losses,
                reason=reason
            )
        self.save()
    
    def set_cooldown_until_next_candle(self, symbol: str, timeframe: str = "15m"):
        """Устанавливает cooldown для символа до закрытия следующей свечи (15 минут для 15m)"""
        from datetime import datetime, timedelta
        
        # Парсим timeframe
        minutes = 15  # По умолчанию 15 минут
        if timeframe.endswith("m"):
            try:
                minutes = int(timeframe[:-1])
            except:
                minutes = 15
        elif timeframe.endswith("h"):
            try:
                minutes = int(timeframe[:-1]) * 60
            except:
                minutes = 15
        
        now = datetime.now()
        
        # Вычисляем время закрытия следующей свечи
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
        
        with self.lock:
            # Проверяем, есть ли уже кулдаун от убытков - если да, берем максимальный
            existing_cooldown = self.cooldowns.get(symbol)
            if existing_cooldown:
                existing_until = datetime.fromisoformat(existing_cooldown.cooldown_until)
                if existing_until > next_close:
                    # Существующий кулдаун длиннее, оставляем его
                    logger.info(f"Keeping existing cooldown for {symbol} until {existing_until.isoformat()} (longer than next candle)")
                    return
            
            self.cooldowns[symbol] = SymbolCooldown(
                symbol=symbol,
                cooldown_until=next_close.isoformat(),
                consecutive_losses=0,
                reason="position_closed"
            )
        self.save()
        logger.info(f"Set cooldown for {symbol} until next candle close: {next_close.isoformat()}")
    
    def remove_cooldown(self, symbol: str):
        """Удаляет cooldown для символа (ручное снятие разморозки)"""
        try:
            # Используем timeout для lock, чтобы избежать зависания
            if not self.lock.acquire(timeout=2.0):
                logger.warning(f"[state] Failed to acquire lock for remove_cooldown({symbol})")
                return
            
            try:
                if symbol in self.cooldowns:
                    del self.cooldowns[symbol]
            finally:
                self.lock.release()
        except Exception as e:
            logger.error(f"[state] Error in remove_cooldown({symbol}): {e}")
            return
        
        # Сохраняем вне lock, чтобы избежать дедлока
        try:
            self.save()
        except Exception as e:
            logger.error(f"[state] Error saving after remove_cooldown({symbol}): {e}")
    
    def get_consecutive_losses(self, symbol: str) -> int:
        """Получает количество последовательных убытков для символа"""
        with self.lock:
            # Получаем последние закрытые сделки для символа
            symbol_trades = [t for t in reversed(self.trades) if t.symbol == symbol and t.status == "closed"]
            
            consecutive_losses = 0
            for trade in symbol_trades:
                if trade.pnl_usd < 0:
                    consecutive_losses += 1
                else:
                    break
            
            return consecutive_losses
    
    def update_trade_on_close(
        self, 
        symbol: str, 
        exit_price: float, 
        pnl_usd: float, 
        pnl_pct: float,
        exit_reason: str = "",
        commission: float = 0.0
    ):
        """
        Обновляет сделку при закрытии и проверяет необходимость cooldown.
        Также экспортирует закрытую сделку в Excel.
        """
        logger.info(f"[{symbol}] 🔄 update_trade_on_close called: exit_price={exit_price:.2f}, pnl_usd={pnl_usd:.2f}, pnl_pct={pnl_pct:.2f}%, commission={commission:.2f}, exit_reason={exit_reason}")
        
        # Log to trades.log
        trade_logger.info(f"TRADE CLOSE: {symbol} | Exit: {exit_price} | PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%) | Fee: ${commission:.2f} | Reason: {exit_reason}")
        
        closed_trade = None
        with self.lock:
            # Находим открытую сделку для символа
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.status == "open":
                    trade.exit_price = exit_price
                    trade.exit_time = datetime.now().isoformat()
                    trade.pnl_usd = pnl_usd
                    trade.pnl_pct = pnl_pct
                    trade.status = "closed"
                    trade.commission = commission
                    if exit_reason:
                        trade.exit_reason = exit_reason
                    closed_trade = trade
                    break
        
        # Экспортируем закрытую сделку в Excel
        if closed_trade:
            logger.info(f"[{symbol}] 📊 Exporting closed trade to Excel...")
            try:
                from bot.trade_exporter import export_trades_to_excel
                filepath = export_trades_to_excel(
                    [closed_trade],
                    output_dir="trade_history",
                    filename=None  # Автоматическое имя файла
                )
                if filepath:
                    logger.info(f"[{symbol}] ✅ Trade exported to Excel: {filepath}")
                else:
                    logger.warning(f"[{symbol}] ⚠️ Trade export returned empty path")
            except ImportError as e:
                logger.warning(f"[{symbol}] ⚠️ Failed to import trade_exporter: {e}. Install openpyxl: pip install openpyxl")
            except Exception as e:
                logger.error(f"[{symbol}] ❌ Failed to export trade to Excel: {e}", exc_info=True)
        else:
            logger.warning(f"[{symbol}] ⚠️ No closed trade found to export (trade might not have been in state)")
        
        # Проверяем, была ли сделка убыточной
        if pnl_usd < 0:
            consecutive_losses = self.get_consecutive_losses(symbol)
            
            # Устанавливаем cooldown при необходимости
            if consecutive_losses > 0:
                reason = f"{consecutive_losses} убыток(ов) подряд"
                self.set_cooldown(symbol, consecutive_losses, reason)
        
        self.save()
    
    def get_open_position(self, symbol: str) -> Optional[TradeRecord]:
        """Получает информацию об открытой позиции для символа"""
        with self.lock:
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.status == "open":
                    return trade
        return None
    
    def update_position(self, symbol: str, new_size: float, new_entry_price: float):
        """Обновляет информацию о позиции (размер, цена входа при добавлении)"""
        with self.lock:
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.status == "open":
                    trade.qty = new_size
                    trade.entry_price = new_entry_price
                    break
        self.save()

    def increment_dca(self, symbol: str):
        """Увеличивает счетчик усреднений для открытой позиции"""
        with self.lock:
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.status == "open":
                    trade.dca_count += 1
                    break
        self.save()

    def add_notification(self, message: str, type: str = "info"):
        """Добавляет уведомление в очередь"""
        import uuid
        with self.lock:
            notif = NotificationRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                message=message,
                type=type,
                read=False
            )
            self.notifications.append(notif)
            # Храним последние 50 уведомлений
            if len(self.notifications) > 50:
                self.notifications.pop(0)
    
    def get_unread_notifications(self) -> List[Dict[str, Any]]:
        """Возвращает непрочитанные уведомления и помечает их как прочитанные"""
        with self.lock:
            unread = [n for n in self.notifications if not n.read]
            result = [asdict(n) for n in unread]
            
            # Mark as read
            for n in self.notifications:
                n.read = True
            
            return result

    def set_strategy_config(self, symbol: str, config: Dict[str, Any]):
        """Sets strategy configuration for a symbol (mode, models, etc)"""
        symbol = symbol.upper()
        with self.lock:
            self.symbol_strategies[symbol] = config
        self.save()

    def get_strategy_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets strategy configuration for a symbol"""
        symbol = symbol.upper()
        with self.lock:
            return self.symbol_strategies.get(symbol)
