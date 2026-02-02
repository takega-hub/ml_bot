import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import threading

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
class SymbolCooldown:
    """Cooldown для символа после убыточных сделок"""
    symbol: str
    cooldown_until: str  # ISO format datetime
    consecutive_losses: int
    reason: str

class BotState:
    def __init__(self, state_file: str = "runtime_state.json"):
        self.state_file = Path(state_file)
        self.lock = threading.Lock()
        
        # Default state
        self.is_running: bool = False
        self.active_symbols: List[str] = ["BTCUSDT"]
        self.symbol_models: Dict[str, str] = {}  # symbol -> model_path
        self.max_active_symbols: int = 5
        
        # History
        self.trades: List[TradeRecord] = []
        self.signals: List[SignalRecord] = []
        
        # Cooldowns для защиты от убытков
        self.cooldowns: Dict[str, SymbolCooldown] = {}  # symbol -> cooldown
        
        self.load()

    def load(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.is_running = data.get("is_running", False)
                self.active_symbols = data.get("active_symbols", ["BTCUSDT"])
                self.symbol_models = data.get("symbol_models", {})
                
                # Load trades
                for t in data.get("trades", []):
                    self.trades.append(TradeRecord(**t))
                
                # Load signals
                for s in data.get("signals", []):
                    self.signals.append(SignalRecord(**s))
                
                # Load cooldowns
                for symbol, cooldown_data in data.get("cooldowns", {}).items():
                    self.cooldowns[symbol] = SymbolCooldown(**cooldown_data)
        except Exception as e:
            print(f"[state] Error loading state: {e}")

    def save(self):
        with self.lock:
            try:
                data = {
                    "is_running": self.is_running,
                    "active_symbols": self.active_symbols,
                    "symbol_models": self.symbol_models,
                    "trades": [asdict(t) for t in self.trades[-500:]],  # Keep last 500
                    "signals": [asdict(s) for s in self.signals[-1000:]],  # Keep last 1000
                    "cooldowns": {symbol: asdict(cooldown) for symbol, cooldown in self.cooldowns.items()}
                }
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[state] Error saving state: {e}")

    def set_running(self, status: bool):
        self.is_running = status
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
        with self.lock:
            self.signals.append(signal)
            if len(self.signals) > 1000:
                self.signals.pop(0)
        self.save()

    def add_trade(self, trade: TradeRecord):
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
        with self.lock:
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
        
        if should_save:
            # Сохраняем вне lock, чтобы избежать дедлока
            self.save()
        return False
    
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
    
    def update_trade_on_close(self, symbol: str, exit_price: float, pnl_usd: float, pnl_pct: float):
        """Обновляет сделку при закрытии и проверяет необходимость cooldown"""
        with self.lock:
            # Находим открытую сделку для символа
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.status == "open":
                    trade.exit_price = exit_price
                    trade.exit_time = datetime.now().isoformat()
                    trade.pnl_usd = pnl_usd
                    trade.pnl_pct = pnl_pct
                    trade.status = "closed"
                    break
        
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
