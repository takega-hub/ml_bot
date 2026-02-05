"""
Entry-Only Trading Environment для PPO.
Агент ТОЛЬКО решает когда входить. Выходы автоматические по TP/SL.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Callable
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Действия агента - только HOLD и входы."""
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2


class PositionSide:
    """Сторона позиции."""
    NONE = None
    LONG = "LONG"
    SHORT = "SHORT"


class TradingEnvV2:
    """
    Entry-Only Trading Environment.
    
    Ключевые отличия от V1:
    - Агент НЕ МОЖЕТ закрывать позиции (только TP/SL)
    - Простая reward function: только PnL сделки
    - Фокус на качестве входов, а не на управлении позицией
    - Трендовый фильтр: вход только по направлению тренда
    - Специализация: allowed_side = "long" / "short" / "both"
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0006,  # 0.06% на сторону (Bybit maker)
        slippage_bps: float = 3.0,
        risk_per_trade_pct: float = 1.0,
        max_leverage: float = 1.0,
        min_bars_between_trades: int = 96,  # 1 день между сделками
        min_adx: Optional[float] = 20.0,  # Усилен фильтр
        min_atr_pct: Optional[float] = 0.12,  # Усилен фильтр
        reward_scale: float = 100.0,  # Масштаб reward (для стабильности обучения)
        use_trend_filter: bool = True,  # Фильтр тренда
        allowed_side: str = "both",  # "long", "short", или "both"
    ):
        """
        Args:
            df: DataFrame с OHLCV и фичами
            initial_capital: Начальный капитал
            commission_rate: Комиссия на сторону (0.0006 = 0.06% maker)
            slippage_bps: Слиппедж в базисных пунктах
            risk_per_trade_pct: Риск на сделку в % от капитала
            max_leverage: Максимальное плечо
            min_bars_between_trades: Минимум баров между сделками (96 = 1 день на 15m)
            min_adx: Минимальный ADX для входа
            min_atr_pct: Минимальный ATR% для входа
            reward_scale: Масштаб reward
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000.0
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.min_bars_between_trades = min_bars_between_trades
        self.min_adx = min_adx
        self.min_atr_pct = min_atr_pct
        self.reward_scale = reward_scale
        self.use_trend_filter = use_trend_filter
        
        # Специализация модели
        self.allowed_side = allowed_side.lower()
        if self.allowed_side not in ["long", "short", "both"]:
            raise ValueError(f"allowed_side must be 'long', 'short', or 'both', got '{allowed_side}'")
        
        # Проверяем колонки
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Состояние среды
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Торговое состояние
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None
        
        # Капитал
        self.equity = initial_capital
        self.cash = initial_capital
        self.max_equity = initial_capital
        
        # История
        self.last_trade_step = -min_bars_between_trades - 1
        self.trades = []
        self._pending_reward = 0.0  # Накопленный reward от закрытой сделки
        self._long_count = 0
        self._short_count = 0
        
        # Feature columns
        exclude_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        if len(self.feature_cols) == 0:
            self.df["price_change"] = self.df["close"].pct_change().fillna(0)
            self.feature_cols = ["price_change"]
    
    def reset(self, start_step: Optional[int] = None) -> np.ndarray:
        """Сброс среды."""
        if start_step is None:
            min_start = 200
            self.current_step = np.random.randint(min_start, self.max_steps - 100)
        else:
            self.current_step = max(0, min(start_step, self.max_steps - 1))
        
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None
        
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.max_equity = self.initial_capital
        
        self.last_trade_step = self.current_step - self.min_bars_between_trades - 1
        self.trades = []
        self._long_count = 0
        self._short_count = 0
        self._pending_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: int, risk_manager_callback=None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Выполняет один шаг среды.
        
        Returns:
            (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, {"reason": "max_steps"}
        
        row = self.df.iloc[self.current_step]
        reward = 0.0
        info = {}
        
        # 1. Проверяем TP/SL (ЕДИНСТВЕННЫЙ способ закрыть позицию!)
        if self.position_side != PositionSide.NONE:
            exit_reason = self._check_exits(row)
            if exit_reason:
                pnl = self._close_position(row, exit_reason)
                # Reward = PnL сделки (нормализованный)
                reward = (pnl / self.initial_capital) * self.reward_scale
                
                # Штраф за дисбаланс LONG/SHORT (поощряем разнообразие)
                total_trades = self._long_count + self._short_count
                if total_trades >= 3:
                    imbalance = abs(self._long_count - self._short_count) / total_trades
                    # Штраф = 10% от reward при сильном дисбалансе
                    reward -= abs(reward) * 0.1 * imbalance
                
                info["trade_closed"] = exit_reason
                info["trade_pnl"] = pnl
        
        # 2. Если нет позиции, обрабатываем действие
        if self.position_side == PositionSide.NONE:
            # Маппинг действий в зависимости от режима специализации
            open_long = False
            open_short = False
            
            if self.allowed_side == "long":
                # Режим LONG-only: action=0 HOLD, action=1 OPEN_LONG
                if action == 1:
                    open_long = True
            elif self.allowed_side == "short":
                # Режим SHORT-only: action=0 HOLD, action=1 OPEN_SHORT
                if action == 1:
                    open_short = True
            else:
                # Режим BOTH: action=0 HOLD, action=1 OPEN_LONG, action=2 OPEN_SHORT
                if action == Action.OPEN_LONG:
                    open_long = True
                elif action == Action.OPEN_SHORT:
                    open_short = True
            
            # Выполняем действие (без бонусов/штрафов - чистый PnL)
            if open_long:
                if self._can_open_trade(row, PositionSide.LONG):
                    self._open_position(PositionSide.LONG, row, risk_manager_callback)
                    info["action"] = "opened_long"
                else:
                    info["action_rejected"] = "filters"
            elif open_short:
                if self._can_open_trade(row, PositionSide.SHORT):
                    self._open_position(PositionSide.SHORT, row, risk_manager_callback)
                    info["action"] = "opened_short"
                else:
                    info["action_rejected"] = "filters"
            else:
                info["action"] = "hold"
        else:
            # Позиция открыта - игнорируем действия открытия
            info["action"] = "holding_position"
        
        # 3. Обновляем equity
        self._update_equity(row)
        
        # 4. Переходим к следующему шагу
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info.update({
            "equity": self.equity,
            "position_side": self.position_side,
            "trade_count": len(self.trades),
        })
        
        return self._get_observation(), reward, done, info
    
    def _can_open_trade(self, row: pd.Series, side: str = None) -> bool:
        """Проверяет, можно ли открыть сделку."""
        # Cooldown
        if self.current_step - self.last_trade_step < self.min_bars_between_trades:
            return False
        
        # ADX filter - только если min_adx > 0
        if self.min_adx is not None and self.min_adx > 0:
            adx = row.get("adx")
            if adx is not None and np.isfinite(adx) and adx < self.min_adx:
                return False
        
        # ATR% filter - только если min_atr_pct > 0
        if self.min_atr_pct is not None and self.min_atr_pct > 0:
            atr_pct = row.get("atr_pct")
            if atr_pct is not None and np.isfinite(atr_pct) and atr_pct < self.min_atr_pct:
                return False
        
        # Фильтры отключены для свободного обучения
        
        return True
    
    def _get_trend_direction(self, row: pd.Series) -> int:
        """
        Определяет направление тренда.
        Returns: 1 = uptrend, -1 = downtrend, 0 = neutral (разрешаем оба направления)
        """
        # Пробуем использовать EMA 1h
        ema_fast = row.get("ema_fast_1h")
        ema_slow = row.get("ema_slow_1h")
        
        if ema_fast is not None and ema_slow is not None:
            if np.isfinite(ema_fast) and np.isfinite(ema_slow):
                # Ослаблен буфер: 0.05% вместо 0.1%
                if ema_fast > ema_slow * 1.0005:
                    return 1
                elif ema_fast < ema_slow * 0.9995:
                    return -1
                # Neutral - разрешаем оба направления
                return 0
        
        # Fallback на 15m EMA
        ema_12 = row.get("ema_12")
        ema_26 = row.get("ema_26")
        
        if ema_12 is not None and ema_26 is not None:
            if np.isfinite(ema_12) and np.isfinite(ema_26):
                if ema_12 > ema_26 * 1.0005:
                    return 1
                elif ema_12 < ema_26 * 0.9995:
                    return -1
        
        return 0  # Neutral - разрешаем оба направления
    
    def _open_position(self, side: str, row: pd.Series, risk_manager_callback: Optional[Callable] = None):
        """Открывает позицию."""
        # Цена входа
        if side == PositionSide.LONG:
            entry_price = row["open"] * (1 + self.slippage_bps)
        else:
            entry_price = row["open"] * (1 - self.slippage_bps)
        
        # Риск
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        
        # TP/SL
        if risk_manager_callback:
            result = risk_manager_callback(
                side, entry_price, row,
                self.df.iloc[:self.current_step + 1] if self.current_step < len(self.df) else self.df
            )
            if isinstance(result, tuple) and len(result) >= 3:
                sl_price, tp_price, rr = result[0], result[1], result[2]
            else:
                sl_price, tp_price, rr = self._fallback_tp_sl(side, entry_price, row)
        else:
            sl_price, tp_price, rr = self._fallback_tp_sl(side, entry_price, row)
        
        # Размер позиции
        if side == PositionSide.LONG:
            risk_per_unit = abs(entry_price - sl_price)
        else:
            risk_per_unit = abs(sl_price - entry_price)
        
        if risk_per_unit <= 0:
            return
        
        # Учитываем комиссию при расчёте размера позиции
        # Для LONG: cash = entry_price * size * (1 + commission_rate)
        if side == PositionSide.LONG:
            # Максимальный размер с учётом комиссии
            max_size_by_cash = self.cash / (entry_price * (1 + self.commission_rate))
            # Размер по риску
            position_size_by_risk = risk_amount / risk_per_unit
            # Ограничение по leverage
            max_notional = self.equity * self.max_leverage
            max_size_by_leverage = max_notional / entry_price if entry_price > 0 else 0
            # Берём минимум
            position_size = min(position_size_by_risk, max_size_by_cash)
            if max_size_by_leverage > 0:
                position_size = min(position_size, max_size_by_leverage)
        else:
            # Для SHORT проще - комиссия вычитается из выручки
            position_size = risk_amount / risk_per_unit
            max_notional = self.equity * self.max_leverage
            max_size = max_notional / entry_price if entry_price > 0 else 0
            if max_size > 0:
                position_size = min(position_size, max_size)
        
        if position_size <= 0:
            return
        
        # Комиссия на вход
        commission = entry_price * position_size * self.commission_rate
        
        # Обновляем cash
        if side == PositionSide.LONG:
            total_cost = entry_price * position_size + commission
            if total_cost > self.cash:
                return
            self.cash -= total_cost
        else:
            self.cash += entry_price * position_size - commission
        
        # Сохраняем позицию
        self.position_side = side
        self.entry_price = entry_price
        self.position_size = position_size
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.entry_step = self.current_step
        self.last_trade_step = self.current_step
        
        # Счетчики для баланса
        if side == PositionSide.LONG:
            self._long_count += 1
        else:
            self._short_count += 1
    
    def _close_position(self, row: pd.Series, reason: str) -> float:
        """Закрывает позицию. Возвращает PnL."""
        if self.position_side == PositionSide.NONE:
            return 0.0
        
        # Определяем цену выхода в зависимости от причины
        if reason == "stop_loss":
            exit_price = self.stop_loss
        elif reason == "take_profit":
            exit_price = self.take_profit
        else:
            # end_of_test или другое
            if self.position_side == PositionSide.LONG:
                exit_price = row["close"] * (1 - self.slippage_bps)
            else:
                exit_price = row["close"] * (1 + self.slippage_bps)
        
        # PnL
        if self.position_side == PositionSide.LONG:
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * self.position_size
        
        # Комиссия на выход
        commission = exit_price * self.position_size * self.commission_rate
        
        # Обновляем cash
        if self.position_side == PositionSide.LONG:
            self.cash += exit_price * self.position_size
        else:
            self.cash -= exit_price * self.position_size
        self.cash -= commission
        
        # Итоговый PnL с учетом комиссий (вход + выход)
        total_commission = commission * 2
        net_pnl = pnl - total_commission
        
        # Записываем сделку
        self.trades.append({
            "entry_step": self.entry_step,
            "exit_step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "side": self.position_side,
            "size": self.position_size,
            "pnl": pnl,
            "commission": total_commission,
            "reason": reason,
            "net_pnl": net_pnl,
        })
        
        # Сброс
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None
        
        return net_pnl
    
    def close_open_position(self, row: pd.Series, reason: str = "end_of_test") -> float:
        """Публичный метод для принудительного закрытия."""
        return self._close_position(row, reason)
    
    def _check_exits(self, row: pd.Series) -> Optional[str]:
        """Проверяет срабатывание TP/SL."""
        if self.position_side == PositionSide.NONE:
            return None
        
        high = row["high"]
        low = row["low"]
        
        # Проверяем что SL/TP установлены
        if self.stop_loss is None or self.take_profit is None:
            return None
        
        if self.position_side == PositionSide.LONG:
            # Проверяем SL первым (консервативно)
            if low <= self.stop_loss:
                return "stop_loss"
            if high >= self.take_profit:
                return "take_profit"
        else:
            if high >= self.stop_loss:
                return "stop_loss"
            if low <= self.take_profit:
                return "take_profit"
        
        return None
    
    def _update_equity(self, row: pd.Series):
        """Обновляет equity."""
        if self.position_side == PositionSide.NONE:
            self.equity = self.cash
        else:
            current_price = row["close"]
            if self.position_side == PositionSide.LONG:
                self.equity = self.cash + (current_price * self.position_size)
            else:
                self.equity = self.cash - (current_price * self.position_size)
        
        if self.equity > self.max_equity:
            self.max_equity = self.equity
    
    def _get_observation(self) -> np.ndarray:
        """Возвращает наблюдение."""
        if self.current_step >= len(self.df):
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.current_step]
        
        # Фичи
        features = []
        for col in self.feature_cols:
            val = row.get(col, 0.0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0.0
            features.append(float(val))
        
        # Позиция (0=нет, 1=LONG, -1=SHORT)
        if self.position_side == PositionSide.LONG:
            position_feature = 1.0
        elif self.position_side == PositionSide.SHORT:
            position_feature = -1.0
        else:
            position_feature = 0.0
        
        # Unrealized PnL %
        if self.position_side != PositionSide.NONE and self.entry_price:
            current_price = row["close"]
            if self.position_side == PositionSide.LONG:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0
        
        # Время с последней сделки (нормализованное)
        bars_since_trade = (self.current_step - self.last_trade_step) / self.min_bars_between_trades
        bars_since_trade = min(bars_since_trade, 2.0)  # Clip
        
        # Можно ли открыть сделку
        can_trade = 1.0 if self.position_side == PositionSide.NONE and bars_since_trade >= 1.0 else 0.0
        
        obs = np.array(
            features + [position_feature, unrealized_pnl_pct, bars_since_trade, can_trade],
            dtype=np.float32
        )
        
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _fallback_tp_sl(self, side: str, entry_price: float, row: pd.Series) -> Tuple[float, float, float]:
        """Fallback TP/SL."""
        atr = row.get("atr", entry_price * 0.01)
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.01
        
        # SL = 1 ATR, TP = 3 ATR (RR=3)
        if side == PositionSide.LONG:
            sl_price = entry_price - atr
            tp_price = entry_price + atr * 3.0
        else:
            sl_price = entry_price + atr
            tp_price = entry_price - atr * 3.0
        
        return sl_price, tp_price, 3.0
    
    def get_state_size(self) -> int:
        """Размер state."""
        return len(self.feature_cols) + 4  # features + position + pnl + bars_since + can_trade
    
    def get_action_size(self) -> int:
        """Размер action space."""
        if self.allowed_side in ["long", "short"]:
            return 2  # HOLD, OPEN_LONG или HOLD, OPEN_SHORT
        return 3  # HOLD, OPEN_LONG, OPEN_SHORT
