"""
Торговое окружение для PPO обучения.
Оффлайн среда на 15m свечах с реалистичной симуляцией исполнения.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Callable
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Действия агента."""
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE = 3


class PositionSide:
    """Сторона позиции."""
    NONE = None
    LONG = "LONG"
    SHORT = "SHORT"


class TradingEnv:
    """
    Торговое окружение для обучения PPO.
    
    Особенности:
    - Работает на 15m свечах
    - Реалистичное исполнение: TP/SL проверяются внутри свечи
    - Комиссии и слиппедж
    - Mark-to-market equity tracking
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% на вход+выход
        slippage_bps: float = 5.0,  # 5 bps слиппедж
        risk_per_trade_pct: float = 1.0,  # 1% капитала на сделку
        max_leverage: float = 1.0,  # Максимальное плечо (1.0 = без плеча)
        reward_churn_penalty: float = 0.01,  # Штраф за частые сделки
        reward_dd_penalty: float = 0.1,  # Штраф за просадку
        min_bars_between_trades: int = 4,  # Минимум баров между сделками
        min_hold_bars: int = 2,  # Минимум баров удержания позиции
        reward_trade_penalty: float = 0.0,  # Фикс. штраф за открытие сделки (в долях капитала)
        min_adx: Optional[float] = None,  # Минимальный ADX для входа
        min_atr_pct: Optional[float] = None,  # Минимальный ATR% для входа
        reward_trend_bonus: float = 0.0,  # Бонус за движение по тренду
        reward_tp_progress: float = 0.0,  # Бонус/штраф за прогресс к TP
        reward_one_sided_penalty: float = 0.0,  # Штраф за одностороннюю торговлю
        action_space: str = "hold",  # hold|no_hold|no_close
    ):
        """
        Args:
            df: DataFrame с OHLCV и фичами (должен иметь DatetimeIndex)
            initial_capital: Начальный капитал
            commission_rate: Комиссия на сделку (0.001 = 0.1%)
            slippage_bps: Слиппедж в базисных пунктах
            risk_per_trade_pct: Риск на сделку в % от капитала
            reward_churn_penalty: Коэффициент штрафа за частые сделки
            reward_dd_penalty: Коэффициент штрафа за просадку
            min_bars_between_trades: Минимум баров между противоположными сделками
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000.0  # Конвертируем в долю
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.reward_churn_penalty = reward_churn_penalty
        self.reward_dd_penalty = reward_dd_penalty
        self.min_bars_between_trades = min_bars_between_trades
        self.min_hold_bars = min_hold_bars
        self.reward_trade_penalty = reward_trade_penalty
        self.min_adx = min_adx
        self.min_atr_pct = min_atr_pct
        self.reward_trend_bonus = reward_trend_bonus
        self.reward_tp_progress = reward_tp_progress
        self.reward_one_sided_penalty = reward_one_sided_penalty
        self.action_space = action_space
        
        # Проверяем необходимые колонки
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Состояние среды
        self.current_step = 0
        self.max_steps = len(self.df) - 1  # -1 чтобы не выходить за границы
        
        # Торговое состояние
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0  # В единицах актива
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None
        
        # Капитал и метрики
        self.equity = initial_capital
        self.cash = initial_capital
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        self.last_dd = 0.0
        
        # История для reward
        self.trade_count = 0
        self.last_trade_step = -min_bars_between_trades - 1
        self._trade_opened_this_step = False
        self._prev_tp_progress = 0.0
        self._long_trades = 0
        self._short_trades = 0
        self._last_close_reason = None  # Для отслеживания причины закрытия
        
        # История сделок
        self.trades = []
        
        # Получаем feature columns (все кроме OHLCV и timestamp)
        exclude_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        if len(self.feature_cols) == 0:
            logger.warning("No feature columns found! Using price-based features only.")
            # Создаем минимальные фичи из цены
            self.df["price_change"] = self.df["close"].pct_change().fillna(0)
            self.feature_cols = ["price_change"]
    
    def reset(self, start_step: Optional[int] = None) -> np.ndarray:
        """
        Сброс среды.
        
        Args:
            start_step: Начальный шаг (None = случайный в допустимом диапазоне)
        
        Returns:
            Начальное наблюдение (state)
        """
        if start_step is None:
            # Начинаем с шага, где достаточно данных для индикаторов
            min_start = 200  # Минимум для расчета индикаторов
            self.current_step = np.random.randint(min_start, self.max_steps - 100)
        else:
            self.current_step = max(0, min(start_step, self.max_steps - 1))
        
        # Сброс торгового состояния
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None
        
        # Сброс капитала
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.last_dd = 0.0
        
        # Сброс метрик
        self.trade_count = 0
        self.last_trade_step = self.current_step - self.min_bars_between_trades - 1
        self.trades = []
        self._prev_tp_progress = 0.0
        self._long_trades = 0
        self._short_trades = 0
        
        return self._get_observation()
    
    def step(self, action: int, risk_manager_callback=None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Выполняет один шаг среды.
        
        Args:
            action: Действие агента (0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3=CLOSE)
            risk_manager_callback: Функция для расчета TP/SL: (side, entry_price, row) -> (sl_price, tp_price, rr)
        
        Returns:
            (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, {"reason": "max_steps"}
        
        row = self.df.iloc[self.current_step]
        prev_equity = self.equity
        
        # Сбрасываем флаг сделки на шаге
        self._trade_opened_this_step = False

        # 1. Проверяем срабатывание TP/SL на текущей свече (если есть позиция)
        if self.position_side != PositionSide.NONE:
            exit_reason = self._check_exits(row)
            if exit_reason:
                # Позиция закрыта по TP/SL
                self._close_position(row, exit_reason)
        
        # 2. Применяем действие агента
        info = {}
        mapped_action = self._map_action(action)

        if mapped_action == Action.OPEN_LONG:
            if self.position_side == PositionSide.NONE:
                # Проверяем cooldown
                if self.current_step - self.last_trade_step < self.min_bars_between_trades:
                    info["action_rejected"] = "cooldown"
                else:
                    self._open_position(PositionSide.LONG, row, risk_manager_callback)
            elif self.position_side == PositionSide.SHORT:
                # Закрываем SHORT и открываем LONG
                if self._can_close_position():
                    self._close_position(row, "flip_to_long")
                    if self.current_step - self.last_trade_step >= self.min_bars_between_trades:
                        self._open_position(PositionSide.LONG, row, risk_manager_callback)
                    else:
                        info["action_rejected"] = "cooldown_after_flip"
                else:
                    info["action_rejected"] = "min_hold_bars"
        
        elif mapped_action == Action.OPEN_SHORT:
            if self.position_side == PositionSide.NONE:
                if self.current_step - self.last_trade_step < self.min_bars_between_trades:
                    info["action_rejected"] = "cooldown"
                else:
                    self._open_position(PositionSide.SHORT, row, risk_manager_callback)
            elif self.position_side == PositionSide.LONG:
                # Закрываем LONG и открываем SHORT
                if self._can_close_position():
                    self._close_position(row, "flip_to_short")
                    if self.current_step - self.last_trade_step >= self.min_bars_between_trades:
                        self._open_position(PositionSide.SHORT, row, risk_manager_callback)
                    else:
                        info["action_rejected"] = "cooldown_after_flip"
                else:
                    info["action_rejected"] = "min_hold_bars"
        
        elif mapped_action == Action.CLOSE:
            if self.position_side != PositionSide.NONE:
                if self._can_close_position():
                    self._close_position(row, "manual_close")
                else:
                    info["action_rejected"] = "min_hold_bars"
        
        # 3. Обновляем equity (mark-to-market)
        self._update_equity(row)
        
        # 4. Вычисляем reward
        reward = self._calculate_reward(prev_equity, row)
        
        # 5. Переходим к следующему шагу
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 6. Собираем info
        info.update({
            "equity": self.equity,
            "position_side": self.position_side,
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown,
        })
        
        return self._get_observation(), reward, done, info
    
    def _open_position(self, side: str, row: pd.Series, risk_manager_callback: Optional[Callable]=None):
        """Открывает позицию."""
        # Фильтры по рыночным условиям (опционально)
        if self.min_adx is not None:
            adx_val = row.get("adx")
            if adx_val is None or not np.isfinite(adx_val) or adx_val < self.min_adx:
                return
        if self.min_atr_pct is not None:
            atr_pct_val = row.get("atr_pct")
            if atr_pct_val is None or not np.isfinite(atr_pct_val) or atr_pct_val < self.min_atr_pct:
                return

        # Цена входа с учетом слиппеджа
        if side == PositionSide.LONG:
            entry_price = row["open"] * (1 + self.slippage_bps)
        else:  # SHORT
            entry_price = row["open"] * (1 - self.slippage_bps)
        
        # Рассчитываем размер позиции на основе риска
        # Риск = risk_per_trade_pct% от капитала
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        
        # Получаем TP/SL от risk manager
        if risk_manager_callback:
            # risk_manager_callback должна возвращать (sl_price, tp_price, rr, metadata)
            result = risk_manager_callback(side, entry_price, row, self.df.iloc[:self.current_step+1] if self.current_step < len(self.df) else self.df)
            if isinstance(result, tuple) and len(result) >= 3:
                sl_price, tp_price, rr = result[0], result[1], result[2]
            else:
                # Fallback если формат неверный
                sl_price, tp_price, rr = self._fallback_tp_sl(side, entry_price, row)
        else:
            sl_price, tp_price, rr = self._fallback_tp_sl(side, entry_price, row)
        
        # Размер позиции: risk_amount / (entry - sl) для LONG, (sl - entry) для SHORT
        if side == PositionSide.LONG:
            risk_per_unit = abs(entry_price - sl_price)
        else:
            risk_per_unit = abs(sl_price - entry_price)
        
        if risk_per_unit <= 0:
            return

        position_size = risk_amount / risk_per_unit

        # Ограничиваем размер позиции по максимальному плечу
        max_notional = self.equity * self.max_leverage
        max_size = max_notional / entry_price if entry_price > 0 else 0.0
        if max_size > 0:
            position_size = min(position_size, max_size)
        else:
            return
        
        # Комиссия на вход
        commission = entry_price * position_size * self.commission_rate
        if commission > self.cash:
            # Недостаточно средств на комиссию
            return

        # Обновляем cash в зависимости от стороны
        if side == PositionSide.LONG:
            total_cost = entry_price * position_size + commission
            if total_cost > self.cash:
                return
            self.cash -= total_cost
        else:  # SHORT: получаем proceeds от продажи
            self.cash += entry_price * position_size
            self.cash -= commission
        
        # Сохраняем состояние позиции
        self.position_side = side
        self.entry_price = entry_price
        self.position_size = position_size
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.entry_step = self.current_step
        self.last_trade_step = self.current_step
        self.trade_count += 1
        self._trade_opened_this_step = True
        if side == PositionSide.LONG:
            self._long_trades += 1
        elif side == PositionSide.SHORT:
            self._short_trades += 1
    
    def _close_position(self, row: pd.Series, reason: str):
        """Закрывает текущую позицию."""
        if self.position_side == PositionSide.NONE:
            return
        
        # Цена выхода с учетом слиппеджа
        if self.position_side == PositionSide.LONG:
            exit_price = row["open"] * (1 - self.slippage_bps)  # Продаем, слиппедж против нас
        else:  # SHORT
            exit_price = row["open"] * (1 + self.slippage_bps)  # Покупаем обратно
        
        # PnL
        if self.position_side == PositionSide.LONG:
            pnl = (exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - exit_price) * self.position_size
        
        # Комиссия на выход
        commission = exit_price * self.position_size * self.commission_rate
        
        # Обновляем cash
        if self.position_side == PositionSide.LONG:
            self.cash += exit_price * self.position_size  # Продаем
        else:  # SHORT
            self.cash -= exit_price * self.position_size  # Покупаем обратно
        
        self.cash -= commission
        
        # Записываем сделку
        trade = {
            "entry_step": self.entry_step,
            "exit_step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "side": self.position_side,
            "size": self.position_size,
            "pnl": pnl,
            "commission": commission * 2,  # Вход + выход
            "reason": reason,
        }
        self.trades.append(trade)
        
        # Сохраняем причину закрытия для reward
        self._last_close_reason = reason
        
        # Сброс позиции
        self.position_side = PositionSide.NONE
        self.entry_price = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.entry_step = None

    def close_open_position(self, row: pd.Series, reason: str = "end_of_test"):
        """Публичный метод для принудительного закрытия позиции."""
        self._close_position(row, reason)
    
    def _check_exits(self, row: pd.Series) -> Optional[str]:
        """
        Проверяет срабатывание TP/SL внутри свечи.
        Возвращает причину выхода или None.
        """
        if self.position_side == PositionSide.NONE:
            return None
        
        high = row["high"]
        low = row["low"]
        
        if self.position_side == PositionSide.LONG:
            # LONG: SL ниже, TP выше
            if low <= self.stop_loss:
                return "stop_loss"
            if high >= self.take_profit:
                return "take_profit"
        else:  # SHORT
            # SHORT: SL выше, TP ниже
            if high >= self.stop_loss:
                return "stop_loss"
            if low <= self.take_profit:
                return "take_profit"
        
        return None
    
    def _update_equity(self, row: pd.Series):
        """Обновляет equity (mark-to-market)."""
        if self.position_side == PositionSide.NONE:
            self.equity = self.cash
        else:
            # Текущая цена позиции
            current_price = row["close"]

            if self.position_side == PositionSide.LONG:
                # LONG: cash отражает затраты на покупку
                self.equity = self.cash + (current_price * self.position_size)
            else:  # SHORT
                # SHORT: cash включает proceeds от продажи, вычитаем текущую стоимость выкупа
                self.equity = self.cash - (current_price * self.position_size)
        
        # Обновляем max equity и drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_dd = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def _calculate_reward(self, prev_equity: float, row: pd.Series) -> float:
        """
        Вычисляет reward для текущего шага.
        
        Reward = Δequity - штрафы за churn и drawdown.
        """
        # Базовый reward: изменение equity
        equity_change = self.equity - prev_equity
        reward = equity_change
        
        # Штраф за частые сделки (churn)
        if self.current_step - self.last_trade_step < self.min_bars_between_trades:
            reward -= self.reward_churn_penalty * abs(equity_change)
        
        # Штраф за увеличение просадки
        current_dd = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        dd_increase = max(0, current_dd - self.last_dd)
        reward -= self.reward_dd_penalty * dd_increase * self.equity
        self.last_dd = current_dd

        # Фиксированный штраф за открытие сделки
        if self._trade_opened_this_step and self.reward_trade_penalty > 0:
            reward -= self.reward_trade_penalty * self.initial_capital

        # Бонус за движение по тренду (если позиция открыта и прибыль положительная)
        if self.reward_trend_bonus > 0 and self.position_side != PositionSide.NONE:
            trend_dir = self._get_trend_direction(row)
            if equity_change > 0:
                if self.position_side == PositionSide.LONG and trend_dir == 1:
                    reward += self.reward_trend_bonus * abs(equity_change)
                elif self.position_side == PositionSide.SHORT and trend_dir == -1:
                    reward += self.reward_trend_bonus * abs(equity_change)

        # Бонус/штраф за прогресс к TP (если позиция открыта)
        if self.reward_tp_progress > 0 and self.position_side != PositionSide.NONE:
            tp_progress = self._get_tp_progress(row)
            delta_progress = tp_progress - self._prev_tp_progress
            # Бонус за изменение прогресса (движение к TP)
            reward += self.reward_tp_progress * delta_progress
            # ДОПОЛНИТЕЛЬНЫЙ бонус за абсолютный прогресс (чем ближе к TP, тем больше бонус)
            # Это мотивирует агента удерживать позицию, когда она уже близка к TP
            if tp_progress > 0.7:  # Если позиция прошла больше 70% пути к TP
                reward += 0.2 * tp_progress * self.initial_capital / 1000  # УСИЛЕННЫЙ бонус за близость к TP
            elif tp_progress > 0.5:  # Если позиция прошла больше 50% пути к TP
                reward += 0.15 * tp_progress * self.initial_capital / 1000  # Бонус за близость к TP
            self._prev_tp_progress = tp_progress
        
        # ДОПОЛНИТЕЛЬНЫЙ бонус за удержание прибыльной позиции (критично для достижения TP!)
        if self.position_side != PositionSide.NONE and equity_change > 0:
            # Чем больше прибыль, тем больше бонус за удержание
            # Это мотивирует агента держать позицию до TP вместо преждевременного закрытия
            if self.entry_price is not None:
                current_price = row["close"]
                if self.position_side == PositionSide.LONG:
                    unrealized_pnl = (current_price - self.entry_price) * self.position_size
                else:  # SHORT
                    unrealized_pnl = (self.entry_price - current_price) * self.position_size
                
                if unrealized_pnl > 0:
                    pnl_pct = unrealized_pnl / self.initial_capital
                    # Проверяем прогресс к TP
                    tp_progress = self._get_tp_progress(row) if self.take_profit is not None else 0.0
                    
                    # Бонус увеличивается с размером прибыли И с прогрессом к TP
                    # Если позиция уже близка к TP (>70%), бонус еще больше
                    if tp_progress > 0.8:  # Если позиция прошла больше 80% пути к TP
                        progress_multiplier = 1.0 + (tp_progress * 1.0)  # ОЧЕНЬ большой множитель
                    elif tp_progress > 0.7:  # Если позиция прошла больше 70% пути к TP
                        progress_multiplier = 1.0 + (tp_progress * 0.75)  # Большой множитель
                    elif tp_progress > 0.5:  # Если позиция прошла больше 50% пути к TP
                        progress_multiplier = 1.0 + (tp_progress * 0.5)  # Средний множитель
                    else:
                        progress_multiplier = 1.0
                    
                    if pnl_pct > 0.02:  # Если прибыль > 2% капитала (ОЧЕНЬ крупная прибыль)
                        # КРИТИЧЕСКИЙ бонус для очень крупных прибылей - они ДОЛЖНЫ доходить до TP!
                        base_bonus = 0.8 * equity_change  # ОЧЕНЬ большой базовый бонус
                        # Дополнительный бонус за близость к TP
                        if tp_progress > 0.8:
                            reward += base_bonus * 2.0  # МАКСИМАЛЬНЫЙ бонус когда близко к TP
                        elif tp_progress > 0.5:
                            reward += base_bonus * 1.5  # Большой бонус
                        else:
                            reward += base_bonus  # Базовый бонус
                    elif pnl_pct > 0.01:  # Если прибыль > 1% капитала
                        reward += 0.5 * equity_change * progress_multiplier  # ОЧЕНЬ большой бонус за удержание крупной прибыли
                    elif pnl_pct > 0.005:  # Если прибыль > 0.5% капитала
                        reward += 0.4 * equity_change * progress_multiplier  # Большой бонус
                    else:
                        reward += 0.25 * equity_change * progress_multiplier  # Средний бонус

        # Штраф за одностороннюю торговлю (дисбаланс LONG/SHORT)
        if self.reward_one_sided_penalty > 0 and (self._long_trades + self._short_trades) >= 3:
            total = self._long_trades + self._short_trades
            imbalance = abs(self._long_trades - self._short_trades) / total if total > 0 else 0
            # Усиленный штраф: квадрат дисбаланса для более сильного эффекта
            # Умножаем на equity для масштабирования
            reward -= self.reward_one_sided_penalty * (imbalance ** 2) * self.initial_capital
        
        # Штраф за преждевременное закрытие (manual_close/flip вместо TP/SL)
        if self._last_close_reason in ["manual_close", "flip_to_long", "flip_to_short"] and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade["reason"] in ["manual_close", "flip_to_long", "flip_to_short"]:
                # Штраф пропорционален тому, насколько далеко была позиция от TP
                # Если позиция была в прибыли, но закрыта до TP - КРИТИЧЕСКИЙ штраф!
                # Если позиция была в убытке - меньший штраф (не мешаем закрывать убытки)
                if last_trade["pnl"] < 0:
                    reward -= 0.1 * abs(last_trade["pnl"])  # Умеренный штраф за закрытие убыточной позиции вручную
                # Если позиция была в прибыли, но не достигла TP - КРИТИЧЕСКИЙ штраф (это главная проблема!)
                elif last_trade["pnl"] > 0:
                    # Штраф увеличивается с размером прибыли - чем больше прибыль, тем больше штраф
                    # Это критично, так как большие прибыли должны доходить до TP
                    pnl_pct = abs(last_trade["pnl"]) / self.initial_capital
                    if pnl_pct > 0.02:  # Если прибыль > 2% капитала (ОЧЕНЬ крупная прибыль)
                        # КРИТИЧЕСКИЙ штраф - такие прибыли ДОЛЖНЫ доходить до TP!
                        reward -= 2.0 * abs(last_trade["pnl"])  # ОЧЕНЬ КРИТИЧЕСКИЙ штраф для очень крупных прибылей
                    elif pnl_pct > 0.01:  # Если прибыль > 1% капитала
                        reward -= 1.0 * abs(last_trade["pnl"])  # КРИТИЧЕСКИЙ штраф для крупных прибылей
                    elif pnl_pct > 0.005:  # Если прибыль > 0.5% капитала
                        reward -= 0.7 * abs(last_trade["pnl"])  # Большой штраф для средних прибылей
                    else:
                        reward -= 0.4 * abs(last_trade["pnl"])  # Увеличенный штраф для небольших прибылей
        
        # Бонус за достижение TP
        if self._last_close_reason == "take_profit" and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade["reason"] == "take_profit" and last_trade["pnl"] > 0:
                # Бонус увеличивается с размером прибыли - чем больше прибыль, тем больше бонус
                pnl_pct = abs(last_trade["pnl"]) / self.initial_capital
                if pnl_pct > 0.02:  # Если прибыль > 2% капитала (ОЧЕНЬ крупная прибыль)
                    # МАКСИМАЛЬНЫЙ бонус - такие прибыли критически важны для PnL!
                    reward += 2.0 * abs(last_trade["pnl"])  # ОЧЕНЬ МАКСИМАЛЬНЫЙ бонус за очень крупные TP
                elif pnl_pct > 0.01:  # Если прибыль > 1% капитала
                    reward += 1.0 * abs(last_trade["pnl"])  # МАКСИМАЛЬНЫЙ бонус за крупные TP
                elif pnl_pct > 0.005:  # Если прибыль > 0.5% капитала
                    reward += 0.7 * abs(last_trade["pnl"])  # Большой бонус за средние TP
                else:
                    reward += 0.6 * abs(last_trade["pnl"])  # Бонус за небольшие TP
        
        # Сбрасываем причину закрытия после обработки
        if self._last_close_reason:
            self._last_close_reason = None
        
        # Нормализуем reward (делим на начальный капитал для стабильности)
        reward = reward / self.initial_capital
        
        return reward

    def _can_close_position(self) -> bool:
        """Проверяет, можно ли закрывать позицию по min_hold_bars."""
        if self.entry_step is None:
            return True
        return (self.current_step - self.entry_step) >= self.min_hold_bars

    def _map_action(self, action: int) -> int:
        """Маппинг действий для альтернативного action space."""
        if self.action_space == "no_hold":
            # 0: LONG, 1: SHORT, 2: CLOSE
            if action == 0:
                return Action.OPEN_LONG
            if action == 1:
                return Action.OPEN_SHORT
            return Action.CLOSE
        if self.action_space == "no_close":
            # 0: HOLD, 1: LONG, 2: SHORT
            if action == 1:
                return Action.OPEN_LONG
            if action == 2:
                return Action.OPEN_SHORT
            return Action.HOLD
        # default: hold space
        return action

    def _get_trend_direction(self, row: pd.Series) -> int:
        """
        Возвращает направление тренда:
        1 = uptrend, -1 = downtrend, 0 = neutral.
        Предпочтение: ema_fast_1h/ema_slow_1h, fallback на ema_12/ema_26.
        """
        ema_fast_1h = row.get("ema_fast_1h")
        ema_slow_1h = row.get("ema_slow_1h")
        if np.isfinite(ema_fast_1h) and np.isfinite(ema_slow_1h):
            if ema_fast_1h > ema_slow_1h:
                return 1
            if ema_fast_1h < ema_slow_1h:
                return -1
            return 0

        ema12 = row.get("ema_12")
        ema26 = row.get("ema_26")
        if np.isfinite(ema12) and np.isfinite(ema26):
            if ema12 > ema26:
                return 1
            if ema12 < ema26:
                return -1
        return 0

    def _get_tp_progress(self, row: pd.Series) -> float:
        """Возвращает прогресс к TP в диапазоне [0, 1]."""
        if self.entry_price is None or self.take_profit is None:
            return 0.0
        current_price = row.get("close")
        if current_price is None or not np.isfinite(current_price):
            return self._prev_tp_progress
        if self.position_side == PositionSide.LONG:
            denom = self.take_profit - self.entry_price
            if denom <= 0:
                return 0.0
            progress = (current_price - self.entry_price) / denom
        else:
            denom = self.entry_price - self.take_profit
            if denom <= 0:
                return 0.0
            progress = (self.entry_price - current_price) / denom
        return float(np.clip(progress, 0.0, 1.0))
    
    def _get_observation(self) -> np.ndarray:
        """Возвращает текущее наблюдение (state)."""
        if self.current_step >= len(self.df):
            # Возвращаем последнее наблюдение
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.current_step]
        
        # Извлекаем фичи
        features = []
        for col in self.feature_cols:
            val = row.get(col, 0.0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0.0
            features.append(float(val))
        
        # Добавляем информацию о позиции (нормализованную)
        if self.position_side == PositionSide.LONG:
            position_feature = 1.0
        elif self.position_side == PositionSide.SHORT:
            position_feature = -1.0
        else:
            position_feature = 0.0
        
        # Добавляем unrealized PnL в % от капитала
        if self.position_side != PositionSide.NONE and self.entry_price:
            current_price = row["close"]
            if self.position_side == PositionSide.LONG:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0
        
        # Добавляем equity в % от начального капитала
        equity_pct = self.equity / self.initial_capital
        
        # Объединяем все фичи
        obs = np.array(features + [position_feature, unrealized_pnl_pct, equity_pct], dtype=np.float32)
        
        # Проверяем на NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def _fallback_tp_sl(self, side: str, entry_price: float, row: pd.Series) -> Tuple[float, float, float]:
        """Fallback расчет TP/SL если risk manager недоступен."""
        atr = row.get("atr", entry_price * 0.01)  # 1% если ATR нет
        if side == PositionSide.LONG:
            sl_price = entry_price * 0.99  # 1% ниже
            tp_price = entry_price * 1.02  # 2% выше
        else:
            sl_price = entry_price * 1.01  # 1% выше
            tp_price = entry_price * 0.98  # 2% ниже
        rr = 2.0
        return sl_price, tp_price, rr
    
    def get_state_size(self) -> int:
        """Возвращает размер state."""
        return len(self.feature_cols) + 3  # features + position + unrealized_pnl + equity_pct
    
    def get_action_size(self) -> int:
        """Возвращает размер action space."""
        if self.action_space == "no_hold":
            return 3
        if self.action_space == "no_close":
            return 3
        return len(Action)
