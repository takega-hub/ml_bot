"""
Бэктест ML стратегии с ТОЧНОЙ имитацией работы сервера.

ВАЖНО: Этот бэктест НЕ исправляет ошибки стратегии!
Он показывает КАК стратегия работает на самом деле.
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

warnings.filterwarnings('ignore')

# Настройка графиков
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-v0_8-darkgrid')

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, ApiSettings
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal, Bias


class ExitReason(Enum):
    """Причины закрытия позиции."""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TIME_LIMIT = "TIME_LIMIT"
    OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"
    MARGIN_CALL = "MARGIN_CALL"
    TRAILING_STOP = "TRAILING_STOP"
    END_OF_BACKTEST = "END_OF_BACKTEST"


@dataclass
class Trade:
    """Сделка в бэктесте."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    action: Action
    size_usd: float
    pnl: float
    pnl_pct: float
    entry_reason: str
    exit_reason: ExitReason
    symbol: str
    confidence: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    entry_volatility: float = 0.0
    exit_volatility: float = 0.0
    signal_tp_pct: Optional[float] = None  # TP% из сигнала
    signal_sl_pct: Optional[float] = None  # SL% из сигнала


@dataclass
class BacktestMetrics:
    """Метрики бэктеста."""
    symbol: str
    model_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_signals: int
    long_signals: int
    short_signals: int
    avg_trade_duration_hours: float
    best_trade_pnl: float
    worst_trade_pnl: float
    consecutive_wins: int
    consecutive_losses: int
    largest_win: float
    largest_loss: float
    avg_confidence: float
    avg_mfe: float
    avg_mae: float
    mfe_mae_ratio: float
    var_95: float
    cvar_95: float
    recovery_factor: float
    expectancy_usd: float
    risk_reward_ratio: float
    trade_frequency_per_day: float
    profitable_days_pct: float
    ulcer_index: float
    kelly_criterion: float
    avg_tp_distance_pct: float = 0.0
    avg_sl_distance_pct: float = 0.0
    avg_rr_ratio: float = 0.0
    signal_quality_score: float = 0.0
    signals_with_tp_sl_pct: float = 100.0  # % сигналов с TP/SL
    signals_with_correct_sl_pct: float = 100.0  # % сигналов с SL=1%
    avg_position_size_usd: float = 0.0


@dataclass
class SignalStats:
    """Статистика сигналов стратегии."""
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    hold_signals: int = 0
    signals_with_tp_sl: int = 0
    signals_without_tp_sl: int = 0
    signals_with_correct_sl: int = 0
    signals_with_wrong_sl: int = 0
    avg_confidence: float = 0.0
    sl_distances: List[float] = field(default_factory=list)
    tp_distances: List[float] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)


class MLBacktestSimulator:
    """
    Симулятор для бэктеста, который ТОЧНО имитирует работу реального бота.
    
    ВАЖНО: Не исправляет ошибки стратегии, только показывает как она работает!
    """
    
    def __init__(
        self,
        initial_balance: float = 100.0,
        risk_per_trade: float = 0.02,
        commission: float = 0.0006,
        max_position_size_pct: float = 0.1,
        leverage: int = 10,
        maintenance_margin_ratio: float = 0.005,
        max_position_hours: float = 48.0,
        # Параметры частичного TP и trailing
        partial_tp_enabled: bool = False,
        partial_tp_pct: float = 0.015,  # 1.5% - первая цель для breakeven
        trailing_activation_pct: float = 0.03,  # 3.0% - активация trailing
        trailing_distance_pct: float = 0.02,  # 2.0% - расстояние trailing
        # Параметры входа по откату (pullback)
        pullback_enabled: bool = False,
        pullback_ema_period: int = 9,  # Период EMA для отката (9 или 20)
        pullback_pct: float = 0.003,  # 0.3% от high/low сигнальной свечи
        pullback_max_bars: int = 3,  # Максимальная задержка входа (1-3 свечи)
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.max_position_size_pct = max_position_size_pct
        self.leverage = leverage
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.max_position_hours = max_position_hours
        
        # Параметры частичного TP и trailing
        self.partial_tp_enabled = partial_tp_enabled
        self.partial_tp_pct = partial_tp_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        
        # Флаги состояния для частичного TP
        self.breakeven_activated = False  # Флаг активации breakeven
        self.trailing_activated = False  # Флаг активации trailing
        
        # Параметры входа по откату
        self.pullback_enabled = pullback_enabled
        self.pullback_ema_period = pullback_ema_period
        self.pullback_pct = pullback_pct
        self.pullback_max_bars = pullback_max_bars
        
        # Ожидающие сигналы (pending signals) для входа по откату
        # Структура: {signal: Signal, signal_time: datetime, signal_high: float, signal_low: float, bars_waited: int}
        self.pending_signals: List[Dict] = []
        
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.equity_curve: List[float] = [initial_balance]
        self.max_equity = initial_balance
        self.drawdowns: List[Dict] = []
        self.current_drawdown_start = None
        self.current_drawdown_peak = initial_balance
        
        # Статистика сигналов
        self.signal_stats = SignalStats()
        self.signal_history: List[Dict] = []
        
        print(f"[Backtest] Режим: ТОЧНАЯ ИМИТАЦИЯ реального сервера")
        print(f"[Backtest] НЕ исправляю ошибки стратегии!")
        if self.partial_tp_enabled:
            print(f"[Backtest] Частичный TP включен: breakeven при {self.partial_tp_pct*100:.2f}%, trailing при {self.trailing_activation_pct*100:.2f}%")
        if self.pullback_enabled:
            print(f"[Backtest] Вход по откату включен: EMA{self.pullback_ema_period}, откат {self.pullback_pct*100:.2f}%, макс. задержка {self.pullback_max_bars} баров")
    
    def analyze_signal(self, signal: Signal, current_price: float):
        """Анализирует сигнал от стратегии (только статистика, без изменений)."""
        self.signal_stats.total_signals += 1
        
        # Записываем причину
        reason_key = signal.reason[:50] if signal.reason else "no_reason"
        self.signal_stats.reasons[reason_key] = self.signal_stats.reasons.get(reason_key, 0) + 1
        
        if signal.action == Action.LONG:
            self.signal_stats.long_signals += 1
        elif signal.action == Action.SHORT:
            self.signal_stats.short_signals += 1
        else:
            self.signal_stats.hold_signals += 1
        
        # Проверяем наличие TP/SL в сигнале
        has_tp_sl = signal.stop_loss is not None and signal.take_profit is not None
        
        if not has_tp_sl and signal.indicators_info:
            # Проверяем indicators_info
            has_tp_sl = (signal.indicators_info.get('stop_loss') is not None and 
                        signal.indicators_info.get('take_profit') is not None)
        
        if has_tp_sl:
            self.signal_stats.signals_with_tp_sl += 1
            
            # Получаем цены TP/SL
            sl_price = signal.stop_loss or signal.indicators_info.get('stop_loss')
            tp_price = signal.take_profit or signal.indicators_info.get('take_profit')
            
            if sl_price and tp_price and current_price > 0:
                # Рассчитываем расстояния
                if signal.action == Action.LONG:
                    sl_distance_pct = (current_price - sl_price) / current_price * 100
                    tp_distance_pct = (tp_price - current_price) / current_price * 100
                else:  # SHORT
                    sl_distance_pct = (sl_price - current_price) / current_price * 100
                    tp_distance_pct = (current_price - tp_price) / current_price * 100
                
                self.signal_stats.sl_distances.append(sl_distance_pct)
                self.signal_stats.tp_distances.append(tp_distance_pct)
                
                # Проверяем SL=1%
                if 0.8 <= sl_distance_pct <= 1.2:  # Допуск ±0.2%
                    self.signal_stats.signals_with_correct_sl += 1
                else:
                    self.signal_stats.signals_with_wrong_sl += 1
                    
                    # Логируем только первые 5 неправильных SL
                    if self.signal_stats.signals_with_wrong_sl <= 5:
                        print(f"⚠️  Сигнал с НЕстандартным SL: {sl_distance_pct:.2f}%")
                        print(f"   Действие: {signal.action.value}, Цена: {current_price:.2f}")
                        print(f"   Причина: {signal.reason}")
        else:
            # HOLD сигналы не должны иметь TP/SL - это нормально
            if signal.action != Action.HOLD:
                self.signal_stats.signals_without_tp_sl += 1
                # Логируем только первые 3 LONG/SHORT сигнала без TP/SL
                if self.signal_stats.signals_without_tp_sl <= 3:
                    print(f"❌ Сигнал БЕЗ TP/SL: {signal.action.value} @ {current_price:.2f}")
                print(f"   Причина: {signal.reason}")
        
        # Записываем в историю
        self.signal_history.append({
            'timestamp': datetime.now(),
            'action': signal.action.value,
            'price': current_price,
            'reason': signal.reason,
            'has_tp_sl': has_tp_sl,
            'confidence': signal.indicators_info.get('confidence', 0) if signal.indicators_info else 0
        })
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, action: Action, 
                               margin_pct_balance: float = 0.20, base_order_usd: float = 100.0) -> Tuple[float, float]:
        """
        Рассчитывает размер позиции ТОЧНО как реальный бот.
        
        Реальный бот использует:
        - Фиксированная сумма позиции: base_order_usd (по умолчанию $100)
        - Маржа = base_order_usd / leverage
        - Количество = base_order_usd / цена входа
        """
        # РАСЧЕТ: Фиксированная сумма позиции с учетом плеча
        # Размер позиции в USD (notional value) = фиксированная сумма
        position_size_usd = base_order_usd
        
        # Требуемая маржа = размер позиции / leverage
        margin_required = position_size_usd / self.leverage
        
        # Проверяем, что маржа не превышает баланс
        if margin_required > self.balance:
            # Если маржа больше баланса, уменьшаем размер позиции
            # Максимальная маржа = баланс
            max_margin = self.balance
            position_size_usd = max_margin * self.leverage
            margin_required = max_margin
        
        return position_size_usd, margin_required
    
    def check_pullback_condition(self, pending_signal: Dict, current_price: float, high: float, low: float, 
                                ema_value: Optional[float] = None) -> bool:
        """
        Проверяет условия отката для pending сигнала.
        
        Args:
            pending_signal: Словарь с информацией о pending сигнале
            current_price: Текущая цена закрытия
            high: High текущей свечи
            low: Low текущей свечи
            ema_value: Значение EMA (если доступно)
        
        Returns:
            True если условия отката выполнены, False иначе
        """
        signal = pending_signal['signal']
        signal_high = pending_signal['signal_high']
        signal_low = pending_signal['signal_low']
        
        if signal.action == Action.LONG:
            # LONG: ждем откат к EMA или к уровню -0.3% от high сигнальной свечи
            pullback_level = signal_high * (1 - self.pullback_pct)
            
            # Проверяем откат к EMA (если доступно)
            if ema_value is not None and not np.isnan(ema_value):
                if low <= ema_value <= high:
                    return True  # Цена коснулась EMA
            
            # Проверяем откат к уровню (low текущей свечи <= pullback_level)
            if low <= pullback_level:
                return True
        else:  # SHORT
            # SHORT: ждем откат вверх к EMA или к уровню +0.3% от low сигнальной свечи
            pullback_level = signal_low * (1 + self.pullback_pct)
            
            # Проверяем откат к EMA (если доступно)
            if ema_value is not None and not np.isnan(ema_value):
                if low <= ema_value <= high:
                    return True  # Цена коснулась EMA
            
            # Проверяем откат к уровню (high текущей свечи >= pullback_level)
            if high >= pullback_level:
                return True
        
        return False
    
    def process_pending_signals(self, current_time: datetime, current_price: float, high: float, low: float,
                                df: pd.DataFrame, current_idx: int) -> Optional[Signal]:
        """
        Обрабатывает pending сигналы и проверяет условия отката.
        
        Returns:
            Signal для открытия позиции, если условия выполнены, None иначе
        """
        if not self.pullback_enabled or not self.pending_signals:
            return None
        
        # Удаляем устаревшие сигналы (превысили максимальную задержку)
        self.pending_signals = [
            ps for ps in self.pending_signals 
            if ps['bars_waited'] < self.pullback_max_bars
        ]
        
        if not self.pending_signals:
            return None
        
        # Получаем EMA значение, если доступно
        # В данных используются ema_short (9) и ema_long (21), адаптируемся
        ema_value = None
        try:
            if current_idx < len(df):
                if self.pullback_ema_period == 9:
                    # Используем ema_short (9)
                    if 'ema_short' in df.columns:
                        ema_value = df.iloc[current_idx]['ema_short']
                elif self.pullback_ema_period == 20 or self.pullback_ema_period == 21:
                    # Используем ema_long (21) для периода 20
                    if 'ema_long' in df.columns:
                        ema_value = df.iloc[current_idx]['ema_long']
                else:
                    # Пробуем найти колонку с нужным периодом
                    ema_col = f'ema_{self.pullback_ema_period}'
                    if ema_col in df.columns:
                        ema_value = df.iloc[current_idx][ema_col]
                
                if pd.isna(ema_value) or ema_value is None:
                    ema_value = None
        except Exception:
            pass
        
        # Проверяем каждый pending сигнал
        for pending_signal in self.pending_signals[:]:  # Копируем список для безопасной итерации
            pending_signal['bars_waited'] += 1
            
            # Проверяем условия отката
            if self.check_pullback_condition(pending_signal, current_price, high, low, ema_value):
                # Условия выполнены - возвращаем сигнал для открытия позиции
                signal = pending_signal['signal']
                self.pending_signals.remove(pending_signal)
                return signal
        
        return None
    
    def add_pending_signal(self, signal: Signal, signal_time: datetime, signal_high: float, signal_low: float):
        """Добавляет сигнал в список ожидающих отката."""
        if not self.pullback_enabled:
            return
        
        self.pending_signals.append({
            'signal': signal,
            'signal_time': signal_time,
            'signal_high': signal_high,
            'signal_low': signal_low,
            'bars_waited': 0,
        })
    
    def open_position(self, signal: Signal, current_time: datetime, symbol: str) -> bool:
        """
        Открывает позицию ТОЧНО как реальный бот.
        
        Использует TP/SL ИЗ СИГНАЛА, без исправлений!
        """
        if self.current_position is not None:
            return False  # Уже есть позиция
        
        if signal.action == Action.HOLD:
            return False
        
        # Сбрасываем флаги частичного TP при открытии новой позиции
        self.breakeven_activated = False
        self.trailing_activated = False
        
        # 1. Получаем TP/SL ИЗ СИГНАЛА (без проверок!)
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        # 2. Если нет, берем из indicators_info
        if (stop_loss is None or take_profit is None) and signal.indicators_info:
            stop_loss = signal.indicators_info.get('stop_loss')
            take_profit = signal.indicators_info.get('take_profit')
        
        # 3. Если ВСЕ ЕЩЕ нет - НЕ открываем позицию (как реальный бот)
        if stop_loss is None or take_profit is None:
            print(f"❌ Не могу открыть позицию: сигнал без TP/SL")
            print(f"   Действие: {signal.action.value}, Цена: {signal.price:.2f}")
            print(f"   Причина: {signal.reason}")
            return False
        
        # 4. Рассчитываем размер позиции (ТОЧНО как реальный бот)
        # Используем фиксированную сумму $100 с учетом плеча
        base_order_usd = getattr(self, '_base_order_usd', 100.0)  # Фиксированная сумма позиции $100
        
        import logging
        logger = logging.getLogger(__name__)
        if hasattr(self, '_open_position_call_count'):
            self._open_position_call_count += 1
        else:
            self._open_position_call_count = 1
        
        if self._open_position_call_count <= 3:
            logger.info(f"[open_position] Расчет размера позиции: base_order_usd={base_order_usd}, price={signal.price:.2f}, sl={stop_loss:.2f}")
        
        try:
            position_size_usd, margin_required = self.calculate_position_size(
                signal.price, stop_loss, signal.action,
                margin_pct_balance=0.20,  # Не используется, оставлено для совместимости
                base_order_usd=base_order_usd
            )
            
            if self._open_position_call_count <= 3:
                logger.info(f"[open_position] Размер позиции рассчитан: size_usd={position_size_usd:.2f}, margin={margin_required:.2f}")
        except Exception as e:
            logger.error(f"[open_position] Ошибка при расчете размера позиции: {e}")
            import traceback
            logger.error(f"[open_position] Traceback:\n{traceback.format_exc()}")
            raise
        
        if position_size_usd <= 0 or margin_required > self.balance:
            if self._open_position_call_count <= 3:
                logger.warning(f"[open_position] Недостаточно средств: size={position_size_usd:.2f}, margin={margin_required:.2f}, balance={self.balance:.2f}")
            print(f"❌ Не могу открыть позицию: недостаточно средств")
            print(f"   Размер: ${position_size_usd:.2f}, Маржа: ${margin_required:.2f}, Баланс: ${self.balance:.2f}")
            return False
        
        # 5. Вычитаем маржу (как реальный бот)
        if self._open_position_call_count <= 3:
            logger.info(f"[open_position] Вычитаем маржу: {margin_required:.2f} из баланса {self.balance:.2f}")
        self.balance -= margin_required
        
        # 6. Рассчитываем расстояния TP/SL для статистики
        if signal.action == Action.LONG:
            sl_distance_pct = (signal.price - stop_loss) / signal.price * 100
            tp_distance_pct = (take_profit - signal.price) / signal.price * 100
        else:
            sl_distance_pct = (stop_loss - signal.price) / signal.price * 100
            tp_distance_pct = (signal.price - take_profit) / signal.price * 100
        
        # 7. Создаем позицию (ТОЧНО с теми TP/SL, что в сигнале)
        confidence = signal.indicators_info.get('confidence', 0.5) if signal.indicators_info else 0.5
        
        if self._open_position_call_count <= 3:
            logger.info(f"[open_position] Создание объекта Trade...")
            print(f"   Создание позиции...")
        
        self.current_position = Trade(
            entry_time=current_time,
            exit_time=None,
            entry_price=signal.price,
            exit_price=None,
            action=signal.action,
            size_usd=position_size_usd,
            pnl=0.0,
            pnl_pct=0.0,
            entry_reason=signal.reason,
            exit_reason=None,
            symbol=symbol,
            confidence=confidence,
            stop_loss=stop_loss,      # ТОЧНО из сигнала
            take_profit=take_profit,  # ТОЧНО из сигнала
            signal_sl_pct=sl_distance_pct,
            signal_tp_pct=tp_distance_pct,
        )
        
        # 8. Логируем (только первые 5 позиций)
        if len(self.trades) < 5:
            print(f"\n📊 Открыта позиция #{len(self.trades) + 1}:")
            print(f"   {signal.action.value} @ ${signal.price:.2f}")
            print(f"   TP: ${take_profit:.2f} ({tp_distance_pct:.2f}%)")
            print(f"   SL: ${stop_loss:.2f} ({sl_distance_pct:.2f}%)")
            print(f"   Размер: ${position_size_usd:.2f}")
            print(f"   Уверенность: {confidence:.1%}")
            print(f"   Причина: {signal.reason}")
            print(f"   Баланс после маржи: ${self.balance:.2f}")
        
        return True
    
    def check_exit(self, current_time: datetime, current_price: float, high: float, low: float) -> bool:
        """Проверяет условия выхода из позиции (как реальный бот)."""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        
        # 1. Проверяем максимальное время удержания
        position_duration = (current_time - pos.entry_time).total_seconds() / 3600
        if position_duration >= self.max_position_hours:
            self.close_position(current_time, current_price, ExitReason.TIME_LIMIT)
            return True
        
        # 2. Проверяем TP/SL (как реальный бот, по high/low свечи)
        if pos.action == Action.LONG:
            # SL: если low <= stop_loss
            if low <= pos.stop_loss:
                exit_price = min(pos.stop_loss, current_price)
                self.close_position(current_time, exit_price, ExitReason.STOP_LOSS)
                return True
            # TP: если high >= take_profit
            elif high >= pos.take_profit:
                exit_price = max(pos.take_profit, current_price)
                self.close_position(current_time, exit_price, ExitReason.TAKE_PROFIT)
                return True
        else:  # SHORT
            # SL: если high >= stop_loss
            if high >= pos.stop_loss:
                exit_price = max(pos.stop_loss, current_price)
                self.close_position(current_time, exit_price, ExitReason.STOP_LOSS)
                return True
            # TP: если low <= take_profit
            elif low <= pos.take_profit:
                exit_price = min(pos.take_profit, current_price)
                self.close_position(current_time, exit_price, ExitReason.TAKE_PROFIT)
                return True
        
        # 3. Обновляем MFE/MAE
        if pos.action == Action.LONG:
            mfe = (high - pos.entry_price) / pos.entry_price
            mae = (low - pos.entry_price) / pos.entry_price
        else:
            mfe = (pos.entry_price - low) / pos.entry_price
            mae = (pos.entry_price - high) / pos.entry_price
        
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, mfe)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, mae)
        
        # 4. Логика частичного TP и trailing (если включено)
        if self.partial_tp_enabled:
            if pos.action == Action.LONG:
                # Breakeven: при достижении partial_tp_pct переводим SL в breakeven
                if not self.breakeven_activated and mfe >= self.partial_tp_pct:
                    # Переводим SL в breakeven (чуть выше цены входа)
                    breakeven_sl = pos.entry_price * 1.001  # 0.1% выше входа
                    if breakeven_sl > pos.stop_loss:
                        pos.stop_loss = breakeven_sl
                        self.breakeven_activated = True
                        # Проверяем, не сработал ли уже breakeven SL
                        if low <= pos.stop_loss:
                            exit_price = min(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.STOP_LOSS)
                            return True
                
                # Trailing: при достижении trailing_activation_pct включаем trailing stop
                if mfe >= self.trailing_activation_pct:
                    if not self.trailing_activated:
                        self.trailing_activated = True
                    # Обновляем trailing stop
                    potential_sl = high * (1 - self.trailing_distance_pct)
                    if potential_sl > pos.stop_loss:
                        pos.stop_loss = potential_sl
                        # Проверяем, не сработал ли уже trailing SL
                        if low <= pos.stop_loss:
                            exit_price = min(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.TRAILING_STOP)
                            return True
                elif self.trailing_activated:
                    # Trailing уже активирован, продолжаем обновлять
                    potential_sl = high * (1 - self.trailing_distance_pct)
                    if potential_sl > pos.stop_loss:
                        pos.stop_loss = potential_sl
                        if low <= pos.stop_loss:
                            exit_price = min(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.TRAILING_STOP)
                            return True
            else:  # SHORT
                # Breakeven: при достижении partial_tp_pct переводим SL в breakeven
                if not self.breakeven_activated and mfe >= self.partial_tp_pct:
                    # Переводим SL в breakeven (чуть ниже цены входа)
                    breakeven_sl = pos.entry_price * 0.999  # 0.1% ниже входа
                    if breakeven_sl < pos.stop_loss:
                        pos.stop_loss = breakeven_sl
                        self.breakeven_activated = True
                        # Проверяем, не сработал ли уже breakeven SL
                        if high >= pos.stop_loss:
                            exit_price = max(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.STOP_LOSS)
                            return True
                
                # Trailing: при достижении trailing_activation_pct включаем trailing stop
                if mfe >= self.trailing_activation_pct:
                    if not self.trailing_activated:
                        self.trailing_activated = True
                    # Обновляем trailing stop
                    potential_sl = low * (1 + self.trailing_distance_pct)
                    if potential_sl < pos.stop_loss:
                        pos.stop_loss = potential_sl
                        # Проверяем, не сработал ли уже trailing SL
                        if high >= pos.stop_loss:
                            exit_price = max(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.TRAILING_STOP)
                            return True
                elif self.trailing_activated:
                    # Trailing уже активирован, продолжаем обновлять
                    potential_sl = low * (1 + self.trailing_distance_pct)
                    if potential_sl < pos.stop_loss:
                        pos.stop_loss = potential_sl
                        if high >= pos.stop_loss:
                            exit_price = max(pos.stop_loss, current_price)
                            self.close_position(current_time, exit_price, ExitReason.TRAILING_STOP)
                            return True
        
        return False
    
    def close_position(self, exit_time: datetime, exit_price: float, exit_reason: ExitReason):
        """Закрывает позицию (как реальный бот)."""
        if self.current_position is None:
            return
        
        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        
        # Рассчитываем PnL
        if pos.action == Action.LONG:
            price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:  # SHORT
            price_change_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        # PnL с учетом плеча: считаем от МАРЖИ, а не от размера позиции!
        # Маржа = размер позиции / плечо
        margin_used = pos.size_usd / self.leverage
        
        # PnL в USD = маржа * процент изменения цены * плечо
        pnl_usd_before_commission = margin_used * price_change_pct * self.leverage
        
        # Комиссии (считаются от суммы сделки при входе и выходе)
        # При входе: сумма = размер позиции в USD
        # При выходе: сумма = количество монет × цена выхода
        # Количество монет = размер позиции при входе / цена входа
        quantity = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0.0
        notional_entry = pos.size_usd  # Сумма сделки при входе
        notional_exit = quantity * exit_price  # Сумма сделки при выходе
        commission_cost = (notional_entry + notional_exit) * self.commission  # Вход + выход
        
        # Итоговый PnL с учетом комиссий
        pnl_usd = pnl_usd_before_commission - commission_cost
        
        # Процент PnL от маржи С УЧЕТОМ комиссий (одинаково для LONG и SHORT)
        # Это реальный процент доходности/убытка от маржи
        if margin_used > 0:
            pnl_pct = (pnl_usd / margin_used) * 100  # Процент от маржи с учетом комиссий
        else:
            pnl_pct = 0.0
        
        # Возвращаем маржу и добавляем PnL
        margin_returned = margin_used
        self.balance += margin_returned + pnl_usd
        
        pos.pnl = pnl_usd
        pos.pnl_pct = pnl_pct  # Процент от маржи с учетом комиссий
        
        # Обновляем кривую капитала
        self.equity_curve.append(self.balance)
        
        # Обновляем максимальный equity
        if self.balance > self.max_equity:
            self.max_equity = self.balance
        
        # Сохраняем сделку
        self.trades.append(pos)
        self.current_position = None
        
        # Логируем все сделки (но с периодическим выводом после 10-й)
        if len(self.trades) <= 10:
            print(f"\n📊 Закрыта позиция #{len(self.trades)}:")
            print(f"   {pos.action.value} @ ${pos.entry_price:.2f} -> ${exit_price:.2f}")
            print(f"   Причина: {exit_reason.value}")
            print(f"   PnL: ${pnl_usd:.2f} ({pos.pnl_pct:.2f}%)")
            print(f"   Новый баланс: ${self.balance:.2f}")
        elif len(self.trades) % 10 == 0:
            # Периодический вывод каждые 10 сделок
            print(f"📊 Сделка #{len(self.trades)}: {pos.action.value} -> {exit_reason.value}, PnL: ${pnl_usd:.2f} ({pos.pnl_pct:.2f}%), Баланс: ${self.balance:.2f}")
    
    def close_all_positions(self, final_time: datetime, final_price: float):
        """Закрывает все позиции в конце бэктеста."""
        if self.current_position is not None:
            self.close_position(final_time, final_price, ExitReason.END_OF_BACKTEST)
    
    def calculate_metrics(self, symbol: str, model_name: str, days_back: int = 0) -> BacktestMetrics:
        """Рассчитывает метрики бэктеста."""
        # Рассчитываем trades_per_day на основе количества дней
        trades_per_day = len(self.trades) / days_back if days_back > 0 and self.trades else 0.0
        
        if not self.trades:
            return BacktestMetrics(
                symbol=symbol,
                model_name=model_name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                total_signals=self.signal_stats.total_signals,
                long_signals=self.signal_stats.long_signals,
                short_signals=self.signal_stats.short_signals,
                avg_trade_duration_hours=0.0,
                best_trade_pnl=0.0,
                worst_trade_pnl=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_confidence=0.0,
                avg_mfe=0.0,
                avg_mae=0.0,
                mfe_mae_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                recovery_factor=0.0,
                expectancy_usd=0.0,
                risk_reward_ratio=0.0,
                trade_frequency_per_day=trades_per_day,
                profitable_days_pct=0.0,
                ulcer_index=0.0,
                kelly_criterion=0.0,
                avg_tp_distance_pct=0.0,
                avg_sl_distance_pct=0.0,
                avg_rr_ratio=0.0,
                signal_quality_score=0.0,
                signals_with_tp_sl_pct=0.0,
                signals_with_correct_sl_pct=0.0,
                avg_position_size_usd=0.0,
            )
        
        # Базовые метрики сделок
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0.0
        total_pnl = self.balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Максимальная просадка
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak = self.initial_balance
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Sharpe Ratio
        sharpe_ratio = 0.0
        if len(self.trades) > 1:
            returns = np.array([t.pnl_pct / 100 for t in self.trades], dtype=float)
            std = float(np.std(returns))
            if std >= 1e-9:
                sharpe_ratio = float(np.mean(returns) / std * np.sqrt(252))
        
        # Статистика TP/SL из сделок
        tp_distances = [t.signal_tp_pct for t in self.trades if t.signal_tp_pct is not None]
        sl_distances = [t.signal_sl_pct for t in self.trades if t.signal_sl_pct is not None]
        
        avg_tp_distance = np.mean(tp_distances) if tp_distances else 0.0
        avg_sl_distance = np.mean(sl_distances) if sl_distances else 0.0
        
        # R/R Ratio
        avg_rr_ratio = 0.0
        if sl_distances and np.mean(sl_distances) > 0:
            avg_rr_ratio = avg_tp_distance / np.mean(sl_distances)
        
        # Статистика сигналов
        # ВАЖНО: Считаем процент только для LONG/SHORT сигналов, так как HOLD не должны иметь TP/SL
        tradable_signals = self.signal_stats.long_signals + self.signal_stats.short_signals
        signals_with_tp_sl_pct = (self.signal_stats.signals_with_tp_sl / 
                                 max(1, tradable_signals)) * 100 if tradable_signals > 0 else 0.0
        
        signals_with_correct_sl_pct = (self.signal_stats.signals_with_correct_sl / 
                                      max(1, self.signal_stats.signals_with_tp_sl)) * 100
        
        # Средний размер позиции
        avg_position_size = np.mean([t.size_usd for t in self.trades]) if self.trades else 0.0
        
        return BacktestMetrics(
            symbol=symbol,
            model_name=model_name,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0.0,  # Упрощенно
            calmar_ratio=total_pnl_pct / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 0 else 0.0,
            total_signals=self.signal_stats.total_signals,
            long_signals=self.signal_stats.long_signals,
            short_signals=self.signal_stats.short_signals,
            avg_trade_duration_hours=0.0,  # Упрощенно
            best_trade_pnl=max([t.pnl for t in self.trades]) if self.trades else 0.0,
            worst_trade_pnl=min([t.pnl for t in self.trades]) if self.trades else 0.0,
            consecutive_wins=0,  # Упрощенно
            consecutive_losses=0,
            largest_win=max([t.pnl for t in winning_trades]) if winning_trades else 0.0,
            largest_loss=min([t.pnl for t in losing_trades]) if losing_trades else 0.0,
            avg_confidence=np.mean([t.confidence for t in self.trades]) if self.trades else 0.0,
            avg_mfe=np.mean([t.max_favorable_excursion for t in self.trades]) * 100 if self.trades else 0.0,  # В процентах
            avg_mae=np.mean([abs(t.max_adverse_excursion) for t in self.trades]) * 100 if self.trades else 0.0,  # В процентах, берем abs
            mfe_mae_ratio=np.mean([t.max_favorable_excursion / abs(t.max_adverse_excursion) if t.max_adverse_excursion != 0 else 0.0 for t in self.trades]) if self.trades else 0.0,
            var_95=0.0,
            cvar_95=0.0,
            recovery_factor=total_pnl / max_drawdown if max_drawdown > 0 else 0.0,
            expectancy_usd=(win_rate/100 * avg_win) - ((100 - win_rate)/100 * abs(avg_loss)),
            risk_reward_ratio=avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0.0,
            trade_frequency_per_day=trades_per_day,
            profitable_days_pct=0.0,
            ulcer_index=0.0,
            kelly_criterion=0.0,
            avg_tp_distance_pct=avg_tp_distance,
            avg_sl_distance_pct=avg_sl_distance,
            avg_rr_ratio=avg_rr_ratio,
            signal_quality_score=0.0,
            signals_with_tp_sl_pct=signals_with_tp_sl_pct,
            signals_with_correct_sl_pct=signals_with_correct_sl_pct,
            avg_position_size_usd=avg_position_size,
        )


def _get_atr_pct_1h_for_time(atr_1h_series: pd.Series, current_time: pd.Timestamp) -> Optional[float]:
    """Возвращает ATR 1h в % от цены для момента current_time (последняя закрытая 1h свеча)."""
    if atr_1h_series is None or atr_1h_series.empty:
        return None
    valid = atr_1h_series.index <= current_time
    if not valid.any():
        return None
    val = atr_1h_series.loc[valid].iloc[-1]
    if pd.isna(val) or np.isnan(val):
        return None
    return float(val)


def run_exact_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    interval: str = "15",
    initial_balance: float = 100.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    atr_filter_enabled: bool = False,
    atr_min_pct: float = 0.3,
    atr_max_pct: float = 2.0,
    # Параметры частичного TP и trailing
    partial_tp_enabled: bool = False,
    partial_tp_pct: float = 0.015,  # 1.5%
    trailing_activation_pct: float = 0.03,  # 3.0%
    trailing_distance_pct: float = 0.02,  # 2.0%
    # Параметры входа по откату (pullback)
    pullback_enabled: bool = False,
    pullback_ema_period: int = 9,  # Период EMA (9 или 20)
    pullback_pct: float = 0.003,  # 0.3% от high/low
    pullback_max_bars: int = 3,  # Максимальная задержка (1-3 свечи)
) -> Optional[BacktestMetrics]:
    """
    Запускает ТОЧНЫЙ бэктест, который имитирует работу сервера.
    
    Args:
        model_path: Путь к ML модели
        symbol: Торговая пара
        days_back: Количество дней для бэктеста
        interval: Интервал свечей
        initial_balance: Начальный баланс
        risk_per_trade: Риск на сделку
        leverage: Плечо
    
    Returns:
        BacktestMetrics с результатами или None при ошибке
    """
    import traceback
    
    try:
        print("=" * 80)
        print("🚀 ТОЧНЫЙ БЭКТЕСТ (полная имитация продакшена)")
        print("=" * 80)
        print(f"Модель: {Path(model_path).name}")
        print(f"Символ: {symbol}")
        print(f"Дней: {days_back}")
        print(f"Интервал: {interval}")
        print(f"Начальный баланс: ${initial_balance:.2f}")
        print(f"Риск на сделку: {risk_per_trade*100:.1f}%")
        print(f"Плечо: {leverage}x")
        print("=" * 80)
        print("✅ БЭКТЕСТ ИСПОЛЬЗУЕТ ТОЧНО ТЕ ЖЕ МЕТОДЫ, ЧТО И РЕАЛЬНЫЙ БОТ:")
        print("   - MLStrategy.generate_signal() - идентично продакшену")
        print("   - Те же параметры из config.py")
        print("   - Те же фильтры (стабильность, RSI, объем)")
        print("   - Тот же расчет TP/SL")
        print("   - То же окно данных (все данные до текущего момента)")
        print("=" * 80)
        print("⚠️  ВАЖНО: Бэктест НЕ исправляет ошибки стратегии!")
        print("          Показывает КАК стратегия работает на самом деле.")
        print("          Результаты бэктеста = результаты на реальных данных.")
        print("=" * 80)
        
        # Проверка модели
        model_file = Path(model_path)
        if not model_file.exists():
            model_file = Path("ml_models") / model_path
            if not model_file.exists():
                print(f"❌ Файл модели не найден: {model_path}")
                return None
        
        # Загружаем настройки
        try:
            settings = load_settings()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[run_exact_backtest] Ошибка загрузки настроек: {e}")
            print(f"❌ Ошибка загрузки настроек: {e}")
            return None
        
        # Создаем клиент
        client = BybitClient(settings.api)
    
        # Получаем исторические данные
        print(f"\n📊 Загрузка исторических данных...")
        try:
            if interval.endswith("m"):
                bybit_interval = interval[:-1]
            else:
                bybit_interval = interval
            
            interval_min = int(bybit_interval)
            candles_per_day = (24 * 60) // interval_min
            total_candles = days_back * candles_per_day
            
            df = client.get_kline_df(symbol, bybit_interval, limit=total_candles)
            
            if df.empty:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"[run_exact_backtest] Нет данных для {symbol}")
                print(f"❌ Нет данных для {symbol}")
                return None
            
            print(f"✅ Загружено {len(df)} свечей")
            print(f"   Период: {df.index[0]} до {df.index[-1]}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[run_exact_backtest] Ошибка загрузки данных для {symbol}: {e}")
            print(f"❌ Ошибка загрузки данных: {e}")
            return None
        
        # Загрузка 1h данных для фильтра волатильности (ATR 1h)
        atr_1h_series: Optional[pd.Series] = None
        if atr_filter_enabled:
            try:
                candles_1h = days_back * 24 + 30  # достаточно для ATR(14)
                df_1h = client.get_kline_df(symbol, "60", limit=candles_1h)
                if df_1h.empty or len(df_1h) < 20:
                    print(f"⚠️ Недостаточно 1h данных для фильтра ATR, фильтр отключен для этого запуска")
                    atr_filter_enabled = False
                else:
                    df_1h = prepare_with_indicators(df_1h)
                    if "atr_pct" in df_1h.columns:
                        atr_1h_series = df_1h["atr_pct"]
                        print(f"✅ Фильтр волатильности: ATR 1h загружен ({len(atr_1h_series)} баров), диапазон {atr_min_pct}–{atr_max_pct}%")
                    else:
                        print(f"⚠️ В 1h данных нет atr_pct, фильтр отключен")
                        atr_filter_enabled = False
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"[run_exact_backtest] Ошибка загрузки 1h для ATR-фильтра: {e}")
                print(f"⚠️ Фильтр волатильности отключен из-за ошибки: {e}")
                atr_filter_enabled = False
        
        # Определяем, является ли модель MTF (multi-timeframe)
        # и устанавливаем переменную окружения если нужно
        model_name = model_file.stem
        is_mtf_model = "_mtf" in model_name.lower()
        if is_mtf_model:
            os.environ["ML_MTF_ENABLED"] = "1"
            print(f"🔧 MTF модель обнаружена, включен MTF режим")
        else:
            # Убеждаемся, что MTF отключен для не-MTF моделей
            os.environ["ML_MTF_ENABLED"] = "0"
        
        # Подготавливаем индикаторы
        print(f"\n🔧 Подготовка индикаторов...")
        try:
            df_with_indicators = prepare_with_indicators(df.copy())
            print(f"✅ Индикаторы подготовлены")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[run_exact_backtest] Ошибка подготовки индикаторов для {symbol}: {e}")
            import traceback
            logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
            print(f"❌ Ошибка подготовки индикаторов: {e}")
            return None
        
        # Готовим ML стратегию (ТОЧНО как реальный бот)
        print(f"\n🤖 Подготовка ML стратегии...")
        try:
            # ВАЖНО: Используем те же параметры, что и реальный бот
            strategy = MLStrategy(
                model_path=str(model_file),
                confidence_threshold=settings.ml_strategy.confidence_threshold,
                min_signal_strength=settings.ml_strategy.min_signal_strength,
                stability_filter=settings.ml_strategy.stability_filter,
                min_signals_per_day=settings.ml_strategy.min_signals_per_day,
                max_signals_per_day=settings.ml_strategy.max_signals_per_day
            )
            
            # Подготавливаем данные (как реальный бот)
            df_work = df_with_indicators.copy()
            if "timestamp" in df_work.columns:
                df_work = df_work.set_index("timestamp")
            
            # Создаем технические индикаторы (как реальный бот)
            df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)
            
            # ВАЛИДАЦИЯ: Проверяем, что стратегия инициализирована правильно
            print(f"   Параметры стратегии:")
            print(f"   - Confidence threshold: {strategy.confidence_threshold}")
            print(f"   - Min signal strength: {strategy.min_signal_strength}")
            print(f"   - Stability filter: {strategy.stability_filter}")
            print(f"   - Min signals/day: {strategy.min_signals_per_day}")
            print(f"   - Max signals/day: {strategy.max_signals_per_day}")
            print(f"   - Target profit (margin): {settings.ml_strategy.target_profit_pct_margin}%")
            print(f"   - Max loss (margin): {settings.ml_strategy.max_loss_pct_margin}%")
            print(f"   - Leverage: {leverage}x")
            print(f"   ✅ Стратегия готова (идентична продакшену)")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[run_exact_backtest] Ошибка подготовки стратегии для {model_path}: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"[run_exact_backtest] Traceback:\n{error_traceback}")
            print(f"❌ Ошибка подготовки стратегии: {e}")
            traceback.print_exc()
            return None
        
        # Создаем симулятор
        simulator = MLBacktestSimulator(
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            leverage=leverage,
            max_position_hours=48.0,
            partial_tp_enabled=partial_tp_enabled,
            partial_tp_pct=partial_tp_pct,
            trailing_activation_pct=trailing_activation_pct,
            trailing_distance_pct=trailing_distance_pct,
            pullback_enabled=pullback_enabled,
            pullback_ema_period=pullback_ema_period,
            pullback_pct=pullback_pct,
            pullback_max_bars=pullback_max_bars,
        )
        
        # Передаем настройки размера позиции в симулятор (как в реальном боте)
        simulator._margin_pct_balance = settings.risk.margin_pct_balance  # 20% от баланса
        # Передаем настройки размера позиции в симулятор (как в реальном боте)
        # Используем фиксированную сумму $100 с учетом плеча
        simulator._base_order_usd = 100.0  # Фиксированная сумма позиции $100
        
        # Подготовка BTCUSDT данных и стратегии для проверки направления (если символ не BTCUSDT)
        btc_strategy = None
        btc_df_with_features = None
        if symbol != "BTCUSDT":
            try:
                print(f"\n📊 Подготовка BTCUSDT для проверки направления...")
                # Загружаем данные BTCUSDT
                btc_df = client.get_kline_df("BTCUSDT", bybit_interval, limit=total_candles)
                if not btc_df.empty:
                    # Подготавливаем индикаторы для BTCUSDT
                    btc_df_with_indicators = prepare_with_indicators(btc_df.copy())
                    btc_df_work = btc_df_with_indicators.copy()
                    if "timestamp" in btc_df_work.columns:
                        btc_df_work = btc_df_work.set_index("timestamp")
                    
                    # Ищем модель BTCUSDT
                    btc_models = list(Path("ml_models").glob("*_BTCUSDT_*.pkl"))
                    if btc_models:
                        btc_model_path = str(btc_models[0])
                        # Определяем, является ли модель MTF
                        btc_is_mtf = "_mtf" in Path(btc_model_path).stem.lower()
                        if btc_is_mtf:
                            os.environ["ML_MTF_ENABLED"] = "1"
                        else:
                            os.environ["ML_MTF_ENABLED"] = "0"
                        
                        # Создаем стратегию BTCUSDT
                        btc_strategy = MLStrategy(
                            model_path=btc_model_path,
                            confidence_threshold=settings.ml_strategy.confidence_threshold,
                            min_signal_strength=settings.ml_strategy.min_signal_strength,
                            stability_filter=settings.ml_strategy.stability_filter,
                            min_signals_per_day=settings.ml_strategy.min_signals_per_day,
                            max_signals_per_day=settings.ml_strategy.max_signals_per_day
                        )
                        
                        # Создаем технические индикаторы для BTCUSDT
                        btc_df_with_features = btc_strategy.feature_engineer.create_technical_indicators(btc_df_work)
                        print(f"✅ BTCUSDT стратегия подготовлена для проверки направления")
                    else:
                        print(f"⚠️  Модель BTCUSDT не найдена, проверка направления отключена")
            except Exception as e:
                logger.warning(f"[run_exact_backtest] Ошибка подготовки BTCUSDT: {e}, проверка направления отключена")
                btc_strategy = None
                btc_df_with_features = None
        
        # Запускаем бэктест
        print(f"\n📈 Запуск точного бэктеста...")
        print(f"   Имитация работы реального бота на сервере")
        print(f"   Используются те же параметры и методы, что и в продакшене")
        
        # Определяем интервал модели из имени файла для адаптивного min_window_size
        model_name = model_file.stem
        model_parts = model_name.split("_")
        model_interval = "15"  # По умолчанию 15 минут
        for part in model_parts:
            if part in ["15", "60", "240", "D"]:
                model_interval = part
                break
        
        # Минимальное окно данных для расчета всех индикаторов (как в реальном боте)
        # Адаптивный размер в зависимости от интервала модели
        if model_interval == "60":  # 1h модели
            min_window_size = 100  # 100 часов = ~4 дня
        elif model_interval == "240":  # 4h модели
            min_window_size = 50  # 50 * 4h = ~8 дней
        else:  # 15m модели
            min_window_size = 200  # 200 * 15m = ~2 дня
        
        print(f"   Определен интервал модели: {model_interval}min")
        print(f"   Минимальное окно данных: {min_window_size} баров")
        
        # ВАЖНО: Используем все данные до текущего момента (как реальный бот)
        # Реальный бот на каждой итерации использует ВСЕ доступные исторические данные
        total_bars = len(df_with_features)
        processed_bars = 0
        
        # Логируем начало бэктеста
        import logging
        logger = logging.getLogger(__name__)
        bars_to_process = total_bars - min_window_size
        logger.info(f"[run_exact_backtest] Начало бэктеста: {total_bars} баров, будет обработано {bars_to_process} баров")
        
        # ДИАГНОСТИКА: Проверяем, достаточно ли данных
        if total_bars < min_window_size:
            error_msg = (
                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Недостаточно данных для бэктеста!\n"
                f"   Загружено баров: {total_bars}\n"
                f"   Требуется минимум: {min_window_size}\n"
                f"   Интервал модели: {model_interval}min\n"
                f"   Период данных: {df_with_features.index[0]} до {df_with_features.index[-1]}\n"
                f"   Для {model_interval}min моделей нужно минимум {min_window_size} баров данных"
            )
            print(error_msg)
            logger.error(error_msg)
            return None
        
        print(f"✅ Достаточно данных: {total_bars} баров (требуется минимум {min_window_size})")
        
        # Прогресс-бар отключен для серверного режима
        # В серверном режиме (деплой, Telegram бот) прогресс-бар не нужен и может вызывать проблемы
        # Используем простой range и логируем прогресс периодически
        bars_to_process = total_bars - min_window_size
        if bars_to_process <= 0:
            error_msg = (
                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Недостаточно данных для обработки!\n"
                f"   Всего баров: {total_bars}\n"
                f"   Минимальное окно: {min_window_size}\n"
                f"   Будет обработано: {bars_to_process}\n"
                f"   Увеличьте период тестирования (--days) или проверьте загрузку данных"
            )
            print(error_msg)
            logger.error(error_msg)
            return None
        
        print(f"📊 Будет обработано {bars_to_process} баров (с {min_window_size} по {total_bars})")
        progress_bar = range(min_window_size, total_bars)
        start_time_loop = None
        try:
            import time
            start_time_loop = time.time()
        except:
            pass
        
        for idx in progress_bar:
            # Пропускаем первые N баров, чтобы накопить достаточно данных для индикаторов
            if idx < min_window_size:
                continue
            
            try:
                current_time = df_with_features.index[idx]
                row = df_with_features.iloc[idx]
                current_price = row['close']
                high = row['high']
                low = row['low']
            except Exception as e:
                logger.error(f"[run_exact_backtest] Ошибка при извлечении данных строки {idx}: {e}")
                import traceback
                logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                raise
            
            # ВАЖНО: Реальный бот использует ВСЕ данные до текущего момента
            # Это критично для правильной работы индикаторов и ML модели
            # Используем данные от начала до текущего индекса (включительно)
            try:
                # Используем view для ускорения (не создаем копию)
                df_window = df_with_features.iloc[:idx+1]
            except Exception as e:
                logger.error(f"[run_exact_backtest] Ошибка при создании df_window для бара {idx}: {e}")
                import traceback
                logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                raise
            
            # Логируем прогресс каждые 1000 баров
            if processed_bars > 0 and processed_bars % 1000 == 0:
                elapsed = time.time() - start_time_loop if start_time_loop else 0
                bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[run_exact_backtest] Прогресс: {processed_bars}/{total_bars - min_window_size} баров "
                    f"({processed_bars*100/(total_bars - min_window_size):.1f}%), "
                    f"сделок: {len(simulator.trades)}, "
                    f"скорость: {bars_per_sec:.1f} бар/сек"
                )
            
            # Определяем текущую позицию (как реальный бот)
            has_position = None
            if simulator.current_position is not None:
                has_position = Bias.LONG if simulator.current_position.action == Action.LONG else Bias.SHORT
            
            # ВАЖНО: Генерируем сигнал ТОЧНО как реальный бот
            # Используем те же параметры из настроек
            # ВАЛИДАЦИЯ: Проверяем, что используем правильный метод
            assert hasattr(strategy, 'generate_signal'), "MLStrategy должен иметь метод generate_signal"
            assert callable(strategy.generate_signal), "generate_signal должен быть вызываемым"
            
            try:
                # ВАЖНО: Вызываем ТОЧНО тот же метод, что и реальный бот
                # ОПТИМИЗАЦИЯ: Фичи уже созданы в df_with_features, поэтому используем skip_feature_creation=True
                # Это значительно ускоряет бэктест (с ~0.6 сек на бар до ~0.01 сек)
                # ВАЖНО: Это корректно, так как фичи уже созданы для всех баров в df_with_features
                # В реальном боте фичи создаются заново, но в бэктесте мы можем оптимизировать
                signal = strategy.generate_signal(
                    row=row,
                    df=df_window,  # Все данные до текущего момента (как реальный бот)
                    has_position=has_position,
                    current_price=current_price,
                    leverage=leverage,
                    target_profit_pct_margin=settings.ml_strategy.target_profit_pct_margin,
                    max_loss_pct_margin=settings.ml_strategy.max_loss_pct_margin,
                    skip_feature_creation=True,  # ОПТИМИЗАЦИЯ: фичи уже созданы
                )
                
                # ВАЛИДАЦИЯ: Проверяем, что сигнал имеет правильный тип
                assert isinstance(signal, Signal), f"Сигнал должен быть типа Signal, получен {type(signal)}"
                
            except AssertionError as e:
                # Критическая ошибка валидации
                logger.error(f"[run_exact_backtest] КРИТИЧЕСКАЯ ОШИБКА ВАЛИДАЦИИ на баре {idx}: {e}")
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ВАЛИДАЦИИ: {e}")
                raise
            except Exception as e:
                # Если ошибка при генерации сигнала, логируем и пропускаем
                # (это может происходить в реальном боте тоже)
                if idx < 10 or processed_bars % 1000 == 0:  # Логируем первые 10 и каждую 1000-ю ошибку
                    logger.warning(f"[run_exact_backtest] Ошибка генерации сигнала на {current_time} (бар {idx}): {e}")
                    if idx < 10:
                        import traceback
                        logger.debug(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                signal = Signal(
                    timestamp=current_time,
                    action=Action.HOLD,
                    reason=f"ml_ошибка_генерации_{str(e)[:30]}",
                    price=current_price
                )
            
            # Анализируем сигнал (только статистика, без изменений)
            try:
                simulator.analyze_signal(signal, current_price)
            except Exception as e:
                logger.error(f"[run_exact_backtest] Ошибка в analyze_signal(): {e}")
                import traceback
                logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                raise
            
            # ВАЖНО: Сначала проверяем выход из позиции (как реальный бот)
            # Это важно, так как может быть сигнал на закрытие текущей позиции
            if simulator.current_position is not None:
                try:
                    exited = simulator.check_exit(current_time, current_price, high, low)
                except Exception as e:
                    logger.error(f"[run_exact_backtest] Ошибка в check_exit(): {e}")
                    import traceback
                    logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                    raise
                
                # Если позиция закрыта, не открываем новую на этой же итерации
                if exited:
                    continue
            
            # Проверка сигнала BTCUSDT для других пар (альткоины следуют за BTC)
            if symbol != "BTCUSDT" and signal.action in (Action.LONG, Action.SHORT) and btc_strategy is not None and btc_df_with_features is not None:
                try:
                    if idx < len(btc_df_with_features):
                        btc_row = btc_df_with_features.iloc[idx]
                        btc_current_price = btc_row['close']
                        btc_df_window = btc_df_with_features.iloc[:idx+1]
                        
                        # Генерируем сигнал BTCUSDT
                        btc_signal = btc_strategy.generate_signal(
                            row=btc_row,
                            df=btc_df_window,
                            has_position=None,  # В бэктесте не проверяем позиции BTC
                            current_price=btc_current_price,
                            leverage=leverage
                        )
                        
                        if btc_signal and btc_signal.action in (Action.LONG, Action.SHORT):
                            # Если сигнал BTC противоположен сигналу текущего символа - игнорируем
                            if (btc_signal.action == Action.LONG and signal.action == Action.SHORT) or \
                               (btc_signal.action == Action.SHORT and signal.action == Action.LONG):
                                logger.debug(
                                    f"[run_exact_backtest] Signal ignored: BTCUSDT={btc_signal.action.value}, "
                                    f"{symbol}={signal.action.value} (opposite direction, following BTC)"
                                )
                                # Пропускаем этот сигнал, но продолжаем обработку
                                processed_bars += 1
                                if processed_bars % 500 == 0:
                                    trades_count = len(simulator.trades)
                                    elapsed = time.time() - start_time_loop if start_time_loop else 0
                                    bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                                    logger.info(
                                        f"[run_exact_backtest] Прогресс: {processed_bars}/{total_bars - min_window_size} баров "
                                        f"({processed_bars*100/(total_bars - min_window_size):.1f}%), "
                                        f"сделок: {trades_count}, баланс: ${simulator.balance:.2f}, "
                                        f"скорость: {bars_per_sec:.1f} бар/сек"
                                    )
                                continue
                except Exception as e:
                    logger.debug(f"[run_exact_backtest] Ошибка проверки BTCUSDT сигнала: {e}")
                    # Продолжаем обработку, если проверка BTC не удалась
            
            # Фильтр по волатильности (ATR 1h): входить только когда «есть движение»
            if atr_filter_enabled and simulator.current_position is None and signal.action in (Action.LONG, Action.SHORT):
                atr_pct_1h = _get_atr_pct_1h_for_time(atr_1h_series, current_time)
                if atr_pct_1h is None:
                    processed_bars += 1
                    if processed_bars % 500 == 0:
                        trades_count = len(simulator.trades)
                        elapsed = time.time() - start_time_loop if start_time_loop else 0
                        bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"[run_exact_backtest] Прогресс: {processed_bars}/{total_bars - min_window_size} баров "
                            f"сделок: {trades_count}, ATR 1h недоступен, пропуск входа"
                        )
                    continue
                if atr_pct_1h < atr_min_pct or atr_pct_1h > atr_max_pct:
                    processed_bars += 1
                    if processed_bars % 500 == 0:
                        trades_count = len(simulator.trades)
                        elapsed = time.time() - start_time_loop if start_time_loop else 0
                        bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"[run_exact_backtest] Прогресс: {processed_bars}/{total_bars - min_window_size} баров "
                            f"сделок: {trades_count}, ATR 1h={atr_pct_1h:.3f}% вне [{atr_min_pct}, {atr_max_pct}], пропуск входа"
                        )
                    continue
            
            # Обрабатываем pending сигналы (вход по откату)
            if simulator.current_position is None and simulator.pullback_enabled:
                try:
                    pullback_signal = simulator.process_pending_signals(
                        current_time, current_price, high, low, df_window, idx
                    )
                    if pullback_signal is not None:
                        # Условия отката выполнены - открываем позицию
                        try:
                            opened = simulator.open_position(pullback_signal, current_time, symbol)
                        except Exception as e:
                            logger.error(f"[run_exact_backtest] Ошибка в open_position() для pullback сигнала: {e}")
                            import traceback
                            logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                            raise
                except Exception as e:
                    logger.error(f"[run_exact_backtest] Ошибка в process_pending_signals(): {e}")
                    import traceback
                    logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                    # Продолжаем обработку, не прерываем бэктест
            
            # Проверяем вход в позицию (только если нет открытой позиции)
            if simulator.current_position is None and signal.action in (Action.LONG, Action.SHORT):
                try:
                    if simulator.pullback_enabled:
                        # Добавляем сигнал в pending вместо немедленного открытия
                        simulator.add_pending_signal(signal, current_time, high, low)
                    else:
                        # Обычный вход без pullback
                        opened = simulator.open_position(signal, current_time, symbol)
                except Exception as e:
                    logger.error(f"[run_exact_backtest] Ошибка в open_position(): {e}")
                    import traceback
                    logger.error(f"[run_exact_backtest] Traceback:\n{traceback.format_exc()}")
                    raise
            
            # Периодический вывод прогресса в лог (каждые 500 баров)
            processed_bars += 1
            if processed_bars % 500 == 0:
                trades_count = len(simulator.trades)
                elapsed = time.time() - start_time_loop if start_time_loop else 0
                bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                progress_pct = processed_bars * 100 / (total_bars - min_window_size) if (total_bars - min_window_size) > 0 else 0
                logger.info(
                    f"[run_exact_backtest] Прогресс: {processed_bars}/{total_bars - min_window_size} баров "
                    f"({progress_pct:.1f}%), сделок: {trades_count}, баланс: ${simulator.balance:.2f}, "
                    f"скорость: {bars_per_sec:.1f} бар/сек"
                    )
        
        # Закрываем все позиции
        if simulator.current_position is not None:
            final_price = df_with_features['close'].iloc[-1]
            final_time = df_with_features.index[-1]
            simulator.close_all_positions(final_time, final_price)
        
        # Рассчитываем метрики
        print(f"\n📊 Расчет метрик...")
        model_name = model_file.stem
        metrics = simulator.calculate_metrics(symbol, model_name, days_back=days_back)
    
        # Выводим результаты
        print("\n" + "=" * 80)
        print("📈 РЕЗУЛЬТАТЫ ТОЧНОГО БЭКТЕСТА")
        print("=" * 80)
        print(f"Символ: {metrics.symbol}")
        print(f"Модель: {metrics.model_name}")
        
        print(f"\n💰 Финансовые метрики:")
        print(f"   Начальный баланс: ${initial_balance:.2f}")
        print(f"   Конечный баланс: ${initial_balance + metrics.total_pnl:.2f}")
        print(f"   Общий PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)")
        print(f"   Макс. просадка: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        
        print(f"\n📊 Статистика сделок:")
        print(f"   Всего сделок: {metrics.total_trades}")
        print(f"   Прибыльных: {metrics.winning_trades}")
        print(f"   Убыточных: {metrics.losing_trades}")
        print(f"   Win Rate: {metrics.win_rate:.2f}%")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        print(f"   Средний выигрыш: ${metrics.avg_win:.2f}")
        print(f"   Средний проигрыш: ${metrics.avg_loss:.2f}")
        
        print(f"\n🎯 АНАЛИЗ СИГНАЛОВ СТРАТЕГИИ:")
        print(f"   Всего сигналов: {metrics.total_signals}")
        print(f"   LONG сигналов: {metrics.long_signals}")
        print(f"   SHORT сигналов: {metrics.short_signals}")
        tradable_count = metrics.long_signals + metrics.short_signals
        if tradable_count > 0:
            print(f"   Сигналов с TP/SL: {metrics.signals_with_tp_sl_pct:.1f}% (от {tradable_count} LONG/SHORT)")
        else:
            print(f"   Сигналов с TP/SL: N/A (нет LONG/SHORT сигналов)")
        print(f"   Сигналов с SL=1%: {metrics.signals_with_correct_sl_pct:.1f}%")
        print(f"   Средний SL в сигналах: {metrics.avg_sl_distance_pct:.2f}%")
        print(f"   Средний TP в сигналах: {metrics.avg_tp_distance_pct:.2f}%")
        print(f"   Средний R/R: {metrics.avg_rr_ratio:.2f}")

        # ТОП причин сигналов (особенно полезно при отсутствии сделок)
        if simulator.signal_stats.reasons:
            print(f"\n🧾 ТОП причин сигналов:")
            top_reasons = sorted(
                simulator.signal_stats.reasons.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for reason, count in top_reasons:
                print(f"   {count:4d}x - {reason}")
        
        print(f"\n📊 Размер позиций:")
        print(f"   Средний размер: ${metrics.avg_position_size_usd:.2f}")
        print(f"   Риск на сделку: {risk_per_trade*100:.1f}% от баланса")
        
        print(f"\n📈 Коэффициенты:")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"   Recovery Factor: {metrics.recovery_factor:.2f}")
        
        print("\n" + "=" * 80)
        
        # КРИТИЧЕСКИЙ АНАЛИЗ СТРАТЕГИИ
        print(f"\n🔍 КРИТИЧЕСКИЙ АНАЛИЗ СТРАТЕГИИ:")
        print(f"   (Анализ основан на ТОЧНОЙ симуляции работы реального бота)")
        
        # Проверяем процент только для LONG/SHORT сигналов (HOLD не должны иметь TP/SL)
        tradable_signals_count = metrics.long_signals + metrics.short_signals
        if tradable_signals_count > 0 and metrics.signals_with_tp_sl_pct < 90:
            print(f"❌ ПРОБЛЕМА: Только {metrics.signals_with_tp_sl_pct:.1f}% LONG/SHORT сигналов имеют TP/SL")
            print(f"   Реальная стратегия на сервере НЕ сможет открыть {100-metrics.signals_with_tp_sl_pct:.1f}% позиций!")
            print(f"   ⚠️  Это означает, что на реальных данных будет такая же проблема!")
        elif tradable_signals_count == 0:
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Нет LONG/SHORT сигналов для анализа TP/SL")
            print(f"   Всего сигналов: {metrics.total_signals}, из них HOLD: {metrics.total_signals}")
        
        if metrics.signals_with_correct_sl_pct < 90:
            print(f"❌ ПРОБЛЕМА: Только {metrics.signals_with_correct_sl_pct:.1f}% сигналов имеют SL=1%")
            print(f"   Стратегия НЕ следует правилу SL=1%!")
            print(f"   Средний SL: {metrics.avg_sl_distance_pct:.2f}% (должен быть 1.0%)")
            print(f"   ⚠️  На реальных данных будет такой же SL!")
        
        if metrics.avg_sl_distance_pct > 2.0:
            print(f"🚨 ОПАСНО: Средний SL {metrics.avg_sl_distance_pct:.2f}% СЛИШКОМ ВЕЛИК!")
            print(f"   Риск на сделку ВЫШЕ чем планировалось!")
            print(f"   ⚠️  На реальных данных риск будет таким же!")
        
        if metrics.avg_rr_ratio < 1.5:
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Средний R/R {metrics.avg_rr_ratio:.2f} слишком низкий")
            print(f"   Нужно R/R > 2.0 для прибыльной торговли")
        
        if metrics.win_rate < 40 and metrics.profit_factor < 1.5:
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Низкий Win Rate ({metrics.win_rate:.1f}%) и Profit Factor ({metrics.profit_factor:.2f})")
            print(f"   Стратегия может быть убыточной на реальных данных")
        
        # РЕКОМЕНДАЦИИ
        print(f"\n📋 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ СТРАТЕГИИ:")
        
        if metrics.signals_with_correct_sl_pct < 90:
            print(f"1. ❗ ИСПРАВИТЬ bot/ml/strategy_ml.py чтобы ВСЕГДА давать SL=1%")
            print(f"   Текущий код должен гарантировать: sl_pct = max_loss_pct_margin / leverage")
        
        tradable_count = metrics.long_signals + metrics.short_signals
        if tradable_count > 0 and metrics.signals_with_tp_sl_pct < 90:
            print(f"2. ❗ ИСПРАВИТЬ bot/ml/strategy_ml.py чтобы ВСЕГДА давать TP/SL в сигналах")
            print(f"   Все сигналы LONG/SHORT должны содержать stop_loss и take_profit")
        
        if metrics.total_trades == 0:
            print(f"3. ❗ СТРАТЕГИЯ НЕ РАБОТАЕТ: 0 сделок за {days_back} дней")
            print(f"   Проверьте:")
            print(f"   - Правильность загрузки модели")
            print(f"   - Пороги confidence_threshold и min_signal_strength")
            print(f"   - Фильтры стратегии (стабильность, RSI, объем)")
        
        # ФИНАЛЬНЫЙ ВЕРДИКТ
        print(f"\n🎯 ФИНАЛЬНЫЙ ВЕРДИКТ:")
        if (metrics.win_rate > 50 and 
            metrics.profit_factor > 2.0 and 
            metrics.signals_with_correct_sl_pct >= 90 and
            (metrics.long_signals + metrics.short_signals == 0 or metrics.signals_with_tp_sl_pct >= 90) and
            metrics.total_trades > 0):
            print(f"✅ СТРАТЕГИЯ ГОТОВА К ПРОДАКШЕНУ!")
            print(f"   Win Rate: {metrics.win_rate:.1f}%")
            print(f"   Profit Factor: {metrics.profit_factor:.2f}")
            print(f"   Правильный SL: {metrics.signals_with_correct_sl_pct:.1f}% сигналов")
            print(f"   Сигналы с TP/SL: {metrics.signals_with_tp_sl_pct:.1f}%")
            print(f"   Всего сделок: {metrics.total_trades}")
            print(f"   📊 Результаты бэктеста = ожидаемые результаты на реальных данных")
        else:
            print(f"🚫 СТРАТЕГИЯ НЕ ГОТОВА К ПРОДАКШЕНУ")
            print(f"   Исправьте проблемы выше и запустите бэктест снова")
            print(f"   ⚠️  Результаты на реальных данных будут аналогичными бэктесту")
        
        print("\n" + "=" * 80)
        print("📝 ВАЖНО: Этот бэктест ТОЧНО симулирует работу реального бота.")
        print("          Все методы, параметры и логика идентичны продакшену.")
        print("          Результаты бэктеста = результаты на реальных данных.")
        print("=" * 80)
        
        return metrics
    
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        # Логируем ошибку через logger, чтобы она попала в логи
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[run_exact_backtest] КРИТИЧЕСКАЯ ОШИБКА для {model_path}: {error_msg}")
        logger.error(f"[run_exact_backtest] Traceback:\n{error_traceback}")
        
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА В БЭКТЕСТЕ:")
        print(f"   {error_msg}")
        print(f"\n📋 Полный traceback:")
        print(error_traceback)
        print("=" * 80)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Точный бэктест ML стратегии (имитация сервера)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:
  # Точный бэктест (показывает КАК стратегия работает)
  python backtest_ml_strategy.py --model ml_models/triple_ensemble_BTCUSDT_15.pkl
  
  # С другими параметрами
  python backtest_ml_strategy.py --model ml_models/ensemble_BTCUSDT_15.pkl --balance 5000 --risk 0.01
  
  # Для другой пары
  python backtest_ml_strategy.py --model ml_models/ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 60
  
  # С частичным TP и trailing (breakeven при 1.5%, trailing при 3.0%)
  python backtest_ml_strategy.py --model ml_models/rf_BTCUSDT_15_15m.pkl --partial-tp --partial-tp-pct 1.5 --trailing-activation-pct 3.0 --trailing-distance-pct 2.0
  
  # С входом по откату (ожидание отката к EMA9 или -0.3% от high)
  python backtest_ml_strategy.py --model ml_models/rf_BTCUSDT_15_15m.pkl --pullback --pullback-ema-period 9 --pullback-pct 0.3 --pullback-max-bars 3
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Путь к файлу модели')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Торговый символ (по умолчанию: BTCUSDT)')
    parser.add_argument('--days', type=int, default=30,
                       help='Количество дней для бэктеста (по умолчанию: 30)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Таймфрейм (по умолчанию: 15m)')
    parser.add_argument('--balance', type=float, default=100.0,
                       help='Начальный баланс (по умолчанию: 100.0)')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Риск на сделку (по умолчанию: 0.02 = 2%%)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Плечо (по умолчанию: 10)')
    parser.add_argument('--atr-filter', action='store_true',
                       help='Включить фильтр волатильности по ATR 1h (вход только в диапазоне)')
    parser.add_argument('--atr-min', type=float, default=None,
                       help='Минимальный ATR 1h в %% (по умолчанию из конфига или 0.3)')
    parser.add_argument('--atr-max', type=float, default=None,
                       help='Максимальный ATR 1h в %% (по умолчанию из конфига или 2.0)')
    parser.add_argument('--partial-tp', action='store_true',
                       help='Включить режим частичного TP (breakeven + trailing)')
    parser.add_argument('--partial-tp-pct', type=float, default=0.015,
                       help='Порог активации breakeven в %% (по умолчанию: 1.5%%)')
    parser.add_argument('--trailing-activation-pct', type=float, default=0.03,
                       help='Порог активации trailing stop в %% (по умолчанию: 3.0%%)')
    parser.add_argument('--trailing-distance-pct', type=float, default=0.02,
                       help='Расстояние trailing stop в %% (по умолчанию: 2.0%%)')
    parser.add_argument('--pullback', action='store_true',
                       help='Включить вход по откату (ожидание отката к EMA или уровню)')
    parser.add_argument('--pullback-ema-period', type=int, default=9,
                       help='Период EMA для отката (по умолчанию: 9, можно 20)')
    parser.add_argument('--pullback-pct', type=float, default=0.3,
                       help='Процент отката от high/low сигнальной свечи (по умолчанию: 0.3%%)')
    parser.add_argument('--pullback-max-bars', type=int, default=3,
                       help='Максимальная задержка входа в свечах (по умолчанию: 3)')
    
    args = parser.parse_args()
    
    # Параметры фильтра ATR из конфига, если не заданы в CLI
    try:
        settings = load_settings()
        atr_filter_enabled = args.atr_filter or settings.ml_strategy.atr_filter_enabled
        atr_min_pct = args.atr_min if args.atr_min is not None else settings.ml_strategy.atr_min_pct
        atr_max_pct = args.atr_max if args.atr_max is not None else settings.ml_strategy.atr_max_pct
    except Exception:
        atr_filter_enabled = args.atr_filter
        atr_min_pct = args.atr_min if args.atr_min is not None else 0.3
        atr_max_pct = args.atr_max if args.atr_max is not None else 2.0
    
    # Запускаем точный бэктест
    metrics = run_exact_backtest(
        model_path=args.model,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        atr_filter_enabled=atr_filter_enabled,
        atr_min_pct=atr_min_pct,
        atr_max_pct=atr_max_pct,
        partial_tp_enabled=args.partial_tp,
        partial_tp_pct=args.partial_tp_pct / 100.0 if args.partial_tp_pct >= 1.0 else args.partial_tp_pct,
        trailing_activation_pct=args.trailing_activation_pct / 100.0 if args.trailing_activation_pct >= 1.0 else args.trailing_activation_pct,
        trailing_distance_pct=args.trailing_distance_pct / 100.0 if args.trailing_distance_pct >= 1.0 else args.trailing_distance_pct,
        pullback_enabled=args.pullback,
        pullback_ema_period=args.pullback_ema_period,
        pullback_pct=args.pullback_pct / 100.0 if args.pullback_pct >= 1.0 else args.pullback_pct,
        pullback_max_bars=args.pullback_max_bars,
    )
    
    if metrics:
        print(f"\n✅ Точный бэктест завершен!")
        print(f"   Результаты показывают КАК стратегия работает на самом деле")
        
        # Финальный вердикт
        tradable_count = metrics.long_signals + metrics.short_signals
        if (metrics.signals_with_correct_sl_pct >= 90 and 
            (tradable_count == 0 or metrics.signals_with_tp_sl_pct >= 90) and
            metrics.total_trades > 0):
            print(f"\n🎯 СТРАТЕГИЯ ПРОШЛА ПРОВЕРКУ")
            print(f"   Можно тестировать на сервере")
        else:
            print(f"\n🚫 СТРАТЕГИЯ НЕ ПРОШЛА ПРОВЕРКУ")
            print(f"   Исправьте ошибки перед тестированием на сервере")
    else:
        print(f"\n❌ Бэктест не удался!")
        sys.exit(1)

def run_ml_backtest(*args, **kwargs):
    """Псевдоним для run_exact_backtest для обратной совместимости."""
    return run_exact_backtest(*args, **kwargs)
if __name__ == "__main__":
    main()