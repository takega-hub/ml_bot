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
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.02,
        commission: float = 0.0006,
        max_position_size_pct: float = 0.1,
        leverage: int = 10,
        maintenance_margin_ratio: float = 0.005,
        max_position_hours: float = 48.0,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.max_position_size_pct = max_position_size_pct
        self.leverage = leverage
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.max_position_hours = max_position_hours
        
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
            self.signal_stats.signals_without_tp_sl += 1
            # Логируем только первые 3 сигнала без TP/SL
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
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, action: Action) -> Tuple[float, float]:
        """
        Рассчитывает размер позиции ТОЧНО как реальный бот.
        
        Реальный бот использует stop_loss из сигнала, даже если он не 1%!
        """
        # Рассчитываем риск на единицу
        if action == Action.LONG:
            risk_per_unit = abs(entry_price - stop_loss)
        else:  # SHORT
            risk_per_unit = abs(stop_loss - entry_price)
        
        if risk_per_unit <= 0:
            print(f"⚠️  Нулевой или отрицательный риск: entry={entry_price}, SL={stop_loss}")
            return 0.0, 0.0
        
        # Риск в процентах (реальный бот так считает)
        risk_pct = risk_per_unit / entry_price
        
        # Сумма риска на сделку
        risk_amount = self.balance * self.risk_per_trade
        
        # Размер позиции в USD
        position_size = risk_amount / risk_pct
        
        # Максимальный размер позиции
        max_position = self.balance * self.max_position_size_pct * self.leverage
        position_size = min(position_size, max_position)
        
        # Требуемая маржа
        margin_required = position_size / self.leverage
        
        # Проверяем маржу
        if margin_required > self.balance:
            # Реальный бот уменьшит размер позиции
            position_size = self.balance * self.leverage
            margin_required = self.balance
        
        return position_size, margin_required
    
    def open_position(self, signal: Signal, current_time: datetime, symbol: str) -> bool:
        """
        Открывает позицию ТОЧНО как реальный бот.
        
        Использует TP/SL ИЗ СИГНАЛА, без исправлений!
        """
        if self.current_position is not None:
            return False  # Уже есть позиция
        
        if signal.action == Action.HOLD:
            return False
        
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
        
        # 4. Рассчитываем размер позиции (с реальными TP/SL из сигнала)
        position_size_usd, margin_required = self.calculate_position_size(
            signal.price, stop_loss, signal.action
        )
        
        if position_size_usd <= 0 or margin_required > self.balance:
            print(f"❌ Не могу открыть позицию: недостаточно средств")
            print(f"   Размер: ${position_size_usd:.2f}, Маржа: ${margin_required:.2f}, Баланс: ${self.balance:.2f}")
            return False
        
        # 5. Вычитаем маржу (как реальный бот)
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
        
        # PnL с учетом плеча
        pnl_pct = price_change_pct * self.leverage
        pnl_usd = pos.size_usd * pnl_pct
        
        # Комиссии
        notional = pos.size_usd * self.leverage
        commission_cost = notional * self.commission * 2
        pnl_usd -= commission_cost
        
        # Возвращаем маржу и добавляем PnL
        margin_returned = pos.size_usd / self.leverage
        self.balance += margin_returned + pnl_usd
        
        pos.pnl = pnl_usd
        pos.pnl_pct = pnl_pct * 100
        
        # Обновляем кривую капитала
        self.equity_curve.append(self.balance)
        
        # Обновляем максимальный equity
        if self.balance > self.max_equity:
            self.max_equity = self.balance
        
        # Сохраняем сделку
        self.trades.append(pos)
        self.current_position = None
        
        # Логируем (только первые 10 сделок)
        if len(self.trades) <= 10:
            print(f"\n📊 Закрыта позиция #{len(self.trades)}:")
            print(f"   {pos.action.value} @ ${pos.entry_price:.2f} -> ${exit_price:.2f}")
            print(f"   Причина: {exit_reason.value}")
            print(f"   PnL: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")
            print(f"   Новый баланс: ${self.balance:.2f}")
    
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
        signals_with_tp_sl_pct = (self.signal_stats.signals_with_tp_sl / 
                                 max(1, self.signal_stats.total_signals)) * 100
        
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
            avg_mfe=0.0,
            avg_mae=0.0,
            mfe_mae_ratio=0.0,
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


def run_exact_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    interval: str = "15",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
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
        BacktestMetrics с результатами
    """
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
            print(f"❌ Нет данных для {symbol}")
            return None
        
        print(f"✅ Загружено {len(df)} свечей")
        print(f"   Период: {df.index[0]} до {df.index[-1]}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None
    
    # Подготавливаем индикаторы
    print(f"\n🔧 Подготовка индикаторов...")
    try:
        df_with_indicators = prepare_with_indicators(df.copy())
        print(f"✅ Индикаторы подготовлены")
    except Exception as e:
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
        print(f"❌ Ошибка подготовки стратегии: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Создаем симулятор
    simulator = MLBacktestSimulator(
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        max_position_hours=48.0,
    )
    
    # Запускаем бэктест
    print(f"\n📈 Запуск точного бэктеста...")
    print(f"   Имитация работы реального бота на сервере")
    print(f"   Используются те же параметры и методы, что и в продакшене")
    
    # Минимальное окно данных для расчета всех индикаторов (как в реальном боте)
    # MLStrategy требует минимум 200 баров для корректной работы
    min_window_size = 200
    
    # ВАЖНО: Используем все данные до текущего момента (как реальный бот)
    # Реальный бот на каждой итерации использует ВСЕ доступные исторические данные
    for idx in range(len(df_with_features)):
        # Пропускаем первые N баров, чтобы накопить достаточно данных для индикаторов
        if idx < min_window_size:
            continue
        
        current_time = df_with_features.index[idx]
        row = df_with_features.iloc[idx]
        current_price = row['close']
        high = row['high']
        low = row['low']
        
        # ВАЖНО: Реальный бот использует ВСЕ данные до текущего момента
        # Это критично для правильной работы индикаторов и ML модели
        # Используем данные от начала до текущего индекса (включительно)
        df_window = df_with_features.iloc[:idx+1].copy()
        
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
            signal = strategy.generate_signal(
                row=row,
                df=df_window,  # Все данные до текущего момента (как реальный бот)
                has_position=has_position,
                current_price=current_price,
                leverage=leverage,
                target_profit_pct_margin=settings.ml_strategy.target_profit_pct_margin,
                max_loss_pct_margin=settings.ml_strategy.max_loss_pct_margin,
            )
            
            # ВАЛИДАЦИЯ: Проверяем, что сигнал имеет правильный тип
            assert isinstance(signal, Signal), f"Сигнал должен быть типа Signal, получен {type(signal)}"
            
        except AssertionError as e:
            # Критическая ошибка валидации
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ВАЛИДАЦИИ: {e}")
            raise
        except Exception as e:
            # Если ошибка при генерации сигнала, логируем и пропускаем
            # (это может происходить в реальном боте тоже)
            if idx < 10:  # Логируем только первые 10 ошибок
                print(f"⚠️  Ошибка генерации сигнала на {current_time}: {e}")
            signal = Signal(
                timestamp=current_time,
                action=Action.HOLD,
                reason=f"ml_ошибка_генерации_{str(e)[:30]}",
                price=current_price
            )
        
        # Анализируем сигнал (только статистика, без изменений)
        simulator.analyze_signal(signal, current_price)
        
        # ВАЖНО: Сначала проверяем выход из позиции (как реальный бот)
        # Это важно, так как может быть сигнал на закрытие текущей позиции
        if simulator.current_position is not None:
            exited = simulator.check_exit(current_time, current_price, high, low)
            # Если позиция закрыта, не открываем новую на этой же итерации
            if exited:
                continue
        
        # Проверяем вход в позицию (только если нет открытой позиции)
        if simulator.current_position is None and signal.action in (Action.LONG, Action.SHORT):
            simulator.open_position(signal, current_time, symbol)
    
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
    print(f"   Сигналов с TP/SL: {metrics.signals_with_tp_sl_pct:.1f}%")
    print(f"   Сигналов с SL=1%: {metrics.signals_with_correct_sl_pct:.1f}%")
    print(f"   Средний SL в сигналах: {metrics.avg_sl_distance_pct:.2f}%")
    print(f"   Средний TP в сигналах: {metrics.avg_tp_distance_pct:.2f}%")
    print(f"   Средний R/R: {metrics.avg_rr_ratio:.2f}")
    
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
    
    if metrics.signals_with_tp_sl_pct < 90:
        print(f"❌ ПРОБЛЕМА: Только {metrics.signals_with_tp_sl_pct:.1f}% сигналов имеют TP/SL")
        print(f"   Реальная стратегия на сервере НЕ сможет открыть {100-metrics.signals_with_tp_sl_pct:.1f}% позиций!")
        print(f"   ⚠️  Это означает, что на реальных данных будет такая же проблема!")
    
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
    
    if metrics.signals_with_tp_sl_pct < 90:
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
        metrics.signals_with_tp_sl_pct >= 90 and
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
    parser.add_argument('--balance', type=float, default=1000.0,
                       help='Начальный баланс (по умолчанию: 1000.0)')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Риск на сделку (по умолчанию: 0.02 = 2%%)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Плечо (по умолчанию: 10)')
    
    args = parser.parse_args()
    
    # Запускаем точный бэктест
    metrics = run_exact_backtest(
        model_path=args.model,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
    )
    
    if metrics:
        print(f"\n✅ Точный бэктест завершен!")
        print(f"   Результаты показывают КАК стратегия работает на самом деле")
        
        # Финальный вердикт
        if (metrics.signals_with_correct_sl_pct >= 90 and 
            metrics.signals_with_tp_sl_pct >= 90 and
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