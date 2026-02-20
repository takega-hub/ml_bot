#!/usr/bin/env python3
"""
Тестирование гипотез стратегий с разными TP/SL параметрами
С реальными историческими данными и ML моделью
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
import sys
from pathlib import Path

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bot.config import load_settings
    from bot.exchange.bybit_client import BybitClient
    from bot.indicators import prepare_with_indicators
    from bot.ml.strategy_ml import MLStrategy
    from bot.strategy import Action, Signal
    HAS_BOT_MODULES = True
except ImportError as e:
    print(f"Warning: Could not import bot modules: {e}")
    print("Using simplified test data instead")
    HAS_BOT_MODULES = False
    # Определяем Action локально если модули недоступны
    from enum import Enum
    class Action(Enum):
        LONG = "LONG"
        SHORT = "SHORT"
        HOLD = "HOLD"

# Убираем локальное определение Action, используем из bot.strategy
# class Action(Enum):
#     LONG = "LONG"
#     SHORT = "SHORT"
#     HOLD = "HOLD"

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    action: Action = Action.HOLD
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion

@dataclass
class BacktestMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

class HypothesisSimulator:
    def __init__(self, strategy_mode: str, initial_balance: float = 100.0, leverage: int = 10, commission: float = 0.0006):
        self.strategy_mode = strategy_mode
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        
        # Dynamic strategy settings
        if strategy_mode == 'dynamic_partial_tp':
            # Режим частичного TP: breakeven при 1.5%, trailing при 3.0%
            self.breakeven_activation = 0.015  # 1.5% - первая цель для breakeven
            self.breakeven_sl = 0.001         # 0.1% profit (чуть выше входа)
            self.trailing_activation = 0.030   # 3.0% - активация trailing
            self.trailing_distance = 0.020     # 2.0% - расстояние trailing
        else:
            # Стандартный dynamic режим
            self.breakeven_activation = 0.020  # 2.0%
            self.breakeven_sl = 0.010         # 1.0% profit
            self.trailing_activation = 0.050   # 5.0%
            self.trailing_distance = 0.020     # 2.0%
        
        # Fixed strategy settings
        if strategy_mode == 'fixed_25_10':
            self.fixed_tp_pct = 0.25  # 25%
            self.fixed_sl_pct = 0.10  # 10%
        elif strategy_mode == 'fixed_15_5':
            self.fixed_tp_pct = 0.15  # 15%
            self.fixed_sl_pct = 0.05  # 5%
        elif strategy_mode == 'fixed_10_10':
            self.fixed_tp_pct = 0.10  # 10%
            self.fixed_sl_pct = 0.10  # 10%
        
        self.current_position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve = []

    def open_position(self, action: Action, entry_time: datetime, entry_price: float):
        if self.current_position is not None:
            return False
        
        self.current_position = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            action=action
        )
        
        # Set initial SL/TP based on strategy
        if self.strategy_mode in ('dynamic', 'dynamic_partial_tp'):
            # Dynamic uses ATR-based SL/TP with breakeven and trailing
            # Более широкие параметры для соответствия консервативной стратегии
            if action == Action.LONG:
                self.current_position.stop_loss = entry_price * (1 - 0.030)  # 3.0% SL
                self.current_position.take_profit = entry_price * (1 + 0.060)  # 6.0% TP
            else:
                self.current_position.stop_loss = entry_price * (1 + 0.030)  # 3.0% SL
                self.current_position.take_profit = entry_price * (1 - 0.060)  # 6.0% TP
        else:
            # Fixed strategies
            if action == Action.LONG:
                self.current_position.stop_loss = entry_price * (1 - self.fixed_sl_pct)
                self.current_position.take_profit = entry_price * (1 + self.fixed_tp_pct)
            else:
                self.current_position.stop_loss = entry_price * (1 + self.fixed_sl_pct)
                self.current_position.take_profit = entry_price * (1 - self.fixed_tp_pct)
        
        return True

    def check_exit(self, current_time: datetime, current_price: float, high: float, low: float) -> bool:
        if self.current_position is None:
            return False
        
        # Track MFE/MAE
        if self.current_position.action == Action.LONG:
            mfe_pct = (high / self.current_position.entry_price - 1)
            mae_pct = (1 - low / self.current_position.entry_price)
        else:
            mfe_pct = (1 - low / self.current_position.entry_price)
            mae_pct = (high / self.current_position.entry_price - 1)
        
        self.current_position.mfe = max(self.current_position.mfe, mfe_pct)
        self.current_position.mae = max(self.current_position.mae, mae_pct)
        
        # Check fixed TP/SL exit conditions
        if self.strategy_mode not in ('dynamic', 'dynamic_partial_tp'):
            if self.current_position.action == Action.LONG:
                if current_price >= self.current_position.take_profit:
                    self.close_position(current_time, current_price, "TP")
                    return True
                elif current_price <= self.current_position.stop_loss:
                    self.close_position(current_time, current_price, "SL")
                    return True
            else:
                if current_price <= self.current_position.take_profit:
                    self.close_position(current_time, current_price, "TP")
                    return True
                elif current_price >= self.current_position.stop_loss:
                    self.close_position(current_time, current_price, "SL")
                    return True
        else:
            # Dynamic strategy logic with breakeven and trailing
            if self.current_position.action == Action.LONG:
                # Breakeven logic
                if mfe_pct >= self.breakeven_activation:
                    new_sl = self.current_position.entry_price * (1 + self.breakeven_sl)
                    if new_sl > self.current_position.stop_loss:
                        self.current_position.stop_loss = new_sl
                
                # Trailing stop logic
                if mfe_pct >= self.trailing_activation:
                    potential_sl = high * (1 - self.trailing_distance)
                    if potential_sl > self.current_position.stop_loss:
                        self.current_position.stop_loss = potential_sl
                
                # Check exit conditions
                if current_price >= self.current_position.take_profit:
                    self.close_position(current_time, current_price, "TP")
                    return True
                elif current_price <= self.current_position.stop_loss:
                    self.close_position(current_time, current_price, "SL")
                    return True
            else:
                # Similar logic for SHORT positions
                if mfe_pct >= self.breakeven_activation:
                    new_sl = self.current_position.entry_price * (1 - self.breakeven_sl)
                    if new_sl < self.current_position.stop_loss:
                        self.current_position.stop_loss = new_sl
                
                if mfe_pct >= self.trailing_activation:
                    potential_sl = low * (1 + self.trailing_distance)
                    if potential_sl < self.current_position.stop_loss:
                        self.current_position.stop_loss = potential_sl
                
                if current_price <= self.current_position.take_profit:
                    self.close_position(current_time, current_price, "TP")
                    return True
                elif current_price >= self.current_position.stop_loss:
                    self.close_position(current_time, current_price, "SL")
                    return True
        
        return False

    def close_position(self, exit_time: datetime, exit_price: float, reason: str = "manual"):
        if self.current_position is None:
            return
        
        # Calculate PnL
        entry_price = self.current_position.entry_price
        if self.current_position.action == Action.LONG:
            pnl_pct = (exit_price / entry_price - 1) - self.commission * 2
        else:
            pnl_pct = (1 - exit_price / entry_price) - self.commission * 2
        
        pnl_usd = self.initial_balance * pnl_pct
        
        # Update balance
        self.balance += pnl_usd
        
        # Complete the trade
        self.current_position.exit_time = exit_time
        self.current_position.exit_price = exit_price
        self.current_position.pnl = pnl_usd
        self.current_position.pnl_pct = pnl_pct
        
        self.trades.append(self.current_position)
        self.equity_curve.append((exit_time, self.balance))
        self.current_position = None

def load_model_predictions(model_path: str, symbol: str, days: int = 30, interval: str = "15m"):
    """Загружаем предсказания модели с реальными историческими данными"""
    # Переменные для подсчета сигналов (объявляем заранее)
    hold_count = 0
    long_count = 0
    short_count = 0
    
    if HAS_BOT_MODULES:
        try:
            print(f"📊 Загрузка реальных исторических данных для {symbol}...")
            
            # Загружаем настройки
            settings = load_settings()
            
            # Создаем клиент
            client = BybitClient(settings.api)
            
            # Получаем исторические данные
            if interval.endswith("m"):
                bybit_interval = interval[:-1]
            else:
                bybit_interval = interval
            
            interval_min = int(bybit_interval)
            candles_per_day = (24 * 60) // interval_min
            total_candles = days * candles_per_day
            
            df = client.get_kline_df(symbol, bybit_interval, limit=total_candles)
            
            if df.empty:
                print(f"❌ Нет данных для {symbol}")
                return None
            
            print(f"✅ Загружено {len(df)} свечей")
            print(f"   Период: {df.index[0]} до {df.index[-1]}")
            
            # Создаем ML стратегию для генерации сигналов
            strategy = MLStrategy(
                model_path=model_path,
                confidence_threshold=0.35,
                min_signal_strength="слабое",
                stability_filter=True,
                min_signals_per_day=1,
                max_signals_per_day=20
            )
            
            # Подготавливаем индикаторы через feature_engineer стратегии
            df_with_indicators = df.copy()
            if "timestamp" in df_with_indicators.columns:
                df_with_indicators = df_with_indicators.set_index("timestamp")
            df_with_indicators = strategy.feature_engineer.create_technical_indicators(df_with_indicators)
            
            # Генерируем сигналы с использованием реальной ML модели
            predictions = []
            confidences = []
            
            print(f"🤖 Генерация сигналов с использованием ML модели...")
            print(f"   Всего баров: {len(df_with_indicators)}")
            print(f"   Колонки: {list(df_with_indicators.columns[:10])}...")
            
            # Проверяем первые данные
            print(f"   Первый ряд: {df_with_indicators.iloc[0]['close']}")
            print(f"   Последний ряд: {df_with_indicators.iloc[-1]['close']}")
            
            # Считаем сколько HOLD vs LONG/SHORT - объявляем заранее
            hold_count = 0
            long_count = 0
            short_count = 0
            debug_count = 0  # Для отладки
            
            for i in range(len(df_with_indicators)):
                if i < 50:
                    # Недостаточно данных для модели
                    predictions.append(Action.HOLD)
                    confidences.append(0.3)
                    continue
                
                # Берем данные до текущей свечи (имитация реального времени)
                df_up_to_now = df_with_indicators.iloc[:i+1].copy()
                current_row = df_up_to_now.iloc[-1]
                current_price = current_row['close']
                
                # Генерируем сигнал с правильными параметрами
                signal = strategy.generate_signal(
                    row=current_row,
                    df=df_up_to_now,
                    has_position=None,  # Нет открытой позиции
                    current_price=current_price,
                    leverage=10,
                    target_profit_pct_margin=25.0,
                    max_loss_pct_margin=10.0,
                    skip_feature_creation=True  # Фичи уже созданы
                )
                
                # Преобразуем Action в наш enum
                action_val = signal.action
                
                if action_val == Action.LONG:
                    prediction = Action.LONG
                    long_count += 1
                elif action_val == Action.SHORT:
                    prediction = Action.SHORT
                    short_count += 1
                else:
                    prediction = Action.HOLD
                    hold_count += 1
                
                predictions.append(prediction)
                confidences.append(signal.indicators_info.get('confidence', 0.5) if signal.indicators_info else 0.5)
                
                # Логируем первые 10 сигналов для отладки
                if i < 60 and i >= 50:
                    print(f"   [{i}] Action: {signal.action.value}, Confidence: {signal.indicators_info.get('confidence', 0.5) if signal.indicators_info else 0.5:.2f}, Reason: {signal.reason}")
            
            # Создаем DataFrame с результатами
            result_df = pd.DataFrame({
                'timestamp': df_with_indicators.index,
                'open': df_with_indicators['open'],
                'high': df_with_indicators['high'],
                'low': df_with_indicators['low'],
                'close': df_with_indicators['close'],
                'prediction': predictions,
                'confidence': confidences
            })
            
            print(f"✅ Сгенерировано {len(result_df)} сигналов")
            print(f"   LONG: {long_count}, SHORT: {short_count}, HOLD: {hold_count}")
            return result_df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки реальных данных: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Используем тестовые данные...")
            # Продолжаем с тестовыми данными
    
    # Если модули бота не доступны или произошла ошибка, используем тестовые данные
    print("📊 Используем тестовые данные...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Генерируем тестовые данные
        dates = pd.date_range(start=start_date, end=end_date, freq='15min')
        n_periods = len(dates)
        
        # Случайные цены вокруг 50000
        base_price = 50000
        prices = base_price + np.random.randn(n_periods) * 100
        
        # Случайные предсказания
        predictions = np.random.choice([Action.LONG, Action.SHORT, Action.HOLD], 
                                      size=n_periods, p=[0.4, 0.4, 0.2])
        confidences = np.random.uniform(0.3, 0.9, n_periods)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 50, n_periods),
            'low': prices - np.random.uniform(0, 50, n_periods),
            'close': prices,
            'prediction': predictions,
            'confidence': confidences
        })
        
        return df
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def run_backtest(strategy_mode: str, predictions_df: pd.DataFrame, initial_balance: float = 100.0):
    """Запускаем бектест для конкретной стратегии"""
    simulator = HypothesisSimulator(strategy_mode, initial_balance)
    
    for _, row in predictions_df.iterrows():
        timestamp = row['timestamp']
        current_price = row['close']
        high = row['high']
        low = row['low']
        prediction = row['prediction']
        confidence = row['confidence']
        
        # Проверяем выход из текущей позиции - сначала по high/low чтобы поймать TP/SL внутри свечи
        if simulator.current_position is not None:
            # Сначала проверяем high для LONG (TP) и low для SHORT (SL)
            if simulator.current_position.action == Action.LONG:
                # Проверяем TP по high
                if high >= simulator.current_position.take_profit:
                    exit_price = simulator.current_position.take_profit
                    simulator.close_position(timestamp, exit_price, "TP")
                # Проверяем SL по low
                elif low <= simulator.current_position.stop_loss:
                    exit_price = simulator.current_position.stop_loss
                    simulator.close_position(timestamp, exit_price, "SL")
                else:
                    # Проверяем динамический выход (breakeven/trailing)
                    simulator.check_exit(timestamp, current_price, high, low)
            else:  # SHORT
                # Проверяем TP по low
                if low <= simulator.current_position.take_profit:
                    exit_price = simulator.current_position.take_profit
                    simulator.close_position(timestamp, exit_price, "TP")
                # Проверяем SL по high
                elif high >= simulator.current_position.stop_loss:
                    exit_price = simulator.current_position.stop_loss
                    simulator.close_position(timestamp, exit_price, "SL")
                else:
                    # Проверяем динамический выход
                    simulator.check_exit(timestamp, current_price, high, low)
        
        # Если нет открытой позиции и хороший сигнал
        if simulator.current_position is None and prediction != Action.HOLD and confidence > 0.35:
            simulator.open_position(prediction, timestamp, current_price)
    
    # Закрываем все открытые позиции в конце
    if simulator.current_position is not None:
        simulator.close_position(
            predictions_df['timestamp'].iloc[-1],
            predictions_df['close'].iloc[-1],
            "end_of_period"
        )
    
    return simulator

def calculate_metrics(simulator: HypothesisSimulator) -> BacktestMetrics:
    """Рассчитываем метрики бектеста"""
    trades = simulator.trades
    if not trades:
        return BacktestMetrics()
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = total_trades - winning_trades
    win_rate_pct = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl_pct = sum(t.pnl_pct for t in trades)
    
    # Profit factor
    total_profit = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Max drawdown
    equity_curve = [simulator.initial_balance]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.pnl)
    
    peak = equity_curve[0]
    max_drawdown_pct = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown_pct = (peak - equity) / peak * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    
    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate_pct,
        total_pnl_pct=total_pnl_pct,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct
    )

def main():
    parser = argparse.ArgumentParser(description='Test trading strategy hypotheses')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--interval', type=str, default='15m', help='Timeframe (default: 15m)')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Загружаем предсказания
    print(f"Loading predictions for {args.symbol}...")
    predictions_df = load_model_predictions(args.model, args.symbol, args.days, args.interval)
    
    if predictions_df is None:
        print("Failed to load predictions")
        return
    
    # Тестируем три стратегии
    strategies = ['dynamic', 'fixed_25_10', 'fixed_15_5', 'fixed_10_10']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        simulator = run_backtest(strategy, predictions_df, args.balance)
        metrics = calculate_metrics(simulator)
        results[strategy] = metrics
        
        print(f"  Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate_pct:.2f}%")
        print(f"  PnL: {metrics.total_pnl_pct:.2f}%")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    
    # Выводим сравнение
    print("\n=== FINAL COMPARISON ===")
    print(f"{'Mode':<15} | {'PnL %':<10} | {'Win Rate':<10} | {'Trades':<8} | {'Drawdown':<10}")
    print("-" * 65)
    
    for strategy in strategies:
        metrics = results[strategy]
        print(f"{strategy:<15} | {metrics.total_pnl_pct:>8.2f}% | {metrics.win_rate_pct:>8.2f}% | {metrics.total_trades:>7} | {metrics.max_drawdown_pct:>8.2f}%")

if __name__ == "__main__":
    main()