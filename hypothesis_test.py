
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal, Bias

# --- Definitions from backtest_ml_strategy.py ---

class ExitReason(Enum):
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TIME_LIMIT = "TIME_LIMIT"
    OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"
    MARGIN_CALL = "MARGIN_CALL"
    TRAILING_STOP = "TRAILING_STOP"
    END_OF_BACKTEST = "END_OF_BACKTEST"

@dataclass
class Trade:
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
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    profit_factor: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float

class HypothesisSimulator:
    def __init__(
        self,
        strategy_mode: str, # 'dynamic', 'fixed_25_10', 'fixed_15_5'
        initial_balance: float = 100.0,
        leverage: int = 10,
        commission: float = 0.0006
    ):
        self.strategy_mode = strategy_mode
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.equity_curve: List[float] = [initial_balance]
        self.max_equity = initial_balance
        
        # Dynamic Strategy Settings (from config.py)
        self.breakeven_activation = 0.005 # 0.5%
        self.breakeven_sl = 0.001        # 0.1% profit
        self.trailing_activation = 0.007 # 0.7%
        self.trailing_distance = 0.003   # 0.3%
        
    def open_position(self, signal: Signal, current_time: datetime, symbol: str) -> bool:
        if self.current_position is not None:
            return False
            
        # Determine TP/SL based on strategy mode
        price = signal.price
        
        if self.strategy_mode == 'fixed_25_10':
            if signal.action == Action.LONG:
                tp = price * 1.25
                sl = price * 0.90
            else:
                tp = price * 0.75
                sl = price * 1.10
        elif self.strategy_mode == 'fixed_15_5':
            if signal.action == Action.LONG:
                tp = price * 1.15
                sl = price * 0.95
            else:
                tp = price * 0.85
                sl = price * 1.05
        else: # dynamic
            # Use signal's TP/SL (which usually come from config)
            # Default fallback if signal missing them
            if signal.action == Action.LONG:
                tp = signal.take_profit or (price * 1.015) # 1.5% default
                sl = signal.stop_loss or (price * 0.97)   # 3% default
            else:
                tp = signal.take_profit or (price * 0.985)
                sl = signal.stop_loss or (price * 1.03)

        # Position Sizing (Fixed $100 margin logic)
        base_order_usd = 100.0
        margin_required = base_order_usd / self.leverage
        position_size_usd = base_order_usd
        
        if margin_required > self.balance:
            return False
            
        self.balance -= margin_required
        
        self.current_position = Trade(
            entry_time=current_time,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            action=signal.action,
            size_usd=position_size_usd,
            pnl=0.0,
            pnl_pct=0.0,
            entry_reason=signal.reason,
            exit_reason=None,
            symbol=symbol,
            confidence=signal.indicators_info.get('confidence', 0) if signal.indicators_info else 0,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop=None
        )
        return True

    def check_exit(self, current_time: datetime, current_price: float, high: float, low: float) -> bool:
        if self.current_position is None:
            return False
            
        pos = self.current_position
        
        # Update MFE/MAE
        if pos.action == Action.LONG:
            mfe = (high - pos.entry_price) / pos.entry_price
            mae = (low - pos.entry_price) / pos.entry_price
        else:
            mfe = (pos.entry_price - low) / pos.entry_price
            mae = (pos.entry_price - high) / pos.entry_price
            
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, mfe)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, mae)
        
        # --- Dynamic Logic (Breakeven + Trailing) ---
        if self.strategy_mode == 'dynamic':
            if pos.action == Action.LONG:
                # Breakeven
                if mfe >= self.breakeven_activation:
                    new_sl = pos.entry_price * (1 + self.breakeven_sl)
                    if new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                
                # Trailing Stop
                if mfe >= self.trailing_activation:
                    # Trailing Logic: High - distance
                    potential_sl = high * (1 - self.trailing_distance)
                    if potential_sl > pos.stop_loss:
                         pos.stop_loss = potential_sl
                         
            else: # SHORT
                # Breakeven
                if mfe >= self.breakeven_activation:
                    new_sl = pos.entry_price * (1 - self.breakeven_sl)
                    if new_sl < pos.stop_loss:
                        pos.stop_loss = new_sl
                        
                # Trailing Stop
                if mfe >= self.trailing_activation:
                    # Trailing Logic: Low + distance
                    potential_sl = low * (1 + self.trailing_distance)
                    if potential_sl < pos.stop_loss:
                        pos.stop_loss = potential_sl

        # --- Standard TP/SL Check ---
        if pos.action == Action.LONG:
            if low <= pos.stop_loss:
                exit_price = min(pos.stop_loss, current_price) # Slippage sim simplified
                # Check if it was a trailing stop hit
                reason = ExitReason.STOP_LOSS
                # Heuristic: if SL is above entry, it's a trailing stop or breakeven
                if pos.stop_loss > pos.entry_price:
                    reason = ExitReason.TRAILING_STOP
                self.close_position(current_time, exit_price, reason)
                return True
            elif high >= pos.take_profit:
                exit_price = max(pos.take_profit, current_price)
                self.close_position(current_time, exit_price, ExitReason.TAKE_PROFIT)
                return True
        else: # SHORT
            if high >= pos.stop_loss:
                exit_price = max(pos.stop_loss, current_price)
                if pos.stop_loss < pos.entry_price:
                    reason = ExitReason.TRAILING_STOP
                else:
                    reason = ExitReason.STOP_LOSS
                self.close_position(current_time, exit_price, reason)
                return True
            elif low <= pos.take_profit:
                exit_price = min(pos.take_profit, current_price)
                self.close_position(current_time, exit_price, ExitReason.TAKE_PROFIT)
                return True
                
        return False

    def close_position(self, exit_time: datetime, exit_price: float, exit_reason: ExitReason):
        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        
        if pos.action == Action.LONG:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            
        margin_used = pos.size_usd / self.leverage
        pnl_usd_raw = margin_used * pnl_pct * self.leverage
        
        # Commission
        notional = pos.size_usd + (pos.size_usd / pos.entry_price * exit_price)
        comm = notional * self.commission
        
        pnl_usd = pnl_usd_raw - comm
        
        self.balance += margin_used + pnl_usd
        pos.pnl = pnl_usd
        pos.pnl_pct = (pnl_usd / margin_used) * 100
        
        self.trades.append(pos)
        self.current_position = None
        
        if self.balance > self.max_equity:
            self.max_equity = self.balance
            
    def get_metrics(self) -> BacktestMetrics:
        if not self.trades:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = self.balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        win_rate = len(winning) / len(self.trades) * 100
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        peak = self.initial_balance
        max_dd = 0
        equity = self.initial_balance
        # Reconstruct equity curve roughly
        for t in self.trades:
            equity += t.pnl  # Simplified, strictly it's sequential
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
            
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        
        return BacktestMetrics(
            len(self.trades), win_rate, total_pnl, total_pnl_pct, 
            profit_factor, max_dd, avg_win, avg_loss
        )

def run_simulation(model_path, symbol, days, strategy_mode):
    print(f"\n--- Running Simulation: {strategy_mode} ---")
    
    settings = load_settings()
    client = BybitClient(settings.api)
    
    # Load Data
    interval = "15"
    limit = days * 96
    df = client.get_kline_df(symbol, interval, limit=limit)
    df = prepare_with_indicators(df)
    
    # Load Strategy
    strategy = MLStrategy(
        model_path=model_path,
        confidence_threshold=settings.ml_strategy.confidence_threshold,
        min_signal_strength=settings.ml_strategy.min_signal_strength
    )
    
    # Feature Engineering
    df_features = strategy.feature_engineer.create_technical_indicators(df)
    
    sim = HypothesisSimulator(strategy_mode, initial_balance=1000.0)
    
    # Run Loop
    min_window = 200
    for i in range(min_window, len(df_features)):
        row = df_features.iloc[i]
        df_slice = df_features.iloc[:i+1]
        
        try:
            signal = strategy.generate_signal(
                row=row, df=df_slice, has_position=None, 
                current_price=row['close'], leverage=10, 
                skip_feature_creation=True
            )
        except Exception:
            continue
            
        # Check Exit
        sim.check_exit(df_features.index[i], row['close'], row['high'], row['low'])
        
        # Check Entry
        if signal.action in [Action.LONG, Action.SHORT]:
             sim.open_position(signal, df_features.index[i], symbol)
             
    metrics = sim.get_metrics()
    print(f"Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print(f"PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:.2f}%)")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    
    modes = ['dynamic', 'fixed_25_10', 'fixed_15_5']
    results = {}
    
    for mode in modes:
        results[mode] = run_simulation(args.model, args.symbol, args.days, mode)
        
    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'Mode':<15} | {'PnL %':<10} | {'Win Rate':<10} | {'Trades':<8} | {'Drawdown':<10}")
    print("-" * 65)
    for mode, m in results.items():
        print(f"{mode:<15} | {m.total_pnl_pct:>9.2f}% | {m.win_rate:>9.2f}% | {m.total_trades:>8} | {m.max_drawdown_pct:>9.2f}%")
