
import os
import sys
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import itertools

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("FAST_BACKTEST", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bot.config import load_settings
from bot.indicators import prepare_with_indicators
from bot.ml.strategy_ml import MLStrategy
from bot.strategy import Action
from backtest_ml_strategy import MLBacktestSimulator, ExitReason

# Reuse the evaluation logic
# We need to import these functions. Since backtest_tp_reentry_delay is a script, 
# importing it might run main if not guarded. I checked and it has if __name__ == "__main__".
from backtest_tp_reentry_delay import _eval_tp_reentry, _auto_find_history_csv, _load_history_from_csv

import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def run_simulation_fast(
    df_with_features: pd.DataFrame,
    strategy: MLStrategy,
    initial_balance: float,
    risk_per_trade: float,
    leverage: int,
    wait_candles: int,
    window_candles: int,
    min_pullback_pct: float,
    max_pullback_pct: float,
    breakout_buffer_pct: float,
    volume_factor: float,
    sr_lookback: int,
    trend_lookback: int,
    min_trend_slope: float,
    symbol: str
) -> Dict[str, Any]:
    
    settings = load_settings()
    simulator = MLBacktestSimulator(
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        max_position_hours=48.0,
    )
    simulator._margin_pct_balance = settings.risk.margin_pct_balance
    simulator._base_order_usd = settings.risk.base_order_usd if hasattr(settings.risk, "base_order_usd") else 100.0

    min_window_size = max(100, int(getattr(strategy.feature_engineer, "min_window_size", 100) or 100))
    total_bars = len(df_with_features)
    
    tp_guard = None
    tp_stats = {
        "wait_candles": int(wait_candles),
        "skipped_wait": 0,
        "skipped_criteria": 0,
        "allowed": 0,
        "expired": 0,
    }

    last_trade_count = 0
    window_size = 1200 
    
    # Iterate through bars
    sys.stderr.write(f"Simulating: wait={wait_candles}, pb={min_pullback_pct}...\n")
    for idx in range(min_window_size, total_bars):
        current_time = df_with_features.index[idx]
        row = df_with_features.iloc[idx]
        current_price = float(row["close"])

        # 1. Manage existing positions
        if simulator.current_position is not None:
            try:
                simulator.check_exit(
                    current_time=current_time,
                    current_price=current_price,
                    high=float(row["high"]),
                    low=float(row["low"]),
                )
            except Exception:
                pass

        # 2. Check for recent TP exit to set guard
        if len(simulator.trades) > last_trade_count:
            last = simulator.trades[-1]
            if last.exit_reason == ExitReason.TAKE_PROFIT and wait_candles > 0:
                tp_guard = {
                    "action": last.action,
                    "exit_idx": idx,
                    "exit_price": float(last.exit_price or current_price),
                    "wait_until_idx": idx + int(wait_candles),
                    "expire_idx": idx + max(0, int(window_candles)),
                }
            last_trade_count = len(simulator.trades)

        # 3. Check guard expiration
        if tp_guard is not None and idx > tp_guard["expire_idx"]:
            tp_stats["expired"] += 1
            tp_guard = None

        # 4. Generate Signal
        start_idx = max(0, idx - window_size + 1)
        df_window = df_with_features.iloc[start_idx: idx + 1]
        
        try:
            signal = strategy.generate_signal(
                row=row,
                df=df_window,
                has_position=None, 
                current_price=current_price,
                leverage=leverage,
                target_profit_pct_margin=settings.ml_strategy.target_profit_pct_margin,
                max_loss_pct_margin=settings.ml_strategy.max_loss_pct_margin,
                stop_loss_pct=settings.risk.stop_loss_pct,
                take_profit_pct=settings.risk.take_profit_pct,
                skip_feature_creation=True,
            )
        except Exception:
            continue

        simulator.analyze_signal(signal, current_price)

        # 5. Execute Entry (with TP Guard check)
        if simulator.current_position is None and signal.action in (Action.LONG, Action.SHORT):
            if tp_guard is not None and signal.action == tp_guard["action"]:
                if idx < tp_guard["wait_until_idx"]:
                    tp_stats["skipped_wait"] += 1
                    continue

                ok, _ = _eval_tp_reentry(
                    df=df_with_features,
                    exit_idx=int(tp_guard["exit_idx"]),
                    current_idx=idx,
                    action=signal.action,
                    exit_price=float(tp_guard["exit_price"]),
                    min_pullback_pct=min_pullback_pct,
                    max_pullback_pct=max_pullback_pct,
                    breakout_buffer_pct=breakout_buffer_pct,
                    volume_factor=volume_factor,
                    sr_lookback=sr_lookback,
                    trend_lookback=trend_lookback,
                    min_trend_slope=min_trend_slope,
                )
                if not ok:
                    tp_stats["skipped_criteria"] += 1
                    continue

                tp_stats["allowed"] += 1
                tp_guard = None

            try:
                simulator.open_position(signal, current_time, symbol)
            except Exception:
                continue

    if simulator.current_position is not None:
        final_price = float(df_with_features["close"].iloc[-1])
        final_time = df_with_features.index[-1]
        simulator.close_all_positions(final_time, final_price)

    metrics = simulator.calculate_metrics(symbol, "optimization", days_back=0)
    result = asdict(metrics)
    result.update(tp_stats)
    
    # Add parameters
    result["min_pullback"] = min_pullback_pct
    result["breakout_buffer"] = breakout_buffer_pct
    result["vol_factor"] = volume_factor
    result["min_trend_slope"] = min_trend_slope
    result["wait_candles"] = wait_candles
    
    return result

def main():
    symbol = "BNBUSDT"
    model_path = "ensemble_BNBUSDT_15_15m.pkl"
    days = 30
    
    print(f"Loading data for {symbol} ({days} days)...")
    
    data_path = _auto_find_history_csv(symbol, "15m")
    if not data_path:
        print("Error: No history data found")
        return

    df = _load_history_from_csv(data_path, days_back=days)
    if df.empty:
        print("Error: Empty history")
        return

    print("Calculating indicators...")
    df_with_indicators = prepare_with_indicators(df.copy())
    if "timestamp" in df_with_indicators.columns:
        df_with_indicators = df_with_indicators.set_index("timestamp")

    settings = load_settings()
    model_file = Path("ml_models") / model_path
    
    # We need to handle if model file is not found
    if not model_file.exists():
        # Try relative path
        model_file = Path("c:/Users/takeg/OneDrive/Документы/vibecodding/ml_bot/ml_models") / model_path
        if not model_file.exists():
             print(f"Model not found: {model_file}")
             return

    print(f"Using model: {model_file}")

    strategy = MLStrategy(
        model_path=str(model_file),
        confidence_threshold=settings.ml_strategy.confidence_threshold,
        min_signal_strength=settings.ml_strategy.min_signal_strength,
        stability_filter=settings.ml_strategy.stability_filter,
        min_signals_per_day=settings.ml_strategy.min_signals_per_day,
        max_signals_per_day=settings.ml_strategy.max_signals_per_day,
    )
    
    print("Feature engineering...")
    df_with_features = strategy.feature_engineer.create_technical_indicators(df_with_indicators)
    df_with_features = df_with_features.fillna(0) # Fill NaNs to avoid errors in prediction
    
    # Parameter Grid
    # We want to iterate:
    # wait: [2, 3]
    # pullback: [0.002, 0.005, 0.008] (0.2% to 0.8%)
    # breakout: [0.001, 0.003]
    # vol: [1.0, 1.2, 1.5]
    
    param_grid = {
        "wait_candles": [3],
        "min_pullback": [0.0],
        "breakout_buffer": [0.0],
        "vol_factor": [1.5],
        "min_trend_slope": [0.0] 
    }
    
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    
    results = []
    
    # Save partial results
    out_file = "optimization_results.csv"
    
    print("Running baseline (wait=0)...")
    with suppress_stdout():
        baseline = run_simulation_fast(
            df_with_features=df_with_features,
            strategy=strategy,
            initial_balance=100.0,
            risk_per_trade=0.02,
            leverage=10,
            wait_candles=0,
            window_candles=8,
            min_pullback_pct=0.001,
            max_pullback_pct=0.02,
            breakout_buffer_pct=0.0005,
            volume_factor=1.1,
            sr_lookback=20,
            trend_lookback=20,
            min_trend_slope=0.0,
            symbol=symbol
        )
    print(f"Baseline (wait=0): PnL={baseline['total_pnl_pct']:.2f}% WinRate={baseline['win_rate']:.2f}% Trades={baseline['total_trades']}")
    results.append(baseline)
    
    # Write header
    pd.DataFrame([baseline]).to_csv(out_file, index=False)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # The combo order is: wait, pullback, breakout, vol, slope
        
        with suppress_stdout():
            res = run_simulation_fast(
                df_with_features=df_with_features,
                strategy=strategy,
                initial_balance=100.0,
                risk_per_trade=0.02,
                leverage=10,
                wait_candles=params["wait_candles"],
                window_candles=8,
                min_pullback_pct=params["min_pullback"],
                max_pullback_pct=0.02,
                breakout_buffer_pct=params["breakout_buffer"],
                volume_factor=params["vol_factor"],
                sr_lookback=20,
                trend_lookback=20,
                min_trend_slope=params["min_trend_slope"],
                symbol=symbol
            )
        
        print(f"[{i+1}/{len(combinations)}] wait={params['wait_candles']} pb={params['min_pullback']} "
              f"brk={params['breakout_buffer']} vol={params['vol_factor']} "
              f"-> PnL={res['total_pnl_pct']:.2f}% WR={res['win_rate']:.2f}% T={res['total_trades']}")
        
        results.append(res)
        
        # Append to CSV (skip header)
        pd.DataFrame([res]).to_csv(out_file, mode='a', header=False, index=False)

    # Sort by PnL
    results.sort(key=lambda x: x['total_pnl_pct'], reverse=True)
    
    print("\nTop 5 Results:")
    for r in results[:5]:
        print(f"Wait={r.get('wait_candles', 0)} PnL={r['total_pnl_pct']:.2f}% WR={r['win_rate']:.2f}% "
              f"Params: pb={r.get('min_pullback')}, brk={r.get('breakout_buffer')}, vol={r.get('vol_factor')}")
        
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
