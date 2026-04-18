"""
Script for optimizing Triple Barrier Method (TBM) hyperparameters.
Uses a grid search approach to find the best PT/SL ratio and Vertical Barrier.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import ApiSettings, load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize TBM Hyperparameters")
    parser.add_argument("--symbol", type=str, default="BNBUSDT", help="Symbol to optimize for")
    parser.add_argument("--days", type=int, default=30, help="Days of data for optimization")
    parser.add_argument("--interval", type=str, default="15m", help="Interval (15m, 1h)")
    args = parser.parse_args()

    load_settings()
    api_settings = ApiSettings()
    collector = DataCollector(api_settings)
    engineer = FeatureEngineer()
    trainer = ModelTrainer()

    symbol = args.symbol
    interval = "15" if args.interval == "15m" else "60"

    safe_print(f"🚀 Starting TBM Hyperparameter Optimization for {symbol} ({args.interval})")

    # 1. Collect Data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    df = collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )

    if df is None or df.empty:
        safe_print("❌ Failed to collect data")
        return

    # 2. Pre-calculate features (so we don't redo it in every loop)
    df_feat = engineer.create_technical_indicators(df)

    # 3. Define Grid
    pt_sl_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    vertical_barriers = [12, 24, 48, 96] if args.interval == "15m" else [4, 8, 12, 24]

    results = []

    safe_print(f"🔍 Grid Search: {len(pt_sl_ratios)} PT/SL ratios x {len(vertical_barriers)} Vertical Barriers")

    for pt_sl in pt_sl_ratios:
        for vb in vertical_barriers:
            safe_print(f"  Testing: pt_sl_ratio={pt_sl}, vertical_barrier={vb}...")

            # Create labels
            df_target = engineer.create_triple_barrier_labels(
                df_feat,
                pt_sl_ratio=pt_sl,
                vertical_barrier_candles=vb
            )

            if df_target.empty or len(df_target["target"].unique()) < 2:
                safe_print(f"    ⚠️ Skipping: Not enough labels or classes")
                continue

            # Prepare data
            exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "target"}
            feature_cols = [col for col in df_target.columns if col not in exclude_cols]
            X = df_target[feature_cols].values
            y = df_target["target"].values

            # Simple CV using ModelTrainer's logic (Random Forest for speed)
            _, metrics = trainer.train_random_forest_classifier(X, y)

            if metrics:
                cv_acc = metrics.get("cv_accuracy", 0)
                f1 = metrics.get("f1-score", metrics.get("f1_macro", 0))
                class_dist = dict(pd.Series(y).value_counts())
                signal_count = class_dist.get(1, 0) + class_dist.get(-1, 0)
                signal_pct = (signal_count / len(y)) * 100

                results.append({
                    "pt_sl_ratio": pt_sl,
                    "vertical_barrier": vb,
                    "cv_accuracy": cv_acc,
                    "f1_macro": f1,
                    "signal_pct": signal_pct,
                    "signal_count": signal_count,
                    "total_samples": len(y)
                })
                safe_print(f"    ✅ CV Acc: {cv_acc:.4f}, F1: {f1:.4f}, Signals: {signal_pct:.1f}%")

    # 4. Sort and Save
    if not results:
        safe_print("❌ No results found")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_macro", ascending=False)

    safe_print("\n" + "="*50)
    safe_print("🏆 TOP 5 PARAMETER COMBINATIONS")
    safe_print("="*50)
    safe_print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]

    # Save to a JSON file that can be used by retrain_ml_optimized.py
    best_config = {
        "tbm_pt_sl_ratio": float(best["pt_sl_ratio"]),
        "tbm_vertical_barrier": int(best["vertical_barrier"]),
        "optimization_date": datetime.now().isoformat(),
        "symbol": symbol,
        "interval": args.interval,
        "cv_accuracy": float(best["cv_accuracy"])
    }

    output_path = f"tbm_optimized_{symbol}_{args.interval}.json"
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=2)

    safe_print(f"\n✅ Best parameters saved to {output_path}")

if __name__ == "__main__":
    main()
