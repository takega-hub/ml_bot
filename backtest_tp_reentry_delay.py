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


def _load_history_from_csv(path: Path, days_back: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.sort_values("timestamp")
        if days_back > 0 and not df.empty:
            end = df["timestamp"].iloc[-1]
            start = end - pd.Timedelta(days=int(days_back))
            df = df.loc[df["timestamp"] >= start]
    else:
        raise ValueError("CSV must contain timestamp column")
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[["timestamp"] + keep].copy()


def _auto_find_history_csv(symbol: str, interval: str) -> Optional[Path]:
    symbol = symbol.upper()
    interval = str(interval).strip().lower()
    suffix = interval
    if interval.endswith("m"):
        suffix = interval[:-1]
    if interval.endswith("h"):
        try:
            suffix = str(int(interval[:-1]) * 60)
        except ValueError:
            suffix = interval
    candidates = []
    ml_data = Path("ml_data")
    if not ml_data.exists():
        return None
    for p in ml_data.glob(f"{symbol}_{suffix}_*.csv"):
        candidates.append(p)
    if not candidates:
        for p in ml_data.glob(f"{symbol}_{suffix}_cache.csv"):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]


def _eval_tp_reentry(
    df: pd.DataFrame,
    exit_idx: int,
    current_idx: int,
    action: Action,
    exit_price: float,
    min_pullback_pct: float,
    max_pullback_pct: float,
    breakout_buffer_pct: float,
    volume_factor: float,
    sr_lookback: int,
    trend_lookback: int,
    min_trend_slope: float,
) -> Tuple[bool, Dict[str, Any]]:
    w = df.iloc[max(0, exit_idx): current_idx + 1].copy()
    lb = max(5, int(sr_lookback))
    tlb = max(5, int(trend_lookback))
    w = w.tail(max(lb, tlb))

    if w.empty or not {"high", "low", "close", "volume"}.issubset(set(w.columns)):
        return True, {"reason": "no_ohlcv"}

    current_price = float(w["close"].iloc[-1])
    exit_price = float(exit_price)
    if exit_price <= 0:
        return True, {"reason": "bad_exit_price"}

    if action == Action.LONG:
        pullback = (exit_price - float(w["low"].min())) / exit_price
        breakout_level = float(w["high"].iloc[:-1].max()) if len(w) > 1 else float(w["high"].max())
        breakout_ok = current_price >= breakout_level * (1.0 + breakout_buffer_pct)
        trend_series = w["close"].tail(tlb).astype(float)
        x = np.arange(len(trend_series))
        slope = float(np.polyfit(x, trend_series.values, 1)[0]) / max(1e-12, float(trend_series.values[-1]))
        trend_ok = slope >= min_trend_slope
    else:
        pullback = (float(w["high"].max()) - exit_price) / exit_price
        breakout_level = float(w["low"].iloc[:-1].min()) if len(w) > 1 else float(w["low"].min())
        breakout_ok = current_price <= breakout_level * (1.0 - breakout_buffer_pct)
        trend_series = w["close"].tail(tlb).astype(float)
        x = np.arange(len(trend_series))
        slope = float(np.polyfit(x, trend_series.values, 1)[0]) / max(1e-12, float(trend_series.values[-1]))
        trend_ok = slope <= -min_trend_slope

    pullback_ok = (pullback >= min_pullback_pct) and (pullback <= max_pullback_pct)

    vol_series = w["volume"].astype(float)
    avg_vol = float(vol_series.tail(min(20, len(vol_series))).mean())
    cur_vol = float(vol_series.iloc[-1])
    vol_ok = (avg_vol <= 0) or (cur_vol >= avg_vol * volume_factor)

    ok = pullback_ok and breakout_ok and trend_ok and vol_ok
    return ok, {
        "ok": ok,
        "pullback": pullback,
        "breakout_ok": breakout_ok,
        "trend_slope": slope,
        "trend_ok": trend_ok,
        "vol_ok": vol_ok,
        "cur_vol": cur_vol,
        "avg_vol": avg_vol,
    }


def run_backtest_with_tp_delay(
    model_path: str,
    symbol: str,
    days_back: int,
    interval: str,
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
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    model_file = Path(model_path)
    if not model_file.exists():
        model_file = Path("ml_models") / model_path
        if not model_file.exists():
            return None, {"error": f"model_not_found:{model_path}"}

    settings = load_settings()

    data_path = _auto_find_history_csv(symbol, interval)
    if not data_path:
        return None, {"error": "no_local_history_csv"}

    df = _load_history_from_csv(data_path, days_back=days_back)
    if df.empty:
        return None, {"error": "empty_local_history_csv", "path": str(data_path)}

    df_with_indicators = prepare_with_indicators(df.copy())
    df_work = df_with_indicators.copy()
    if "timestamp" in df_work.columns:
        df_work = df_work.set_index("timestamp")

    strategy = MLStrategy(
        model_path=str(model_file),
        confidence_threshold=settings.ml_strategy.confidence_threshold,
        min_signal_strength=settings.ml_strategy.min_signal_strength,
        stability_filter=settings.ml_strategy.stability_filter,
        min_signals_per_day=settings.ml_strategy.min_signals_per_day,
        max_signals_per_day=settings.ml_strategy.max_signals_per_day,
    )

    df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)

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
    if total_bars <= min_window_size + 10:
        return None, {"error": "not_enough_bars", "bars": total_bars}

    tp_guard = None
    tp_stats = {
        "wait_candles": int(wait_candles),
        "window_candles": int(window_candles),
        "skipped_wait": 0,
        "skipped_criteria": 0,
        "allowed": 0,
        "expired": 0,
    }

    last_trade_count = 0

    window_size = 1200
    for idx in range(min_window_size, total_bars):
        current_time = df_with_features.index[idx]
        row = df_with_features.iloc[idx]
        current_price = float(row["close"])

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

        if tp_guard is not None and idx > tp_guard["expire_idx"]:
            tp_stats["expired"] += 1
            tp_guard = None

        start_idx = max(0, idx - window_size + 1)
        df_window = df_with_features.iloc[start_idx: idx + 1]
        row_window = df_with_features.iloc[idx]

        try:
            signal = strategy.generate_signal(
                row=row_window,
                df=df_window,
                has_position=None,
                current_price=current_price,
                leverage=leverage,
                target_profit_pct_margin=settings.ml_strategy.target_profit_pct_margin,
                max_loss_pct_margin=settings.ml_strategy.max_loss_pct_margin,
                stop_loss_pct=settings.risk.stop_loss_pct,
                take_profit_pct=settings.risk.take_profit_pct,
            )
        except Exception:
            continue

        simulator.analyze_signal(signal, current_price)

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

        if idx % 500 == 0:
            print(f"progress: {idx}/{total_bars} trades={len(simulator.trades)} bal={simulator.balance:.2f}")

    if simulator.current_position is not None:
        final_price = float(df_with_features["close"].iloc[-1])
        final_time = df_with_features.index[-1]
        simulator.close_all_positions(final_time, final_price)

    model_name = model_file.stem
    metrics = simulator.calculate_metrics(symbol, model_name, days_back=days_back)
    result = asdict(metrics)
    result.update(tp_stats)
    return result, tp_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--interval", default="15")
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--risk", type=float, default=0.02)
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--wait", default="0,2,3")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--min-pullback", type=float, default=0.001)
    p.add_argument("--max-pullback", type=float, default=0.006)
    p.add_argument("--breakout-buffer", type=float, default=0.0005)
    p.add_argument("--vol-factor", type=float, default=1.1)
    p.add_argument("--sr-lookback", type=int, default=20)
    p.add_argument("--trend-lookback", type=int, default=20)
    p.add_argument("--min-trend-slope", type=float, default=0.0)
    args = p.parse_args()

    wait_list = [int(x.strip()) for x in str(args.wait).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for wc in wait_list:
        res, stats = run_backtest_with_tp_delay(
            model_path=args.model,
            symbol=args.symbol,
            days_back=args.days,
            interval=args.interval,
            initial_balance=args.balance,
            risk_per_trade=args.risk,
            leverage=args.leverage,
            wait_candles=wc,
            window_candles=args.window,
            min_pullback_pct=args.min_pullback,
            max_pullback_pct=args.max_pullback,
            breakout_buffer_pct=args.breakout_buffer,
            volume_factor=args.vol_factor,
            sr_lookback=args.sr_lookback,
            trend_lookback=args.trend_lookback,
            min_trend_slope=args.min_trend_slope,
        )
        if res is None:
            print(f"wait={wc}: failed: {stats}")
            continue
        print(
            f"wait={wc}: pnl_pct={res.get('total_pnl_pct'):.2f} trades={res.get('total_trades')} "
            f"win_rate={res.get('win_rate'):.2f}% pf={res.get('profit_factor'):.2f} "
            f"skipped_wait={res.get('skipped_wait')} skipped_criteria={res.get('skipped_criteria')} allowed={res.get('allowed')}"
        )
        rows.append(res)

    if rows:
        out = Path(f"tp_reentry_backtest_{args.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
