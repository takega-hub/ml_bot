
import os
import sys
import json
import subprocess
import argparse
import time
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bot.experiment_management import (
    ExperimentAnalyzer,
    ExperimentCriteria,
    ExperimentStore,
    build_hyperparameter_search_strategy,
    build_regime_plan_context,
    compute_unified_score,
    derive_market_regime_from_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("research.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("research_runner")

_FILE_LOCK = threading.Lock()
_ACTIVITY_LOCK = threading.Lock()
_LAST_OUTPUT_MONO = time.monotonic()
_ACTIVE_SUBPROCS = 0


def _touch_output():
    global _LAST_OUTPUT_MONO
    with _ACTIVITY_LOCK:
        _LAST_OUTPUT_MONO = time.monotonic()


def _set_active_subprocs(delta: int):
    global _ACTIVE_SUBPROCS
    with _ACTIVITY_LOCK:
        _ACTIVE_SUBPROCS = max(0, _ACTIVE_SUBPROCS + int(delta))


def _get_activity_state():
    with _ACTIVITY_LOCK:
        return _LAST_OUTPUT_MONO, _ACTIVE_SUBPROCS


def _try_with_file_lock(fn, timeout_sec: float = 2.0, retries: int = 3):
    for _ in range(max(1, int(retries))):
        acquired = False
        try:
            acquired = _FILE_LOCK.acquire(timeout=timeout_sec)
            if not acquired:
                time.sleep(0.1)
                continue
            fn()
            return True
        except Exception as e:
            logger.error(f"Failed to write experiments.json: {e}")
            return False
        finally:
            if acquired:
                try:
                    _FILE_LOCK.release()
                except Exception:
                    pass
    return False


def update_experiment_status(experiment_id: str, status: str, details: dict = None):
    try:
        def _write():
            file_path = Path("experiments.json")
            data = {}
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        pass
            if experiment_id not in data:
                data[experiment_id] = {
                    "id": experiment_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            data[experiment_id]["status"] = status
            data[experiment_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if details:
                for k, v in details.items():
                    data[experiment_id][k] = v
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        _try_with_file_lock(_write)
    except Exception as e:
        logger.error(f"Failed to update experiment status: {e}")


def patch_experiment(experiment_id: str, fields: Dict[str, Any]):
    try:
        def _write():
            file_path = Path("experiments.json")
            data: Dict[str, Any] = {}
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            exp = data.get(experiment_id) if isinstance(data.get(experiment_id), dict) else {"id": experiment_id}
            if "created_at" not in exp:
                exp["created_at"] = datetime.now(timezone.utc).isoformat()
            exp["updated_at"] = datetime.now(timezone.utc).isoformat()
            for k, v in (fields or {}).items():
                exp[k] = v
            data[experiment_id] = exp
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        _try_with_file_lock(_write)
    except Exception as e:
        logger.error(f"Failed to patch experiment: {e}")


def run_process_with_heartbeat(
    cmd,
    log_prefix: str,
    experiment_id: str,
    status: str,
    base_details: dict,
    heartbeat_interval_sec: int = 10,
):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    waiter_done = threading.Event()
    waiter_result: Dict[str, Any] = {"rc": None}

    def _waiter():
        try:
            waiter_result["rc"] = process.wait()
        except Exception:
            waiter_result["rc"] = process.poll()
        finally:
            waiter_done.set()

    threading.Thread(target=_waiter, daemon=True).start()
    _set_active_subprocs(+1)

    q_out: "queue.Queue[str]" = queue.Queue()
    q_err: "queue.Queue[str]" = queue.Queue()

    def _reader(stream, q):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                q.put(line.rstrip("\n"))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(process.stdout, q_out), daemon=True)
    t_err = threading.Thread(target=_reader, args=(process.stderr, q_err), daemon=True)
    t_out.start()
    t_err.start()

    lines: List[str] = []
    last_hb = time.monotonic()
    last_log = None
    last_output_write = 0.0

    try:
        while True:
            now = time.monotonic()
            try:
                line = q_out.get(timeout=0.5)
                last_log = line.strip()
                if last_log:
                    _touch_output()
                    logger.info(f"[{log_prefix}] {last_log}")
                    lines.append(last_log)
                    if len(lines) > 200:
                        lines = lines[-200:]
                    if now - last_output_write >= 5:
                        try:
                            patch_experiment(
                                experiment_id,
                                {
                                    "last_output_at": datetime.now(timezone.utc).isoformat(),
                                    "runner_step": log_prefix,
                                    "last_log": last_log,
                                },
                            )
                            last_output_write = now
                        except Exception:
                            pass
            except queue.Empty:
                pass

            try:
                err_line = q_err.get_nowait()
                if err_line:
                    err_line = err_line.strip()
                    _touch_output()
                    logger.info(f"[{log_prefix}-ERR] {err_line}")
                    if now - last_output_write >= 5:
                        try:
                            patch_experiment(
                                experiment_id,
                                {
                                    "last_output_at": datetime.now(timezone.utc).isoformat(),
                                    "runner_step": f"{log_prefix}-ERR",
                                    "last_log": err_line,
                                },
                            )
                            last_output_write = now
                        except Exception:
                            pass
            except queue.Empty:
                pass

            if now - last_hb >= heartbeat_interval_sec:
                patch = dict(base_details or {})
                patch.update(
                    {
                        "status": status,
                        "last_output_at": datetime.now(timezone.utc).isoformat(),
                        "runner_step": log_prefix,
                        "last_log": last_log,
                    }
                )
                patch_experiment(experiment_id, patch)
                last_hb = now

            if waiter_done.is_set() and q_out.empty() and q_err.empty():
                break
    finally:
        _set_active_subprocs(-1)

    return int(waiter_result["rc"] if waiter_result["rc"] is not None else (process.poll() or 0)), lines, process.pid


def _run_python_json(output_path: Path) -> Optional[Dict[str, Any]]:
    if not output_path.exists():
        return None
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _safe_kill(pid: Optional[int]):
    if not pid:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def _build_campaign_notebook_payload(root_experiment_id: str) -> Dict[str, Any]:
    store = ExperimentStore(Path("experiments.json"))
    analyzer = ExperimentAnalyzer(store.list())
    return analyzer.build_campaign_notebook(root_experiment_id)


def _resolve_existing_train_script() -> List[str]:
    candidates = [
        Path("retrain_ml_optimized.py"),
        Path("train_model.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return [sys.executable, str(candidate)]
    raise FileNotFoundError("No training script found (expected retrain_ml_optimized.py or train_model.py)")


def _resolve_backtest_script(use_mtf: bool) -> List[str]:
    candidates = [Path("backtest_mtf_strategy.py")] if use_mtf else [Path("backtest_ml_strategy.py"), Path("backtest.py")]
    for candidate in candidates:
        if candidate.exists():
            return [sys.executable, str(candidate)]
    raise FileNotFoundError(
        "No backtest script found"
        + (" for MTF (expected backtest_mtf_strategy.py)" if use_mtf else " (expected backtest_ml_strategy.py or backtest.py)")
    )


def _find_model_path(symbol: str, interval: str) -> Optional[Path]:
    model_dir = Path("ml_models")
    if not model_dir.exists():
        return None
    if interval == "1h":
        patterns = [
            f"*{symbol}*60*1h*.pkl",
            f"*{symbol}*1h*.pkl",
        ]
    else:
        patterns = [
            f"*{symbol}*15*15m*.pkl",
            f"*{symbol}*15*.pkl",
        ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))
    seen = set()
    ordered: List[Path] = []
    for m in matches:
        key = str(m.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(m)
    return ordered[0] if ordered else None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _metrics_quality_score(metrics: Dict[str, Any]) -> float:
    if not isinstance(metrics, dict):
        return -1.0
    cv_f1 = _to_float(metrics.get("cv_f1_mean"))
    f1 = _to_float(metrics.get("f1_score"))
    cv_mean = _to_float(metrics.get("cv_mean"))
    accuracy = _to_float(metrics.get("accuracy"))
    return (4.0 * cv_f1) + (3.0 * f1) + (2.0 * cv_mean) + accuracy


def _select_model_for_experiment(
    symbol: str,
    interval: str,
    model_suffix: str,
    training_report: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    model_dir = Path("ml_models")
    if not model_dir.exists():
        return None
    base_interval = "60" if interval == "1h" else "15"
    candidates = [
        ("quad_ensemble", "quad_metrics", 5),
        ("triple_ensemble", "triple_ensemble_metrics", 4),
        ("ensemble", "ensemble_metrics", 3),
        ("xgb", "xgb_metrics", 2),
        ("rf", "rf_metrics", 1),
    ]
    ranked: List[Dict[str, Any]] = []
    report = training_report if isinstance(training_report, dict) else {}
    for prefix, metrics_key, tie_rank in candidates:
        metrics = report.get(metrics_key)
        if not isinstance(metrics, dict):
            continue
        filename = f"{prefix}_{symbol}_{base_interval}_{interval}{model_suffix}.pkl"
        ranked.append(
            {
                "model_type": prefix,
                "filename": filename,
                "path": model_dir / filename,
                "score": _metrics_quality_score(metrics),
                "tie_rank": tie_rank,
            }
        )
    ranked.sort(key=lambda item: (item["score"], item["tie_rank"]), reverse=True)
    for item in ranked:
        if item["path"].exists():
            return {
                "model_type": item["model_type"],
                "path": str(item["path"]),
                "score": float(item["score"]),
            }
    for prefix, _, _ in candidates:
        fallback_path = model_dir / f"{prefix}_{symbol}_{base_interval}_{interval}{model_suffix}.pkl"
        if fallback_path.exists():
            return {
                "model_type": prefix,
                "path": str(fallback_path),
                "score": None,
            }
    legacy = _find_model_path(symbol, interval)
    if legacy:
        return {
            "model_type": "legacy_latest",
            "path": str(legacy),
            "score": None,
        }
    return None


def _collect_experiment_models(
    symbol: str,
    interval: str,
    model_suffix: str,
    training_report: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    model_dir = Path("ml_models")
    if not model_dir.exists():
        return []
    base_interval = "60" if interval == "1h" else "15"
    candidates = [
        ("quad_ensemble", "quad_metrics", 5),
        ("triple_ensemble", "triple_ensemble_metrics", 4),
        ("ensemble", "ensemble_metrics", 3),
        ("xgb", "xgb_metrics", 2),
        ("rf", "rf_metrics", 1),
    ]
    report = training_report if isinstance(training_report, dict) else {}
    collected: List[Dict[str, Any]] = []
    for prefix, metrics_key, tie_rank in candidates:
        filename = f"{prefix}_{symbol}_{base_interval}_{interval}{model_suffix}.pkl"
        path = model_dir / filename
        if not path.exists():
            continue
        metrics = report.get(metrics_key) if isinstance(report.get(metrics_key), dict) else {}
        collected.append(
            {
                "model_type": prefix,
                "path": str(path),
                "score": _metrics_quality_score(metrics),
                "tie_rank": tie_rank,
            }
        )
    collected.sort(key=lambda item: (float(item.get("score") or 0.0), int(item.get("tie_rank") or 0)), reverse=True)
    return collected


def main():
    parser = argparse.ArgumentParser(description="Run AI research experiment")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--type", required=True, choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--no-mtf", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--metadata-path")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    symbol = args.symbol.upper().strip()
    exp_type = args.type
    metadata_path = Path(args.metadata_path) if args.metadata_path else None
    details: Dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                details = loaded
        except Exception as e:
            logger.warning(f"Failed to read metadata: {e}")

    def _status_payload(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "type": exp_type,
        }
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    artifacts_dir = Path("artifacts") / experiment_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []

    watchdog_stop = threading.Event()
    subprocess_pids: List[int] = []

    def _watchdog():
        idle_limit_sec = 20 * 60
        while not watchdog_stop.wait(15):
            try:
                last_output, active_subprocs = _get_activity_state()
                idle_for = time.monotonic() - last_output
                if active_subprocs > 0:
                    continue
                if idle_for >= idle_limit_sec:
                    patch_experiment(
                        experiment_id,
                        {
                            "status": "failed",
                            "error": f"Watchdog timeout: no activity for {int(idle_for)} sec",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    for pid in list(subprocess_pids):
                        _safe_kill(pid)
                    os._exit(1)
            except Exception:
                pass

    threading.Thread(target=_watchdog, daemon=True).start()

    try:
        requested_backtest_days = int(details.get("backtest_days") or ((details.get("ai_plan") or {}).get("backtest_days") if isinstance(details.get("ai_plan"), dict) else 7) or 7)
        backtest_days = max(3, min(120, requested_backtest_days))
        market_regime = details.get("market_regime") if isinstance(details.get("market_regime"), dict) else derive_market_regime_from_metrics(symbol)

        update_experiment_status(experiment_id, "training", _status_payload({
            "symbol": symbol,
            "type": exp_type,
            "params": {
                "interval": args.interval,
                "no_mtf": bool(args.no_mtf),
                "safe_mode": bool(args.safe_mode),
                "backtest_days": backtest_days,
            },
            "market_regime": market_regime,
            "progress": 5,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }))

        model_15m_path = None
        model_1h_path = None
        training_reports: Dict[str, Any] = {}
        interval_candidate_models: Dict[str, List[Dict[str, Any]]] = {}
        hyperparams_payload = details.get("hyperparams") if isinstance(details.get("hyperparams"), dict) else None

        intervals = ["15m"] if args.no_mtf else ["15m", "1h"]
        train_script = _resolve_existing_train_script()
        for interval in intervals:
            report_path = artifacts_dir / f"training_{interval}.json"
            model_suffix = f"__{experiment_id}_{interval}"
            train_cmd = train_script + [
                "--symbol", symbol,
                "--interval", interval,
                "--report-json", str(report_path),
                "--model-suffix", model_suffix,
            ]
            if args.no_mtf:
                train_cmd.append("--no-mtf")
            if args.safe_mode:
                train_cmd.append("--safe-mode")
            if isinstance(hyperparams_payload, dict) and hyperparams_payload:
                hp_path = artifacts_dir / f"hyperparams_{interval}.json"
                with open(hp_path, "w", encoding="utf-8") as f:
                    json.dump(hyperparams_payload, f, ensure_ascii=False, indent=2, default=str)
                train_cmd.extend(["--hyperparams-json", str(hp_path)])
            logger.info(f"Running training command ({interval}): {' '.join(train_cmd)}")
            return_code, out_lines, pid = run_process_with_heartbeat(
                train_cmd,
                f"train_{interval}",
                experiment_id,
                "training",
                _status_payload({
                    "progress": 15 if interval == "15m" else 35,
                    "current_phase": f"training_{interval}",
                    "market_regime": market_regime,
                }),
            )
            subprocess_pids.append(pid)
            logs.extend(out_lines[-20:])
            if return_code != 0:
                raise RuntimeError(f"Training failed for interval {interval} with code {return_code}")
            report_payload = _run_python_json(report_path) or {}
            interval_models = _collect_experiment_models(symbol, interval, model_suffix, report_payload)
            if not interval_models:
                selected_model = _select_model_for_experiment(symbol, interval, model_suffix, report_payload)
                if selected_model and selected_model.get("path"):
                    interval_models = [selected_model]
            if not interval_models:
                raise RuntimeError(f"Model artifacts not found after training for interval {interval}")
            interval_candidate_models[interval] = interval_models
            selected_model = interval_models[0]
            found_model = Path(str(selected_model.get("path")))
            if interval == "15m":
                model_15m_path = str(found_model)
            else:
                model_1h_path = str(found_model)
            if isinstance(report_payload, dict):
                report_payload["selected_model_path"] = str(found_model)
                report_payload["selected_model_type"] = selected_model.get("model_type")
                report_payload["model_suffix"] = model_suffix
                report_payload["model_selection_score"] = selected_model.get("score")
                report_payload["candidate_models"] = interval_models
            training_reports[interval] = report_payload

        update_experiment_status(experiment_id, "training_completed", _status_payload({
            "progress": 45,
            "current_phase": "training_completed",
            "models": {"15m": model_15m_path, "1h": model_1h_path},
            "training_report": training_reports,
            "market_regime": market_regime,
        }))

        def _run_single_backtest(name: str, model_path: Optional[str], interval: str, days_override: Optional[int] = None) -> Optional[Dict[str, Any]]:
            if not model_path:
                return None
            bt_path = artifacts_dir / f"backtest_{name}.json"
            bt_script = _resolve_backtest_script(False)
            bt_cmd = bt_script + [
                "--symbol", symbol,
                "--interval", interval,
                "--model", str(model_path),
                "--days", str(days_override or backtest_days),
                "--save",
                "--out-json", str(bt_path),
            ]
            logger.info(f"Running backtest ({name}): {' '.join(bt_cmd)}")
            rc, bt_lines, pid = run_process_with_heartbeat(
                bt_cmd,
                f"backtest_{name}",
                experiment_id,
                "backtesting",
                _status_payload({
                    "progress": 55 if name == "15m" else 62,
                    "current_phase": f"backtest_{name}",
                    "market_regime": market_regime,
                }),
            )
            subprocess_pids.append(pid)
            logs.extend(bt_lines[-20:])
            if rc != 0:
                return None
            return _run_python_json(bt_path)

        def _evaluate_interval_models(interval: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
            evaluated: List[Dict[str, Any]] = []
            for idx, item in enumerate(candidates):
                model_path = str(item.get("path") or "")
                model_type = str(item.get("model_type") or f"model_{idx + 1}")
                if not model_path:
                    continue
                metrics = _run_single_backtest(f"{interval}_{model_type}_{idx + 1}", model_path, interval)
                if not isinstance(metrics, dict):
                    continue
                evaluated.append(
                    {
                        "model_path": model_path,
                        "model_type": model_type,
                        "train_score": item.get("score"),
                        "metrics": metrics,
                    }
                )
            evaluated.sort(
                key=lambda x: (
                    _to_float((x.get("metrics") or {}).get("total_pnl_pct")),
                    _to_float((x.get("metrics") or {}).get("win_rate")),
                    _to_float((x.get("metrics") or {}).get("profit_factor")),
                    _to_float((x.get("metrics") or {}).get("total_trades")),
                ),
                reverse=True,
            )
            return {
                "best": evaluated[0] if evaluated else None,
                "candidates": evaluated,
            }

        update_experiment_status(experiment_id, "backtesting", _status_payload({"progress": 50, "market_regime": market_regime}))
        tactics: Dict[str, Any] = {}
        interval_model_results: Dict[str, Any] = {}
        eval_15m = _evaluate_interval_models("15m", interval_candidate_models.get("15m") or ([{"path": model_15m_path, "model_type": "fallback"}] if model_15m_path else []))
        interval_model_results["15m"] = eval_15m
        best_15m = eval_15m.get("best") if isinstance(eval_15m, dict) else None
        if isinstance(best_15m, dict) and isinstance(best_15m.get("metrics"), dict):
            model_15m_path = str(best_15m.get("model_path") or model_15m_path or "")
            tactics["single_15m"] = best_15m.get("metrics")
        eval_1h = _evaluate_interval_models("1h", interval_candidate_models.get("1h") or ([{"path": model_1h_path, "model_type": "fallback"}] if model_1h_path else []))
        interval_model_results["1h"] = eval_1h
        best_1h = eval_1h.get("best") if isinstance(eval_1h, dict) else None
        if isinstance(best_1h, dict) and isinstance(best_1h.get("metrics"), dict):
            model_1h_path = str(best_1h.get("model_path") or model_1h_path or "")
            tactics["single_1h"] = best_1h.get("metrics")

        if not args.no_mtf and model_15m_path and model_1h_path:
            mtf_path = artifacts_dir / "backtest_mtf.json"
            mtf_script = _resolve_backtest_script(True)
            mtf_cmd = mtf_script + [
                "--symbol", symbol,
                "--days", str(backtest_days),
                "--model-1h", str(model_1h_path),
                "--model-15m", str(model_15m_path),
                "--save",
                "--out-json", str(mtf_path),
            ]
            rc, mtf_lines, pid = run_process_with_heartbeat(
                mtf_cmd,
                "backtest_mtf",
                experiment_id,
                "backtesting",
                _status_payload({
                    "progress": 70,
                    "current_phase": "backtest_mtf",
                    "market_regime": market_regime,
                }),
            )
            subprocess_pids.append(pid)
            logs.extend(mtf_lines[-20:])
            if rc == 0 and mtf_path.exists():
                try:
                    with open(mtf_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, dict):
                        tactics["mtf"] = payload
                except Exception:
                    pass

        oos_days = max(3, min(30, backtest_days // 3 or 3))
        walk_forward: Dict[str, Any] = {}
        oos_validation: Dict[str, Any] = {}
        for tactic_name, tactic_metrics in list(tactics.items()):
            model_path = model_15m_path if tactic_name in {"single_15m", "mtf"} else model_1h_path
            interval = "15m" if tactic_name in {"single_15m", "mtf"} else "1h"
            wf_windows = []
            for wf_days in (max(3, backtest_days // 2), backtest_days):
                metrics = _run_single_backtest(f"{tactic_name}_wf_{wf_days}", model_path, interval, wf_days)
                if isinstance(metrics, dict):
                    wf_windows.append({
                        "days": wf_days,
                        "total_pnl_pct": metrics.get("total_pnl_pct"),
                        "max_drawdown_pct": metrics.get("max_drawdown_pct") or metrics.get("max_drawdown"),
                        "win_rate": metrics.get("win_rate"),
                    })
            if wf_windows:
                pnls = [float(x.get("total_pnl_pct") or 0.0) for x in wf_windows]
                spread = (max(pnls) - min(pnls)) if pnls else 0.0
                walk_forward[tactic_name] = {
                    "windows": wf_windows,
                    "stability_pass": spread <= 12.0,
                    "pnl_spread": spread,
                }

            if model_path:
                oos_path = artifacts_dir / f"backtest_{tactic_name}_oos.json"
                oos_script = _resolve_backtest_script(False)
                oos_cmd = oos_script + [
                    "--symbol", symbol,
                    "--interval", interval,
                    "--model", str(model_path),
                    "--days", str(oos_days),
                    "--save",
                    "--out-json", str(oos_path),
                ]
                rc, oos_lines, pid = run_process_with_heartbeat(
                    oos_cmd,
                    f"backtest_{tactic_name}_oos",
                    experiment_id,
                    "backtesting",
                    {
                        "progress": 78,
                        "current_phase": f"backtest_{tactic_name}_oos",
                        "market_regime": market_regime,
                    },
                )
                subprocess_pids.append(pid)
                logs.extend(oos_lines[-20:])
                if rc == 0:
                    payload = _run_python_json(oos_path)
                    if isinstance(payload, dict):
                        eval_payload = compute_unified_score(payload, criteria=ExperimentCriteria())
                        oos_validation[tactic_name] = {
                            "metrics": payload,
                            "evaluation": eval_payload,
                            "passed": bool((eval_payload.get("gates") or {}).get("passed")),
                        }

        selection: Dict[str, Any] = {"candidates": []}
        recommended_tactic = None
        best_score = None
        for tactic_name, metrics in tactics.items():
            eval_payload = compute_unified_score(metrics, criteria=ExperimentCriteria())
            score = float(eval_payload.get("score") or 0.0)
            oos_pass = bool(((oos_validation.get(tactic_name) or {}).get("passed"))) if tactic_name in oos_validation else False
            wf_pass = bool(((walk_forward.get(tactic_name) or {}).get("stability_pass"))) if tactic_name in walk_forward else False
            candidate = {
                "tactic": tactic_name,
                "score": score,
                "metrics": metrics,
                "quality_gates": eval_payload.get("gates"),
                "oos_pass": oos_pass,
                "walk_forward_stability_pass": wf_pass,
            }
            selection["candidates"].append(candidate)
            composite = score + (2.0 if oos_pass else -2.0) + (1.5 if wf_pass else -1.5)
            if best_score is None or composite > best_score:
                best_score = composite
                recommended_tactic = tactic_name

        result_data: Dict[str, Any] = {}
        if recommended_tactic and isinstance(tactics.get(recommended_tactic), dict):
            result_data.update(tactics[recommended_tactic])
        result_data["recommended_tactic"] = recommended_tactic
        result_data["model_comparison"] = interval_model_results
        training_report_15m = training_reports.get("15m") if isinstance(training_reports.get("15m"), dict) else {}
        training_report_1h = training_reports.get("1h") if isinstance(training_reports.get("1h"), dict) else {}
        best_15m_payload = (interval_model_results.get("15m") or {}).get("best") if isinstance(interval_model_results.get("15m"), dict) else {}
        best_1h_payload = (interval_model_results.get("1h") or {}).get("best") if isinstance(interval_model_results.get("1h"), dict) else {}
        selected_path_15m = str((best_15m_payload or {}).get("model_path") or training_report_15m.get("selected_model_path") or model_15m_path or "")
        selected_path_1h = str((best_1h_payload or {}).get("model_path") or training_report_1h.get("selected_model_path") or model_1h_path or "")
        selected_type_15m = str((best_15m_payload or {}).get("model_type") or training_report_15m.get("selected_model_type") or "")
        selected_type_1h = str((best_1h_payload or {}).get("model_type") or training_report_1h.get("selected_model_type") or "")
        selected_score_15m = (
            _to_float(((best_15m_payload or {}).get("metrics") or {}).get("total_pnl_pct"))
            if isinstance((best_15m_payload or {}).get("metrics"), dict)
            else training_report_15m.get("model_selection_score")
        )
        selected_score_1h = (
            _to_float(((best_1h_payload or {}).get("metrics") or {}).get("total_pnl_pct"))
            if isinstance((best_1h_payload or {}).get("metrics"), dict)
            else training_report_1h.get("model_selection_score")
        )
        best_model_path = selected_path_15m if recommended_tactic in {"single_15m", "mtf"} else selected_path_1h
        best_model_type = selected_type_15m if recommended_tactic in {"single_15m", "mtf"} else selected_type_1h
        best_model_score = selected_score_15m if recommended_tactic in {"single_15m", "mtf"} else selected_score_1h
        result_data["best_model_path"] = best_model_path
        result_data["best_model_type"] = best_model_type
        result_data["best_model"] = {
            "recommended_tactic": recommended_tactic,
            "primary_path": best_model_path,
            "primary_type": best_model_type,
            "primary_score": best_model_score,
            "paths": {"15m": selected_path_15m, "1h": selected_path_1h},
            "types": {"15m": selected_type_15m, "1h": selected_type_1h},
            "scores": {"15m": selected_score_15m, "1h": selected_score_1h},
        }
        result_data["selection"] = {
            "recommended_tactic": recommended_tactic,
            "recommended_score": best_score,
            "candidates": selection["candidates"],
        }
        result_data["walk_forward"] = walk_forward
        result_data["oos_validation"] = oos_validation
        recommended_oos = (
            oos_validation.get(recommended_tactic)
            if isinstance(recommended_tactic, str) and isinstance(oos_validation.get(recommended_tactic), dict)
            else {}
        )
        result_data["oos_metrics"] = {
            "recommended_tactic": recommended_tactic,
            "metrics": recommended_oos.get("metrics") if isinstance(recommended_oos.get("metrics"), dict) else {},
            "evaluation": recommended_oos.get("evaluation") if isinstance(recommended_oos.get("evaluation"), dict) else {},
            "passed": bool(recommended_oos.get("passed")),
            "source": "oos_validation",
        }
        recommended_wf = (
            walk_forward.get(recommended_tactic)
            if isinstance(recommended_tactic, str) and isinstance(walk_forward.get(recommended_tactic), dict)
            else {}
        )
        pnl_spread = float(recommended_wf.get("pnl_spread_pct") or 0.0) if isinstance(recommended_wf, dict) else 0.0
        stability_pass = bool(recommended_wf.get("stability_pass")) if isinstance(recommended_wf, dict) else False
        drift_level = "low"
        if not stability_pass or pnl_spread >= 8.0:
            drift_level = "high"
        elif pnl_spread >= 4.0:
            drift_level = "medium"
        drift_score = max(0.0, min(1.0, (pnl_spread / 10.0) + (0.35 if not stability_pass else 0.0)))
        result_data["drift_signals"] = {
            "recommended_tactic": recommended_tactic,
            "level": drift_level,
            "drift_score": drift_score,
            "walk_forward_pnl_spread_pct": pnl_spread,
            "walk_forward_stability_pass": stability_pass,
            "signals": {
                "wf_instability": not stability_pass,
                "large_pnl_spread": pnl_spread >= 6.0,
                "oos_gate_failed": not bool((result_data.get("oos_metrics") or {}).get("passed")),
            },
        }
        base_pnl = float(result_data.get("total_pnl_pct") or 0.0)
        base_dd = float(result_data.get("max_drawdown_pct") or result_data.get("max_drawdown") or 0.0)
        base_pf = float(result_data.get("profit_factor") or 0.0)
        exec_realism = details.get("execution_realism") if isinstance(details.get("execution_realism"), dict) else {}
        spread_bps = float(exec_realism.get("spread_bps") or 4.0)
        slippage_bps = float(exec_realism.get("slippage_bps") or 8.0)
        funding_bps_daily = float(exec_realism.get("funding_bps_daily") or 3.0)
        scenarios = [
            {"name": "mild", "cost_mult": 1.0},
            {"name": "moderate", "cost_mult": 1.6},
            {"name": "harsh", "cost_mult": 2.3},
        ]
        stress_cases = []
        for sc in scenarios:
            mult = float(sc["cost_mult"])
            cost_penalty_pct = (
                ((spread_bps + slippage_bps) * mult) / 100.0
                + (funding_bps_daily * mult * max(float(backtest_days), 1.0)) / 10000.0
            )
            stressed_pnl = base_pnl - cost_penalty_pct
            stressed_dd = base_dd + (cost_penalty_pct * 0.5)
            stressed_pf = max(0.0, base_pf - (0.06 * mult))
            stress_cases.append(
                {
                    "scenario": sc["name"],
                    "cost_penalty_pct": cost_penalty_pct,
                    "stressed_pnl_pct": stressed_pnl,
                    "stressed_max_drawdown_pct": stressed_dd,
                    "stressed_profit_factor": stressed_pf,
                    "passed": bool(stressed_pnl > -5.0 and stressed_dd <= 35.0 and stressed_pf >= 0.9),
                }
            )
        passed_cases = len([x for x in stress_cases if bool(x.get("passed"))])
        robust_score = passed_cases / float(len(stress_cases) or 1)
        result_data["stress_results"] = {
            "recommended_tactic": recommended_tactic,
            "robustness_score": robust_score,
            "stress_passed": bool(passed_cases >= 2),
            "assumptions": {
                "spread_bps": spread_bps,
                "slippage_bps": slippage_bps,
                "funding_bps_daily": funding_bps_daily,
                "backtest_days": backtest_days,
            },
            "scenarios": stress_cases,
        }
        ranked_candidates = sorted(
            [
                c
                for c in selection["candidates"]
                if isinstance(c, dict)
            ],
            key=lambda x: (x.get("score") is None, -(float(x.get("score") or 0.0))),
        )
        chosen_candidate = next((c for c in ranked_candidates if str(c.get("tactic") or "") == str(recommended_tactic or "")), None)
        alternatives = [c for c in ranked_candidates if str(c.get("tactic") or "") != str(recommended_tactic or "")][:2]
        chosen_score = float(((chosen_candidate or {}).get("score")) or 0.0)
        result_data["decision_trace"] = {
            "chosen_tactic": recommended_tactic,
            "chosen_score": best_score,
            "why_this_tactic": f"Выбран {recommended_tactic or 'unknown'} по максимальному composite score с учетом OOS/WF.",
            "why_not_others": [
                {
                    "tactic": str((alt or {}).get("tactic") or "unknown"),
                    "score": float((alt or {}).get("score") or 0.0),
                    "delta_vs_chosen": chosen_score - float((alt or {}).get("score") or 0.0),
                }
                for alt in alternatives
                if isinstance(alt, dict)
            ],
            "top_tradeoffs": [
                {"metric": "quality_gates_passed", "value": bool(((chosen_candidate or {}).get("quality_gates") or {}).get("passed"))},
                {"metric": "oos_pass", "value": bool((chosen_candidate or {}).get("oos_pass"))},
                {"metric": "walk_forward_stability_pass", "value": bool((chosen_candidate or {}).get("walk_forward_stability_pass"))},
            ],
            "candidate_rankings": [
                {
                    "tactic": str((c or {}).get("tactic") or "unknown"),
                    "score": float((c or {}).get("score") or 0.0),
                    "oos_pass": bool((c or {}).get("oos_pass")),
                    "walk_forward_stability_pass": bool((c or {}).get("walk_forward_stability_pass")),
                    "quality_gates_passed": bool((((c or {}).get("quality_gates") or {}).get("passed"))),
                }
                for c in ranked_candidates[:4]
            ],
        }
        result_data["campaign_stop_reason"] = None
        result_data["campaign_stop_reason_detail"] = None
        for k in [
            "ai_plan",
            "next_experiments",
            "experiment_description",
            "market_regime",
            "regime_memory",
            "risk_profile",
            "execution_realism",
            "ai_risk_guard",
            "goal",
            "constraints",
            "budget",
        ]:
            if k in details:
                result_data[k] = details[k]
        result_data["analysis_summary"] = (
            f"AI выбрал тактику {recommended_tactic or 'unknown'} на окне {backtest_days} дней. "
            f"PnL={float(result_data.get('total_pnl_pct') or 0.0):.2f}% "
            f"WR={float(result_data.get('win_rate') or 0.0):.1f}% "
            f"Trades={int(result_data.get('total_trades') or 0)} "
            f"Score={float(best_score or 0.0):.2f} "
            f"Regime={str((market_regime or {}).get('regime') or 'unknown')}"
        )
        if "mtf" in tactics and isinstance(tactics["mtf"], dict):
            mtf_metrics = tactics["mtf"]
            result_data["best_combo"] = {
                "tactic": "mtf",
                "total_pnl_pct": mtf_metrics.get("total_pnl_pct"),
                "win_rate": mtf_metrics.get("win_rate"),
                "profit_factor": mtf_metrics.get("profit_factor"),
            }
        for single_key in ("single_15m", "single_1h"):
            if single_key in tactics and isinstance(tactics[single_key], dict):
                single_metrics = tactics[single_key]
                result_data["best_single"] = {
                    "tactic": single_key,
                    "total_pnl_pct": single_metrics.get("total_pnl_pct"),
                    "win_rate": single_metrics.get("win_rate"),
                    "profit_factor": single_metrics.get("profit_factor"),
                }
                break
        if result_data.get("best_combo") and result_data.get("best_single"):
            combo_pnl = float((result_data.get("best_combo") or {}).get("total_pnl_pct") or 0.0)
            single_pnl = float((result_data.get("best_single") or {}).get("total_pnl_pct") or 0.0)
            result_data["combo_vs_single_delta_pnl"] = combo_pnl - single_pnl

        update_experiment_status(experiment_id, "completed", _status_payload({
            "progress": 100,
            "current_phase": "completed",
            "results": result_data,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "market_regime": market_regime,
        }))

        def _to_bool(v: Any, default: bool = False) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
            if isinstance(v, (int, float)):
                return bool(v)
            return default

        ai_campaign = details.get("ai_campaign") if isinstance(details.get("ai_campaign"), dict) else {}
        remaining_steps = int(ai_campaign.get("remaining_steps") or 0)
        iteration = int(ai_campaign.get("iteration") or 1)
        max_steps = int(ai_campaign.get("max_steps") or max(iteration, iteration + remaining_steps))
        auto_chain = bool(ai_campaign.get("auto_chain", details.get("auto_iterate", True)))
        if auto_chain and remaining_steps > 0:
            try:
                from bot.ai_agent_service import AIAgentService
                planner = AIAgentService()
                store = ExperimentStore(Path("experiments.json"))
                analyzer = ExperimentAnalyzer(store.list())
                regime_memory_summary = analyzer.summarize_regime_memory(symbol=symbol)
                next_regime_context = build_regime_plan_context(
                    symbol=symbol,
                    experiment_type=exp_type,
                    summary=analyzer.summarize(symbol=symbol),
                    regime_memory=regime_memory_summary,
                    market_regime=market_regime,
                )
                prev_hyperparams = (
                    details.get("hyperparams")
                    if isinstance(details.get("hyperparams"), dict)
                    else ((details.get("ai_plan") or {}).get("hyperparams") if isinstance(details.get("ai_plan"), dict) and isinstance((details.get("ai_plan") or {}).get("hyperparams"), dict) else None)
                )
                hp_search = build_hyperparameter_search_strategy(
                    symbol=symbol,
                    experiment_type=exp_type,
                    market_regime=market_regime if isinstance(market_regime, dict) else next_regime_context.get("market_regime"),
                    regime_memory=next_regime_context.get("regime_memory") if isinstance(next_regime_context.get("regime_memory"), dict) else None,
                    selection=result_data.get("selection") if isinstance(result_data.get("selection"), dict) else None,
                    previous_hyperparams=prev_hyperparams if isinstance(prev_hyperparams, dict) else None,
                )
                chosen_hp = {}
                chosen_candidate = hp_search.get("chosen_candidate") if isinstance(hp_search.get("chosen_candidate"), dict) else {}
                if isinstance(chosen_candidate.get("hyperparams"), dict):
                    chosen_hp = dict(chosen_candidate.get("hyperparams") or {})
                next_campaign = {
                    "auto_chain": True,
                    "remaining_steps": max(0, remaining_steps - 1),
                    "iteration": iteration + 1,
                    "max_steps": max_steps,
                    "root_experiment_id": str(ai_campaign.get("root_experiment_id") or experiment_id),
                    "parent_experiment_id": experiment_id,
                }
                next_meta = {
                    "ai_campaign": next_campaign,
                    "auto_iterate": True,
                    "allow_regime_memory_soft_block": True,
                    "max_steps": max_steps,
                    "campaign_parallel_limit": details.get("campaign_parallel_limit", 2),
                    "baseline": {"parent_experiment_id": experiment_id},
                    "param_changes": {
                        "mode": "auto_iteration",
                        "from_experiment": experiment_id,
                        "regime_context": next_regime_context.get("plan_defaults"),
                        "hyperparameter_search_candidate": chosen_candidate.get("candidate_id"),
                        "hyperparameter_search_strategy": hp_search.get("version"),
                    },
                    "hyperparameter_search": hp_search,
                    "hyperparams": chosen_hp,
                    "hypothesis": f"Итерация {iteration + 1}/{max_steps}: улучшить score относительно baseline {experiment_id} в режиме {str((market_regime or {}).get('regime') or 'unknown')}",
                    "expected_outcome": "Улучшить PnL и/или снизить drawdown при достаточном числе сделок.",
                    "rationale": f"Автоматическая AI-кампания продолжает локальный multi-objective поиск гиперпараметров ({str(chosen_candidate.get('strategy') or 'local_base')}) с учётом режима рынка.",
                    "experiment_description": f"AI campaign iteration {iteration + 1} for {symbol}",
                    "backtest_days": requested_backtest_days,
                    "market_regime": market_regime,
                    "regime_memory_summary": regime_memory_summary,
                    "next_experiments": [
                        str((c or {}).get("rationale"))
                        for c in list(hp_search.get("candidates") or [])[:3]
                        if isinstance(c, dict) and isinstance((c or {}).get("rationale"), str)
                    ],
                    "risk_profile": details.get("risk_profile") if isinstance(details.get("risk_profile"), dict) else None,
                    "execution_realism": details.get("execution_realism") if isinstance(details.get("execution_realism"), dict) else None,
                    "ai_risk_guard": details.get("ai_risk_guard") if isinstance(details.get("ai_risk_guard"), dict) else None,
                }
                res = planner.start_research_experiment(
                    symbol=symbol,
                    experiment_type=exp_type,
                    metadata=next_meta,
                    allow_duplicate=True,
                    safe_mode=_to_bool((details.get("params") or {}).get("safe_mode"), True),
                )
                if not res.get("ok"):
                    result_data["campaign_stop_reason"] = "failed_to_schedule_next"
                    result_data["campaign_stop_reason_detail"] = {
                        "error": res.get("error"),
                        "blocked_reasons": res.get("blocked_reasons"),
                    }
                    patch_experiment(
                        experiment_id,
                        {
                            "results": result_data,
                            "campaign_status": "failed_to_schedule_next",
                            "next_experiment_error": res.get("error"),
                            "next_experiment_blocked_reasons": res.get("blocked_reasons"),
                        },
                    )
                else:
                    result_data["next_experiment_id"] = res.get("experiment_id")
                    result_data["campaign_stop_reason"] = "scheduled_next"
                    patch_experiment(
                        experiment_id,
                        {
                            "results": result_data,
                            "campaign_status": "scheduled_next",
                            "next_experiment_id": res.get("experiment_id"),
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to schedule next campaign iteration: {e}")
                result_data["campaign_stop_reason"] = "failed_to_schedule_next_exception"
                result_data["campaign_stop_reason_detail"] = {"error": str(e)}
                patch_experiment(
                    experiment_id,
                    {
                        "results": result_data,
                        "campaign_status": "failed_to_schedule_next",
                        "next_experiment_error": str(e),
                    },
                )
        else:
            if not auto_chain:
                result_data["campaign_stop_reason"] = "manual_stop"
            elif remaining_steps <= 0:
                result_data["campaign_stop_reason"] = "max_steps_reached"
            else:
                result_data["campaign_stop_reason"] = "no_next_iteration"
            patch_experiment(experiment_id, {"results": result_data})

        notebook_payload = _build_campaign_notebook_payload(str((details.get("ai_campaign") or {}).get("root_experiment_id") or experiment_id))
        patch_experiment(
            experiment_id,
            {
                "research_notebook": notebook_payload,
                "analysis_summary": result_data.get("analysis_summary"),
            },
        )
    except Exception as e:
        logger.error(f"Research experiment failed: {e}", exc_info=True)
        update_experiment_status(
            experiment_id,
            "failed",
            _status_payload({
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }),
        )
        raise
    finally:
        watchdog_stop.set()


if __name__ == "__main__":
    main()
