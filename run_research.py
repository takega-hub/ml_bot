
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

# Configure logging
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
_LAST_ACTIVITY_MONO = time.monotonic()
_ACTIVE_SUBPROCS = 0

def _touch_activity():
    global _LAST_ACTIVITY_MONO
    with _ACTIVITY_LOCK:
        _LAST_ACTIVITY_MONO = time.monotonic()

def _set_active_subprocs(delta: int):
    global _ACTIVE_SUBPROCS
    with _ACTIVITY_LOCK:
        _ACTIVE_SUBPROCS = max(0, _ACTIVE_SUBPROCS + int(delta))

def _get_activity_state():
    with _ACTIVITY_LOCK:
        return _LAST_ACTIVITY_MONO, _ACTIVE_SUBPROCS

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
    """Updates the status of an experiment in experiments.json"""
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
        _touch_activity()
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
        _touch_activity()
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
                    _touch_activity()
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
                    _touch_activity()
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
                patch["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                if last_log:
                    patch["last_log"] = last_log
                update_experiment_status(experiment_id, status, patch)
                last_hb = now

            if waiter_done.is_set() and q_out.empty() and q_err.empty():
                break
    finally:
        try:
            waiter_done.wait(timeout=0.5)
        except Exception:
            pass
        _set_active_subprocs(-1)

    return_code = waiter_result.get("rc")
    if return_code is None:
        return_code = process.poll()
    return return_code, lines, process

def main():
    parser = argparse.ArgumentParser(description="Run Research Experiment (Train + Backtest)")
    parser.add_argument("--symbol", required=True, help="Trading pair")
    parser.add_argument("--type", required=True, help="Experiment type (aggressive, conservative, balanced)")
    parser.add_argument("--experiment-id", required=True, help="Unique ID for the experiment")
    parser.add_argument("--metadata-path", default=None, help="Optional JSON file with experiment metadata")
    
    # Pass-through args for retraining
    parser.add_argument("--interval", default="15m", help="Base interval")
    parser.add_argument("--no-mtf", action="store_true", help="Disable MTF features")
    
    args = parser.parse_args()
    
    experiment_id = args.experiment_id
    symbol = args.symbol
    exp_type = args.type
    
    logger.info(f"Starting research experiment {experiment_id} for {symbol} ({exp_type})")
    details = {
        "symbol": symbol,
        "type": exp_type,
        "params": {
            "interval": args.interval,
            "no_mtf": args.no_mtf
        }
    }
    if args.metadata_path:
        try:
            with open(args.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if k not in {"status", "updated_at", "created_at"}:
                        details[k] = v
        except Exception as e:
            logger.error(f"Failed to read metadata file: {e}")
    update_experiment_status(experiment_id, "starting", details)
    
    stop_hb = threading.Event()
    current_phase: Dict[str, str] = {"status": "starting", "step": "starting"}
    hb_started = False
    _touch_activity()

    try:
        artifacts_dir = Path("experiment_artifacts") / experiment_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        training_reports: Dict[str, Any] = {}
        if not hb_started:
            def _hb_loop():
                while not stop_hb.is_set():
                    patch_experiment(
                        experiment_id,
                        {
                            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                            "runner_phase": current_phase.get("status"),
                            "runner_step": current_phase.get("step"),
                        },
                    )
                    stop_hb.wait(10)
            threading.Thread(target=_hb_loop, daemon=True).start()
            hb_started = True

        def _checkpoint(step: str, status_patch: Optional[Dict[str, Any]] = None):
            current_phase["step"] = step
            fields: Dict[str, Any] = {
                "runner_phase": current_phase.get("status"),
                "runner_step": step,
                "last_output_at": datetime.now(timezone.utc).isoformat(),
            }
            if isinstance(status_patch, dict):
                fields.update(status_patch)
            patch_experiment(experiment_id, fields)

        def _watchdog_loop():
            while not stop_hb.is_set():
                last_mono, active = _get_activity_state()
                idle = time.monotonic() - last_mono
                if active == 0 and idle > 1800 and current_phase.get("status") in {"training", "backtesting"}:
                    try:
                        update_experiment_status(
                            experiment_id,
                            "failed",
                            {
                                "error": f"Runner idle for {int(idle)}s with no active subprocesses",
                                "failed_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        pass
                    try:
                        stop_hb.set()
                    except Exception:
                        pass
                    os._exit(1)
                stop_hb.wait(10)

        threading.Thread(target=_watchdog_loop, daemon=True).start()

        intervals = ["15m", "1h"]
        primary = (args.interval or "15m").strip().lower()
        if primary in ("1h", "60m"):
            intervals = ["1h", "15m"]

        # 1. Training Phase
        current_phase["status"] = "training"
        logger.info("Phase 1: Training Models...")
        _checkpoint("training_start")
        update_experiment_status(experiment_id, "training", {"progress": 10})
        
        suffix = f"_{exp_type}_exp"

        logs = []
        for idx, interval in enumerate(intervals):
            current_phase["step"] = f"training_{interval}"
            _checkpoint(f"training_{interval}_start")
            progress = 10 + int(((idx + 1) / len(intervals)) * 40)
            update_experiment_status(
                experiment_id,
                "training",
                {"progress": progress, "last_log": f"Training {interval} started", "heartbeat_at": datetime.now(timezone.utc).isoformat()},
            )

            train_cmd = [
                sys.executable,
                "-u",
                "retrain_ml_optimized.py",
                "--symbol",
                symbol,
                "--model-suffix",
                suffix,
                "--interval",
                interval,
            ]
            training_report_path = artifacts_dir / f"training_report_{interval}.json"
            train_cmd += ["--report-json", str(training_report_path)]

            if args.no_mtf:
                train_cmd.append("--no-mtf")

            logger.info(f"Running training command ({interval}): {' '.join(train_cmd)}")
            return_code, out_lines, _ = run_process_with_heartbeat(
                train_cmd,
                f"TRAIN-{interval}",
                experiment_id,
                "training",
                {"progress": progress},
            )
            _checkpoint(f"training_{interval}_done")
            logs.extend(out_lines[-20:])
            if return_code != 0:
                raise Exception(f"Training failed ({interval}) with code {return_code}")

            try:
                if training_report_path.exists():
                    with open(training_report_path, "r", encoding="utf-8") as f:
                        rep = json.load(f)
                    if isinstance(rep, dict):
                        training_reports[interval] = rep
            except Exception as e:
                logger.error(f"Failed to load training report ({interval}): {e}")

        logger.info("Training completed successfully.")
        _checkpoint("training_done")
        update_experiment_status(experiment_id, "training_completed", {"progress": 50, "last_log": "Training completed"})
        
        # 2. Backtesting Phase (Virtual Trading)
        current_phase["status"] = "backtesting"
        current_phase["step"] = "backtesting_mtf"
        logger.info("Phase 2: Virtual Trading (Backtest)...")
        _checkpoint("backtesting_start")
        update_experiment_status(experiment_id, "backtesting", {"progress": 60})
        
        # Construct model names based on naming convention in retrain_ml_optimized.py
        # rf_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl
        # mode_suffix is "15m" (since no-mtf is default now)
        # model_suffix is "_aggressive_exp"
        
        # We need to find the actual files created to be sure
        models_dir = Path("ml_models")
        
        # Find the newly created models
        # Naming pattern: *_{symbol}_*_{suffix}.pkl
        # Example: rf_BTCUSDT_15_15m_aggressive_exp.pkl
        
        # For MTF Strategy backtest, we need 1h and 15m models.
        # If we trained 15m, we might need a 1h model.
        # If the experiment trained ONLY 15m (which is likely for aggressive), 
        # we need to pair it with an existing 1h model OR use the newly trained one if we trained both?
        # retrain_ml_optimized.py trains ONE interval.
        
        # Assumption: We are testing the newly trained model against the market.
        # If we trained a 15m model, we should use it as the 15m component.
        # We need a 1h component. We can find the best existing 1h model.
        
        # Let's use backtest_mtf_strategy.py which has logic to find models.
        # We will pass the SPECIFIC 15m model we just trained.
        
        # Find the specific 15m model file
        candidates_15m = list(models_dir.glob(f"*_{symbol}_15_*{suffix}.pkl"))
        # Prefer Ensemble or Quad if available
        model_15m_path = None
        for name in ["quad_ensemble", "triple_ensemble", "ensemble", "rf", "xgb"]:
            for c in candidates_15m:
                if name in c.name:
                    model_15m_path = str(c)
                    break
            if model_15m_path: break
            
        if not model_15m_path and candidates_15m:
            model_15m_path = str(candidates_15m[0])
            
        if not model_15m_path:
             logger.warning("Could not find the trained 15m model! Checking for 1h...")
             # Maybe we trained 1h?
             pass

        # Prepare backtest command
        backtest_days = 7
        backtest_cmd = [
            sys.executable, "-u", "backtest_mtf_strategy.py",
            "--symbol", symbol,
            "--days", str(backtest_days),
            "--save",
            "--out-json",
            str(artifacts_dir / "backtest_mtf.json"),
        ]
        
        if model_15m_path:
            backtest_cmd.extend(["--model-15m", model_15m_path])
            logger.info(f"Using new 15m model: {model_15m_path}")
        
        # If we trained a 1h model, pass it
        candidates_1h = list(models_dir.glob(f"*_{symbol}_60_*{suffix}.pkl"))
        if not candidates_1h:
             candidates_1h = list(models_dir.glob(f"*_{symbol}_1h_*{suffix}.pkl"))
             
        model_1h_path = None
        for name in ["quad_ensemble", "triple_ensemble", "ensemble", "rf", "xgb"]:
            for c in candidates_1h:
                if name in c.name:
                    model_1h_path = str(c)
                    break
            if model_1h_path: break

        single_results: Dict[str, Any] = {}
        def _run_single_backtest(tag: str, model_path: str, interval: str) -> Optional[Dict[str, Any]]:
            try:
                out_path = artifacts_dir / f"backtest_single_{tag}.json"
                cmd_single = [
                    sys.executable,
                    "-u",
                    "backtest_ml_strategy.py",
                    "--model",
                    model_path,
                    "--symbol",
                    symbol,
                    "--days",
                    str(backtest_days),
                    "--interval",
                    interval,
                    "--save",
                    "--out-json",
                    str(out_path),
                ]
                logger.info(f"Running single backtest ({tag}): {' '.join(cmd_single)}")
                rc, _, _ = run_process_with_heartbeat(
                    cmd_single,
                    f"SINGLE-{tag}",
                    experiment_id,
                    "backtesting",
                    {"progress": 68},
                )
                if rc != 0:
                    logger.error(f"Single backtest failed ({tag}) with code {rc}")
                    return None
                if out_path.exists():
                    with open(out_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data if isinstance(data, dict) else None
                return None
            except Exception as e:
                logger.error(f"Single backtest error ({tag}): {e}", exc_info=True)
                return None

        if model_15m_path:
            current_phase["step"] = "backtesting_single_15m"
            _checkpoint("backtesting_single_15m_start")
            r15 = _run_single_backtest("15m", model_15m_path, "15m")
            if r15:
                single_results["15m"] = r15
            _checkpoint("backtesting_single_15m_done")

        if model_1h_path:
            current_phase["step"] = "backtesting_single_1h"
            _checkpoint("backtesting_single_1h_start")
            backtest_cmd.extend(["--model-1h", model_1h_path])
            logger.info(f"Using new 1h model: {model_1h_path}")
            r1h = _run_single_backtest("1h", model_1h_path, "1h")
            if r1h:
                single_results["1h"] = r1h
            _checkpoint("backtesting_single_1h_done")
        else:
            logger.info("Using best available 1h model (automatic selection)")
            
        logger.info(f"Running backtest command: {' '.join(backtest_cmd)}")
        _checkpoint("backtesting_mtf_start")
        
        return_code_bt, bt_logs, _ = run_process_with_heartbeat(
            backtest_cmd,
            "BACKTEST",
            experiment_id,
            "backtesting",
            {"progress": 70},
        )
        _checkpoint("backtesting_mtf_done")
        if return_code_bt != 0:
            raise Exception(f"Backtest failed with code {return_code_bt}")

        # 3. Harvest Results
        logger.info("Phase 3: Harvesting Results...")
        _checkpoint("harvest_start")
        
        # Find the generated result JSON
        # backtest_mtf_strategy.py saves to backtest_reports/backtest_mtf_{symbol}_{timestamp}.json
        results_dir = Path("backtest_reports")
        # Find the most recent file for this symbol
        json_files = sorted(
            results_dir.glob(f"backtest_mtf_{symbol}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        mtf_report = {}
        mtf_path = artifacts_dir / "backtest_mtf.json"
        if mtf_path.exists():
            try:
                with open(mtf_path, "r", encoding="utf-8") as f:
                    mtf_report = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load mtf report: {e}")
        elif json_files:
            latest_file = json_files[0]
            logger.info(f"Found result file: {latest_file}")
            with open(latest_file, "r", encoding="utf-8") as f:
                mtf_report = json.load(f)
        else:
            logger.warning("No result JSON found!")
            
        def _score(metrics: Dict[str, Any]) -> float:
            pnl = float(metrics.get("total_pnl_pct") or 0.0)
            dd = float(metrics.get("max_drawdown_pct") or 0.0)
            trades = float(metrics.get("total_trades") or 0.0)
            return pnl - (0.5 * dd) + (0.01 * trades)

        tactics: Dict[str, Any] = {}
        if isinstance(mtf_report, dict) and mtf_report:
            tactics["mtf"] = mtf_report
        if "15m" in single_results:
            tactics["single_15m"] = single_results["15m"]
        if "1h" in single_results:
            tactics["single_1h"] = single_results["1h"]

        recommended_tactic = None
        best_metrics: Dict[str, Any] = {}
        best_score = None
        for name, payload in tactics.items():
            if not isinstance(payload, dict):
                continue
            s = _score(payload)
            if best_score is None or s > best_score:
                best_score = s
                recommended_tactic = name
                best_metrics = payload

        result_data = dict(best_metrics) if isinstance(best_metrics, dict) else {}
        result_data["tactics"] = tactics
        result_data["recommended_tactic"] = recommended_tactic
        result_data["evaluation_window_days"] = backtest_days

        # Add training logs/metrics to result
        result_data["training_logs"] = logs[-20:] # Last 20 lines of training log
        if training_reports:
            result_data["training_report"] = training_reports
        result_data["models"] = {
            "15m": model_15m_path,
            "1h": model_1h_path
        }
        
        update_experiment_status(experiment_id, "completed", {
            "progress": 100,
            "results": result_data,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        
        logger.info("Experiment completed successfully.")
        stop_hb.set()
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        update_experiment_status(experiment_id, "failed", {
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat()
        })
        stop_hb.set()
        sys.exit(1)

if __name__ == "__main__":
    main()
