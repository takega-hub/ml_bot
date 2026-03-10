
import os
import sys
import json
import subprocess
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

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

def update_experiment_status(experiment_id: str, status: str, details: dict = None):
    """Updates the status of an experiment in experiments.json"""
    try:
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
                "created_at": datetime.now().isoformat(),
            }
            
        data[experiment_id]["status"] = status
        data[experiment_id]["updated_at"] = datetime.now().isoformat()
        
        if details:
            for k, v in details.items():
                data[experiment_id][k] = v
                
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Failed to update experiment status: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Research Experiment (Train + Backtest)")
    parser.add_argument("--symbol", required=True, help="Trading pair")
    parser.add_argument("--type", required=True, help="Experiment type (aggressive, conservative, balanced)")
    parser.add_argument("--experiment-id", required=True, help="Unique ID for the experiment")
    
    # Pass-through args for retraining
    parser.add_argument("--interval", default="15m", help="Base interval")
    parser.add_argument("--no-mtf", action="store_true", help="Disable MTF features")
    
    args = parser.parse_args()
    
    experiment_id = args.experiment_id
    symbol = args.symbol
    exp_type = args.type
    
    logger.info(f"Starting research experiment {experiment_id} for {symbol} ({exp_type})")
    update_experiment_status(experiment_id, "starting", {
        "symbol": symbol,
        "type": exp_type,
        "params": {
            "interval": args.interval,
            "no_mtf": args.no_mtf
        }
    })
    
    try:
        # 1. Training Phase
        logger.info("Phase 1: Training Models...")
        update_experiment_status(experiment_id, "training", {"progress": 10})
        
        suffix = f"_{exp_type}_exp"
        
        train_cmd = [
            sys.executable, "retrain_ml_optimized.py",
            "--symbol", symbol,
            "--model-suffix", suffix,
            "--interval", args.interval
        ]
        
        if args.no_mtf:
            train_cmd.append("--no-mtf")
            
        logger.info(f"Running training command: {' '.join(train_cmd)}")
        
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Stream output to capture logs
        logs = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                logger.info(f"[TRAIN] {line}")
                logs.append(line)
                # Update status occasionally with last log
                if len(logs) % 10 == 0:
                     update_experiment_status(experiment_id, "training", {"last_log": line})

        return_code = process.poll()
        if return_code != 0:
            stderr = process.stderr.read()
            raise Exception(f"Training failed with code {return_code}: {stderr}")
            
        logger.info("Training completed successfully.")
        update_experiment_status(experiment_id, "training_completed", {"progress": 50})
        
        # 2. Backtesting Phase (Virtual Trading)
        logger.info("Phase 2: Virtual Trading (Backtest)...")
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
        backtest_cmd = [
            sys.executable, "backtest_mtf_strategy.py",
            "--symbol", symbol,
            "--days", "30", # Standard test period
            "--save"
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
            
        if model_1h_path:
            backtest_cmd.extend(["--model-1h", model_1h_path])
            logger.info(f"Using new 1h model: {model_1h_path}")
        else:
            logger.info("Using best available 1h model (automatic selection)")
            
        logger.info(f"Running backtest command: {' '.join(backtest_cmd)}")
        
        process_bt = subprocess.Popen(
            backtest_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        bt_logs = []
        while True:
            line = process_bt.stdout.readline()
            if not line and process_bt.poll() is not None:
                break
            if line:
                line = line.strip()
                logger.info(f"[BACKTEST] {line}")
                bt_logs.append(line)
                
        return_code_bt = process_bt.poll()
        if return_code_bt != 0:
            stderr_bt = process_bt.stderr.read()
            logger.error(f"Backtest failed: {stderr_bt}")
            # Don't fail the whole experiment, just the backtest part? 
            # No, backtest is crucial for "Virtual Trading" status.
            raise Exception(f"Backtest failed with code {return_code_bt}")

        # 3. Harvest Results
        logger.info("Phase 3: Harvesting Results...")
        
        # Find the generated result JSON
        # backtest_mtf_strategy.py saves to backtest_reports/backtest_mtf_{symbol}_{timestamp}.json
        results_dir = Path("backtest_reports")
        # Find the most recent file for this symbol
        json_files = sorted(
            results_dir.glob(f"backtest_mtf_{symbol}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        result_data = {}
        if json_files:
            latest_file = json_files[0]
            logger.info(f"Found result file: {latest_file}")
            with open(latest_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
        else:
            logger.warning("No result JSON found!")
            
        # Add training logs/metrics to result
        result_data["training_logs"] = logs[-20:] # Last 20 lines of training log
        result_data["models"] = {
            "15m": model_15m_path,
            "1h": model_1h_path
        }
        
        update_experiment_status(experiment_id, "completed", {
            "progress": 100,
            "results": result_data,
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info("Experiment completed successfully.")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        update_experiment_status(experiment_id, "failed", {
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        sys.exit(1)

if __name__ == "__main__":
    main()
