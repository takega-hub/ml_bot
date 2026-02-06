import os
import subprocess
import sys
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from bot.config import AppSettings
from bot.state import BotState
from bot.ml.strategy_ml import MLStrategy

class ModelManager:
    def __init__(self, settings: AppSettings, state: BotState):
        self.settings = settings
        self.state = state
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(exist_ok=True)

    def train_and_compare(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Запускает переобучение моделей для символа и возвращает отчет.
        Это обертка над существующими скриптами обучения.
        """
        symbol = symbol.upper()
        print(f"[model_manager] Starting training for {symbol}...")
        
        # Вызываем существующий оптимизированный скрипт обучения
        # Мы используем subprocess для изоляции процесса обучения
        try:
            # Нам нужно убедиться, что venv используется
            python_exe = sys.executable
            cmd = [python_exe, "retrain_ml_optimized.py", "--symbol", symbol]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[model_manager] Training output: {result.stdout[-500:]}") # Последние 500 символов
            
            # После обучения ищем лучшую новую модель
            # retrain_ml_optimized сохраняет модели как {type}_{symbol}_{interval}_{mode}.pkl
            new_models = list(self.models_dir.glob(f"*_{symbol}_*.pkl"))
            if not new_models:
                return None
                
            # Сортируем по времени изменения, чтобы найти самые свежие
            new_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Для каждой новой модели мы могли бы запустить бэктест
            # Но retrain_ml_optimized уже выводит CV Score.
            # В идеале мы запускаем compare_ml_models.py
            
            compare_cmd = [python_exe, "compare_ml_models.py", "--symbols", symbol, "--days", "14", "--output", "csv"]
            subprocess.run(compare_cmd, capture_output=True, text=True)
            
            # Ищем последний CSV отчет сравнения
            reports = list(Path(".").glob(f"ml_models_comparison_*.csv"))
            if not reports:
                return None
            reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            df = pd.read_csv(reports[0])
            symbol_results = df[df['symbol'] == symbol]
            
            if symbol_results.empty:
                return None
                
            best_new = symbol_results.iloc[0].to_dict()
            
            # Получаем метрики текущей модели для сравнения, если она есть
            current_model_path = self.state.symbol_models.get(symbol)
            comparison = {
                "symbol": symbol,
                "new_model": best_new,
                "current_model_path": current_model_path
            }
            
            return comparison
            
        except Exception as e:
            print(f"[model_manager] Error during training/comparison for {symbol}: {e}")
            return None

    def find_models_for_symbol(self, symbol: str) -> list:
        """Находит все доступные модели для символа"""
        symbol = symbol.upper()
        models = []
        
        # Ищем модели в формате: {type}_{SYMBOL}_*.pkl
        patterns = [
            f"*_{symbol}_*.pkl",
            f"*{symbol}*.pkl"  # Более широкий паттерн
        ]
        
        for pattern in patterns:
            for model_file in self.models_dir.glob(pattern):
                if model_file.is_file() and model_file not in models:
                    models.append(model_file)
        
        # Сортируем по времени изменения (новые первыми)
        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return models

    def test_model(self, model_path: str, symbol: str, days: int = 14) -> Optional[Dict[str, Any]]:
        """Тестирует модель на исторических данных и возвращает метрики"""
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        
        try:
            from backtest_ml_strategy import run_exact_backtest
            
            logger.info(f"[test_model] Starting backtest for {model_path} on {symbol} ({days} days)")
            
            metrics = run_exact_backtest(
                model_path=str(model_path),
                symbol=symbol,
                days_back=days,
                interval="15",  # 15 минут в формате для backtest
                initial_balance=1000.0,
                risk_per_trade=0.02,
                leverage=10,
            )
            
            if metrics:
                logger.info(f"[test_model] Backtest completed successfully for {model_path}")
                return {
                    "total_pnl_pct": metrics.total_pnl_pct,
                    "win_rate": metrics.win_rate,
                    "total_trades": metrics.total_trades,
                    "trades_per_day": metrics.trade_frequency_per_day,
                    "profit_factor": metrics.profit_factor,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                }
            else:
                logger.warning(f"[test_model] Backtest returned None for {model_path}")
                return None
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"[model_manager] Error testing model {model_path}: {error_msg}")
            logger.error(f"[model_manager] Traceback: {error_traceback}")
            print(f"[model_manager] Error testing model {model_path}: {error_msg}")
            print(f"[model_manager] Traceback: {error_traceback}")
            return None

    def get_model_test_results(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Получает сохраненные результаты тестов для всех моделей символа"""
        results_file = Path(f"model_test_results_{symbol}.json")
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[model_manager] Error loading test results: {e}")
        return {}

    def save_model_test_result(self, symbol: str, model_path: str, results: Dict[str, Any]):
        """Сохраняет результаты теста модели"""
        results_file = Path(f"model_test_results_{symbol}.json")
        all_results = self.get_model_test_results(symbol)
        all_results[str(model_path)] = results
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[model_manager] Error saving test results: {e}")

    def apply_model(self, symbol: str, model_path: str):
        with self.state.lock:
            self.state.symbol_models[symbol] = model_path
        self.state.save()
        print(f"[model_manager] Applied model {model_path} for {symbol}")
