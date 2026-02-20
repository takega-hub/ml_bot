"""
Оптимизированный скрипт для выбора лучших MTF комбинаций:
1. Тренирует модели (1h и 15m)
2. Делает предсказания (быстрый скрининг топ-15)
3. Реально тестирует только топ-15 предсказанных комбинаций
4. Выбирает лучшую и сохраняет для использования в боте

Использование:
    python optimize_mtf_strategies.py --symbols BTCUSDT,ETHUSDT
    python optimize_mtf_strategies.py --skip-training  # Пропустить обучение
    python optimize_mtf_strategies.py --full  # Полный цикл
"""
import argparse
import subprocess
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import traceback

# Настройка логирования
log_file = f'optimize_mtf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорты
from bot.state import BotState
from bot.config import load_settings
from backtest_mtf_strategy import run_mtf_backtest, find_all_models_for_symbol
from predict_mtf_from_single import MTFPredictor


class OptimizedMTFOptimizer:
    """Оптимизированный оптимизатор MTF стратегий"""
    
    def __init__(
        self,
        symbols: List[str],
        days: int = 30,
        output_dir: str = "mtf_optimization",
        skip_training: bool = False,
        skip_prediction: bool = False,
        top_n_predictions: int = 15,  # Тестируем только топ-15 предсказанных
        apply_to_bot: bool = True,  # Автоматически применять к боту
    ):
        self.symbols = [s.upper() for s in symbols]
        self.days = days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.skip_training = skip_training
        self.skip_prediction = skip_prediction
        self.top_n_predictions = top_n_predictions
        self.apply_to_bot = apply_to_bot
        
        self.python_exe = sys.executable
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Результаты
        self.training_results: Dict[str, bool] = {}
        self.prediction_results: Dict[str, pd.DataFrame] = {}
        self.backtest_results: Dict[str, pd.DataFrame] = {}
        self.best_combinations: Dict[str, Dict[str, Any]] = {}
        
        # Ошибки
        self.errors: List[Dict[str, Any]] = []
    
    def log_error(self, stage: str, symbol: str, error: Exception):
        """Логирует ошибку"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "symbol": symbol,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        self.errors.append(error_info)
        logger.error(f"[{stage}] {symbol}: {error}", exc_info=True)
    
    def train_models(self, symbol: str) -> bool:
        """Обучает модели для символа (15m и 1h)"""
        logger.info(f"[TRAINING] Начало обучения моделей для {symbol}")
        
        try:
            # Обучаем 15m модели
            logger.info(f"[TRAINING] {symbol}: Обучение 15m моделей...")
            cmd_15m = [
                self.python_exe,
                "retrain_ml_optimized.py",
                "--symbol", symbol,
                "--no-mtf"
            ]
            result_15m = subprocess.run(
                cmd_15m,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result_15m.returncode != 0:
                logger.error(f"[TRAINING] {symbol}: Ошибка обучения 15m моделей")
                logger.error(f"STDERR: {result_15m.stderr[-500:]}")
                return False
            
            # Обучаем 1h модели
            logger.info(f"[TRAINING] {symbol}: Обучение 1h моделей...")
            cmd_1h = [
                self.python_exe,
                "retrain_ml_optimized.py",
                "--symbol", symbol,
                "--no-mtf",
                "--interval", "60m"
            ]
            result_1h = subprocess.run(
                cmd_1h,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result_1h.returncode != 0:
                logger.error(f"[TRAINING] {symbol}: Ошибка обучения 1h моделей")
                logger.error(f"STDERR: {result_1h.stderr[-500:]}")
                return False
            
            logger.info(f"[TRAINING] {symbol}: Обучение завершено успешно")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"[TRAINING] {symbol}: Таймаут при обучении")
            return False
        except Exception as e:
            self.log_error("TRAINING", symbol, e)
            return False
    
    def test_single_models(self) -> bool:
        """Тестирует одиночные модели для получения результатов"""
        logger.info("[TESTING] Начало тестирования одиночных моделей")
        
        try:
            cmd = [
                self.python_exe,
                "compare_ml_models.py",
                "--symbols", ",".join(self.symbols),
                "--days", str(self.days),
                "--output", "csv",
                "--detailed-analysis"
            ]
            
            logger.info(f"[TESTING] Команда: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            if result.returncode != 0:
                logger.error("[TESTING] Ошибка тестирования моделей")
                logger.error(f"STDERR: {result.stderr[-500:]}")
                return False
            
            logger.info("[TESTING] Тестирование завершено успешно")
            return True
            
        except Exception as e:
            logger.error(f"[TESTING] Ошибка: {e}", exc_info=True)
            return False
    
    def predict_best_combinations(self) -> bool:
        """Предсказывает лучшие MTF комбинации на основе одиночных моделей"""
        logger.info("[PREDICTION] Начало предсказания лучших комбинаций")
        
        try:
            predictor = MTFPredictor(
                symbols=self.symbols,
                days=self.days,
                top_n=self.top_n_predictions * 2,  # Берем больше для запаса
                skip_testing=False  # Нужно протестировать одиночные модели
            )
            
            # Запускаем предсказание
            success = predictor.run()
            
            if not success:
                logger.error("[PREDICTION] Ошибка предсказания")
                return False
            
            # Сохраняем результаты предсказаний
            for symbol in self.symbols:
                if symbol in predictor.predictions:
                    self.prediction_results[symbol] = predictor.predictions[symbol]
                    logger.info(f"[PREDICTION] {symbol}: Найдено {len(predictor.predictions[symbol])} предсказанных комбинаций")
            
            # Сохраняем лучшие предсказанные комбинации
            if hasattr(predictor, 'best_combinations'):
                self.best_combinations = predictor.best_combinations.copy()
            
            logger.info("[PREDICTION] Предсказание завершено успешно")
            return True
            
        except Exception as e:
            logger.error(f"[PREDICTION] Ошибка: {e}", exc_info=True)
            return False
    
    def backtest_top_predictions(self, symbol: str) -> Optional[pd.DataFrame]:
        """Реально тестирует топ-N предсказанных комбинаций"""
        logger.info(f"[BACKTEST] {symbol}: Начало реального тестирования топ-{self.top_n_predictions} комбинаций")
        
        try:
            # Получаем предсказанные комбинации
            if symbol not in self.prediction_results:
                logger.warning(f"[BACKTEST] {symbol}: Нет предсказанных комбинаций")
                return None
            
            predictions_df = self.prediction_results[symbol]
            
            # Берем топ-N по predicted_score
            top_predictions = predictions_df.nlargest(self.top_n_predictions, 'predicted_score')
            
            logger.info(f"[BACKTEST] {symbol}: Тестируем {len(top_predictions)} комбинаций из {len(predictions_df)} предсказанных")
            
            # Находим все модели
            models_1h, models_15m = find_all_models_for_symbol(symbol)
            
            if not models_1h or not models_15m:
                logger.error(f"[BACKTEST] {symbol}: Не найдено моделей")
                return None
            
            # Создаем маппинг имен моделей к путям
            model_1h_map = {Path(m).stem: m for m in models_1h}
            model_15m_map = {Path(m).stem: m for m in models_15m}
            
            # Результаты реального тестирования
            backtest_results = []
            
            for idx, row in top_predictions.iterrows():
                model_1h_name = row['model_1h']
                model_15m_name = row['model_15m']
                
                # Находим пути к моделям
                model_1h_path = model_1h_map.get(model_1h_name)
                model_15m_path = model_15m_map.get(model_15m_name)
                
                if not model_1h_path or not model_15m_path:
                    logger.warning(f"[BACKTEST] {symbol}: Модели не найдены: {model_1h_name}, {model_15m_name}")
                    continue
                
                logger.info(f"[BACKTEST] {symbol}: Тестируем {model_1h_name} + {model_15m_name}")
                
                try:
                    # Запускаем реальный бэктест
                    result = run_mtf_backtest(
                        symbol=symbol,
                        days_back=self.days,
                        model_1h_path=model_1h_path,
                        model_15m_path=model_15m_path,
                        confidence_threshold_1h=0.50,
                        confidence_threshold_15m=0.35,
                        alignment_mode="strict",
                        require_alignment=True
                    )
                    
                    if result:
                        # win_rate в BacktestMetrics это процент (0-100), конвертируем в долю (0-1)
                        win_rate_decimal = result.win_rate / 100.0 if result.win_rate > 1.0 else result.win_rate
                        
                        backtest_results.append({
                            'model_1h': model_1h_name,
                            'model_15m': model_15m_name,
                            'total_pnl_pct': result.total_pnl_pct,
                            'win_rate': win_rate_decimal,
                            'profit_factor': result.profit_factor,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown_pct': result.max_drawdown_pct,
                            'total_trades': result.total_trades,
                            'predicted_pnl_pct': row.get('predicted_pnl_pct', 0),
                            'predicted_score': row.get('predicted_score', 0),
                        })
                        logger.info(f"[BACKTEST] {symbol}: {model_1h_name} + {model_15m_name}: "
                                  f"PnL={result.total_pnl_pct:.2f}%, WR={result.win_rate*100:.1f}%, "
                                  f"PF={result.profit_factor:.2f}")
                
                except Exception as e:
                    logger.error(f"[BACKTEST] {symbol}: Ошибка тестирования {model_1h_name} + {model_15m_name}: {e}")
                    continue
            
            if not backtest_results:
                logger.warning(f"[BACKTEST] {symbol}: Нет успешных результатов")
                return None
            
            # Создаем DataFrame с результатами
            df_results = pd.DataFrame(backtest_results)
            df_results = df_results.sort_values('total_pnl_pct', ascending=False)
            
            logger.info(f"[BACKTEST] {symbol}: Завершено, протестировано {len(df_results)} комбинаций")
            
            return df_results
            
        except Exception as e:
            self.log_error("BACKTEST", symbol, e)
            return None
    
    def select_best_combination(self, symbol: str, backtest_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Выбирает лучшую комбинацию на основе реальных результатов"""
        if backtest_df.empty:
            return None
        
        # Выбираем лучшую по composite score (комбинация PnL, WR, PF, Sharpe)
        backtest_df['composite_score'] = (
            backtest_df['total_pnl_pct'] * 0.4 +
            backtest_df['win_rate'] * 100 * 0.2 +
            backtest_df['profit_factor'] * 10 * 0.2 +
            backtest_df['sharpe_ratio'] * 0.2
        )
        
        best_row = backtest_df.loc[backtest_df['composite_score'].idxmax()]
        
        # Находим пути к моделям
        models_1h, models_15m = find_all_models_for_symbol(symbol)
        model_1h_map = {Path(m).stem: m for m in models_1h}
        model_15m_map = {Path(m).stem: m for m in models_15m}
        
        model_1h_path = model_1h_map.get(best_row['model_1h'])
        model_15m_path = model_15m_map.get(best_row['model_15m'])
        
        if not model_1h_path or not model_15m_path:
            logger.error(f"[SELECT] {symbol}: Модели не найдены для лучшей комбинации")
            return None
        
        best_combination = {
            'symbol': symbol,
            'model_1h': best_row['model_1h'],
            'model_15m': best_row['model_15m'],
            'model_1h_path': str(model_1h_path),
            'model_15m_path': str(model_15m_path),
            'total_pnl_pct': float(best_row['total_pnl_pct']),
            'win_rate': float(best_row['win_rate']),
            'profit_factor': float(best_row['profit_factor']),
            'sharpe_ratio': float(best_row['sharpe_ratio']),
            'max_drawdown_pct': float(best_row['max_drawdown_pct']),
            'total_trades': int(best_row['total_trades']),
            'composite_score': float(best_row['composite_score']),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"[SELECT] {symbol}: Выбрана лучшая комбинация:")
        logger.info(f"  1h: {best_combination['model_1h']}")
        logger.info(f"  15m: {best_combination['model_15m']}")
        logger.info(f"  PnL: {best_combination['total_pnl_pct']:.2f}%")
        logger.info(f"  WR: {best_combination['win_rate']*100:.1f}%")
        logger.info(f"  PF: {best_combination['profit_factor']:.2f}")
        logger.info(f"  Sharpe: {best_combination['sharpe_ratio']:.2f}")
        
        return best_combination
    
    def save_results(self):
        """Сохраняет результаты оптимизации"""
        logger.info("[SAVE] Сохранение результатов оптимизации")
        
        try:
            # Сохраняем лучшие комбинации в JSON
            results_file = self.output_dir / f"best_mtf_combinations_{self.timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.best_combinations, f, indent=2, ensure_ascii=False)
            logger.info(f"[SAVE] Лучшие комбинации сохранены в {results_file}")
            
            # Сохраняем результаты бэктестов в CSV
            for symbol, df in self.backtest_results.items():
                csv_file = self.output_dir / f"backtest_results_{symbol}_{self.timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"[SAVE] Результаты бэктеста для {symbol} сохранены в {csv_file}")
            
            # Сохраняем в формате для бота (ml_settings.json)
            if self.apply_to_bot:
                self.save_to_bot_config()
            
        except Exception as e:
            logger.error(f"[SAVE] Ошибка сохранения: {e}", exc_info=True)
    
    def save_to_bot_config(self):
        """Сохраняет лучшие комбинации в ml_settings.json для использования в боте"""
        logger.info("[SAVE] Сохранение в конфигурацию бота")
        
        try:
            settings_file = Path("ml_settings.json")
            
            # Загружаем существующие настройки
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                settings = {}
            
            # Добавляем/обновляем секцию с лучшими MTF моделями
            if 'mtf_models' not in settings:
                settings['mtf_models'] = {}
            
            for symbol, combo in self.best_combinations.items():
                if not combo:
                    logger.warning(f"[SAVE] {symbol}: Комбинация пустая, пропускаем")
                    continue
                
                # Проверяем наличие обязательных ключей
                required_keys = ['model_1h', 'model_15m', 'model_1h_path', 'model_15m_path']
                missing_keys = [key for key in required_keys if key not in combo]
                
                if missing_keys:
                    logger.warning(f"[SAVE] {symbol}: Отсутствуют ключи {missing_keys}, пропускаем сохранение")
                    continue
                
                try:
                    settings['mtf_models'][symbol] = {
                        'model_1h': combo['model_1h'],
                        'model_15m': combo['model_15m'],
                        'model_1h_path': combo['model_1h_path'],
                        'model_15m_path': combo['model_15m_path'],
                        'metrics': {
                            'total_pnl_pct': combo.get('total_pnl_pct', 0),
                            'win_rate': combo.get('win_rate', 0),
                            'profit_factor': combo.get('profit_factor', 0),
                            'sharpe_ratio': combo.get('sharpe_ratio', 0),
                            'max_drawdown_pct': combo.get('max_drawdown_pct', 0),
                            'total_trades': combo.get('total_trades', 0)
                        },
                        'optimized_at': combo.get('timestamp', datetime.now().isoformat())
                    }
                    logger.info(f"[SAVE] {symbol}: Сохранена лучшая комбинация в ml_settings.json")
                except Exception as e:
                    logger.error(f"[SAVE] {symbol}: Ошибка при сохранении комбинации: {e}")
                    continue
            
            # Сохраняем обновленные настройки
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SAVE] Настройки сохранены в {settings_file}")
            
        except Exception as e:
            logger.error(f"[SAVE] Ошибка сохранения в конфигурацию бота: {e}", exc_info=True)
    
    def run(self) -> bool:
        """Запускает полный цикл оптимизации"""
        logger.info("=" * 80)
        logger.info("🚀 НАЧАЛО ОПТИМИЗАЦИИ MTF СТРАТЕГИЙ")
        logger.info("=" * 80)
        logger.info(f"Символы: {', '.join(self.symbols)}")
        logger.info(f"Дни бэктеста: {self.days}")
        logger.info(f"Топ-N для тестирования: {self.top_n_predictions}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Этап 1: Обучение моделей
        if not self.skip_training:
            logger.info("\n[ЭТАП 1] ОБУЧЕНИЕ МОДЕЛЕЙ")
            logger.info("-" * 80)
            for symbol in self.symbols:
                success = self.train_models(symbol)
                self.training_results[symbol] = success
                if not success:
                    logger.error(f"[TRAINING] {symbol}: Обучение не удалось, пропускаем символ")
        else:
            logger.info("\n[ЭТАП 1] ОБУЧЕНИЕ МОДЕЛЕЙ - ПРОПУЩЕНО")
            for symbol in self.symbols:
                self.training_results[symbol] = True
        
        # Этап 2: Тестирование одиночных моделей
        logger.info("\n[ЭТАП 2] ТЕСТИРОВАНИЕ ОДИНОЧНЫХ МОДЕЛЕЙ")
        logger.info("-" * 80)
        if not self.test_single_models():
            logger.error("[TESTING] Ошибка тестирования одиночных моделей")
            return False
        
        # Этап 3: Предсказание лучших комбинаций
        if not self.skip_prediction:
            logger.info("\n[ЭТАП 3] ПРЕДСКАЗАНИЕ ЛУЧШИХ КОМБИНАЦИЙ")
            logger.info("-" * 80)
            if not self.predict_best_combinations():
                logger.error("[PREDICTION] Ошибка предсказания")
                return False
        else:
            logger.info("\n[ЭТАП 3] ПРЕДСКАЗАНИЕ - ПРОПУЩЕНО")
            # Загружаем существующие предсказания из файлов
            try:
                prediction_files = sorted(
                    Path("mtf_predictions").glob("predicted_mtf_*_*.csv"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True
                )
                
                for symbol in self.symbols:
                    # Ищем файл для символа
                    symbol_file = None
                    for f in prediction_files:
                        if symbol in f.name:
                            symbol_file = f
                            break
                    
                    if symbol_file and symbol_file.exists():
                        df = pd.read_csv(symbol_file)
                        self.prediction_results[symbol] = df
                        logger.info(f"[PREDICTION] {symbol}: Загружено {len(df)} предсказаний из {symbol_file.name}")
            except Exception as e:
                logger.warning(f"[PREDICTION] Ошибка загрузки существующих предсказаний: {e}")
        
        # Этап 4: Реальное тестирование топ-комбинаций
        logger.info("\n[ЭТАП 4] РЕАЛЬНОЕ ТЕСТИРОВАНИЕ ТОП-КОМБИНАЦИЙ")
        logger.info("-" * 80)
        
        # Очищаем best_combinations перед реальным тестированием
        # (предыдущие значения из predictor могут иметь другую структуру)
        self.best_combinations = {}
        
        for symbol in self.symbols:
            if symbol not in self.prediction_results:
                logger.warning(f"[BACKTEST] {symbol}: Нет предсказаний, пропускаем")
                continue
            
            backtest_df = self.backtest_top_predictions(symbol)
            if backtest_df is not None and not backtest_df.empty:
                self.backtest_results[symbol] = backtest_df
                
                # Выбираем лучшую комбинацию
                best_combo = self.select_best_combination(symbol, backtest_df)
                if best_combo:
                    self.best_combinations[symbol] = best_combo
        
        # Этап 5: Сохранение результатов
        logger.info("\n[ЭТАП 5] СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        logger.info("-" * 80)
        self.save_results()
        
        # Итоговая сводка
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        logger.info("\n" + "=" * 80)
        logger.info("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        logger.info("=" * 80)
        logger.info(f"Время выполнения: {elapsed:.1f} минут")
        logger.info(f"Обработано символов: {len(self.symbols)}")
        logger.info(f"Найдено лучших комбинаций: {len(self.best_combinations)}")
        
        if self.best_combinations:
            logger.info("\n🏆 ЛУЧШИЕ КОМБИНАЦИИ:")
            for symbol, combo in self.best_combinations.items():
                logger.info(f"\n{symbol}:")
                logger.info(f"  1h: {combo['model_1h']}")
                logger.info(f"  15m: {combo['model_15m']}")
                logger.info(f"  PnL: {combo['total_pnl_pct']:.2f}%")
                logger.info(f"  WR: {combo['win_rate']*100:.1f}%")
                logger.info(f"  PF: {combo['profit_factor']:.2f}")
                logger.info(f"  Sharpe: {combo['sharpe_ratio']:.2f}")
        
        logger.info("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Оптимизированная оптимизация MTF стратегий")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT",
                       help="Символы для оптимизации (через запятую)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для бэктеста")
    parser.add_argument("--skip-training", action="store_true",
                       help="Пропустить обучение моделей")
    parser.add_argument("--skip-prediction", action="store_true",
                       help="Пропустить предсказание (использовать существующие)")
    parser.add_argument("--top-n", type=int, default=15,
                       help="Количество топ-комбинаций для реального тестирования")
    parser.add_argument("--no-apply", action="store_true",
                       help="Не применять результаты к боту")
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    optimizer = OptimizedMTFOptimizer(
        symbols=symbols,
        days=args.days,
        skip_training=args.skip_training,
        skip_prediction=args.skip_prediction,
        top_n_predictions=args.top_n,
        apply_to_bot=not args.no_apply
    )
    
    success = optimizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
