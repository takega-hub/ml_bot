"""
Автоматическая оптимизация стратегий: обучение моделей, сравнение, тестирование MTF комбинаций
и автоматический выбор лучших стратегий.

Использование:
    python auto_strategy_optimizer.py --symbols BTCUSDT,ETHUSDT
    python auto_strategy_optimizer.py --skip-training  # Пропустить обучение
    python auto_strategy_optimizer.py --full  # Полный цикл
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

# Настройка логирования с безопасной обработкой эмодзи для Windows
import sys
import codecs

# Безопасная функция для логирования (убирает эмодзи для Windows)
def safe_log_message(msg: str) -> str:
    """Убирает эмодзи из сообщения для совместимости с Windows"""
    if sys.platform == 'win32':
        # Заменяем основные эмодзи на текстовые метки
        replacements = {
            '🚀': '[START]',
            '📊': '[INFO]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '🔄': '[RETRAIN]',
            '📦': '[DATA]',
            '🤖': '[MODEL]',
            '🎯': '[TARGET]',
            '📈': '[CHART]',
            '🧠': '[ML]',
            '💡': '[TIP]',
            '🔍': '[SEARCH]',
            '🏆': '[BEST]',
            '📥': '[DOWNLOAD]',
            '🔧': '[ENGINEERING]',
            '⏳': '[WAIT]',
            '🔥': '[HOT]',
            '🌲': '[RF]',
            '⚡': '[XGB]',
            '🎉': '[SUCCESS]',
            '📋': '[LIST]',
            '📝': '[NOTE]',
            '💪': '[STRONG]',
            '🔹': '[INFO]',
            'ℹ️': '[INFO]',
        }
        for emoji, replacement in replacements.items():
            msg = msg.replace(emoji, replacement)
    return msg

class SafeStreamHandler(logging.StreamHandler):
    """Обработчик логов, который безопасно обрабатывает эмодзи"""
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = safe_log_message(msg)
            stream = self.stream
            # Пытаемся записать с обработкой ошибок кодировки
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Если все еще ошибка, удаляем все не-ASCII символы
                msg_clean = ''.join(c for c in msg if ord(c) < 128)
                stream.write(msg_clean + self.terminator)
            self.flush()
        except Exception as e:
            # В случае критической ошибки просто пропускаем
            try:
                stream.write(f"[LOG ERROR: {type(e).__name__}]\n")
            except:
                pass

# Настройка логирования
log_file = f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        SafeStreamHandler()  # Используем безопасный обработчик для консоли
    ]
)
logger = logging.getLogger(__name__)

# Импорты для работы с ботом
from bot.state import BotState
from bot.config import load_settings
from backtest_mtf_strategy import run_mtf_backtest_all_combinations, find_all_models_for_symbol


class StrategyOptimizer:
    """Класс для автоматической оптимизации стратегий"""
    
    def __init__(
        self,
        symbols: List[str],
        days: int = 30,
        output_dir: str = "optimization_results",
        skip_training: bool = False,
        skip_comparison: bool = False,
        skip_mtf_testing: bool = False,
        mtf_top_n: int = 5,
        full_mtf_testing: bool = False,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.days = days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.skip_training = skip_training
        self.skip_comparison = skip_comparison
        self.skip_mtf_testing = skip_mtf_testing
        self.mtf_top_n = mtf_top_n
        self.full_mtf_testing = full_mtf_testing
        
        self.python_exe = sys.executable
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Результаты
        self.training_results: Dict[str, bool] = {}
        self.comparison_results: Dict[str, Optional[str]] = {}  # symbol -> csv_path
        self.mtf_results: Dict[str, Optional[pd.DataFrame]] = {}  # symbol -> DataFrame
        self.best_strategies: Dict[str, Dict[str, Any]] = {}
        
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

    def run_command(self, cmd: List[str], timeout: int = 7200) -> int:
        """Запускает команду с выводом логов в реальном времени"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            # Читаем вывод в реальном времени
            if process.stdout:
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        logger.info(f"[SUBPROCESS] {line}")
            
            # Ждем завершения
            return_code = process.wait(timeout=timeout)
            return return_code
            
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"[SUBPROCESS] Таймаут выполнения команды (>{timeout}с)")
            return -1
        except Exception as e:
            logger.error(f"[SUBPROCESS] Ошибка запуска команды: {e}")
            return -1
    
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
                "--no-mtf"  # 15m модели без MTF
            ]
            
            return_code_15m = self.run_command(cmd_15m, timeout=7200)
            
            if return_code_15m != 0:
                logger.error(f"[TRAINING] {symbol}: Ошибка обучения 15m моделей")
                return False
            
            # Обучаем 1h модели
            logger.info(f"[TRAINING] {symbol}: Обучение 1h моделей...")
            cmd_1h = [
                self.python_exe,
                "retrain_ml_optimized.py",
                "--symbol", symbol,
                "--no-mtf",
                "--interval", "60m"  # 1h интервал (60m или 1h)
            ]
            
            return_code_1h = self.run_command(cmd_1h, timeout=7200)
            
            if return_code_1h != 0:
                logger.error(f"[TRAINING] {symbol}: Ошибка обучения 1h моделей")
                return False
            
            logger.info(f"[TRAINING] {symbol}: Обучение завершено успешно")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"[TRAINING] {symbol}: Таймаут при обучении")
            return False
        except Exception as e:
            self.log_error("TRAINING", symbol, e)
            return False
    
    def compare_models(self) -> bool:
        """Сравнивает все модели (15m и 1h вместе)"""
        logger.info("[COMPARISON] Начало сравнения моделей")
        logger.info("[COMPARISON] Тестирование всех моделей (15m и 1h) для всех символов...")
        
        try:
            # Тестируем все модели сразу (15m и 1h вместе)
            # compare_ml_models.py автоматически определит интервал из имени модели
            cmd = [
                self.python_exe,
                "compare_ml_models.py",
                "--symbols", ",".join(self.symbols),
                "--days", str(self.days),
                "--output", "csv",
                "--interval", "15m",  # Базовый интервал, но скрипт определит правильный из имени модели
                "--detailed-analysis"
            ]
            
            logger.info(f"[COMPARISON] Команда: {' '.join(cmd)}")
            
            return_code = self.run_command(cmd, timeout=7200)
            
            if return_code != 0:
                logger.error("[COMPARISON] Ошибка сравнения моделей")
                return False
            
            # Находим последний файл сравнения
            comparison_files = sorted(
                Path(".").glob("ml_models_comparison_*.csv"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            
            if comparison_files:
                latest_file = comparison_files[0]
                logger.info(f"[COMPARISON] Последний файл сравнения: {latest_file}")
                
                # Проверяем, что файл содержит и 15m, и 1h модели
                try:
                    df_check = pd.read_csv(latest_file)
                    has_15m = df_check['model_filename'].str.contains('_15_|_15m', na=False).any()
                    has_1h = df_check['model_filename'].str.contains('_60_|_1h', na=False).any()
                    
                    if has_15m and has_1h:
                        logger.info("[COMPARISON] ✅ Файл содержит и 15m, и 1h модели")
                    elif has_15m:
                        logger.warning("[COMPARISON] ⚠️  Файл содержит только 15m модели")
                    elif has_1h:
                        logger.warning("[COMPARISON] ⚠️  Файл содержит только 1h модели")
                except Exception as e:
                    logger.warning(f"[COMPARISON] Не удалось проверить содержимое файла: {e}")
                
                return True
            else:
                logger.warning("[COMPARISON] Файлы сравнения не найдены")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[COMPARISON] Таймаут при сравнении")
            return False
        except Exception as e:
            self.log_error("COMPARISON", "ALL", e)
            return False
    
    def select_top_models(self, symbol: str, timeframe: str, top_n: int = 5) -> List[str]:
        """
        Выбирает топ-N моделей для символа и таймфрейма на основе composite score.
        
        Args:
            symbol: Символ
            timeframe: '1h' или '15m'
            top_n: Количество топ-моделей
            
        Returns:
            Список имен моделей (без .pkl)
        """
        # Загружаем результаты сравнения моделей
        comparison_files = sorted(
            Path(".").glob("ml_models_comparison_*.csv"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        
        if not comparison_files:
            logger.warning(f"[SELECTION] Не найдено файлов сравнения для {symbol}")
            return []
        
        try:
            df_comparison = pd.read_csv(comparison_files[0])
            symbol_data = df_comparison[df_comparison['symbol'] == symbol].copy()
            
            # Фильтруем по таймфрейму
            if timeframe == '1h':
                filtered = symbol_data[
                    (symbol_data['mode_suffix'] == '1h') |
                    (symbol_data['model_filename'].str.contains('_60_|_1h', na=False))
                ].copy()
            else:  # 15m
                filtered = symbol_data[
                    (symbol_data['mode_suffix'] == '15m') |
                    (symbol_data['model_filename'].str.contains('_15_|_15m', na=False))
                ].copy()
            
            if filtered.empty:
                logger.warning(f"[SELECTION] Не найдено {timeframe} моделей для {symbol}")
                return []
            
            # Вычисляем composite score
            filtered['composite_score'] = filtered.apply(
                lambda row: self.calculate_composite_score({
                    'total_pnl_pct': row.get('total_pnl_pct', 0),
                    'win_rate': row.get('win_rate_pct', 0),
                    'profit_factor': row.get('profit_factor', 0),
                    'sharpe_ratio': row.get('sharpe_ratio', 0),
                    'max_drawdown_pct': row.get('max_drawdown_pct', 100),
                }),
                axis=1
            )
            
            # Сортируем по score
            filtered = filtered.sort_values('composite_score', ascending=False)
            
            # Берем топ-N
            top_models = filtered.head(top_n)
            
            # Извлекаем имена моделей (без .pkl)
            model_names = [name.replace('.pkl', '') for name in top_models['model_filename'].tolist()]
            
            logger.info(f"[SELECTION] {symbol} {timeframe}: Выбрано {len(model_names)} топ-моделей")
            for i, model in enumerate(model_names, 1):
                score = top_models.iloc[i-1]['composite_score']
                pnl = top_models.iloc[i-1]['total_pnl_pct']
                logger.info(f"   {i}. {model} (score: {score:.2f}, PnL: {pnl:.2f}%)")
            
            return model_names
            
        except Exception as e:
            self.log_error("SELECTION", symbol, e)
            return []
    
    def test_optimized_mtf_combinations(self, symbol: str, top_n: int = 5) -> Optional[pd.DataFrame]:
        """
        Тестирует только топ-N комбинаций MTF стратегий для символа.
        Выбирает топ-модели каждого таймфрейма на основе результатов одиночных тестов.
        """
        logger.info(f"[MTF TESTING] Начало оптимизированного тестирования MTF для {symbol}")
        
        try:
            # Выбираем топ-модели
            top_1h = self.select_top_models(symbol, '1h', top_n)
            top_15m = self.select_top_models(symbol, '15m', top_n)
            
            if not top_1h or not top_15m:
                logger.warning(f"[MTF TESTING] {symbol}: Не удалось выбрать топ-модели")
                logger.info(f"[MTF TESTING] {symbol}: Запускаем полное тестирование всех комбинаций")
                # Fallback: тестируем все комбинации
                return self.test_mtf_combinations(symbol)
            
            logger.info(f"[MTF TESTING] {symbol}: Будет протестировано {len(top_1h)} × {len(top_15m)} = {len(top_1h) * len(top_15m)} комбинаций")
            logger.info(f"[MTF TESTING] {symbol}: Вместо всех комбинаций (ускорение в ~{max(1, (5*5)/(len(top_1h)*len(top_15m))):.1f}x)")
            
            # Находим все модели для символа
            from backtest_mtf_strategy import find_all_models_for_symbol
            all_models_1h, all_models_15m = find_all_models_for_symbol(symbol)
            
            # Фильтруем только выбранные модели
            selected_1h = []
            selected_15m = []
            
            for model_path in all_models_1h:
                model_name = Path(model_path).stem
                for top_model in top_1h:
                    if top_model in model_name or model_name in top_model:
                        selected_1h.append(model_path)
                        break
            
            for model_path in all_models_15m:
                model_name = Path(model_path).stem
                for top_model in top_15m:
                    if top_model in model_name or model_name in top_model:
                        selected_15m.append(model_path)
                        break
            
            if not selected_1h or not selected_15m:
                logger.warning(f"[MTF TESTING] {symbol}: Не удалось найти файлы выбранных моделей")
                logger.info(f"[MTF TESTING] {symbol}: Запускаем полное тестирование")
                return self.test_mtf_combinations(symbol)
            
            # Тестируем только выбранные комбинации
            from backtest_mtf_strategy import run_mtf_backtest
            results = []
            
            for model_1h_path in selected_1h:
                for model_15m_path in selected_15m:
                    model_1h_name = Path(model_1h_path).name
                    model_15m_name = Path(model_15m_path).name
                    
                    logger.info(f"[MTF TESTING] {symbol}: Тестирование {model_1h_name} + {model_15m_name}")
                    
                    try:
                        metrics = run_mtf_backtest(
                            symbol=symbol,
                            days_back=self.days,
                            initial_balance=100.0,
                            risk_per_trade=0.02,
                            leverage=10,
                            model_1h_path=str(model_1h_path),
                            model_15m_path=str(model_15m_path),
                            confidence_threshold_1h=0.50,
                            confidence_threshold_15m=0.35,
                            alignment_mode="strict",
                            require_alignment=True,
                        )
                        
                        if metrics:
                            results.append({
                                'model_1h': model_1h_name,
                                'model_15m': model_15m_name,
                                'symbol': symbol,
                                'total_trades': metrics.total_trades,
                                'winning_trades': metrics.winning_trades,
                                'losing_trades': metrics.losing_trades,
                                'win_rate': metrics.win_rate,
                                'total_pnl': metrics.total_pnl,
                                'total_pnl_pct': metrics.total_pnl_pct,
                                'avg_win': metrics.avg_win,
                                'avg_loss': metrics.avg_loss,
                                'profit_factor': metrics.profit_factor,
                                'max_drawdown_pct': metrics.max_drawdown_pct,
                                'sharpe_ratio': metrics.sharpe_ratio,
                            })
                            logger.info(f"[MTF TESTING] {symbol}: {model_1h_name} + {model_15m_name} - "
                                      f"PnL: {metrics.total_pnl_pct:.2f}%, WR: {metrics.win_rate:.1f}%")
                    except Exception as e:
                        logger.error(f"[MTF TESTING] {symbol}: Ошибка тестирования {model_1h_name} + {model_15m_name}: {e}")
            
            if not results:
                logger.warning(f"[MTF TESTING] {symbol}: Нет результатов")
                return None
            
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('total_pnl_pct', ascending=False)
            
            logger.info(f"[MTF TESTING] {symbol}: Протестировано {len(results)} комбинаций")
            logger.info(f"[MTF TESTING] {symbol}: Лучшая комбинация: {df_results.iloc[0]['model_1h']} + {df_results.iloc[0]['model_15m']} "
                      f"(PnL: {df_results.iloc[0]['total_pnl_pct']:.2f}%)")
            
            # Сохраняем результаты
            filename = self.output_dir / f"mtf_combinations_{symbol}_{self.timestamp}.csv"
            df_results.to_csv(filename, index=False)
            logger.info(f"[MTF TESTING] {symbol}: Результаты сохранены в {filename}")
            
            return df_results
                
        except Exception as e:
            self.log_error("MTF_TESTING", symbol, e)
            return None
    
    def test_mtf_combinations(self, symbol: str) -> Optional[pd.DataFrame]:
        """Тестирует все комбинации MTF стратегий для символа (fallback метод)"""
        logger.info(f"[MTF TESTING] Начало полного тестирования MTF комбинаций для {symbol}")
        
        try:
            df_results = run_mtf_backtest_all_combinations(
                symbol=symbol,
                days_back=self.days,
                initial_balance=100.0,
                risk_per_trade=0.02,
                leverage=10,
                confidence_threshold_1h=0.50,
                confidence_threshold_15m=0.35,
                alignment_mode="strict",
                require_alignment=True,
            )
            
            if df_results is not None and not df_results.empty:
                logger.info(f"[MTF TESTING] {symbol}: Протестировано {len(df_results)} комбинаций")
                # Сохраняем результаты
                filename = self.output_dir / f"mtf_combinations_{symbol}_{self.timestamp}.csv"
                df_results.to_csv(filename, index=False)
                logger.info(f"[MTF TESTING] {symbol}: Результаты сохранены в {filename}")
                return df_results
            else:
                logger.warning(f"[MTF TESTING] {symbol}: Нет результатов")
                return None
                
        except Exception as e:
            self.log_error("MTF_TESTING", symbol, e)
            return None
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Вычисляет composite score для выбора лучшей стратегии"""
        total_pnl_pct = metrics.get('total_pnl_pct', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown_pct = metrics.get('max_drawdown_pct', 100)
        
        composite_score = (
            total_pnl_pct * 0.4 +
            win_rate * 0.2 +
            profit_factor * 20.0 * 0.2 +  # Нормализуем profit_factor
            sharpe_ratio * 0.1 +
            (100 - max_drawdown_pct) * 0.1
        )
        
        return composite_score
    
    def select_best_strategies(self):
        """Выбирает лучшие стратегии для каждого символа"""
        logger.info("[SELECTION] Начало выбора лучших стратегий")
        
        # Загружаем результаты сравнения моделей
        comparison_files = sorted(
            Path(".").glob("ml_models_comparison_*.csv"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        
        df_comparison = None
        if comparison_files:
            try:
                df_comparison = pd.read_csv(comparison_files[0])
                logger.info(f"[SELECTION] Загружен файл сравнения: {comparison_files[0]}")
            except Exception as e:
                logger.error(f"[SELECTION] Ошибка загрузки файла сравнения: {e}")
        
        for symbol in self.symbols:
            logger.info(f"[SELECTION] Анализ стратегий для {symbol}")
            
            best_strategy = None
            best_score = float('-inf')
            
            # 1. Проверяем MTF комбинации
            if symbol in self.mtf_results and self.mtf_results[symbol] is not None:
                df_mtf = self.mtf_results[symbol]
                if not df_mtf.empty:
                    # Выбираем лучшую MTF комбинацию
                    for _, row in df_mtf.iterrows():
                        metrics = {
                            'total_pnl_pct': row.get('total_pnl_pct', 0),
                            'win_rate': row.get('win_rate', 0),
                            'profit_factor': row.get('profit_factor', 0),
                            'sharpe_ratio': row.get('sharpe_ratio', 0),
                            'max_drawdown_pct': row.get('max_drawdown_pct', 100),
                        }
                        score = self.calculate_composite_score(metrics)
                        
                        if score > best_score:
                            best_score = score
                            best_strategy = {
                                "strategy_type": "mtf",
                                "model_1h": row['model_1h'],
                                "model_15m": row['model_15m'],
                                "confidence_threshold_1h": 0.50,
                                "confidence_threshold_15m": 0.35,
                                "alignment_mode": "strict",
                                "require_alignment": True,
                                "metrics": metrics,
                                "source": "mtf_combinations_test"
                            }
            
            # 2. Проверяем лучшие single модели из сравнения
            if df_comparison is not None:
                symbol_comparison = df_comparison[df_comparison['symbol'] == symbol].copy()
                if not symbol_comparison.empty:
                    # Фильтруем по 15m моделям
                    symbol_15m = symbol_comparison[
                        (symbol_comparison.get('mode_suffix', '') == '15m') |
                        (symbol_comparison['model_filename'].str.contains('_15_|_15m', na=False))
                    ]
                    
                    if not symbol_15m.empty:
                        # Сортируем по total_pnl_pct
                        symbol_15m = symbol_15m.sort_values('total_pnl_pct', ascending=False)
                        best_single = symbol_15m.iloc[0]
                        
                        single_metrics = {
                            'total_pnl_pct': best_single.get('total_pnl_pct', 0),
                            'win_rate': best_single.get('win_rate_pct', 0),
                            'profit_factor': best_single.get('profit_factor', 0),
                            'sharpe_ratio': best_single.get('sharpe_ratio', 0),
                            'max_drawdown_pct': best_single.get('max_drawdown_pct', 100),
                        }
                        single_score = self.calculate_composite_score(single_metrics)
                        
                        # Если single стратегия лучше MTF на >20%, используем её
                        if single_score > best_score * 1.2:
                            best_score = single_score
                            best_strategy = {
                                "strategy_type": "single",
                                "model": best_single['model_filename'],
                                "confidence_threshold": 0.40,  # Можно оптимизировать
                                "metrics": single_metrics,
                                "source": "model_comparison"
                            }
            
            if best_strategy:
                self.best_strategies[symbol] = best_strategy
                logger.info(f"[SELECTION] {symbol}: Выбрана стратегия {best_strategy['strategy_type']} "
                          f"(score: {best_score:.2f}, PnL: {best_strategy['metrics'].get('total_pnl_pct', 0):.2f}%)")
            else:
                logger.warning(f"[SELECTION] {symbol}: Не удалось выбрать стратегию")
    
    def save_best_strategies(self) -> Path:
        """Сохраняет лучшие стратегии в JSON файл"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "optimization_version": "1.0",
            "backtest_days": self.days,
            "symbols": self.best_strategies
        }
        
        filename = self.output_dir / f"best_strategies_{self.timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVE] Лучшие стратегии сохранены в {filename}")
        return filename
    
    def send_notification(self, message: str):
        """Отправляет уведомление в Telegram (если настроено)"""
        try:
            from bot.config import load_settings
            from bot.notification_manager import NotificationManager
            
            settings = load_settings()
            if settings.telegram_token:
                notifier = NotificationManager(None, settings)
                notifier.send_notification(message, level="HIGH")
                logger.info("[NOTIFICATION] Уведомление отправлено")
        except Exception as e:
            logger.warning(f"[NOTIFICATION] Не удалось отправить уведомление: {e}")
    
    def run(self):
        """Запускает полный цикл оптимизации"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("[START] НАЧАЛО АВТОМАТИЧЕСКОЙ ОПТИМИЗАЦИИ СТРАТЕГИЙ")
        logger.info("=" * 80)
        logger.info(f"Символы: {', '.join(self.symbols)}")
        logger.info(f"Дни бэктеста: {self.days}")
        logger.info(f"Пропуск обучения: {self.skip_training}")
        logger.info(f"Пропуск сравнения: {self.skip_comparison}")
        logger.info(f"Пропуск MTF тестирования: {self.skip_mtf_testing}")
        logger.info("=" * 80)
        
        # Этап 1: Обучение моделей
        if not self.skip_training:
            logger.info("\n[ЭТАП 1] ОБУЧЕНИЕ МОДЕЛЕЙ")
            logger.info("-" * 80)
            for symbol in self.symbols:
                success = self.train_models(symbol)
                self.training_results[symbol] = success
                if not success:
                    logger.warning(f"[TRAINING] {symbol}: Обучение не удалось, используем существующие модели")
        else:
            logger.info("\n[ЭТАП 1] ОБУЧЕНИЕ МОДЕЛЕЙ - ПРОПУЩЕНО")
            for symbol in self.symbols:
                self.training_results[symbol] = True  # Предполагаем, что модели уже есть
        
        # Этап 2: Сравнение моделей
        if not self.skip_comparison:
            logger.info("\n[ЭТАП 2] СРАВНЕНИЕ МОДЕЛЕЙ")
            logger.info("-" * 80)
            success = self.compare_models()
            if not success:
                logger.warning("[COMPARISON] Сравнение не удалось")
        else:
            logger.info("\n[ЭТАП 2] СРАВНЕНИЕ МОДЕЛЕЙ - ПРОПУЩЕНО")
        
        # Этап 3: Тестирование MTF комбинаций (оптимизированное)
        if not self.skip_mtf_testing:
            logger.info("\n[ЭТАП 3] ТЕСТИРОВАНИЕ MTF КОМБИНАЦИЙ (ОПТИМИЗИРОВАННОЕ)")
            logger.info("-" * 80)
            if self.full_mtf_testing:
                logger.info("[INFO] Используется полное тестирование всех MTF комбинаций")
            else:
                logger.info("[INFO] Используется оптимизированный подход:")
                logger.info(f"  1. Выбираются топ-{self.mtf_top_n} моделей каждого таймфрейма на основе composite score")
                logger.info(f"  2. Тестируются только {self.mtf_top_n * self.mtf_top_n} комбинаций вместо всех")
                logger.info("  3. Ускорение процесса в 4-5 раз")
            logger.info("-" * 80)
            for symbol in self.symbols:
                if self.full_mtf_testing:
                    df_results = self.test_mtf_combinations(symbol)
                else:
                    df_results = self.test_optimized_mtf_combinations(symbol, top_n=self.mtf_top_n)
                self.mtf_results[symbol] = df_results
        else:
            logger.info("\n[ЭТАП 3] ТЕСТИРОВАНИЕ MTF КОМБИНАЦИЙ - ПРОПУЩЕНО")
        
        # Этап 4: Выбор лучших стратегий
        logger.info("\n[ЭТАП 4] ВЫБОР ЛУЧШИХ СТРАТЕГИЙ")
        logger.info("-" * 80)
        self.select_best_strategies()
        
        # Сохранение результатов
        logger.info("\n[СОХРАНЕНИЕ] Сохранение результатов...")
        strategy_file = self.save_best_strategies()
        
        # Отчет
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # минуты
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        logger.info("=" * 80)
        logger.info(f"Время выполнения: {duration:.1f} минут")
        logger.info(f"Обработано символов: {len(self.symbols)}")
        logger.info(f"Выбрано стратегий: {len(self.best_strategies)}")
        logger.info(f"Ошибок: {len(self.errors)}")
        logger.info(f"Файл стратегий: {strategy_file}")
        logger.info("=" * 80)
        
        # Формируем отчет для уведомления
        report_lines = [
            "[INFO] ОТЧЕТ ОБ ОПТИМИЗАЦИИ СТРАТЕГИЙ",
            f"Время: {duration:.1f} минут",
            f"Символов: {len(self.symbols)}",
            "",
            "[BEST] ЛУЧШИЕ СТРАТЕГИИ:"
        ]
        
        for symbol, strategy in self.best_strategies.items():
            strategy_type = strategy['strategy_type']
            metrics = strategy['metrics']
            pnl = metrics.get('total_pnl_pct', 0)
            wr = metrics.get('win_rate', 0)
            
            if strategy_type == "mtf":
                report_lines.append(
                    f"{symbol}: MTF ({strategy['model_1h']} + {strategy['model_15m']})"
                )
            else:
                report_lines.append(
                    f"{symbol}: Single ({strategy['model']})"
                )
            report_lines.append(f"  PnL: {pnl:.2f}%, WR: {wr:.1f}%")
        
        if self.errors:
            report_lines.append(f"\n[WARN] Ошибок: {len(self.errors)}")
        
        report = "\n".join(report_lines)
        logger.info(f"\n{report}")
        
        # Отправляем уведомление
        self.send_notification(report)
        
        return strategy_file


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическая оптимизация стратегий",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--symbols", type=str, default=None,
                       help="Список символов через запятую (по умолчанию из state.active_symbols)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для бэктеста (по умолчанию 30)")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                       help="Директория для сохранения результатов")
    parser.add_argument("--skip-training", action="store_true",
                       help="Пропустить обучение моделей")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Пропустить сравнение моделей")
    parser.add_argument("--skip-mtf-testing", action="store_true",
                       help="Пропустить тестирование MTF комбинаций")
    parser.add_argument("--mtf-top-n", type=int, default=5,
                       help="Количество топ-моделей каждого таймфрейма для MTF тестирования (по умолчанию 5)")
    parser.add_argument("--full-mtf-testing", action="store_true",
                       help="Тестировать все MTF комбинации вместо оптимизированного подхода")
    
    args = parser.parse_args()
    
    # Определяем символы
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        # Загружаем из state
        state = BotState()
        symbols = state.active_symbols
        if not symbols:
            symbols = ["BTCUSDT"]  # Fallback
    
    # Создаем оптимизатор
    optimizer = StrategyOptimizer(
        symbols=symbols,
        days=args.days,
        output_dir=args.output_dir,
        skip_training=args.skip_training,
        skip_comparison=args.skip_comparison,
        skip_mtf_testing=args.skip_mtf_testing,
        mtf_top_n=args.mtf_top_n,
        full_mtf_testing=args.full_mtf_testing,
    )
    
    # Запускаем оптимизацию
    try:
        optimizer.run()
    except KeyboardInterrupt:
        logger.info("\n[WARN] Оптимизация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
