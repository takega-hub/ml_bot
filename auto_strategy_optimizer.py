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
    ):
        self.symbols = [s.upper() for s in symbols]
        self.days = days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.skip_training = skip_training
        self.skip_comparison = skip_comparison
        self.skip_mtf_testing = skip_mtf_testing
        
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
            result_15m = subprocess.run(
                cmd_15m,
                capture_output=True,
                text=True,
                timeout=3600  # 1 час таймаут
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
                "--interval", "60"  # 1h интервал
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
    
    def compare_models(self) -> bool:
        """Сравнивает модели (15m и 1h отдельно)"""
        logger.info("[COMPARISON] Начало сравнения моделей")
        
        try:
            # Сравниваем 15m модели
            logger.info("[COMPARISON] Сравнение 15m моделей...")
            cmd_15m = [
                self.python_exe,
                "compare_ml_models.py",
                "--symbols", ",".join(self.symbols),
                "--days", str(self.days),
                "--output", "csv",
                "--interval", "15m"
            ]
            
            result_15m = subprocess.run(
                cmd_15m,
                capture_output=True,
                text=True,
                timeout=7200  # 2 часа таймаут
            )
            
            if result_15m.returncode != 0:
                logger.error("[COMPARISON] Ошибка сравнения 15m моделей")
                logger.error(f"STDERR: {result_15m.stderr[-500:]}")
            
            # Сравниваем 1h модели
            logger.info("[COMPARISON] Сравнение 1h моделей...")
            cmd_1h = [
                self.python_exe,
                "compare_ml_models.py",
                "--symbols", ",".join(self.symbols),
                "--days", str(self.days),
                "--output", "csv",
                "--interval", "15m",  # Используем 15m данные для агрегации
                "--only-1h"  # Только 1h модели
            ]
            
            result_1h = subprocess.run(
                cmd_1h,
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            if result_1h.returncode != 0:
                logger.error("[COMPARISON] Ошибка сравнения 1h моделей")
                logger.error(f"STDERR: {result_1h.stderr[-500:]}")
            
            # Находим последний файл сравнения
            comparison_files = sorted(
                Path(".").glob("ml_models_comparison_*.csv"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            
            if comparison_files:
                latest_file = comparison_files[0]
                logger.info(f"[COMPARISON] Последний файл сравнения: {latest_file}")
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
    
    def test_mtf_combinations(self, symbol: str) -> Optional[pd.DataFrame]:
        """Тестирует все комбинации MTF стратегий для символа"""
        logger.info(f"[MTF TESTING] Начало тестирования MTF комбинаций для {symbol}")
        
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
        
        # Этап 3: Тестирование MTF комбинаций
        if not self.skip_mtf_testing:
            logger.info("\n[ЭТАП 3] ТЕСТИРОВАНИЕ MTF КОМБИНАЦИЙ")
            logger.info("-" * 80)
            for symbol in self.symbols:
                df_results = self.test_mtf_combinations(symbol)
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
