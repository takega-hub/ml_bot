"""
Скрипт для предсказания лучших MTF комбинаций на основе результатов одиночных моделей.
Тестирует 15m и 1h модели отдельно, анализирует результаты и предсказывает лучшие комбинации
БЕЗ реального тестирования MTF стратегий.
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
log_file = f'predict_mtf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорты для работы с ботом
from bot.state import BotState


class MTFPredictor:
    """Класс для предсказания лучших MTF комбинаций на основе одиночных моделей"""
    
    def __init__(
        self,
        symbols: List[str],
        days: int = 30,
        output_dir: str = "mtf_predictions",
        top_n: int = 10,
        skip_testing: bool = False,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.days = days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.top_n = top_n
        self.skip_testing = skip_testing
        
        self.python_exe = sys.executable
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Результаты
        self.single_results: Optional[pd.DataFrame] = None
        self.predictions: Dict[str, pd.DataFrame] = {}
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
    
    def check_data_freshness(self, file_path: Path) -> Tuple[bool, bool]:
        """
        Проверяет свежесть данных в файле.
        Возвращает (is_fresh, has_all_symbols)
        """
        if not file_path.exists():
            return False, False
        
        # Проверяем возраст файла (1 день = 86400 секунд)
        file_time = file_path.stat().st_mtime
        current_time = datetime.now().timestamp()
        age_hours = (current_time - file_time) / 3600
        
        is_fresh = age_hours < 24  # Меньше 24 часов
        
        # Проверяем наличие данных для всех символов
        try:
            df = pd.read_csv(file_path)
            if 'symbol' not in df.columns:
                return is_fresh, False
            
            file_symbols = set(df['symbol'].unique())
            required_symbols = set(self.symbols)
            has_all_symbols = required_symbols.issubset(file_symbols)
            
            missing_symbols = required_symbols - file_symbols
            if missing_symbols:
                logger.info(f"[TESTING] В файле отсутствуют данные для: {', '.join(missing_symbols)}")
            
            return is_fresh, has_all_symbols
        except Exception as e:
            logger.warning(f"[TESTING] Ошибка при проверке файла {file_path}: {e}")
            return False, False
    
    def test_single_models(self) -> bool:
        """Тестирует все одиночные модели (15m и 1h) или загружает существующие результаты"""
        logger.info("[TESTING] Поиск результатов тестирования одиночных моделей")
        logger.info(f"[TESTING] Символы: {', '.join(self.symbols)}")
        logger.info(f"[TESTING] Дни бэктеста: {self.days}")
        
        # Сначала проверяем, есть ли уже файл результатов
        comparison_files = sorted(
            Path(".").glob("ml_models_comparison_*.csv"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        
        symbols_to_test = []
        existing_data = None
        
        # Инициализируем existing_data, если файл есть
        if comparison_files:
            try:
                existing_data = pd.read_csv(comparison_files[0])
            except:
                existing_data = None
        
        if comparison_files and self.skip_testing:
            # Режим пропуска тестирования - используем существующий файл
            logger.info(f"[TESTING] Пропуск тестирования, используется существующий файл: {comparison_files[0]}")
            try:
                self.single_results = pd.read_csv(comparison_files[0])
                logger.info(f"[TESTING] Загружено {len(self.single_results)} результатов из существующего файла")
                return True
            except Exception as e:
                logger.warning(f"[TESTING] Ошибка загрузки существующего файла: {e}")
                return False
        
        if comparison_files:
            # Проверяем свежесть и полноту данных
            latest_file = comparison_files[0]
            is_fresh, has_all_symbols = self.check_data_freshness(latest_file)
            
            if is_fresh and has_all_symbols:
                # Данные свежие и полные - используем их
                logger.info(f"[TESTING] Найден свежий файл с полными данными: {latest_file}")
                if existing_data is not None:
                    self.single_results = existing_data
                    logger.info(f"[TESTING] Загружено {len(self.single_results)} результатов")
                    return True
                else:
                    # Ошибка загрузки - собираем данные заново
                    logger.warning(f"[TESTING] Ошибка загрузки файла - собираем данные заново")
                    symbols_to_test = self.symbols.copy()
            
            elif not is_fresh:
                # Данные устарели - нужно обновить все
                logger.info(f"[TESTING] Данные устарели (старше 24 часов) - требуется обновление")
                symbols_to_test = self.symbols.copy()
            elif not has_all_symbols:
                # Данные свежие, но неполные - собираем только недостающие
                if existing_data is not None:
                    existing_symbols = set(existing_data['symbol'].unique())
                    symbols_to_test = [s for s in self.symbols if s not in existing_symbols]
                    logger.info(f"[TESTING] Данные свежие, но неполные. Нужно собрать данные для: {', '.join(symbols_to_test)}")
                else:
                    logger.warning(f"[TESTING] Ошибка чтения существующего файла - собираем все данные")
                    symbols_to_test = self.symbols.copy()
        else:
            # Файла нет - собираем все данные
            logger.info("[TESTING] Файл результатов не найден - требуется сбор данных")
            symbols_to_test = self.symbols.copy()
        
        # Если нужно собрать данные
        if not symbols_to_test:
            # Данные уже есть и свежие - используем их
            if existing_data is not None:
                self.single_results = existing_data
            else:
                # Загружаем из последнего файла
                try:
                    self.single_results = pd.read_csv(comparison_files[0])
                except Exception as e:
                    logger.error(f"[TESTING] Ошибка загрузки данных: {e}")
                    return False
            
            logger.info(f"[TESTING] Используются существующие свежие данные: {len(self.single_results)} результатов")
            return True
        
        # Нужно собрать данные для символов
        logger.info(f"[TESTING] Запуск тестирования для символов: {', '.join(symbols_to_test)}")
        
        try:
            # Тестируем модели для нужных символов
            cmd = [
                self.python_exe,
                "compare_ml_models.py",
                "--symbols", ",".join(symbols_to_test),
                "--days", str(self.days),
                "--output", "csv",
                "--interval", "15m",  # Базовый интервал
                "--detailed-analysis"
            ]
            
            logger.info(f"[TESTING] Команда: {' '.join(cmd)}")
            
            # Запускаем с обработкой кодировки для Windows
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Заменяем проблемные символы
                    timeout=7200  # 2 часа таймаут
                )
            except Exception as e:
                logger.warning(f"[TESTING] Ошибка при запуске процесса: {e}")
                result = None
            
            # Проверяем, создан ли файл результатов после запуска
            comparison_files_after = sorted(
                Path(".").glob("ml_models_comparison_*.csv"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            
            # Используем файл, если он существует (главный критерий успеха)
            if comparison_files_after:
                file_time = comparison_files_after[0].stat().st_mtime
                current_time = datetime.now().timestamp()
                time_diff = current_time - file_time
                
                logger.info(f"[TESTING] Файл результатов найден ({time_diff/60:.1f} минут назад)")
                
                # Загружаем новые результаты
                new_data = pd.read_csv(comparison_files_after[0])
                
                # Объединяем с существующими данными (если есть)
                if existing_data is not None:
                    # Удаляем старые данные для символов, которые мы обновили
                    existing_data = existing_data[~existing_data['symbol'].isin(symbols_to_test)]
                    # Объединяем
                    self.single_results = pd.concat([existing_data, new_data], ignore_index=True)
                    logger.info(f"[TESTING] Объединены существующие и новые данные: {len(self.single_results)} результатов")
                else:
                    self.single_results = new_data
                    logger.info(f"[TESTING] Загружены новые данные: {len(self.single_results)} результатов")
                
                # Логируем ошибки, но продолжаем работу (файл есть)
                if result and result.returncode != 0:
                    logger.warning("[TESTING] Процесс завершился с ошибкой кодировки, но файл создан - продолжаем")
                    try:
                        if result.stderr:
                            stderr_safe = result.stderr[-500:].encode('ascii', 'replace').decode('ascii')
                            logger.debug(f"STDERR: {stderr_safe}")
                    except:
                        pass
            else:
                # Файл не создан - это ошибка
                logger.error("[TESTING] Файл результатов не создан после запуска")
                if result:
                    try:
                        if result.stderr:
                            stderr_safe = result.stderr[-500:].encode('ascii', 'replace').decode('ascii')
                            logger.error(f"STDERR: {stderr_safe}")
                    except:
                        pass
                
                # Если есть существующие данные - используем их
                if existing_data is not None:
                    logger.warning("[TESTING] Используются существующие данные (новые не собраны)")
                    self.single_results = existing_data
                    return True
                
                return False
            
            # Проверяем наличие 15m и 1h моделей
            has_15m = self.single_results['model_filename'].str.contains('_15_|_15m', na=False).any()
            has_1h = self.single_results['model_filename'].str.contains('_60_|_1h', na=False).any()
            
            if not has_15m:
                logger.warning("[TESTING] ⚠️  Не найдено 15m моделей в результатах")
            if not has_1h:
                logger.warning("[TESTING] ⚠️  Не найдено 1h моделей в результатах")
            
            if has_15m and has_1h:
                logger.info("[TESTING] ✅ Файл содержит и 15m, и 1h модели")
                logger.info(f"[TESTING] Всего моделей: {len(self.single_results)}")
                return True
            else:
                logger.error("[TESTING] ❌ Недостаточно данных для предсказания")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[TESTING] Таймаут при тестировании")
            # Если есть существующие данные - используем их
            if existing_data is not None:
                logger.warning("[TESTING] Используются существующие данные (таймаут при обновлении)")
                self.single_results = existing_data
                return True
            return False
        except Exception as e:
            self.log_error("TESTING", "ALL", e)
            # Если есть существующие данные - используем их
            if existing_data is not None:
                logger.warning("[TESTING] Используются существующие данные (ошибка при обновлении)")
                self.single_results = existing_data
                return True
            return False
    
    def calculate_composite_score(self, row: pd.Series) -> float:
        """Вычисляет composite score для модели"""
        pnl = row.get('total_pnl_pct', 0)
        wr = row.get('win_rate_pct', row.get('win_rate', 0))
        pf = row.get('profit_factor', 0)
        sharpe = row.get('sharpe_ratio', 0)
        dd = row.get('max_drawdown_pct', 100)
        
        score = (
            pnl * 0.4 +
            wr * 0.2 +
            pf * 20.0 * 0.2 +
            sharpe * 0.1 +
            (100 - dd) * 0.1
        )
        return score
    
    def predict_mtf_combinations(self, symbol: str) -> pd.DataFrame:
        """
        Предсказывает лучшие MTF комбинации на основе результатов одиночных моделей.
        Не тестирует реальные MTF стратегии, только анализирует одиночные результаты.
        """
        logger.info(f"[PREDICTION] Предсказание MTF комбинаций для {symbol}")
        
        if self.single_results is None:
            logger.error("[PREDICTION] Нет результатов одиночных моделей")
            return pd.DataFrame()
        
        symbol_data = self.single_results[self.single_results['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            logger.warning(f"[PREDICTION] Нет данных для {symbol} в файле сравнения")
            logger.info(f"[PREDICTION] Для получения данных запустите: python compare_ml_models.py --symbols {symbol} --days {self.days}")
            return pd.DataFrame()
        
        # Разделяем на 1h и 15m модели
        models_1h = symbol_data[
            (symbol_data['mode_suffix'] == '1h') |
            (symbol_data['model_filename'].str.contains('_60_|_1h', na=False))
        ].copy()
        
        models_15m = symbol_data[
            (symbol_data['mode_suffix'] == '15m') |
            (symbol_data['model_filename'].str.contains('_15_|_15m', na=False))
        ].copy()
        
        if models_1h.empty or models_15m.empty:
            logger.warning(f"[PREDICTION] Недостаточно моделей для {symbol}")
            logger.info(f"  1h моделей: {len(models_1h)}, 15m моделей: {len(models_15m)}")
            return pd.DataFrame()
        
        # Вычисляем composite score для всех моделей
        models_1h['composite_score'] = models_1h.apply(self.calculate_composite_score, axis=1)
        models_15m['composite_score'] = models_15m.apply(self.calculate_composite_score, axis=1)
        
        # Сортируем по score
        models_1h = models_1h.sort_values('composite_score', ascending=False)
        models_15m = models_15m.sort_values('composite_score', ascending=False)
        
        logger.info(f"[PREDICTION] {symbol}: Найдено {len(models_1h)} 1h моделей, {len(models_15m)} 15m моделей")
        
        # Генерируем все комбинации и предсказываем результаты
        predictions = []
        
        for _, row_1h in models_1h.iterrows():
            for _, row_15m in models_15m.iterrows():
                model_1h_name = row_1h['model_filename'].replace('.pkl', '')
                model_15m_name = row_15m['model_filename'].replace('.pkl', '')
                
                # Предсказанные метрики на основе одиночных моделей
                # Используем различные стратегии предсказания
                
                # Стратегия 1: Среднее арифметическое
                predicted_pnl_avg = (row_1h['total_pnl_pct'] + row_15m['total_pnl_pct']) / 2
                predicted_wr_avg = (row_1h.get('win_rate_pct', 0) + row_15m.get('win_rate_pct', 0)) / 2
                
                # Стратегия 2: Взвешенное среднее (1h важнее для тренда, 15m для входа)
                predicted_pnl_weighted = row_1h['total_pnl_pct'] * 0.4 + row_15m['total_pnl_pct'] * 0.6
                predicted_wr_weighted = (row_1h.get('win_rate_pct', 0) * 0.3 + 
                                        row_15m.get('win_rate_pct', 0) * 0.7)
                
                # Стратегия 3: Улучшение за счет синергии (оптимистичная оценка)
                # MTF обычно дает улучшение на 20-50% от суммы одиночных
                synergy_factor = 1.3  # 30% улучшение за счет синергии
                predicted_pnl_synergy = (row_1h['total_pnl_pct'] + row_15m['total_pnl_pct']) * synergy_factor / 2
                
                # Используем взвешенное среднее как основное предсказание
                predicted_pnl = predicted_pnl_weighted
                predicted_wr = predicted_wr_weighted
                
                # Предсказанный composite score
                predicted_score_1h = row_1h['composite_score']
                predicted_score_15m = row_15m['composite_score']
                predicted_score = (predicted_score_1h + predicted_score_15m) / 2
                
                # Дополнительные предсказанные метрики
                predicted_pf = (row_1h['profit_factor'] + row_15m['profit_factor']) / 2
                predicted_sharpe = (row_1h['sharpe_ratio'] + row_15m['sharpe_ratio']) / 2
                predicted_dd = max(row_1h.get('max_drawdown_pct', 100), row_15m.get('max_drawdown_pct', 100))
                
                predictions.append({
                    'model_1h': model_1h_name,
                    'model_15m': model_15m_name,
                    'symbol': symbol,
                    # Одиночные метрики
                    'single_1h_pnl': row_1h['total_pnl_pct'],
                    'single_15m_pnl': row_15m['total_pnl_pct'],
                    'single_1h_wr': row_1h.get('win_rate_pct', 0),
                    'single_15m_wr': row_15m.get('win_rate_pct', 0),
                    'single_1h_score': predicted_score_1h,
                    'single_15m_score': predicted_score_15m,
                    # Предсказанные MTF метрики
                    'predicted_pnl_pct': predicted_pnl,
                    'predicted_wr': predicted_wr,
                    'predicted_pnl_avg': predicted_pnl_avg,
                    'predicted_pnl_synergy': predicted_pnl_synergy,
                    'predicted_score': predicted_score,
                    'predicted_profit_factor': predicted_pf,
                    'predicted_sharpe': predicted_sharpe,
                    'predicted_max_drawdown_pct': predicted_dd,
                    # Дополнительная информация
                    'single_1h_trades': row_1h.get('total_trades', 0),
                    'single_15m_trades': row_15m.get('total_trades', 0),
                    'estimated_mtf_trades': min(row_1h.get('total_trades', 0), row_15m.get('total_trades', 0)) * 0.8,
                })
        
        if not predictions:
            logger.warning(f"[PREDICTION] Не удалось сгенерировать предсказания для {symbol}")
            return pd.DataFrame()
        
        df_predictions = pd.DataFrame(predictions)
        
        # Сортируем по predicted_score
        df_predictions = df_predictions.sort_values('predicted_score', ascending=False)
        
        logger.info(f"[PREDICTION] {symbol}: Сгенерировано {len(df_predictions)} предсказаний")
        logger.info(f"[PREDICTION] {symbol}: Лучшая предсказанная комбинация:")
        best = df_predictions.iloc[0]
        logger.info(f"   {best['model_1h']} + {best['model_15m']}")
        logger.info(f"   Предсказанный PnL: {best['predicted_pnl_pct']:.2f}%")
        logger.info(f"   Предсказанный WR: {best['predicted_wr']:.1f}%")
        logger.info(f"   Предсказанный Score: {best['predicted_score']:.2f}")
        
        return df_predictions
    
    def select_best_combinations(self, symbol: str, df_predictions: pd.DataFrame) -> Dict[str, Any]:
        """Выбирает лучшие комбинации для символа"""
        if df_predictions.empty:
            return {}
        
        # Берем топ-N комбинаций
        top_combinations = df_predictions.head(self.top_n).copy()
        
        best = top_combinations.iloc[0]
        
        result = {
            "symbol": symbol,
            "best_combination": {
                "model_1h": best['model_1h'],
                "model_15m": best['model_15m'],
                "predicted_pnl_pct": best['predicted_pnl_pct'],
                "predicted_wr": best['predicted_wr'],
                "predicted_score": best['predicted_score'],
                "predicted_profit_factor": best['predicted_profit_factor'],
                "predicted_sharpe": best['predicted_sharpe'],
                "predicted_max_drawdown_pct": best['predicted_max_drawdown_pct'],
            },
            "single_models_performance": {
                "1h": {
                    "model": best['model_1h'],
                    "pnl": best['single_1h_pnl'],
                    "wr": best['single_1h_wr'],
                    "score": best['single_1h_score'],
                },
                "15m": {
                    "model": best['model_15m'],
                    "pnl": best['single_15m_pnl'],
                    "wr": best['single_15m_wr'],
                    "score": best['single_15m_score'],
                }
            },
            "top_combinations": top_combinations.to_dict('records'),
        }
        
        return result
    
    def save_results(self):
        """Сохраняет результаты предсказаний"""
        # Сохраняем предсказания для каждого символа
        for symbol, df_pred in self.predictions.items():
            if not df_pred.empty:
                filename = self.output_dir / f"predicted_mtf_{symbol}_{self.timestamp}.csv"
                df_pred.to_csv(filename, index=False)
                logger.info(f"[SAVE] Предсказания для {symbol} сохранены в {filename}")
        
        # Сохраняем лучшие комбинации
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction_version": "1.0",
            "backtest_days": self.days,
            "top_n": self.top_n,
            "method": "prediction_from_single_models",
            "note": "Эти результаты основаны на предсказаниях, а не на реальном тестировании MTF",
            "best_combinations": self.best_combinations,
        }
        
        filename = self.output_dir / f"best_predicted_mtf_{self.timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"[SAVE] Лучшие предсказанные комбинации сохранены в {filename}")
        return filename
    
    def print_summary(self):
        """Печатает сводку результатов"""
        print("\n" + "=" * 100)
        print("📊 СВОДКА ПРЕДСКАЗАНИЙ MTF КОМБИНАЦИЙ")
        print("=" * 100)
        print()
        print("⚠️  ВНИМАНИЕ: Результаты основаны на предсказаниях, а не на реальном тестировании MTF")
        print("   Для проверки предсказаний рекомендуется протестировать топ-комбинации реально")
        print()
        
        symbols_with_results = [s for s in self.symbols if s in self.best_combinations]
        symbols_without_results = [s for s in self.symbols if s not in self.best_combinations]
        
        if symbols_without_results:
            print(f"\n⚠️  Символы без данных: {', '.join(symbols_without_results)}")
            print("   Для получения данных запустите:")
            print(f"   python compare_ml_models.py --symbols {','.join(symbols_without_results)} --days {self.days}")
            print()
        
        for symbol in symbols_with_results:
            best = self.best_combinations[symbol]
            combo = best['best_combination']
            single = best['single_models_performance']
            
            print(f"🎯 {symbol}")
            print("-" * 100)
            print(f"Лучшая предсказанная комбинация:")
            print(f"  1h: {combo['model_1h']}")
            print(f"     Одиночный PnL: {single['1h']['pnl']:.2f}%, WR: {single['1h']['wr']:.1f}%")
            print(f"  15m: {combo['model_15m']}")
            print(f"     Одиночный PnL: {single['15m']['pnl']:.2f}%, WR: {single['15m']['wr']:.1f}%")
            print()
            print(f"Предсказанные MTF метрики:")
            print(f"  PnL: {combo['predicted_pnl_pct']:.2f}%")
            print(f"  Win Rate: {combo['predicted_wr']:.1f}%")
            print(f"  Profit Factor: {combo['predicted_profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {combo['predicted_sharpe']:.2f}")
            print(f"  Max Drawdown: {combo['predicted_max_drawdown_pct']:.2f}%")
            print(f"  Composite Score: {combo['predicted_score']:.2f}")
            print()
        
        print("=" * 100)
        print()
    
    def run(self):
        """Запускает полный цикл предсказания"""
        start_time = datetime.now()
        logger.info("=" * 100)
        logger.info("🚀 НАЧАЛО ПРЕДСКАЗАНИЯ MTF КОМБИНАЦИЙ")
        logger.info("=" * 100)
        logger.info(f"Символы: {', '.join(self.symbols)}")
        logger.info(f"Дни бэктеста: {self.days}")
        logger.info(f"Топ-N комбинаций: {self.top_n}")
        logger.info("=" * 100)
        
        # Этап 1: Тестирование одиночных моделей
        logger.info("\n[ЭТАП 1] ТЕСТИРОВАНИЕ ОДИНОЧНЫХ МОДЕЛЕЙ")
        logger.info("-" * 100)
        success = self.test_single_models()
        if not success:
            logger.error("[ERROR] Не удалось протестировать одиночные модели")
            return None
        
        # Этап 2: Предсказание MTF комбинаций
        logger.info("\n[ЭТАП 2] ПРЕДСКАЗАНИЕ MTF КОМБИНАЦИЙ")
        logger.info("-" * 100)
        
        symbols_without_data = []
        for symbol in self.symbols:
            df_pred = self.predict_mtf_combinations(symbol)
            self.predictions[symbol] = df_pred
            
            if not df_pred.empty:
                best_combo = self.select_best_combinations(symbol, df_pred)
                if best_combo:
                    self.best_combinations[symbol] = best_combo
            else:
                symbols_without_data.append(symbol)
        
        # Если есть символы без данных - собираем их автоматически
        if symbols_without_data and not self.skip_testing:
            logger.info(f"\n[АВТОСБОР] Обнаружены символы без данных: {', '.join(symbols_without_data)}")
            logger.info("[АВТОСБОР] Запуск автоматического сбора данных...")
            
            # Собираем данные для недостающих символов
            try:
                cmd = [
                    self.python_exe,
                    "compare_ml_models.py",
                    "--symbols", ",".join(symbols_without_data),
                    "--days", str(self.days),
                    "--output", "csv",
                    "--interval", "15m",
                    "--detailed-analysis"
                ]
                
                logger.info(f"[АВТОСБОР] Команда: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=7200
                )
                
                # Запоминаем время перед запуском для проверки новых файлов
                process_start_time = datetime.now().timestamp()
                
                # Ждем немного, чтобы файл успел создаться
                import time
                time.sleep(2)
                
                # Проверяем, создан ли новый файл (созданный после запуска процесса)
                comparison_files_after = sorted(
                    Path(".").glob("ml_models_comparison_*.csv"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True
                )
                
                # Ищем файл, созданный после запуска процесса
                new_file = None
                for file_path in comparison_files_after:
                    file_time = file_path.stat().st_mtime
                    # Файл должен быть создан не более чем за 5 секунд до запуска процесса
                    # (с учетом времени на запуск)
                    if file_time >= (process_start_time - 5):
                        new_file = file_path
                        break
                
                # Если не нашли новый файл, берем самый свежий
                if new_file is None and comparison_files_after:
                    new_file = comparison_files_after[0]
                    logger.warning(f"[АВТОСБОР] Не найден явно новый файл, используем самый свежий: {new_file}")
                
                if new_file and new_file.exists():
                    # Загружаем новые данные
                    new_data = pd.read_csv(new_file)
                    logger.info(f"[АВТОСБОР] Загружен файл: {new_file} (создан {datetime.fromtimestamp(new_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
                    
                    # Проверяем, какие символы есть в новых данных
                    new_symbols = set(new_data['symbol'].unique()) if 'symbol' in new_data.columns else set()
                    logger.info(f"[АВТОСБОР] В новых данных найдены символы: {', '.join(sorted(new_symbols))}")
                    
                    # Проверяем, есть ли нужные символы
                    missing_in_new = set(symbols_without_data) - new_symbols
                    if missing_in_new:
                        logger.warning(f"[АВТОСБОР] В новых данных отсутствуют: {', '.join(missing_in_new)}")
                        # Пробуем найти данные в других файлах
                        logger.info("[АВТОСБОР] Поиск данных в других файлах...")
                        for file_path in comparison_files_after[1:5]:  # Проверяем еще 4 файла
                            try:
                                check_data = pd.read_csv(file_path)
                                check_symbols = set(check_data['symbol'].unique()) if 'symbol' in check_data.columns else set()
                                found_missing = set(missing_in_new) & check_symbols
                                if found_missing:
                                    logger.info(f"[АВТОСБОР] Найдены данные для {', '.join(found_missing)} в файле {file_path}")
                                    # Добавляем найденные данные
                                    found_data = check_data[check_data['symbol'].isin(found_missing)]
                                    new_data = pd.concat([new_data, found_data], ignore_index=True)
                                    new_symbols.update(found_missing)
                                    missing_in_new -= found_missing
                            except Exception as e:
                                logger.debug(f"[АВТОСБОР] Ошибка при проверке {file_path}: {e}")
                        
                        if missing_in_new:
                            logger.warning(f"[АВТОСБОР] Не удалось найти данные для: {', '.join(missing_in_new)}")
                        else:
                            logger.info("[АВТОСБОР] Все недостающие данные найдены в других файлах")
                    
                    # Объединяем с существующими
                    if self.single_results is not None:
                        # Удаляем старые данные для этих символов (если есть)
                        self.single_results = self.single_results[
                            ~self.single_results['symbol'].isin(symbols_without_data)
                        ]
                        # Объединяем
                        self.single_results = pd.concat([self.single_results, new_data], ignore_index=True)
                    else:
                        self.single_results = new_data
                    
                    # Проверяем итоговые данные
                    final_symbols = set(self.single_results['symbol'].unique()) if 'symbol' in self.single_results.columns else set()
                    logger.info(f"[АВТОСБОР] Данные собраны и объединены: {len(self.single_results)} результатов")
                    logger.info(f"[АВТОСБОР] Итоговые символы в данных: {', '.join(sorted(final_symbols))}")
                    
                    # Повторяем предсказание для этих символов
                    logger.info("[АВТОСБОР] Повторное предсказание для собранных символов...")
                    for symbol in symbols_without_data:
                        # Проверяем наличие данных перед предсказанием
                        if symbol in final_symbols:
                            df_pred = self.predict_mtf_combinations(symbol)
                            self.predictions[symbol] = df_pred
                            
                            if not df_pred.empty:
                                best_combo = self.select_best_combinations(symbol, df_pred)
                                if best_combo:
                                    self.best_combinations[symbol] = best_combo
                                    logger.info(f"[АВТОСБОР] ✅ Данные собраны и предсказание выполнено для {symbol}")
                            else:
                                logger.warning(f"[АВТОСБОР] ⚠️  Данные для {symbol} есть, но предсказание пустое (возможно, нет моделей 15m или 1h)")
                        else:
                            logger.warning(f"[АВТОСБОР] ⚠️  Символ {symbol} отсутствует в итоговых данных")
                else:
                    logger.warning(f"[АВТОСБОР] Не удалось собрать данные для {', '.join(symbols_without_data)}")
                    
            except Exception as e:
                logger.error(f"[АВТОСБОР] Ошибка при автоматическом сборе данных: {e}")
                self.log_error("AUTO_COLLECT", ",".join(symbols_without_data), e)
        
        # Сохранение результатов
        logger.info("\n[СОХРАНЕНИЕ] Сохранение результатов...")
        result_file = self.save_results()
        
        # Печать сводки
        self.print_summary()
        
        # Отчет
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # минуты
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 100)
        logger.info(f"Время выполнения: {duration:.1f} минут")
        logger.info(f"Обработано символов: {len(self.symbols)}")
        logger.info(f"Предсказано комбинаций: {sum(len(df) for df in self.predictions.values())}")
        logger.info(f"Выбрано лучших комбинаций: {len(self.best_combinations)}")
        logger.info(f"Файл результатов: {result_file}")
        logger.info("=" * 100)
        
        return result_file


def main():
    parser = argparse.ArgumentParser(
        description="Предсказание лучших MTF комбинаций на основе одиночных моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--symbols", type=str, default=None,
                       help="Список символов через запятую (по умолчанию из state.active_symbols)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для бэктеста (по умолчанию 30)")
    parser.add_argument("--output-dir", type=str, default="mtf_predictions",
                       help="Директория для сохранения результатов")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Количество топ-комбинаций для сохранения (по умолчанию 10)")
    parser.add_argument("--skip-testing", action="store_true",
                       help="Пропустить тестирование, использовать существующий файл сравнения")
    
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
    
    # Создаем предсказатель
    predictor = MTFPredictor(
        symbols=symbols,
        days=args.days,
        output_dir=args.output_dir,
        top_n=args.top_n,
        skip_testing=args.skip_testing,
    )
    
    # Запускаем предсказание
    try:
        predictor.run()
    except KeyboardInterrupt:
        logger.info("\n[WARN] Предсказание прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
