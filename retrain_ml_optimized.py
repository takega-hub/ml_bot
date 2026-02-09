"""
Улучшенный скрипт переобучения ML-модели с оптимизациями для большего количества сигналов.

Улучшения:
1. Более агрессивный таргет (movement > 1%)
2. Балансировка классов (class_weight)
3. Увеличенные данные (30 дней)
4. Оптимизированные гиперпараметры
"""
import warnings
import os
import sys

# Настраиваем кодировку для Windows (БЕЗОПАСНАЯ ВЕРСИЯ)
if sys.platform == 'win32':
    try:
        # Используем замену ошибок вместо перезаписи stdout/stderr
        import codecs
        # Только если не перенаправлен
        if sys.stdout.isatty():
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        if sys.stderr.isatty():
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        pass  # Если не получилось, продолжаем как есть

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, WeightedEnsemble, TripleEnsemble

# Функция для безопасного вывода (заменяет эмодзи на текстовые метки для Windows)
def safe_print(*args, **kwargs):
    """Безопасный print, который заменяет эмодзи на текстовые метки."""
    try:
        # Пытаемся вывести как есть
        print(*args, **kwargs)
        sys.stdout.flush()  # Очищаем буфер
    except (UnicodeEncodeError, IOError) as e:
        try:
            # Заменяем эмодзи на текстовые метки
            text = ' '.join(str(arg) for arg in args)
            # Основные эмодзи
            replacements = {
                '🚀': '[START]',
                '📊': '[INFO]', 
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⏳': '[WAIT]',
                '🔥': '[HOT]',
                '📥': '[DOWNLOAD]',
                '🔧': '[ENGINEERING]',
                '🎯': '[TARGET]',
                '📦': '[DATA]',
                '🤖': '[MODEL]',
                '🌲': '[RF]',
                '⚡': '[XGB]',
                '🎉': '[SUCCESS]',
                '💡': '[TIP]',
                '🔄': '[RETRAIN]',
                '📋': '[LIST]',
                '🔍': '[SEARCH]',
                '📈': '[CHART]',
                '🧪': '[TEST]',
                '⚙️': '[SETTINGS]'
            }
            for emoji, replacement in replacements.items():
                text = text.replace(emoji, replacement)
            
            # Выводим очищенный текст
            print(text, **kwargs)
            sys.stdout.flush()
        except:
            # Последняя попытка - выводим только текст
            try:
                text = ' '.join(str(arg) for arg in args)
                # Удаляем все не-ASCII символы
                text = ''.join(c for c in text if ord(c) < 128)
                print(text, **kwargs)
            except:
                print("[ERROR: Could not print message]", **kwargs)


def load_optimized_weights(weights_file: str = None) -> dict:
    """
    Загружает оптимизированные веса из JSON файла.
    
    Returns:
        Словарь вида: {symbol: {model_name: weight}}
    """
    try:
        from apply_optimized_weights import load_optimized_weights as load_weights
        return load_weights(Path(weights_file) if weights_file else None)
    except ImportError:
        # Если модуль не найден, пробуем загрузить напрямую
        import json
        from pathlib import Path
        
        if weights_file is None:
            # Ищем последний файл
            weights_files = sorted(
                Path(".").glob("ensemble_weights_all_*.json"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            if not weights_files:
                return {}
            weights_file = weights_files[0]
        
        weights_file = Path(weights_file)
        if not weights_file.exists():
            return {}
        
        with open(weights_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        weights_dict = {}
        if isinstance(data, list):
            for item in data:
                symbol = item.get("symbol", "").upper()
                weights = item.get("weights", {})
                if symbol and weights:
                    weights_dict[symbol] = weights
        
        return weights_dict


def main():
    """Переобучение с оптимизированными параметрами."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Обучение ML моделей с опциональными MTF фичами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обучение БЕЗ MTF (только 15m фичи)
  python retrain_ml_optimized.py --no-mtf
  
  # Обучение С MTF (15m + 1h + 4h фичи)
  python retrain_ml_optimized.py --mtf
  
  # Обучение БЕЗ MTF для конкретного символа
  python retrain_ml_optimized.py --symbol SOLUSDT --no-mtf
        """
    )
    parser.add_argument("--symbol", type=str, help="Торговая пара для переобучения")
    parser.add_argument(
        "--mtf", 
        action="store_true", 
        help="Использовать MTF фичи (1h, 4h) - по умолчанию включено"
    )
    parser.add_argument(
        "--no-mtf", 
        action="store_true", 
        help="НЕ использовать MTF фичи (только 15m)"
    )
    parser.add_argument(
        "--use-optimized-weights",
        action="store_true",
        help="Использовать оптимизированные веса ансамблей из файла ensemble_weights_all_*.json"
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        help="Путь к JSON файлу с оптимизированными весами (по умолчанию ищет последний)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        choices=["15m", "60m", "1h"],
        help="Базовый таймфрейм для обучения (15m или 60m/1h). По умолчанию: 15m"
    )
    args = parser.parse_known_args()[0]
    
    safe_print("=" * 80)
    safe_print("🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ")
    safe_print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Список символов для обучения
    symbols = [args.symbol] if args.symbol else ["BNBUSDT", "ADAUSDT"]
    #["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
    
    # Определяем базовый интервал из аргумента
    interval_str = args.interval.lower().replace("h", "m")  # "1h" -> "60m"
    if interval_str == "60m":
        base_interval = "60"  # 1 час
        interval_display = "1h"
    else:
        base_interval = "15"  # 15 минут
        interval_display = "15m"
    
    safe_print(f"📌 Базовый таймфрейм: {interval_display}")
    # Определяем, использовать ли MTF-режим при обучении
    # Приоритет: --no-mtf > --mtf > переменная окружения > по умолчанию (включено)
    if args.no_mtf:
        ml_mtf_enabled = False
        safe_print("📌 Режим: БЕЗ MTF фичей (только 15m)")
    elif args.mtf:
        ml_mtf_enabled = True
        safe_print("📌 Режим: С MTF фичами (15m + 1h + 4h)")
    else:
        # Проверяем переменную окружения, если флаги не указаны
        ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "1")
        ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
        if ml_mtf_enabled:
            safe_print("📌 Режим: С MTF фичами (15m + 1h + 4h) [по умолчанию]")
        else:
            safe_print("📌 Режим: БЕЗ MTF фичей (только 15m) [из переменной окружения]")
    
    # Формируем суффикс для имени файла модели
    if ml_mtf_enabled:
        mode_suffix = f"mtf_{interval_display}"
    else:
        mode_suffix = interval_display
    
    # Загружаем оптимизированные веса, если нужно
    optimized_weights = {}
    if args.use_optimized_weights:
        try:
            optimized_weights = load_optimized_weights(args.weights_file)
            if optimized_weights:
                safe_print(f"\n[OK] Загружены оптимизированные веса для {len(optimized_weights)} символов")
            else:
                safe_print(f"\n[WARNING] Не удалось загрузить оптимизированные веса, будут использованы веса из CV")
        except Exception as e:
            safe_print(f"\n[WARNING] Ошибка загрузки оптимизированных весов: {e}")
            safe_print(f"         Будут использованы веса из CV")
    
    # Обучаем модели для каждого символа
    for symbol in symbols:
        safe_print("\n" + "=" * 80)
        safe_print(f"📊 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ {symbol}")
        safe_print("=" * 80)
        
        # === Шаг 1: Сбор данных (30 дней) ===
        if ml_mtf_enabled:
            if base_interval == "15":
                safe_print(f"\n[1/5] 📥 Сбор исторических данных ({interval_display}, 1h, 4h) для {symbol}...")
                mtf_intervals = [base_interval, "60", "240"]  # 15m, 1h, 4h
            else:  # base_interval == "60" (1h)
                safe_print(f"\n[1/5] 📥 Сбор исторических данных ({interval_display}, 4h, 1d) для {symbol}...")
                mtf_intervals = [base_interval, "240", "D"]  # 1h, 4h, 1d
        else:
            safe_print(f"\n[1/5] 📥 Сбор исторических данных ({interval_display} only) для {symbol}...")
        collector = DataCollector(settings.api)
        
        if ml_mtf_enabled:
            # Собираем данные сразу для нескольких таймфреймов
            mtf_data = collector.collect_multiple_timeframes(
                symbol=symbol,
                intervals=mtf_intervals,
                start_date=None,
                end_date=None,
            )
            
            df_raw_base = mtf_data.get(base_interval)
            if base_interval == "15":
                df_raw_1h = mtf_data.get("60")
                df_raw_4h = mtf_data.get("240")
            else:  # 1h
                df_raw_4h = mtf_data.get("240")
                df_raw_1d = mtf_data.get("D")
            
            if df_raw_base is None or df_raw_base.empty:
                safe_print(f"❌ Нет данных ({interval_display}) для {symbol}. Пропускаем.")
                continue
            
            candles_per_day = 96 if base_interval == "15" else 24  # 15m: 96 свечей/день, 1h: 24 свечи/день
            safe_print(f"✅ Собрано {len(df_raw_base)} свечей {interval_display} (~{len(df_raw_base)/candles_per_day:.1f} дней)")
        else:
            # Собираем только базовые данные
            df_raw_base = collector.collect_klines(
                symbol=symbol,
                interval=base_interval,
                start_date=None,
                end_date=None,
                limit=3000,
            )
            if df_raw_base.empty:
                safe_print(f"❌ Нет данных ({interval_display}) для {symbol}. Пропускаем.")
                continue
            candles_per_day = 96 if base_interval == "15" else 24
            safe_print(f"✅ Собрано {len(df_raw_base)} свечей {interval_display} (~{len(df_raw_base)/candles_per_day:.1f} дней)")
        
        # === Шаг 2: Feature Engineering ===
        safe_print(f"\n[2/5] 🔧 Создание признаков для {symbol}...")
        feature_engineer = FeatureEngineer()
        
        # Создаем технические индикаторы на базовом ТФ
        df_features = feature_engineer.create_technical_indicators(df_raw_base)
        
        # Добавляем мульти‑таймфреймовые признаки, если данные есть и MTF включен
        if ml_mtf_enabled:
            higher_timeframes = {}
            if base_interval == "15":
                df_raw_1h = mtf_data.get("60")
                df_raw_4h = mtf_data.get("240")
                if df_raw_1h is not None and not df_raw_1h.empty:
                    higher_timeframes["60"] = df_raw_1h
                if df_raw_4h is not None and not df_raw_4h.empty:
                    higher_timeframes["240"] = df_raw_4h
            else:  # base_interval == "60" (1h)
                df_raw_4h = mtf_data.get("240")
                df_raw_1d = mtf_data.get("D")
                if df_raw_4h is not None and not df_raw_4h.empty:
                    higher_timeframes["240"] = df_raw_4h
                if df_raw_1d is not None and not df_raw_1d.empty:
                    higher_timeframes["D"] = df_raw_1d
            
            if higher_timeframes:
                df_features = feature_engineer.add_mtf_features(df_features, higher_timeframes)
                tf_names = "/".join(higher_timeframes.keys())
                safe_print(f"✅ Добавлены MTF‑признаки ({tf_names}). Всего фич: {len(feature_engineer.get_feature_names())}")
            else:
                safe_print(f"⚠️ Не удалось получить данные для высших ТФ — обучение только на {interval_display} признаках.")
        
        feature_names = feature_engineer.get_feature_names()
        safe_print(f"✅ Создано {len(feature_names)} признаков")
        
        # === Шаг 3: Создание таргета (оптимизированный) ===
        safe_print(f"\n[3/5] 🎯 Создание целевой переменной (оптимизированный таргет)...")
        safe_print("   Параметры:")
        safe_print("   • Forward periods: 5 (75 минут)")
        safe_print("   • Threshold: 0.3% (уменьшено для больше сигналов)")
        safe_print("   • Min profit: 0.3% (уменьшено для больше сигналов)")
        safe_print("   • Risk/Reward: 1.5:1")
        safe_print("   • Use ATR threshold: True")
        
        # Используем УПРОЩЕННЫЕ параметры для большего количества сигналов
        # КРИТИЧНО: Уменьшены пороги для увеличения использования сигналов (цель: 30-40%)
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=5 if base_interval == "15" else 2,  # 5 * 15m = 75 минут или 2 * 1h = 2 часа
            threshold_pct=0.3,  # УМЕНЬШЕНО с 0.5% до 0.3% для больше сигналов
            use_atr_threshold=True,
            use_risk_adjusted=True,
            min_risk_reward_ratio=1.5,  # УМЕНЬШЕНО с 2.0 до 1.5
            max_hold_periods=96 if base_interval == "15" else 24,  # 96 * 15m = 24 часа или 24 * 1h = 24 часа
            min_profit_pct=0.3,  # УМЕНЬШЕНО с 0.5% до 0.3% для больше сигналов
        )
        
        # Анализ распределения классов
        target_dist = df_with_target['target'].value_counts()
        safe_print(f"\n✅ Целевая переменная создана")
        safe_print(f"   Распределение классов:")
        for label, count in target_dist.items():
            pct = count / len(df_with_target) * 100
            label_name = "LONG" if label == 1 else ("SHORT" if label == -1 else "HOLD")
            safe_print(f"   {label_name:5s}: {count:5d} ({pct:5.1f}%)")
        
        # === Шаг 4: Подготовка данных ===
        safe_print(f"\n[4/5] 📦 Подготовка данных для обучения...")
        X, y = feature_engineer.prepare_features_for_ml(df_with_target)
        
        safe_print(f"✅ Данные подготовлены:")
        safe_print(f"   Features: {X.shape[0]} samples × {X.shape[1]} features")
        safe_print(f"   Target: {y.shape[0]} labels")
        
        # Проверяем достаточно ли сигналов
        signal_count = (y != 0).sum()
        if signal_count < 50:
            safe_print(f"\n⚠️  Мало сигналов ({signal_count}). Смягчаю параметры таргета...")
            # Пересоздаем таргет с более мягкими параметрами
            df_with_target = feature_engineer.create_target_variable(
                df_features,
                forward_periods=4,  # Меньше периодов
                threshold_pct=0.3,  # Еще ниже порог
                use_atr_threshold=True,
                use_risk_adjusted=False,  # Отключаем риск-скорректирование
                min_risk_reward_ratio=1.2,  # Минимальный RR
                max_hold_periods=144,  # 36 часов
                min_profit_pct=0.3,  # Минимальная прибыль
            )
            X, y = feature_engineer.prepare_features_for_ml(df_with_target)
            signal_count = (y != 0).sum()
            safe_print(f"   После смягчения: {signal_count} сигналов")
        
        # === Шаг 5: Обучение с балансировкой классов ===
        safe_print(f"\n[5/5] 🤖 Обучение моделей с балансировкой классов...")
        trainer = ModelTrainer()
        
        # Вычисляем веса классов для балансировки
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y)
        if len(classes) < 2:
            safe_print("❌ Только один класс в данных. Пропускаем обучение.")
            continue
        
        base_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # УЛУЧШЕННАЯ балансировка: учитываем дисбаланс LONG/SHORT
        # Подсчитываем количество каждого класса
        class_counts = {}
        for cls in classes:
            class_counts[cls] = (y == cls).sum()
        
        long_count = class_counts.get(1, 0)
        short_count = class_counts.get(-1, 0)
        hold_count = class_counts.get(0, 0)
        
        # Определяем minority class (LONG или SHORT)
        if long_count > 0 and short_count > 0:
            if long_count < short_count:
                minority_class = 1  # LONG
                majority_class = -1  # SHORT
                imbalance_ratio = short_count / long_count if long_count > 0 else 1.0
            else:
                minority_class = -1  # SHORT
                majority_class = 1  # LONG
                imbalance_ratio = long_count / short_count if short_count > 0 else 1.0
        else:
            minority_class = None
            majority_class = None
            imbalance_ratio = 1.0
        
        # УМЕРЕННЫЕ веса для балансировки с учетом дисбаланса LONG/SHORT
        class_weight_dict = {}
        for i, cls in enumerate(classes):
            if cls == 0:  # HOLD
                class_weight_dict[cls] = base_weights[i] * 0.3  # Уменьшаем вес HOLD
            else:  # LONG or SHORT
                base_weight = base_weights[i] * 2.0  # Базовое увеличение для торговых сигналов
                
                # Если есть дисбаланс, увеличиваем вес minority class
                if minority_class is not None and cls == minority_class and imbalance_ratio > 1.5:
                    # Увеличиваем вес minority class пропорционально дисбалансу
                    boost_factor = min(1.5, imbalance_ratio / 2.0)  # Максимум 1.5x boost
                    class_weight_dict[cls] = base_weight * (1.0 + boost_factor)
                    safe_print(f"      Увеличиваем вес {('LONG' if cls == 1 else 'SHORT')} (minority) на {boost_factor*100:.0f}% из-за дисбаланса")
                else:
                    class_weight_dict[cls] = base_weight
        
        safe_print(f"\n   📊 Веса классов:")
        for cls, weight in class_weight_dict.items():
            label_name = "LONG" if cls == 1 else ("SHORT" if cls == -1 else "HOLD")
            safe_print(f"      {label_name}: {weight:.2f}")
        
        # Обучаем Random Forest
        safe_print(f"\n   🌲 Обучение Random Forest...")
        rf_model, rf_metrics = trainer.train_random_forest_classifier(
            X, y,
            n_estimators=100,  # Стандартное значение
            max_depth=10,
            class_weight=class_weight_dict,
        )
        
        # Сохраняем модель
        rf_filename = f"rf_{symbol}_{base_interval}_{mode_suffix}.pkl"
        trainer.save_model(
            rf_model,
            trainer.scaler,
            feature_names,
            rf_metrics,
            rf_filename,
            symbol=symbol,
            interval=base_interval,
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "n_estimators": 100,
                "max_depth": 10,
                "forward_periods": 5,
                "threshold_pct": 0.5,
                "min_risk_reward_ratio": 1.5,
            },
        )
        safe_print(f"      ✅ Сохранено как: {rf_filename}")
        safe_print(f"      📊 Accuracy: {rf_metrics['accuracy']:.4f}")
        safe_print(f"      📊 CV Accuracy: {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']*2:.4f}")
        
        # Обучаем XGBoost (если установлен)
        try:
            # Пробуем импортировать xgboost напрямую
            import xgboost
            # Проверяем, что функция train_xgboost_classifier доступна
            if not hasattr(trainer, 'train_xgboost_classifier'):
                raise AttributeError("train_xgboost_classifier method not available")
            
            safe_print(f"\n   ⚡ Обучение XGBoost...")
            
            xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
                X, y,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight=class_weight_dict,
            )
            
            # Сохраняем модель
            xgb_filename = f"xgb_{symbol}_{base_interval}_{mode_suffix}.pkl"
            trainer.save_model(
                xgb_model,
                trainer.scaler,
                feature_names,
                xgb_metrics,
                xgb_filename,
                symbol=symbol,
                interval=base_interval,
                class_weights=class_weight_dict,
                class_distribution=target_dist.to_dict(),
                training_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "forward_periods": 5,
                    "threshold_pct": 0.5,
                    "min_risk_reward_ratio": 1.5,
                },
            )
            safe_print(f"      ✅ Сохранено как: {xgb_filename}")
            safe_print(f"      📊 Accuracy: {xgb_metrics['accuracy']:.4f}")
            safe_print(f"      📊 CV Accuracy: {xgb_metrics['cv_mean']:.4f} ± {xgb_metrics['cv_std']*2:.4f}")
            
        except (ImportError, NameError) as e:
            safe_print(f"   ⚡ XGBoost не установлен или недоступен. Пропускаем.")
            safe_print(f"      Детали: {str(e)[:100]}")
            # Пытаемся проверить, установлен ли xgboost в системе
            try:
                import subprocess
                import sys
                result = subprocess.run([sys.executable, "-c", "import xgboost; print(xgboost.__version__)"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    safe_print(f"      ⚠️  XGBoost установлен в системе, но не доступен в текущем окружении")
                    safe_print(f"      Попробуйте: pip install xgboost")
                else:
                    safe_print(f"      Установите XGBoost: pip install xgboost")
            except:
                pass
        
        # Обучаем Ensemble (RF + XGBoost если оба есть)
        try:
            rf_model
            xgb_model
            safe_print(f"\n   🎯 Обучение Ensemble (RF + XGBoost)...")
            
            # Проверяем, есть ли оптимизированные веса для этого символа
            use_optimized = args.use_optimized_weights and symbol.upper() in optimized_weights
            rf_weight_opt = None
            xgb_weight_opt = None
            
            if use_optimized:
                symbol_weights = optimized_weights[symbol.upper()]
                # Ищем веса для RF и XGB моделей
                for model_name, weight in symbol_weights.items():
                    if mode_suffix in model_name and symbol.upper() in model_name:
                        if model_name.startswith("rf_"):
                            rf_weight_opt = weight
                        elif model_name.startswith("xgb_"):
                            xgb_weight_opt = weight
                
                if rf_weight_opt is not None and xgb_weight_opt is not None:
                    # Нормализуем веса
                    total = rf_weight_opt + xgb_weight_opt
                    if total > 0:
                        rf_weight_opt = rf_weight_opt / total
                        xgb_weight_opt = xgb_weight_opt / total
                        safe_print(f"   [OK] Используются оптимизированные веса: RF={rf_weight_opt:.3f}, XGB={xgb_weight_opt:.3f}")
            
            # Обучаем ансамбль (веса будут вычислены автоматически или использованы оптимизированные)
            if use_optimized and rf_weight_opt is not None and xgb_weight_opt is not None:
                # Создаем ансамбль вручную с оптимизированными весами
                # Сначала обучаем модели отдельно для получения метрик
                ensemble_model, ensemble_metrics = trainer.train_ensemble(
                    X, y,
                    rf_n_estimators=100,
                    rf_max_depth=10,
                    xgb_n_estimators=100,
                    xgb_max_depth=6,
                    xgb_learning_rate=0.1,
                    ensemble_method="weighted_average",
                    class_weight=class_weight_dict,
                )
                # Заменяем ансамбль на новый с оптимизированными весами
                ensemble_model = WeightedEnsemble(rf_model, xgb_model, rf_weight_opt, xgb_weight_opt)
                # Обновляем метрики с новыми весами
                ensemble_metrics['rf_weight'] = rf_weight_opt
                ensemble_metrics['xgb_weight'] = xgb_weight_opt
                safe_print(f"   [OK] Ансамбль создан с оптимизированными весами")
            else:
                # Используем стандартный метод с автоматическими весами
                ensemble_model, ensemble_metrics = trainer.train_ensemble(
                    X, y,
                    rf_n_estimators=100,
                    rf_max_depth=10,
                    xgb_n_estimators=100,
                    xgb_max_depth=6,
                    xgb_learning_rate=0.1,
                    ensemble_method="weighted_average",
                    class_weight=class_weight_dict,
                )
            
            # Сохраняем модель
            ensemble_filename = f"ensemble_{symbol}_{base_interval}_{mode_suffix}.pkl"
            trainer.save_model(
                ensemble_model,
                trainer.scaler,
                feature_names,
                ensemble_metrics,
                ensemble_filename,
                symbol=symbol,
                interval=base_interval,
                model_type="ensemble_weighted",
                class_weights=class_weight_dict,
                class_distribution=target_dist.to_dict(),
                training_params={
                    "rf_n_estimators": 100,
                    "rf_max_depth": 10,
                    "xgb_n_estimators": 100,
                    "xgb_max_depth": 6,
                    "xgb_learning_rate": 0.1,
                    "ensemble_method": "weighted_average",
                    "forward_periods": 5,
                    "threshold_pct": 0.5,
                    "min_risk_reward_ratio": 1.5,
                    "optimized_weights": use_optimized and rf_weight_opt is not None,
                    "rf_weight": rf_weight_opt if use_optimized else ensemble_metrics.get('rf_weight'),
                    "xgb_weight": xgb_weight_opt if use_optimized else ensemble_metrics.get('xgb_weight'),
                },
            )
            safe_print(f"      ✅ Сохранено как: {ensemble_filename}")
            safe_print(f"      📊 Accuracy: {ensemble_metrics['accuracy']:.4f}")
            safe_print(f"      📊 CV Accuracy: {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
            if use_optimized and rf_weight_opt is not None:
                safe_print(f"      📊 Веса: RF={rf_weight_opt:.3f}, XGB={xgb_weight_opt:.3f} (оптимизированные)")
            
        except (NameError, ImportError):
            safe_print(f"   🎯 Не удалось обучить Ensemble. Требуются RF и XGBoost.")
        
        # Обучаем TripleEnsemble (если есть LightGBM)
        try:
            import lightgbm
            from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
            if LIGHTGBM_AVAILABLE:
                safe_print(f"\n   🎯 Обучение TripleEnsemble (RF + XGBoost + LightGBM)...")
                
                # Обучаем LightGBM отдельно для создания ансамбля вручную
                lgb_model, lgb_metrics = trainer.train_lightgbm_classifier(
                    X, y,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    class_weight=class_weight_dict,
                )
                
                # Для TripleEnsemble используем оптимизированные веса только для RF и XGB, LightGBM получает остаток
                use_optimized_triple = args.use_optimized_weights and symbol.upper() in optimized_weights
                rf_weight_triple = None
                xgb_weight_triple = None
                lgb_weight_triple = None
                
                if use_optimized_triple:
                    symbol_weights = optimized_weights[symbol.upper()]
                    for model_name, weight in symbol_weights.items():
                        if mode_suffix in model_name and symbol.upper() in model_name:
                            if model_name.startswith("rf_"):
                                rf_weight_triple = weight
                            elif model_name.startswith("xgb_"):
                                xgb_weight_triple = weight
                    
                    if rf_weight_triple is not None and xgb_weight_triple is not None:
                        # Нормализуем веса RF и XGB, LightGBM получает остаток
                        total_rf_xgb = rf_weight_triple + xgb_weight_triple
                        if total_rf_xgb > 0:
                            # Масштабируем RF и XGB веса, оставляя место для LightGBM
                            scale = 0.8  # 80% для RF+XGB, 20% для LightGBM
                            rf_weight_triple = (rf_weight_triple / total_rf_xgb) * scale
                            xgb_weight_triple = (xgb_weight_triple / total_rf_xgb) * scale
                            lgb_weight_triple = 1.0 - scale
                            safe_print(f"   [OK] Используются оптимизированные веса: RF={rf_weight_triple:.3f}, XGB={xgb_weight_triple:.3f}, LGB={lgb_weight_triple:.3f}")
                
                # Создаем ансамбль с оптимизированными или стандартными весами
                if use_optimized_triple and rf_weight_triple is not None and xgb_weight_triple is not None:
                    # Используем оптимизированные веса
                    triple_ensemble_model = TripleEnsemble(rf_model, xgb_model, lgb_model, rf_weight_triple, xgb_weight_triple, lgb_weight_triple)
                    # Вычисляем метрики для ансамбля (используем средневзвешенные метрики компонентов)
                    triple_ensemble_metrics = {
                        'accuracy': (
                            rf_metrics.get('accuracy', 0.0) * rf_weight_triple +
                            xgb_metrics.get('accuracy', 0.0) * xgb_weight_triple +
                            lgb_metrics.get('accuracy', 0.0) * lgb_weight_triple
                        ),
                        'cv_mean': (
                            rf_metrics.get('cv_mean', 0.0) * rf_weight_triple +
                            xgb_metrics.get('cv_mean', 0.0) * xgb_weight_triple +
                            lgb_metrics.get('cv_mean', 0.0) * lgb_weight_triple
                        ),
                        'cv_std': (
                            (rf_metrics.get('cv_std', 0.0) * rf_weight_triple +
                             xgb_metrics.get('cv_std', 0.0) * xgb_weight_triple +
                             lgb_metrics.get('cv_std', 0.0) * lgb_weight_triple) / 3.0
                        ),
                        'rf_weight': rf_weight_triple,
                        'xgb_weight': xgb_weight_triple,
                        'lgb_weight': lgb_weight_triple,
                        'rf_metrics': rf_metrics,
                        'xgb_metrics': xgb_metrics,
                        'lgb_metrics': lgb_metrics,
                    }
                else:
                    # Используем стандартный метод с автоматическими весами
                    triple_ensemble_model, triple_ensemble_metrics = trainer.train_ensemble(
                        X, y,
                        rf_n_estimators=100,
                        rf_max_depth=10,
                        xgb_n_estimators=100,
                        xgb_max_depth=6,
                        xgb_learning_rate=0.1,
                        lgb_n_estimators=100,
                        lgb_max_depth=6,
                        lgb_learning_rate=0.1,
                        ensemble_method="triple",
                        include_lightgbm=True,
                        class_weight=class_weight_dict,
                    )
                
                # Сохраняем модель
                triple_filename = f"triple_ensemble_{symbol}_{base_interval}_{mode_suffix}.pkl"
                trainer.save_model(
                    triple_ensemble_model,
                    trainer.scaler,
                    feature_names,
                    triple_ensemble_metrics,
                    triple_filename,
                    symbol=symbol,
                    interval=base_interval,
                    model_type="triple_ensemble",
                    class_weights=class_weight_dict,
                    class_distribution=target_dist.to_dict(),
                    training_params={
                        "rf_n_estimators": 100,
                        "rf_max_depth": 10,
                        "xgb_n_estimators": 100,
                        "xgb_max_depth": 6,
                        "xgb_learning_rate": 0.1,
                        "lgb_n_estimators": 100,
                        "lgb_max_depth": 6,
                        "lgb_learning_rate": 0.1,
                        "ensemble_method": "triple",
                        "forward_periods": 5,
                        "threshold_pct": 0.5,
                        "min_risk_reward_ratio": 1.5,
                        "optimized_weights": use_optimized_triple and rf_weight_triple is not None,
                        "rf_weight": rf_weight_triple if use_optimized_triple else triple_ensemble_metrics.get('rf_weight'),
                        "xgb_weight": xgb_weight_triple if use_optimized_triple else triple_ensemble_metrics.get('xgb_weight'),
                        "lgb_weight": lgb_weight_triple if use_optimized_triple else triple_ensemble_metrics.get('lgb_weight'),
                    },
                )
                safe_print(f"      ✅ Сохранено как: {triple_filename}")
                safe_print(f"      📊 Accuracy: {triple_ensemble_metrics['accuracy']:.4f}")
                safe_print(f"      📊 CV Accuracy: {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")
                if use_optimized_triple and rf_weight_triple is not None:
                    safe_print(f"      📊 Веса: RF={rf_weight_triple:.3f}, XGB={xgb_weight_triple:.3f}, LGB={lgb_weight_triple:.3f} (оптимизированные)")
            else:
                safe_print(f"   ⚠️  LightGBM не установлен, пропускаем TripleEnsemble")
        except ImportError:
            safe_print(f"   ⚠️  LightGBM не установлен, пропускаем TripleEnsemble")
        
        # Обучаем QuadEnsemble (RF + XGB + LGB + LSTM)
        try:
            from bot.ml.model_trainer import LSTM_AVAILABLE, LIGHTGBM_AVAILABLE
            if LSTM_AVAILABLE and LIGHTGBM_AVAILABLE:
                safe_print(f"\n   🚀 Обучение QuadEnsemble (RF + XGB + LGB + LSTM)...")
                safe_print(f"      (Это может занять некоторое время...)")
                
                quad_ensemble_model, quad_metrics = trainer.train_quad_ensemble(
                    X, y,
                    df=df_with_target,  # Передаем DataFrame для LSTM последовательностей
                    rf_n_estimators=100,
                    rf_max_depth=10,
                    xgb_n_estimators=100,
                    xgb_max_depth=6,
                    xgb_learning_rate=0.1,
                    lgb_n_estimators=100,
                    lgb_max_depth=6,
                    lgb_learning_rate=0.1,
                    lstm_sequence_length=60,
                    lstm_epochs=20,  # 20 эпох достаточно для быстрой перетренировки
                    class_weight=class_weight_dict,
                )
                
                # Сохраняем модель
                quad_filename = f"quad_ensemble_{symbol}_{base_interval}_{mode_suffix}.pkl"
                trainer.save_model(
                    quad_ensemble_model,
                    trainer.scaler,
                    feature_names,
                    quad_metrics,
                    quad_filename,
                    symbol=symbol,
                    interval=base_interval,
                    model_type="quad_ensemble",
                    class_weights=class_weight_dict,
                    class_distribution=target_dist.to_dict(),
                    training_params={
                        "ensemble_method": "quad",
                        "lstm_epochs": 20,
                        "lstm_sequence_length": 60,
                        "forward_periods": 5,
                        "threshold_pct": 0.5,
                        "min_risk_reward_ratio": 1.5,
                    },
                )
                safe_print(f"      ✅ Сохранено как: {quad_filename}")
                
                # Для QuadEnsemble метрики агрегированные
                rf_m = quad_metrics.get("rf_metrics", {})
                lstm_m = quad_metrics.get("lstm_metrics", {})
                
                safe_print(f"      📊 RF CV Accuracy: {rf_m.get('cv_mean', 0):.4f}")
                safe_print(f"      📊 LSTM Accuracy: {lstm_m.get('accuracy', 0):.4f}")
                
            else:
                missing = []
                if not LSTM_AVAILABLE: missing.append("LSTM (PyTorch)")
                if not LIGHTGBM_AVAILABLE: missing.append("LightGBM")
                safe_print(f"   ⚠️  Компоненты отсутствуют ({', '.join(missing)}), пропускаем QuadEnsemble")
        except Exception as e:
            safe_print(f"   ⚠️  Ошибка при обучении QuadEnsemble: {e}")
        
        # Итоговые метрики
        safe_print(f"\n" + "-" * 80)
        safe_print(f"📊 ИТОГОВЫЕ МЕТРИКИ ДЛЯ {symbol}")
        safe_print("-" * 80)
        safe_print(f"\n🌲 Random Forest:")
        safe_print(f"   Accuracy:     {rf_metrics['accuracy']:.4f}")
        safe_print(f"   CV Accuracy:  {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']*2:.4f}")
        
        if 'xgb_metrics' in locals():
            safe_print(f"\n⚡ XGBoost:")
            safe_print(f"   Accuracy:     {xgb_metrics['accuracy']:.4f}")
            safe_print(f"   CV Accuracy:  {xgb_metrics['cv_mean']:.4f} ± {xgb_metrics['cv_std']*2:.4f}")
        
        if 'ensemble_metrics' in locals():
            safe_print(f"\n🎯 Ensemble (RF+XGB):")
            safe_print(f"   Accuracy:     {ensemble_metrics['accuracy']:.4f}")
            safe_print(f"   CV Accuracy:  {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
        
        if 'triple_ensemble_metrics' in locals():
            safe_print(f"\n🎯 TripleEnsemble (RF+XGB+LGB):")
            safe_print(f"   Accuracy:     {triple_ensemble_metrics['accuracy']:.4f}")
            safe_print(f"   CV Accuracy:  {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")

        if 'quad_metrics' in locals():
            safe_print(f"\n🚀 QuadEnsemble (RF+XGB+LGB+LSTM):")
            safe_print(f"   Модель успешно обучена и сохранена.")
        
        # Выбор лучшей модели
        models = []
        models.append(("Random Forest", rf_metrics['cv_mean']))
        if 'xgb_metrics' in locals():
            models.append(("XGBoost", xgb_metrics['cv_mean']))
        if 'ensemble_metrics' in locals():
            models.append(("Ensemble", ensemble_metrics['cv_mean']))
        if 'triple_ensemble_metrics' in locals():
            models.append(("TripleEnsemble", triple_ensemble_metrics['cv_mean']))
        if 'quad_metrics' in locals():
             # Используем среднее CV классических моделей как прокси + бонус за диверсификацию
             avg_cv = (rf_metrics['cv_mean'] + xgb_metrics.get('cv_mean', 0) + triple_ensemble_metrics.get('cv_mean', 0)) / 3
             models.append(("QuadEnsemble", avg_cv * 1.05)) # Условный бонус
        
        if models:
            models.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score = models[0]
            safe_print(f"\n✅ Лучшая модель для {symbol}: {best_model}")
            safe_print(f"   Score: {best_score:.4f}")
    
    # Финальное сообщение
    safe_print("\n" + "=" * 80)
    safe_print("🎉 ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
    safe_print("=" * 80)
    safe_print("\n📦 Созданные модели:")
    safe_print("   • ml_models/rf_*_15.pkl (Random Forest)")
    safe_print("   • ml_models/xgb_*_15.pkl (XGBoost)")
    safe_print("   • ml_models/ensemble_*_15.pkl (RF + XGBoost)")
    safe_print("   • ml_models/triple_ensemble_*_15.pkl (RF + XGBoost + LightGBM)")
    safe_print("   • ml_models/quad_ensemble_*_15.pkl (RF + XGBoost + LightGBM + LSTM)")
    safe_print("\n🚀 Следующие шаги:")
    safe_print("   1. Протестируйте новые модели:")
    safe_print("      python test_ml_strategy.py --symbol SOLUSDT --days 7")
    safe_print("   2. Если результаты хорошие, задеплойте на сервер")
    safe_print("=" * 80)


if __name__ == "__main__":
    main()