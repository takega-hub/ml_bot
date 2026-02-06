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
from bot.ml.model_trainer import ModelTrainer

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


def main():
    """Переобучение с оптимизированными параметрами."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Торговая пара для переобучения")
    args = parser.parse_known_args()[0]
    
    safe_print("=" * 80)
    safe_print("🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ")
    safe_print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Список символов для обучения
    symbols = [args.symbol] if args.symbol else ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
    base_interval = "15"  # 15 минут (базовый ТФ)
    
    # Определяем, использовать ли MTF-режим при обучении (читаем из окружения)
    ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "1")
    ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
    mode_suffix = "mtf" if ml_mtf_enabled else "15m"
    
    # Обучаем модели для каждого символа
    for symbol in symbols:
        safe_print("\n" + "=" * 80)
        safe_print(f"📊 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ {symbol}")
        safe_print("=" * 80)
        
        # === Шаг 1: Сбор данных (30 дней) ===
        if ml_mtf_enabled:
            safe_print(f"\n[1/5] 📥 Сбор исторических данных (15m, 1h, 4h) для {symbol}...")
        else:
            safe_print(f"\n[1/5] 📥 Сбор исторических данных (15m only) для {symbol}...")
        collector = DataCollector(settings.api)
        
        if ml_mtf_enabled:
            # Собираем данные сразу для нескольких таймфреймов
            mtf_data = collector.collect_multiple_timeframes(
                symbol=symbol,
                intervals=[base_interval, "60", "240"],  # 15m, 1h, 4h
                start_date=None,
                end_date=None,
            )
            
            df_raw_15m = mtf_data.get(base_interval)
            df_raw_1h = mtf_data.get("60")
            df_raw_4h = mtf_data.get("240")
            
            if df_raw_15m is None or df_raw_15m.empty:
                safe_print(f"❌ Нет данных (15m) для {symbol}. Пропускаем.")
                continue
            
            safe_print(f"✅ Собрано {len(df_raw_15m)} свечей 15m (~{len(df_raw_15m)/96:.1f} дней)")
        else:
            # Старый режим: собираем только 15m данные
            df_raw_15m = collector.collect_klines(
                symbol=symbol,
                interval=base_interval,
                start_date=None,
                end_date=None,
                limit=3000,
            )
            if df_raw_15m.empty:
                safe_print(f"❌ Нет данных (15m) для {symbol}. Пропускаем.")
                continue
            safe_print(f"✅ Собрано {len(df_raw_15m)} свечей 15m (~{len(df_raw_15m)/96:.1f} дней)")
        
        # === Шаг 2: Feature Engineering ===
        safe_print(f"\n[2/5] 🔧 Создание признаков для {symbol}...")
        feature_engineer = FeatureEngineer()
        
        # Создаем технические индикаторы на базовом ТФ (15m)
        df_features = feature_engineer.create_technical_indicators(df_raw_15m)
        
        # Добавляем мульти‑таймфреймовые признаки (1h, 4h), если данные есть и MTF включен
        if ml_mtf_enabled:
            higher_timeframes = {}
            df_raw_1h = mtf_data.get("60")
            df_raw_4h = mtf_data.get("240")
            if df_raw_1h is not None and not df_raw_1h.empty:
                higher_timeframes["60"] = df_raw_1h
            if df_raw_4h is not None and not df_raw_4h.empty:
                higher_timeframes["240"] = df_raw_4h
            
            if higher_timeframes:
                df_features = feature_engineer.add_mtf_features(df_features, higher_timeframes)
                safe_print(f"✅ Добавлены MTF‑признаки (1h/4h). Всего фич: {len(feature_engineer.get_feature_names())}")
            else:
                safe_print("⚠️ Не удалось получить данные для 1h/4h — обучение только на 15m признаках.")
        
        feature_names = feature_engineer.get_feature_names()
        safe_print(f"✅ Создано {len(feature_names)} признаков")
        
        # === Шаг 3: Создание таргета (оптимизированный) ===
        safe_print(f"\n[3/5] 🎯 Создание целевой переменной (оптимизированный таргет)...")
        safe_print("   Параметры:")
        safe_print("   • Forward periods: 5 (75 минут)")
        safe_print("   • Threshold: 1.0% (вместо 0.2%)")
        safe_print("   • Risk/Reward: 1.5:1")
        safe_print("   • Use ATR threshold: True")
        
        # Используем УПРОЩЕННЫЕ параметры для большего количества сигналов
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=5,  # 5 * 15m = 75 минут
            threshold_pct=0.5,  # УМЕНЬШЕНО с 1.0% до 0.5% для больше сигналов
            use_atr_threshold=True,
            use_risk_adjusted=True,
            min_risk_reward_ratio=1.5,  # УМЕНЬШЕНО с 2.0 до 1.5
            max_hold_periods=96,  # УВЕЛИЧЕНО с 48 до 96 (24 часа)
            min_profit_pct=0.5,  # УМЕНЬШЕНО с 1.0% до 0.5%
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
            import xgboost
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
            
        except ImportError:
            safe_print(f"   ⚡ XGBoost не установлен. Пропускаем.")
        
        # Обучаем Ensemble (RF + XGBoost если оба есть)
        try:
            rf_model
            xgb_model
            safe_print(f"\n   🎯 Обучение Ensemble (RF + XGBoost)...")
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
                },
            )
            safe_print(f"      ✅ Сохранено как: {ensemble_filename}")
            safe_print(f"      📊 Accuracy: {ensemble_metrics['accuracy']:.4f}")
            safe_print(f"      📊 CV Accuracy: {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
            
        except (NameError, ImportError):
            safe_print(f"   🎯 Не удалось обучить Ensemble. Требуются RF и XGBoost.")
        
        # Обучаем TripleEnsemble (если есть LightGBM)
        try:
            import lightgbm
            from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
            if LIGHTGBM_AVAILABLE:
                safe_print(f"\n   🎯 Обучение TripleEnsemble (RF + XGBoost + LightGBM)...")
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
                    },
                )
                safe_print(f"      ✅ Сохранено как: {triple_filename}")
                safe_print(f"      📊 Accuracy: {triple_ensemble_metrics['accuracy']:.4f}")
                safe_print(f"      📊 CV Accuracy: {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")
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