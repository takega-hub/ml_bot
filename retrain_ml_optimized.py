"""
Улучшенный скрипт переобучения ML-модели с оптимизациями для большего количества сигналов.

Улучшения:
1. Более агрессивный таргет (movement > 1%)
2. Балансировка классов (class_weight)
3. Увеличенные данные:
   - 15m модели: 30 дней
   - 1h модели: 180 дней (для лучшего качества обучения)
4. Оптимизированные гиперпараметры
"""
import warnings
import os
import sys
import json
import gc
from datetime import datetime, timedelta

if sys.platform == 'win32':
    pass

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import ApiSettings, load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE, LSTM_AVAILABLE


def safe_print(*args, **kwargs):
        try:
            text = ' '.join(str(arg) for arg in args)
            replacements = {
                '🚀': '[START]', '📊': '[INFO]', '✅': '[OK]', '❌': '[ERROR]',
                '⏳': '[WAIT]', '🔥': '[HOT]', '📥': '[DOWNLOAD]', '🔧': '[ENGINEERING]',
                '🎯': '[TARGET]', '📦': '[DATA]', '🤖': '[MODEL]', '🌲': '[RF]',
                '⚡': '[XGB]', '🎉': '[SUCCESS]', '💡': '[TIP]', '🔄': '[RETRAIN]',
                '📋': '[LIST]', '🔍': '[SEARCH]', '📈': '[CHART]', '🧪': '[TEST]',
                '⚙️': '[SETTINGS]', '⚠️': '[WARN]', 'ℹ️': '[INFO]', '💪': '[STRONG]',
                '🔹': '[INFO]'
            }
            for emoji, replacement in replacements.items():
                text = text.replace(emoji, replacement)
            if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
                text = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(text, **kwargs)
            sys.stdout.flush()
        except Exception:
            try:
                print(*args, **kwargs)
            except Exception:
                pass


def _collect_by_days(collector: DataCollector, symbol: str, interval: str, days_back: int):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
        save_to_file=True,
    )


def _prepare_training_matrix(trainer: ModelTrainer, df_target):
    exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "target"}
    feature_cols = [col for col in df_target.columns if col not in exclude_cols]
    X = df_target[feature_cols].values
    y = df_target["target"].values
    trainer.feature_engineer.feature_names = feature_cols
    return X, y, feature_cols


def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return default


def _clamp_float(v, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


def _load_hyperparams(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Обучение ML моделей с опциональными MTF фичами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", type=str, help="Торговая пара для переобучения")
    parser.add_argument("--mtf", action="store_true", help="Использовать MTF фичи (1h, 4h)")
    parser.add_argument("--no-mtf", action="store_true", help="НЕ использовать MTF фичи (только 15m)")
    parser.add_argument("--interval", type=str, default="15m", choices=["15m", "60m", "1h"])
    parser.add_argument("--model-suffix", type=str, default="")
    parser.add_argument("--report-json", type=str)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--hyperparams-json", type=str)
    parser.add_argument("--use-triple-barrier", action="store_true", help="Использовать Triple Barrier Method для разметки")
    parser.add_argument("--use-meta-labeling", action="store_true", help="Использовать Meta-Labeling для фильтрации сигналов")
    parser.add_argument("--prune-features", action="store_true", help="Удалять малозначимые признаки")
    args = parser.parse_known_args()[0]
    hyperparams = _load_hyperparams(args.hyperparams_json)

    safe_print("=" * 80)
    safe_print("🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ")
    safe_print("=" * 80)

    load_settings()
    api_settings = ApiSettings()
    symbols = [args.symbol] if args.symbol else ["BNBUSDT", "ADAUSDT"]
    interval_arg = (args.interval or "15m").strip().lower()
    if interval_arg in ("1h", "60m"):
        base_interval = "60"
        interval_display = "1h"
    else:
        base_interval = "15"
        interval_display = "15m"

    safe_print(f"📌 Базовый таймфрейм: {interval_display}")
    if args.no_mtf:
        ml_mtf_enabled = False
        safe_print("📌 Режим: БЕЗ MTF фичей (только 15m)")
    elif args.mtf:
        ml_mtf_enabled = True
        safe_print("📌 Режим: С MTF фичами (15m + 1h + 4h)")
    else:
        ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "0")
        ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
        safe_print("📌 Режим: С MTF фичами (15m + 1h + 4h) [из переменной окружения]" if ml_mtf_enabled else "📌 Режим: БЕЗ MTF фичей (только 15m) [по умолчанию]")

    trainer = ModelTrainer()
    rf_metrics = None
    xgb_metrics = None
    ensemble_metrics = None
    triple_ensemble_metrics = None
    quad_metrics = None

    for symbol in symbols:
        # Принудительная очистка памяти перед каждой парой
        gc.collect()

        safe_print("\n" + "=" * 80)
        safe_print(f"📊 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ {symbol}")
        safe_print("=" * 80)
        collector = DataCollector(api_settings)
        if base_interval == "60":
            days = _clamp_int(hyperparams.get("training_days_1h"), 60, 365, 180)
            safe_print(f"\n[1/5] 📥 Сбор исторических данных ({interval_display}) для {symbol}...")
            safe_print(f"   Период обучения: {days} дней (для 1h моделей)")
            df = _collect_by_days(collector, symbol, interval="60", days_back=days)
            safe_print(f"✅ Собрано {len(df)} свечей 1h (~{len(df) / 24:.1f} дней)")
        else:
            days = _clamp_int(hyperparams.get("training_days_15m"), 20, 180, 30)
            safe_print(f"\n[1/5] 📥 Сбор исторических данных ({interval_display}) для {symbol}...")
            safe_print(f"   Период обучения: {days} дней (для 15m моделей)")
            df = _collect_by_days(collector, symbol, interval="15", days_back=days)
            safe_print(f"✅ Собрано {len(df)} свечей 15m (~{len(df) / 96:.1f} дней)")
        if df is None or df.empty:
            safe_print(f"❌ Не удалось собрать данные для {symbol}")
            continue

        safe_print(f"\n[2/5] 🔧 Создание признаков для {symbol}...")
        engineer = FeatureEngineer()
        df_feat = engineer.create_technical_indicators(df)
        if ml_mtf_enabled:
            safe_print("⚠️ MTF-фичи временно пропущены в smoke-режиме обучения")
        safe_print(f"✅ Создано {len(df_feat.columns)} признаков")

        safe_print(f"\n[3/5] 🎯 Создание целевой переменной для {symbol}...")

        # Check if we should use Triple Barrier Method
        use_triple_barrier = args.use_triple_barrier or os.getenv("USE_TRIPLE_BARRIER", "0") in ("1", "true", "True", "yes")
        use_meta_labeling = args.use_meta_labeling or hyperparams.get("use_meta_labeling", False) or os.getenv("USE_META_LABELING", "0") in ("1", "true", "True", "yes")

        if use_triple_barrier:
            safe_print("📌 Используется Triple Barrier Method (TBM)")
            pt_sl_ratio = _clamp_float(hyperparams.get("tbm_pt_sl_ratio"), 1.0, 5.0, 2.0)
            volatility_lookback = _clamp_int(hyperparams.get("tbm_volatility_lookback"), 10, 100, 20)
            vertical_barrier = _clamp_int(hyperparams.get("tbm_vertical_barrier"), 12, 96, 24)

            df_target = engineer.create_triple_barrier_labels(
                df_feat,
                pt_sl_ratio=pt_sl_ratio,
                volatility_lookback=volatility_lookback,
                vertical_barrier_candles=vertical_barrier
            )
        else:
            if base_interval == "60":
                forward_periods = _clamp_int(hyperparams.get("forward_periods_1h"), 4, 20, 8)
                threshold_pct = _clamp_float(hyperparams.get("threshold_pct_1h"), 0.2, 2.5, 0.8)
                min_profit_pct = _clamp_float(hyperparams.get("min_profit_pct_1h"), 0.2, 2.0, 0.8)
            else:
                forward_periods = _clamp_int(hyperparams.get("forward_periods_15m"), 3, 16, 5)
                threshold_pct = _clamp_float(hyperparams.get("threshold_pct_15m"), 0.15, 1.5, 0.3)
                min_profit_pct = _clamp_float(hyperparams.get("min_profit_pct_15m"), 0.1, 1.2, 0.3)

            df_target = engineer.create_target_variable(
                df_feat,
                forward_periods=forward_periods,
                threshold_pct=threshold_pct,
                min_profit_pct=min_profit_pct,
            )

        if df_target is None or df_target.empty:
            safe_print(f"❌ Не удалось создать целевую переменную для {symbol}")
            continue

        safe_print("✅ Целевая переменная создана")

        safe_print(f"\n[4/5] 📦 Подготовка данных для обучения...")
        X, y, feature_names = _prepare_training_matrix(trainer, df_target)

        if args.prune_features:
            kept_features = trainer.prune_features(X, y, feature_names)
            feature_names = kept_features
            # Обновляем X после удаления признаков
            X = df_target[feature_names].values
            trainer.feature_engineer.feature_names = feature_names

        safe_print(f"✅ Данные подготовлены:\n   Features: {X.shape[0]} samples × {X.shape[1]} features\n   Target: {len(y)} labels")

        safe_print(f"\n[5/5] 🤖 Обучение моделей с балансировкой классов...")
        rf_n_estimators = _clamp_int(hyperparams.get("rf_n_estimators"), 80, 500, 100)
        rf_max_depth = _clamp_int(hyperparams.get("rf_max_depth"), 4, 20, 10)
        xgb_n_estimators = _clamp_int(hyperparams.get("xgb_n_estimators"), 80, 600, 100)
        xgb_max_depth = _clamp_int(hyperparams.get("xgb_max_depth"), 3, 12, 6)
        xgb_learning_rate = _clamp_float(hyperparams.get("xgb_learning_rate"), 0.02, 0.3, 0.1)
        safe_print("\n   🌲 Обучение Random Forest...")
        rf_model, rf_metrics = trainer.train_random_forest_classifier(
            X,
            y,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
        )
        mode_suffix = "1h" if base_interval == "60" else "15m"
        model_suffix = args.model_suffix or ""
        rf_filename = f"rf_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl"
        trainer.save_model(rf_model, trainer.scaler, feature_names, rf_metrics, rf_filename)
        safe_print(f"      ✅ Сохранено как: {rf_filename}")

        try:
            safe_print("\n   ⚡ Обучение XGBoost...")
            xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
                X,
                y,
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
            )
            xgb_filename = f"xgb_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl"
            trainer.save_model(xgb_model, trainer.scaler, feature_names, xgb_metrics, xgb_filename)
            safe_print(f"      ✅ Сохранено как: {xgb_filename}")
        except Exception as e:
            xgb_metrics = None
            safe_print(f"      ⚠️  XGBoost недоступен: {e}")

        if xgb_metrics is not None:
            safe_print("\n   🎯 Обучение Ensemble (RF + XGBoost)...")
            ensemble_model, ensemble_metrics = trainer.train_ensemble(
                X,
                y,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
                xgb_n_estimators=xgb_n_estimators,
                xgb_max_depth=xgb_max_depth,
                xgb_learning_rate=xgb_learning_rate,
            )
            ensemble_filename = f"ensemble_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl"
            trainer.save_model(ensemble_model, trainer.scaler, feature_names, ensemble_metrics, ensemble_filename)
            safe_print(f"      ✅ Сохранено как: {ensemble_filename}")
        else:
            ensemble_metrics = None

        if xgb_metrics is not None and LIGHTGBM_AVAILABLE:
            safe_print("\n   🎯 Обучение TripleEnsemble (RF + XGBoost + LightGBM)...")
            lgb_n_estimators = _clamp_int(hyperparams.get("lgb_n_estimators"), 80, 600, 100)
            lgb_max_depth = _clamp_int(hyperparams.get("lgb_max_depth"), 3, 12, 6)
            lgb_learning_rate = _clamp_float(hyperparams.get("lgb_learning_rate"), 0.02, 0.3, 0.1)
            triple_model, triple_ensemble_metrics = trainer.train_ensemble(
                X,
                y,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
                xgb_n_estimators=xgb_n_estimators,
                xgb_max_depth=xgb_max_depth,
                xgb_learning_rate=xgb_learning_rate,
                lgb_n_estimators=lgb_n_estimators,
                lgb_max_depth=lgb_max_depth,
                lgb_learning_rate=lgb_learning_rate,
                ensemble_method="triple",
                include_lightgbm=True,
            )
            triple_filename = f"triple_ensemble_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl"
            trainer.save_model(
                triple_model,
                trainer.scaler,
                feature_names,
                triple_ensemble_metrics,
                triple_filename,
                symbol=symbol,
                interval=base_interval,
                model_type="triple_ensemble",
            )
            safe_print(f"      ✅ Сохранено как: {triple_filename}")
        else:
            triple_ensemble_metrics = None

        if xgb_metrics is not None and LIGHTGBM_AVAILABLE and LSTM_AVAILABLE:
            gc.collect() # Чистим перед самой тяжелой моделью
            safe_print("\n   🚀 Обучение QuadEnsemble (RF + XGBoost + LightGBM + LSTM)...")
            lgb_n_estimators = _clamp_int(hyperparams.get("lgb_n_estimators"), 80, 600, 100)
            lgb_max_depth = _clamp_int(hyperparams.get("lgb_max_depth"), 3, 12, 6)
            lgb_learning_rate = _clamp_float(hyperparams.get("lgb_learning_rate"), 0.02, 0.3, 0.1)
            lstm_sequence_length = _clamp_int(hyperparams.get("lstm_sequence_length"), 30, 120, 60)
            lstm_epochs = _clamp_int(hyperparams.get("lstm_epochs"), 5, 80, 20 if args.safe_mode else 40)
            quad_model, quad_metrics = trainer.train_quad_ensemble(
                X,
                y,
                df=df_target,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
                xgb_n_estimators=xgb_n_estimators,
                xgb_max_depth=xgb_max_depth,
                xgb_learning_rate=xgb_learning_rate,
                lgb_n_estimators=lgb_n_estimators,
                lgb_max_depth=lgb_max_depth,
                lgb_learning_rate=lgb_learning_rate,
                lstm_sequence_length=lstm_sequence_length,
                lstm_epochs=lstm_epochs,
            )

            # --- Добавляем оптимизацию порогов и мета-лейблинг ---
            safe_print("\n   🔍 Оптимизация порогов уверенности...")
            optimal_threshold = trainer.optimize_thresholds(quad_model, X, y, df_history=df_target)
            quad_metrics["optimal_threshold"] = optimal_threshold

            if use_meta_labeling:
                safe_print("\n   🔍 Обучение Meta-Labeling Filter...")
                y_pred_primary_raw = quad_model.predict_proba(X, df_history=df_target)
                y_pred_primary = np.argmax(y_pred_primary_raw, axis=1) - 1 # (-1, 0, 1)

                meta_labels = engineer.compute_meta_labels(df_target, y_pred_primary)
                y_meta = meta_labels.values

                if np.sum(y_meta == 1) > 10:
                    meta_model, meta_metrics = trainer.train_meta_classifier(
                        X,
                        y_meta,
                        n_estimators=100,
                        max_depth=4,
                    )
                    quad_metrics["meta_metrics"] = meta_metrics
                    quad_metrics["meta_model"] = meta_model
                    safe_print(f"      ✅ Meta-Filter обучен. Accuracy: {meta_metrics['accuracy']:.3f}")
                else:
                    safe_print("      ⚠️ Недостаточно положительных примеров для Meta-Filter")

            quad_filename = f"quad_ensemble_{symbol}_{base_interval}_{mode_suffix}{model_suffix}.pkl"
            trainer.save_model(
                quad_model,
                trainer.scaler,
                feature_names,
                quad_metrics,
                quad_filename,
                symbol=symbol,
                interval=base_interval,
                model_type="quad_ensemble",
                meta_model=quad_metrics.get("meta_model"),
                optimal_threshold=quad_metrics.get("optimal_threshold")
            )
            safe_print(f"      ✅ Сохранено как: {quad_filename} (с мета-данными)")

        else:
            quad_metrics = None

    report = {
        "symbols": symbols,
        "interval_display": interval_display,
        "ml_mtf_enabled": ml_mtf_enabled,
        "hyperparams": hyperparams,
        "rf_metrics": rf_metrics,
    }
    if xgb_metrics is not None:
        report["xgb_metrics"] = xgb_metrics
    if ensemble_metrics is not None:
        report["ensemble_metrics"] = ensemble_metrics
    if triple_ensemble_metrics is not None:
        report["triple_ensemble_metrics"] = triple_ensemble_metrics
    if quad_metrics is not None:
        report["quad_metrics"] = quad_metrics
    if args.report_json:
        try:
            p = Path(args.report_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            safe_print(f"[WARNING] Не удалось сохранить отчёт эксперимента: {e}")

    safe_print("\n" + "=" * 80)
    safe_print("🎉 ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
    safe_print("=" * 80)


if __name__ == "__main__":
    main()
