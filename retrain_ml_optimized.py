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
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform == 'win32':
    pass

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from bot.config import ApiSettings, load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE, LSTM_AVAILABLE

# Настройка логирования для обучения
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
training_log = log_dir / "training.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(training_log, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Trainer")

def safe_print(*args, **kwargs):
    """Оставляем для совместимости, но перенаправляем в logger"""
    text = ' '.join(str(arg) for arg in args)
    logger.info(text)

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
    parser.add_argument("--mtf", action="store_true", help="Использовать MTF фичей (1h, 4h)")
    parser.add_argument("--no-mtf", action="store_true", help="НЕ использовать MTF фичи (только 15m)")
    parser.add_argument("--interval", type=str, default="15m", choices=["15m", "60m", "1h"])
    parser.add_argument("--model-suffix", type=str, default="")
    parser.add_argument("--report-json", type=str)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--hyperparams-json", type=str)
    parser.add_argument("--use-triple-barrier", action="store_true", help="Использовать Triple Barrier Method для разметки")
    parser.add_argument("--use-meta-labeling", action="store_true", help="Использовать Meta-Labeling для фильтрации сигналов")
    parser.add_argument("--prune-features", action="store_true", help="Удалять малозначимые признаки")
    parser.add_argument("--cpu-only", action="store_true", help="Принудительно использовать CPU для LSTM")
    args = parser.parse_known_args()[0]
    hyperparams = _load_hyperparams(args.hyperparams_json)

    logger.info("=" * 80)
    logger.info("🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ")
    logger.info("=" * 80)

    load_settings()
    api_settings = ApiSettings()

    if args.symbol:
        symbols = [args.symbol]
    else:
        # Пытаемся взять список из настроек или дефолт
        symbols = ["BNBUSDT", "ADAUSDT"]

    interval_arg = (args.interval or "15m").strip().lower()
    if interval_arg in ("1h", "60m"):
        base_interval = "60"
        interval_display = "1h"
    else:
        base_interval = "15"
        interval_display = "15m"

    logger.info(f"📌 Базовый таймфрейм: {interval_display}")

    ml_mtf_enabled = False
    if args.no_mtf:
        ml_mtf_enabled = False
        logger.info("📌 Режим: БЕЗ MTF фичей (только 15m)")
    elif args.mtf:
        ml_mtf_enabled = True
        logger.info("📌 Режим: С MTF фичами (15m + 1h + 4h)")
    else:
        ml_mtf_enabled = os.getenv("ML_MTF_ENABLED", "0") in ("1", "true", "True")
        logger.info(f"📌 Режим MTF: {ml_mtf_enabled} (из окружения)")

    trainer = ModelTrainer()
    rf_metrics = None
    xgb_metrics = None
    ensemble_metrics = None
    triple_ensemble_metrics = None
    quad_metrics = None

    for symbol in symbols:
        try:
            gc.collect()
            logger.info("\n" + "=" * 80)
            logger.info(f"📊 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ {symbol}")
            logger.info("=" * 80)

            collector = DataCollector(api_settings)
            if base_interval == "60":
                days = _clamp_int(hyperparams.get("training_days_1h"), 60, 365, 180)
                logger.info(f"[1/5] 📥 Сбор исторических данных ({interval_display}) для {symbol}...")
                df = _collect_by_days(collector, symbol, interval="60", days_back=days)
            else:
                days = _clamp_int(hyperparams.get("training_days_15m"), 20, 180, 30)
                logger.info(f"[1/5] 📥 Сбор исторических данных ({interval_display}) для {symbol}...")
                df = _collect_by_days(collector, symbol, interval="15", days_back=days)

            if df is None or df.empty:
                logger.error(f"❌ Не удалось собрать данные для {symbol}")
                continue

            logger.info(f"✅ Собрано {len(df)} свечей")

            logger.info(f"[2/5] 🔧 Создание признаков для {symbol}...")
            engineer = FeatureEngineer()
            df_feat = engineer.create_technical_indicators(df)
            logger.info(f"✅ Создано {len(df_feat.columns)} признаков")

            logger.info(f"[3/5] 🎯 Создание целевой переменной для {symbol}...")
            use_triple_barrier = args.use_triple_barrier or os.getenv("USE_TRIPLE_BARRIER", "0") in ("1", "true", "True")
            use_meta_labeling = args.use_meta_labeling or hyperparams.get("use_meta_labeling", False)

            if use_triple_barrier:
                logger.info("📌 Используется Triple Barrier Method (TBM)")
                df_target = engineer.create_triple_barrier_labels(df_feat)
            else:
                df_target = engineer.create_target_variable(df_feat)

            if df_target is None or df_target.empty:
                logger.error(f"❌ Не удалось создать таргет для {symbol}")
                continue

            logger.info(f"[4/5] 📦 Подготовка данных...")
            X, y, feature_names = _prepare_training_matrix(trainer, df_target)

            if args.prune_features:
                feature_names = trainer.prune_features(X, y, feature_names)
                X = df_target[feature_names].values

            logger.info(f"✅ Подготовлено: {X.shape[0]} образцов, {X.shape[1]} признаков")

            logger.info(f"[5/5] 🤖 Обучение моделей...")

            # RF
            rf_model, rf_metrics = trainer.train_random_forest_classifier(X, y)
            mode_suffix = "1h" if base_interval == "60" else "15m"
            rf_filename = f"rf_{symbol}_{base_interval}_{mode_suffix}{args.model_suffix}.pkl"
            trainer.save_model(rf_model, trainer.scaler, feature_names, rf_metrics, rf_filename)

            # XGB
            try:
                xgb_model, xgb_metrics = trainer.train_xgboost_classifier(X, y)
                xgb_filename = f"xgb_{symbol}_{base_interval}_{mode_suffix}{args.model_suffix}.pkl"
                trainer.save_model(xgb_model, trainer.scaler, feature_names, xgb_metrics, xgb_filename)
            except Exception as e:
                logger.warning(f"⚠️ XGBoost ошибка: {e}")
                xgb_metrics = None

            # QuadEnsemble (если доступны ресурсы)
            if xgb_metrics and LIGHTGBM_AVAILABLE and LSTM_AVAILABLE:
                logger.info("🚀 Обучение QuadEnsemble...")
                quad_model, quad_metrics = trainer.train_quad_ensemble(
                    X, y, df=df_target,
                    force_cpu_lstm=args.cpu_only or hyperparams.get("force_cpu_lstm", True)
                )

                # Оптимизация порогов
                optimal_threshold = trainer.optimize_thresholds(quad_model, X, y, df_history=df_target)
                quad_metrics["optimal_threshold"] = optimal_threshold

                if use_meta_labeling:
                    logger.info("🛡 Обучение Meta-Filter...")
                    y_pred_raw = quad_model.predict_proba(X, df_history=df_target)
                    y_pred = np.argmax(y_pred_raw, axis=1) - 1
                    meta_labels = engineer.compute_meta_labels(df_target, y_pred)
                    if np.sum(meta_labels == 1) > 10:
                        meta_model, meta_metrics = trainer.train_meta_classifier(X, meta_labels.values)
                        quad_metrics["meta_model"] = meta_model
                        logger.info(f"✅ Meta-Filter готов. Acc: {meta_metrics['accuracy']:.3f}")

                quad_filename = f"quad_ensemble_{symbol}_{base_interval}_{mode_suffix}{args.model_suffix}.pkl"
                trainer.save_model(
                    quad_model, trainer.scaler, feature_names, quad_metrics, quad_filename,
                    symbol=symbol, interval=base_interval, model_type="quad_ensemble",
                    meta_model=quad_metrics.get("meta_model"),
                    optimal_threshold=quad_metrics.get("optimal_threshold")
                )

        except Exception as e:
            import traceback
            logger.error(f"💥 Ошибка на {symbol}: {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info("\n" + "=" * 80)
    logger.info("🎉 ВСЕ ЗАДАЧИ ЗАВЕРШЕНЫ!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
