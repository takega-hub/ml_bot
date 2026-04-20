"""
Скрипт для анализа чувствительности порогов вероятности (predict_proba).
Помогает найти оптимальный порог для входа в сделку, балансируя между точностью и количеством сигналов.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# Настройка логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"threshold_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.model_trainer import ModelTrainer
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.config import load_settings, ApiSettings

def analyze_sensitivity(symbol="BNBUSDT", interval="15", model_path=None):
    try:
        logger.info(f"🚀 Запуск анализа чувствительности для {symbol} ({interval}m)")
        load_settings()
        api_settings = ApiSettings()
        collector = DataCollector(api_settings)
        engineer = FeatureEngineer()
        trainer = ModelTrainer()

        if not model_path:
            # Пытаемся найти последнюю модель
            model_dir = Path("ml_models")
            models = list(model_dir.glob(f"quad_ensemble_{symbol}_{interval}m_*.pkl"))
            if not models:
                logger.error(f"❌ Модели для {symbol} {interval}m не найдены в {model_dir.absolute()}")
                return
            model_path = str(sorted(models, key=os.path.getmtime)[-1])

        logger.info(f"📂 Загрузка модели: {model_path}")
        model_data = trainer.load_model(model_path)
        if not model_data:
            logger.error("❌ Не удалось загрузить модель (пустой результат или ошибка загрузки)")
            return

        model = model_data["model"]
        feature_names = model_data["feature_names"]
        logger.info(f"✅ Модель загружена. Фичей: {len(feature_names)}")

        # Собираем свежие данные для валидации (последние 30 дней)
        logger.info(f"📥 Сбор данных для валидации {symbol}...")
        df = collector.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            use_cache=True
        )

        if df is None or df.empty:
            logger.error("❌ Данные не собраны (df is None or empty)")
            return

        logger.info(f"📊 Собрано свечей: {len(df)}")

        # Создаем фичи и таргет
        logger.info("🛠 Генерация технических индикаторов...")
        df_feat = engineer.create_technical_indicators(df)

        logger.info("🎯 Разметка данных (Triple Barrier)...")
        # Используем Triple Barrier для разметки валидационного сета
        df_target = engineer.create_triple_barrier_labels(df_feat)

        if df_target is None or df_target.empty:
            logger.error("❌ Ошибка при разметке данных (target labels missing)")
            return

        X = df_target[feature_names].values
        y_true = df_target["target"].values

        # Получаем вероятности
        logger.info("🔮 Получение предсказаний (predict_proba)...")
        if hasattr(model, "predict_proba"):
            try:
                # Пытаемся вызвать с df_history если это QuadEnsemble
                probas = model.predict_proba(X, df_history=df_target)
            except TypeError:
                logger.warning("⚠️ predict_proba не принял df_history, пробуем обычный вызов")
                probas = model.predict_proba(X)
        else:
            logger.error("❌ Модель не поддерживает predict_proba")
            return

        # Проверка наличия Meta-Filter
        meta_model = model_data.get("meta_model")
        meta_probas = None
        if meta_model:
            logger.info("🛡 Обнаружен Meta-Filter. Анализируем совместную работу...")
            if hasattr(meta_model, "predict_proba"):
                # Мета-модель обучается предсказывать успех (1) или провал (0) основного сигнала
                meta_probas = meta_model.predict_proba(X)[:, 1]

        logger.info(f"🧪 Анализ {len(np.linspace(0.33, 0.8, 47))} вариантов порогов...")
        thresholds = np.linspace(0.33, 0.8, 47)
        results = []

        for t in thresholds:
            # Базовые маски сигналов
            long_mask = probas[:, 2] > t
            short_mask = probas[:, 0] > t
            base_signals = long_mask | short_mask
            total_base = np.sum(base_signals)

            if total_base == 0:
                results.append({
                    "threshold": t,
                    "precision": 0.0,
                    "precision_meta": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "signals": 0,
                    "signals_meta": 0
                })
                continue

            # Точность базовой модели
            correct_base = np.sum((long_mask & (y_true == 1)) | (short_mask & (y_true == -1)))
            precision_base = correct_base / total_base

            # Точность с учетом Meta-Filter (порог 0.5)
            precision_meta = precision_base
            total_meta = total_base
            if meta_probas is not None:
                meta_mask = meta_probas > 0.5
                filtered_signals = base_signals & meta_mask
                total_meta = np.sum(filtered_signals)
                if total_meta > 0:
                    correct_meta = np.sum(filtered_signals & ((long_mask & (y_true == 1)) | (short_mask & (y_true == -1))))
                    precision_meta = correct_meta / total_meta
                else:
                    precision_meta = 0.0

            # Recall и F1 (на основе базовой модели)
            total_positives = np.sum(y_true != 0)
            recall = correct_base / total_positives if total_positives > 0 else 0
            f1 = 2 * (precision_base * recall) / (precision_base + recall) if (precision_base + recall) > 0 else 0

            results.append({
                "threshold": t,
                "precision": precision_base,
                "precision_meta": precision_meta,
                "recall": recall,
                "f1": f1,
                "signals": int(total_base),
                "signals_meta": int(total_meta)
            })

        res_df = pd.DataFrame(results)

        # Вывод результатов
        logger.info("\n" + "="*50)
        logger.info(f"📊 АНАЛИЗ ПОРОГОВ ДЛЯ {symbol}")
        logger.info("="*50)
        top_results = res_df[res_df["signals"] > 5].sort_values("precision_meta", ascending=False).head(10)
        logger.info(f"\n{top_results.to_string()}")

        # Сохраняем отчет
        report_path = f"artifacts/threshold_analysis_{symbol}_{interval}m.csv"
        os.makedirs("artifacts", exist_ok=True)
        res_df.to_csv(report_path, index=False)
        logger.info(f"✅ Отчет сохранен: {report_path}")

        # Визуализация
        try:
            plt.figure(figsize=(12, 7))
            plt.plot(res_df["threshold"], res_df["precision"], 'b--', label="Base Precision")
            plt.plot(res_df["threshold"], res_df["precision_meta"], 'g-', linewidth=2, label="Meta-Filter Precision")
            plt.plot(res_df["threshold"], res_df["f1"], 'r:', label="F1 Score")

            plt.twinx()
            plt.bar(res_df["threshold"], res_df["signals"], alpha=0.1, color='blue', label="Base Signals")
            plt.bar(res_df["threshold"], res_df["signals_meta"], alpha=0.2, color='green', label="Meta Signals")

            plt.title(f"Threshold & Meta-Filter Analysis - {symbol}")
            plt.xlabel("Confidence Threshold")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"artifacts/threshold_plot_{symbol}.png")
            logger.info(f"📈 График сохранен: artifacts/threshold_plot_{symbol}.png")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось построить график: {e}")

    except Exception as e:
        logger.error(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BNBUSDT")
    parser.add_argument("--interval", type=str, default="15")
    parser.add_argument("--model", type=str, help="Путь к файлу модели")
    args = parser.parse_args()

    analyze_sensitivity(args.symbol, args.interval, args.model)
