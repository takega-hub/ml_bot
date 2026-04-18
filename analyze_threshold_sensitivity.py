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
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.model_trainer import ModelTrainer
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.config import load_settings, ApiSettings

def analyze_sensitivity(symbol="BNBUSDT", interval="15", model_path=None):
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
            print(f"❌ Модели для {symbol} {interval}m не найдены.")
            return
        model_path = str(sorted(models, key=os.path.getmtime)[-1])

    print(f"📂 Загрузка модели: {model_path}")
    model_data = trainer.load_model(model_path)
    if not model_data:
        print("❌ Не удалось загрузить модель.")
        return

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Собираем свежие данные для валидации (последние 30 дней)
    print(f"📥 Сбор данных для валидации {symbol}...")
    df = collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        use_cache=True
    )

    if df is None or df.empty:
        print("❌ Данные не собраны.")
        return

    # Создаем фичи и таргет
    df_feat = engineer.create_technical_indicators(df)
    # Используем Triple Barrier для разметки валидационного сета
    df_target = engineer.create_triple_barrier_labels(df_feat)

    X = df_target[feature_names].values
    y_true = df_target["target"].values

    # Получаем вероятности
    print("🔮 Получение предсказаний...")
    # Для QuadEnsemble нужен df_history
    if hasattr(model, "predict_proba"):
        try:
            # Пытаемся вызвать с df_history если это QuadEnsemble
            probas = model.predict_proba(X, df_history=df_target)
        except TypeError:
            probas = model.predict_proba(X)
    else:
        print("❌ Модель не поддерживает predict_proba")
        return

    thresholds = np.linspace(0.33, 0.8, 47) # От 0.33 (случайно) до 0.8
    results = []

    for t in thresholds:
        # LONG signals: proba[2] > t
        long_signals = probas[:, 2] > t
        # SHORT signals: proba[0] > t
        short_signals = probas[:, 0] > t

        total_signals = np.sum(long_signals) + np.sum(short_signals)

        if total_signals == 0:
            results.append({
                "threshold": t,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "signals": 0
            })
            continue

        # Accuracy of signals
        correct_long = np.sum((long_signals) & (y_true == 1))
        correct_short = np.sum((short_signals) & (y_true == -1))

        precision = (correct_long + correct_short) / total_signals

        # Recall (относительно всех возможных прибыльных движений в таргете)
        total_positives = np.sum(y_true != 0)
        recall = (correct_long + correct_short) / total_positives if total_positives > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "signals": int(total_signals)
        })

    res_df = pd.DataFrame(results)

    # Вывод результатов
    print("\n" + "="*50)
    print(f"📊 АНАЛИЗ ПОРОГОВ ДЛЯ {symbol}")
    print("="*50)
    print(res_df[res_df["signals"] > 5].sort_values("precision", ascending=False).head(10))

    # Сохраняем отчет
    report_path = f"artifacts/threshold_analysis_{symbol}_{interval}m.csv"
    os.makedirs("artifacts", exist_ok=True)
    res_df.to_csv(report_path, index=False)
    print(f"\n✅ Отчет сохранен: {report_path}")

    # Визуализация (опционально, если есть куда выводить)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(res_df["threshold"], res_df["precision"], label="Precision (Accuracy of Signals)")
        plt.plot(res_df["threshold"], res_df["f1"], label="F1 Score")
        plt.twinx()
        plt.bar(res_df["threshold"], res_df["signals"], alpha=0.2, color='gray', label="Signal Count")
        plt.title(f"Threshold Sensitivity Analysis - {symbol}")
        plt.xlabel("Confidence Threshold")
        plt.legend()
        plt.savefig(f"artifacts/threshold_plot_{symbol}.png")
        print(f"📈 График сохранен: artifacts/threshold_plot_{symbol}.png")
    except Exception as e:
        print(f"⚠️ Не удалось построить график: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BNBUSDT")
    parser.add_argument("--interval", type=str, default="15")
    parser.add_argument("--model", type=str, help="Путь к файлу модели")
    args = parser.parse_args()

    analyze_sensitivity(args.symbol, args.interval, args.model)
