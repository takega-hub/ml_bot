"""
Анализ feature importance и корреляций для ML моделей.
"""
import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

def load_model_metadata(model_path: str) -> Dict:
    """Загружает метаданные модели."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        print(f"❌ Ошибка загрузки {model_path}: {e}")
        return None

def extract_feature_importance(model_data: Dict, model_name: str) -> pd.DataFrame:
    """Извлекает feature importance из модели."""
    model = model_data.get('model')
    feature_names = model_data.get('feature_names', [])
    
    if not feature_names:
        print(f"⚠️  Нет feature_names в {model_name}")
        return pd.DataFrame()
    
    importance_dict = {}
    
    # Для разных типов моделей
    if hasattr(model, 'feature_importances_'):
        # Random Forest, XGBoost, LightGBM
        importances = model.feature_importances_
        for i, feature in enumerate(feature_names):
            if i < len(importances):
                importance_dict[feature] = importances[i]
    
    elif hasattr(model, 'rf_model') and hasattr(model, 'xgb_model'):
        # Ensemble: берем среднее importance от всех моделей
        rf_imp = model.rf_model.feature_importances_ if hasattr(model.rf_model, 'feature_importances_') else None
        xgb_imp = model.xgb_model.feature_importances_ if hasattr(model.xgb_model, 'feature_importances_') else None
        
        if rf_imp is not None and xgb_imp is not None:
            avg_imp = (rf_imp + xgb_imp) / 2
            for i, feature in enumerate(feature_names):
                if i < len(avg_imp):
                    importance_dict[feature] = avg_imp[i]
        elif rf_imp is not None:
            for i, feature in enumerate(feature_names):
                if i < len(rf_imp):
                    importance_dict[feature] = rf_imp[i]
        elif xgb_imp is not None:
            for i, feature in enumerate(feature_names):
                if i < len(xgb_imp):
                    importance_dict[feature] = xgb_imp[i]
    
    elif hasattr(model, 'rf_model') and hasattr(model, 'xgb_model') and hasattr(model, 'lgb_model'):
        # TripleEnsemble: берем среднее от всех трех
        rf_imp = model.rf_model.feature_importances_ if hasattr(model.rf_model, 'feature_importances_') else None
        xgb_imp = model.xgb_model.feature_importances_ if hasattr(model.xgb_model, 'feature_importances_') else None
        lgb_imp = model.lgb_model.feature_importances_ if hasattr(model.lgb_model, 'feature_importances_') else None
        
        imps = [imp for imp in [rf_imp, xgb_imp, lgb_imp] if imp is not None]
        if imps:
            avg_imp = np.mean(imps, axis=0)
            for i, feature in enumerate(feature_names):
                if i < len(avg_imp):
                    importance_dict[feature] = avg_imp[i]
    
    elif hasattr(model, 'metrics') and 'feature_importance' in model.metrics:
        # Из метрик
        importance_dict = model.metrics['feature_importance']
    
    if not importance_dict:
        print(f"⚠️  Не удалось извлечь importance из {model_name}")
        return pd.DataFrame()
    
    # Создаем DataFrame
    df = pd.DataFrame([
        {'feature': feat, 'importance': imp}
        for feat, imp in importance_dict.items()
    ])
    df = df.sort_values('importance', ascending=False)
    df['model'] = model_name
    
    return df

def analyze_correlations(feature_names: List[str], data_path: str = None) -> pd.DataFrame:
    """Анализирует корреляции между фичами."""
    # Если есть путь к данным, загружаем их
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Выбираем только фичи
            feature_cols = [col for col in feature_names if col in df.columns]
            if feature_cols:
                corr_matrix = df[feature_cols].corr()
                return corr_matrix
        except Exception as e:
            print(f"⚠️  Ошибка загрузки данных для корреляций: {e}")
    
    return pd.DataFrame()

def main():
    print("=" * 80)
    print("📊 АНАЛИЗ FEATURE IMPORTANCE И КОРРЕЛЯЦИЙ")
    print("=" * 80)
    
    # Ищем все модели в ml_models/
    models_dir = Path("ml_models")
    if not models_dir.exists():
        print(f"❌ Директория {models_dir} не найдена")
        return
    
    # Фильтруем только MTF модели BTCUSDT (лучшие результаты)
    model_files = list(models_dir.glob("*BTCUSDT*mtf*.pkl"))
    
    if not model_files:
        print(f"⚠️  Не найдено MTF моделей для BTCUSDT")
        # Пробуем любые модели
        model_files = list(models_dir.glob("*.pkl"))[:5]
    
    print(f"\n📦 Найдено {len(model_files)} моделей для анализа")
    
    all_importances = []
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n🔍 Анализ: {model_name}")
        
        model_data = load_model_metadata(str(model_file))
        if model_data is None:
            continue
        
        # Извлекаем feature importance
        importance_df = extract_feature_importance(model_data, model_name)
        if not importance_df.empty:
            all_importances.append(importance_df)
            print(f"   ✅ Извлечено {len(importance_df)} фичей")
            
            # Показываем топ-10
            top10 = importance_df.head(10)
            print(f"\n   📈 ТОП-10 важных фичей:")
            for idx, row in top10.iterrows():
                print(f"      {row['feature']:<30} {row['importance']:>8.4f}")
        else:
            print(f"   ⚠️  Не удалось извлечь importance")
    
    if not all_importances:
        print("\n❌ Не удалось извлечь importance ни из одной модели")
        return
    
    # Объединяем все importance
    combined_df = pd.concat(all_importances, ignore_index=True)
    
    # Группируем по фичам и считаем среднее importance
    feature_avg = combined_df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
    feature_avg = feature_avg.sort_values('mean', ascending=False)
    
    print("\n" + "=" * 80)
    print("📊 СРЕДНЯЯ IMPORTANCE ПО ВСЕМ МОДЕЛЯМ")
    print("=" * 80)
    print(f"\n{'Фича':<40} | {'Среднее':<10} | {'Std':<10} | {'Моделей':<8}")
    print("-" * 75)
    
    for idx, row in feature_avg.head(30).iterrows():
        print(f"{row['feature']:<40} | {row['mean']:>9.4f} | {row['std']:>9.4f} | {int(row['count']):>7}")
    
    # Анализ категорий фичей
    print("\n" + "=" * 80)
    print("📊 АНАЛИЗ ПО КАТЕГОРИЯМ ФИЧЕЙ")
    print("=" * 80)
    
    categories = {
        'MTF': [f for f in feature_avg['feature'] if any(x in f for x in ['_60', '_240', 'rsi_60', 'rsi_240'])],
        'Волатильность': [f for f in feature_avg['feature'] if any(x in f for x in ['volatility', 'atr', 'parkinson'])],
        'Паттерны': [f for f in feature_avg['feature'] if any(x in f for x in ['is_', 'doji', 'hammer', 'engulfing'])],
        'S/R': [f for f in feature_avg['feature'] if any(x in f for x in ['support', 'resistance', 'local_'])],
        'Тренд': [f for f in feature_avg['feature'] if any(x in f for x in ['ema', 'sma', 'adx', 'di_', 'trend'])],
        'RSI': [f for f in feature_avg['feature'] if 'rsi' in f],
        'Объем': [f for f in feature_avg['feature'] if 'volume' in f],
        'Микроструктура': [f for f in feature_avg['feature'] if any(x in f for x in ['spread', 'imbalance', 'momentum'])],
    }
    
    for category, features in categories.items():
        if features:
            cat_importance = feature_avg[feature_avg['feature'].isin(features)]['mean'].mean()
            print(f"\n{category:<20}: {len(features):>3} фичей, средняя importance: {cat_importance:.4f}")
            top_cat = feature_avg[feature_avg['feature'].isin(features)].head(5)
            for idx, row in top_cat.iterrows():
                print(f"   {row['feature']:<35} {row['mean']:>8.4f}")
    
    # Сохраняем результаты
    output_dir = Path("backtest_reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"feature_importance_analysis_{timestamp}.csv"
    
    feature_avg.to_csv(output_file, index=False)
    print(f"\n💾 Результаты сохранены в: {output_file}")
    
    # Сохраняем детальный отчет в JSON
    json_file = output_dir / f"feature_importance_analysis_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'models_analyzed': len(model_files),
        'total_features': len(feature_avg),
        'top_features': feature_avg.head(30).to_dict('records'),
        'categories': {
            cat: {
                'count': len(features),
                'avg_importance': float(feature_avg[feature_avg['feature'].isin(features)]['mean'].mean()) if features else 0.0,
                'top_features': feature_avg[feature_avg['feature'].isin(features)].head(5).to_dict('records') if features else []
            }
            for cat, features in categories.items()
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON отчет сохранен в: {json_file}")
    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)

if __name__ == "__main__":
    main()
