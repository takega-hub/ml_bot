"""
Анализ качества target labeling для ML моделей.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer

def analyze_target_distribution(df_with_target: pd.DataFrame) -> dict:
    """Анализирует распределение классов в target."""
    if 'target' not in df_with_target.columns:
        return {}
    
    target_dist = df_with_target['target'].value_counts()
    total = len(df_with_target)
    
    result = {
        'total_samples': int(total),
        'long_count': int(target_dist.get(1, 0)),
        'short_count': int(target_dist.get(-1, 0)),
        'hold_count': int(target_dist.get(0, 0)),
        'long_pct': float(target_dist.get(1, 0) / total * 100) if total > 0 else 0.0,
        'short_pct': float(target_dist.get(-1, 0) / total * 100) if total > 0 else 0.0,
        'hold_pct': float(target_dist.get(0, 0) / total * 100) if total > 0 else 0.0,
        'imbalance_ratio': float(target_dist.get(-1, 0) / target_dist.get(1, 0)) if target_dist.get(1, 0) > 0 else 0.0,
    }
    
    return result

def analyze_target_achievability(df_with_target: pd.DataFrame, forward_periods: int = 5) -> dict:
    """Проверяет, достижимы ли TP для меток LONG/SHORT."""
    if 'target' not in df_with_target.columns or 'close' not in df_with_target.columns:
        return {}
    
    long_labels = df_with_target[df_with_target['target'] == 1].copy()
    short_labels = df_with_target[df_with_target['target'] == -1].copy()
    
    result = {
        'long_analyzed': 0,
        'long_achievable': 0,
        'long_achievable_pct': 0.0,
        'short_analyzed': 0,
        'short_achievable': 0,
        'short_achievable_pct': 0.0,
    }
    
    # Анализ LONG меток
    if len(long_labels) > 0:
        for idx, row in long_labels.iterrows():
            current_price = row['close']
            result['long_analyzed'] += 1
            
            # Проверяем, достиг ли цена TP в будущем
            future_idx = df_with_target.index.get_loc(idx) + forward_periods
            if future_idx < len(df_with_target):
                future_prices = df_with_target.iloc[future_idx:future_idx+10]['close']  # Проверяем 10 свечей вперед
                if len(future_prices) > 0:
                    max_future_price = future_prices.max()
                    # TP считается достигнутым, если цена выросла хотя бы на 0.5%
                    if max_future_price >= current_price * 1.005:
                        result['long_achievable'] += 1
        
        if result['long_analyzed'] > 0:
            result['long_achievable_pct'] = result['long_achievable'] / result['long_analyzed'] * 100
    
    # Анализ SHORT меток
    if len(short_labels) > 0:
        for idx, row in short_labels.iterrows():
            current_price = row['close']
            result['short_analyzed'] += 1
            
            # Проверяем, достиг ли цена TP в будущем
            future_idx = df_with_target.index.get_loc(idx) + forward_periods
            if future_idx < len(df_with_target):
                future_prices = df_with_target.iloc[future_idx:future_idx+10]['close']
                if len(future_prices) > 0:
                    min_future_price = future_prices.min()
                    # TP считается достигнутым, если цена упала хотя бы на 0.5%
                    if min_future_price <= current_price * 0.995:
                        result['short_achievable'] += 1
        
        if result['short_analyzed'] > 0:
            result['short_achievable_pct'] = result['short_achievable'] / result['short_analyzed'] * 100
    
    return result

def main():
    print("=" * 80)
    print("📊 АНАЛИЗ КАЧЕСТВА TARGET LABELING")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Символы для анализа
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"📈 Анализ {symbol}")
        print(f"{'='*80}\n")
        
        try:
            # Собираем данные
            print(f"📥 Сбор данных для {symbol}...")
            collector = DataCollector(settings.api)
            df_raw = collector.collect_klines(
                symbol=symbol,
                interval="15",
                start_date=None,
                end_date=None,
                limit=3000,
                save_to_file=False,
            )
            
            if df_raw.empty:
                print(f"⚠️  Нет данных для {symbol}, пропускаем")
                continue
            
            print(f"✅ Собрано {len(df_raw)} свечей")
            
            # Создаем фичи
            print(f"🔧 Создание фичей...")
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.create_technical_indicators(df_raw)
            
            # Создаем target с текущими параметрами
            print(f"🎯 Создание target variable...")
            df_with_target = feature_engineer.create_target_variable(
                df_features,
                forward_periods=5,
                threshold_pct=0.5,
                use_atr_threshold=True,
                use_risk_adjusted=True,
                min_risk_reward_ratio=1.5,
                max_hold_periods=96,
                min_profit_pct=0.5,
            )
            
            if df_with_target.empty or 'target' not in df_with_target.columns:
                print(f"⚠️  Не удалось создать target для {symbol}")
                continue
            
            # Анализ распределения
            print(f"\n📊 Распределение классов:")
            dist = analyze_target_distribution(df_with_target)
            print(f"   Всего образцов: {dist['total_samples']}")
            print(f"   LONG:  {dist['long_count']:5d} ({dist['long_pct']:5.1f}%)")
            print(f"   SHORT: {dist['short_count']:5d} ({dist['short_pct']:5.1f}%)")
            print(f"   HOLD:  {dist['hold_count']:5d} ({dist['hold_pct']:5.1f}%)")
            
            if dist['long_count'] > 0 and dist['short_count'] > 0:
                imbalance = dist['imbalance_ratio']
                print(f"   Дисбаланс LONG/SHORT: {imbalance:.2f}:1")
                if imbalance > 2.0 or imbalance < 0.5:
                    print(f"   ⚠️  КРИТИЧЕСКИЙ дисбаланс! Нужна балансировка.")
            
            # Анализ достижимости
            print(f"\n🎯 Анализ достижимости TP:")
            achievability = analyze_target_achievability(df_with_target, forward_periods=5)
            print(f"   LONG меток проанализировано: {achievability['long_analyzed']}")
            print(f"   LONG меток достижимы: {achievability['long_achievable']} ({achievability['long_achievable_pct']:.1f}%)")
            print(f"   SHORT меток проанализировано: {achievability['short_analyzed']}")
            print(f"   SHORT меток достижимы: {achievability['short_achievable']} ({achievability['short_achievable_pct']:.1f}%)")
            
            if achievability['long_achievable_pct'] < 60 or achievability['short_achievable_pct'] < 60:
                print(f"   ⚠️  НИЗКАЯ достижимость! Многие метки могут быть нереалистичными.")
            
            # Сохраняем результаты
            result = {
                'symbol': symbol,
                'distribution': dist,
                'achievability': achievability,
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Ошибка при анализе {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Итоговый отчет
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    if all_results:
        print(f"\n{'Символ':<10} | {'LONG %':<8} | {'SHORT %':<9} | {'Дисбаланс':<10} | {'LONG TP %':<10} | {'SHORT TP %':<11}")
        print("-" * 80)
        
        for result in all_results:
            dist = result['distribution']
            ach = result['achievability']
            imbalance_str = f"{dist['imbalance_ratio']:.2f}:1" if dist['imbalance_ratio'] > 0 else "N/A"
            print(f"{result['symbol']:<10} | {dist['long_pct']:>6.1f}% | {dist['short_pct']:>7.1f}% | {imbalance_str:<10} | {ach['long_achievable_pct']:>8.1f}% | {ach['short_achievable_pct']:>9.1f}%")
        
        # Сохраняем в JSON
        output_dir = Path("backtest_reports")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"target_labeling_analysis_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены в: {json_file}")
        
        # Рекомендации
        print("\n" + "=" * 80)
        print("💡 РЕКОМЕНДАЦИИ")
        print("=" * 80)
        
        avg_imbalance = np.mean([r['distribution']['imbalance_ratio'] for r in all_results if r['distribution']['imbalance_ratio'] > 0])
        avg_long_ach = np.mean([r['achievability']['long_achievable_pct'] for r in all_results if r['achievability']['long_analyzed'] > 0])
        avg_short_ach = np.mean([r['achievability']['short_achievable_pct'] for r in all_results if r['achievability']['short_analyzed'] > 0])
        
        if avg_imbalance > 1.5 or avg_imbalance < 0.67:
            print(f"\n1. ⚠️  Дисбаланс LONG/SHORT: {avg_imbalance:.2f}:1")
            print(f"   Рекомендация: Увеличить вес minority class в class_weight")
        
        if avg_long_ach < 60 or avg_short_ach < 60:
            print(f"\n2. ⚠️  Низкая достижимость TP:")
            print(f"   LONG: {avg_long_ach:.1f}%, SHORT: {avg_short_ach:.1f}%")
            print(f"   Рекомендация: Увеличить forward_periods или уменьшить threshold_pct")
        
        if avg_long_ach > 80 and avg_short_ach > 80:
            print(f"\n✅ Достижимость TP хорошая: LONG {avg_long_ach:.1f}%, SHORT {avg_short_ach:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)

if __name__ == "__main__":
    main()
