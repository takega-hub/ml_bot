"""
Скрипт для оптимизации target labeling параметров для 1h моделей.
Версия со строгими параметрами для получения 15-25% сигналов.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from bot.config import load_settings
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.data_collector import DataCollector

def test_target_labeling_params(
    symbol: str,
    base_interval: str = "60",
    forward_periods: int = 4,
    threshold_pct: float = 0.5,
    min_profit_pct: float = 0.5,
    min_risk_reward_ratio: float = 2.0,
    max_hold_periods: int = 48,
):
    """Тестирует параметры target labeling и возвращает статистику."""
    settings = load_settings()
    collector = DataCollector(settings.api)
    feature_engineer = FeatureEngineer()
    
    # Собираем данные
    print(f"📥 Сбор данных для {symbol} ({base_interval})...")
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=180)
    
    df_raw = collector.collect_klines(
        symbol=symbol,
        interval=base_interval,
        start_date=start_date,
        end_date=None,
        limit=180 * 24,
        save_to_file=False,
    )
    
    if df_raw.empty:
        return None
    
    # Создаем фичи
    df_features = feature_engineer.create_technical_indicators(df_raw)
    
    # Создаем target с тестируемыми параметрами
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=forward_periods,
        threshold_pct=threshold_pct,
        use_atr_threshold=True,
        use_risk_adjusted=True,
        min_risk_reward_ratio=min_risk_reward_ratio,
        max_hold_periods=max_hold_periods,
        min_profit_pct=min_profit_pct,
    )
    
    # Анализируем распределение
    target_dist = df_with_target['target'].value_counts()
    total = len(df_with_target)
    
    long_count = target_dist.get(1, 0)
    short_count = target_dist.get(-1, 0)
    hold_count = target_dist.get(0, 0)
    
    long_pct = (long_count / total) * 100 if total > 0 else 0
    short_pct = (short_count / total) * 100 if total > 0 else 0
    hold_pct = (hold_count / total) * 100 if total > 0 else 0
    signal_pct = ((long_count + short_count) / total) * 100 if total > 0 else 0
    
    return {
        'forward_periods': forward_periods,
        'threshold_pct': threshold_pct,
        'min_profit_pct': min_profit_pct,
        'min_risk_reward_ratio': min_risk_reward_ratio,
        'max_hold_periods': max_hold_periods,
        'total_samples': total,
        'long_count': long_count,
        'short_count': short_count,
        'hold_count': hold_count,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'hold_pct': hold_pct,
        'signal_pct': signal_pct,
        'balance_ratio': long_count / short_count if short_count > 0 else float('inf'),
    }

def main():
    print("=" * 80)
    print("🎯 ОПТИМИЗАЦИЯ TARGET LABELING ДЛЯ 1h МОДЕЛЕЙ (СТРОГИЕ ПАРАМЕТРЫ)")
    print("=" * 80)
    print("Цель: получить 15-25% сигналов (вместо текущих 56-62%)")
    print()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]
    
    # Варианты параметров для тестирования (более строгие)
    test_configs = [
        # Вариант 1: Строгие параметры для 15-20% сигналов
        {
            'name': 'Вариант 1: Строгие (цель 15-20%)',
            'forward_periods': 6,
            'threshold_pct': 0.8,
            'min_profit_pct': 0.8,
            'min_risk_reward_ratio': 2.5,
            'max_hold_periods': 48,
        },
        # Вариант 2: Умеренно строгие для 20-25% сигналов
        {
            'name': 'Вариант 2: Умеренно строгие (цель 20-25%)',
            'forward_periods': 5,
            'threshold_pct': 0.7,
            'min_profit_pct': 0.7,
            'min_risk_reward_ratio': 2.2,
            'max_hold_periods': 48,
        },
        # Вариант 3: Средние строгие для 18-22% сигналов
        {
            'name': 'Вариант 3: Средние строгие (цель 18-22%)',
            'forward_periods': 4,
            'threshold_pct': 0.6,
            'min_profit_pct': 0.6,
            'min_risk_reward_ratio': 2.0,
            'max_hold_periods': 48,
        },
        # Вариант 4: Компромиссные для 15-18% сигналов
        {
            'name': 'Вариант 4: Компромиссные (цель 15-18%)',
            'forward_periods': 5,
            'threshold_pct': 0.75,
            'min_profit_pct': 0.75,
            'min_risk_reward_ratio': 2.3,
            'max_hold_periods': 48,
        },
        # Вариант 5: Очень строгие для 10-15% сигналов
        {
            'name': 'Вариант 5: Очень строгие (цель 10-15%)',
            'forward_periods': 6,
            'threshold_pct': 1.0,
            'min_profit_pct': 1.0,
            'min_risk_reward_ratio': 3.0,
            'max_hold_periods': 48,
        },
    ]
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"📈 {symbol}")
        print(f"{'='*80}")
        
        for config in test_configs:
            print(f"\n🧪 Тестирование: {config['name']}")
            print(f"   forward_periods={config['forward_periods']}, "
                  f"threshold_pct={config['threshold_pct']}, "
                  f"min_profit_pct={config['min_profit_pct']}, "
                  f"min_risk_reward_ratio={config['min_risk_reward_ratio']}, "
                  f"max_hold_periods={config['max_hold_periods']}")
            
            result = test_target_labeling_params(
                symbol=symbol,
                base_interval="60",
                **{k: v for k, v in config.items() if k != 'name'}
            )
            
            if result:
                result['symbol'] = symbol
                result['config_name'] = config['name']
                all_results.append(result)
                
                print(f"   ✅ Результаты:")
                print(f"      Всего образцов: {result['total_samples']}")
                print(f"      LONG: {result['long_count']} ({result['long_pct']:.2f}%)")
                print(f"      SHORT: {result['short_count']} ({result['short_pct']:.2f}%)")
                print(f"      HOLD: {result['hold_count']} ({result['hold_pct']:.2f}%)")
                print(f"      Сигналов: {result['long_count'] + result['short_count']} ({result['signal_pct']:.2f}%)")
                print(f"      Баланс LONG/SHORT: {result['balance_ratio']:.2f}")
                
                # Оценка качества
                if 15 <= result['signal_pct'] <= 25:
                    print(f"      ✅ ИДЕАЛЬНО: % сигналов в целевом диапазоне!")
                elif result['signal_pct'] < 15:
                    print(f"      ⚠️  Слишком мало сигналов (< 15%)")
                else:
                    print(f"      ⚠️  Слишком много сигналов (> 25%)")
            else:
                print(f"   ❌ Ошибка при тестировании")
    
    # Анализ результатов
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print(f"\n{'='*80}")
        print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
        print(f"{'='*80}")
        
        # Группируем по конфигурациям
        for config_name in df_results['config_name'].unique():
            config_data = df_results[df_results['config_name'] == config_name]
            avg_signal_pct = config_data['signal_pct'].mean()
            avg_balance = config_data['balance_ratio'].mean()
            
            print(f"\n📋 {config_name}:")
            print(f"   Средний % сигналов: {avg_signal_pct:.2f}%")
            print(f"   Средний % LONG: {config_data['long_pct'].mean():.2f}%")
            print(f"   Средний % SHORT: {config_data['short_pct'].mean():.2f}%")
            print(f"   Средний баланс LONG/SHORT: {avg_balance:.2f}")
            
            # Оценка
            if 15 <= avg_signal_pct <= 25:
                print(f"   ✅ ИДЕАЛЬНО: в целевом диапазоне 15-25%")
            elif avg_signal_pct < 15:
                print(f"   ⚠️  Слишком мало сигналов")
            else:
                print(f"   ⚠️  Слишком много сигналов")
        
        # Находим лучший вариант
        print(f"\n🏆 РЕКОМЕНДАЦИЯ:")
        best_configs = df_results[
            (df_results['signal_pct'] >= 15) & 
            (df_results['signal_pct'] <= 25) &
            (df_results['balance_ratio'] >= 0.8) &
            (df_results['balance_ratio'] <= 1.2)
        ]
        
        if not best_configs.empty:
            # Группируем по конфигурациям и выбираем лучшую
            config_scores = best_configs.groupby('config_name').agg({
                'signal_pct': 'mean',
                'balance_ratio': lambda x: abs(x.mean() - 1.0).min(),  # Ближе к 1.0
            }).sort_values('signal_pct')
            
            best_config_name = config_scores.index[0]
            print(f"   ✅ Лучший вариант: {best_config_name}")
            best_config_data = best_configs[best_configs['config_name'] == best_config_name].iloc[0]
            print(f"   Параметры:")
            print(f"      forward_periods={int(best_config_data['forward_periods'])}")
            print(f"      threshold_pct={best_config_data['threshold_pct']}")
            print(f"      min_profit_pct={best_config_data['min_profit_pct']}")
            print(f"      min_risk_reward_ratio={best_config_data['min_risk_reward_ratio']}")
            print(f"      max_hold_periods={int(best_config_data['max_hold_periods'])}")
            print(f"   Средний % сигналов: {best_configs[best_configs['config_name'] == best_config_name]['signal_pct'].mean():.2f}%")
        else:
            print(f"   ⚠️  Нет конфигураций в идеальном диапазоне")
            print(f"   Рекомендуется использовать наиболее близкую к 15-25%")
            # Находим ближайшую к 20%
            closest = df_results.iloc[(df_results['signal_pct'] - 20).abs().argsort()[:1]]
            print(f"   Ближайшая к 20%: {closest['config_name'].values[0]}")
            print(f"   % сигналов: {closest['signal_pct'].values[0]:.2f}%")
        
        # Сохраняем результаты
        output_file = "1h_target_labeling_optimization_strict.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Результаты сохранены в: {output_file}")

if __name__ == "__main__":
    main()
