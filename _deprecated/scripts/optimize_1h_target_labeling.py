"""
Скрипт для оптимизации target labeling параметров для 1h моделей.

Тестирует разные комбинации параметров и выбирает лучшие.
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
    df_raw = collector.collect_klines(
        symbol=symbol,
        interval=base_interval,
        start_date=None,
        end_date=None,
        limit=3000,
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
    print("🎯 ОПТИМИЗАЦИЯ TARGET LABELING ДЛЯ 1h МОДЕЛЕЙ")
    print("=" * 80)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]
    
    # Варианты параметров для тестирования
    test_configs = [
        # Вариант 1: Более длинные периоды, выше пороги
        {
            'name': 'Вариант 1: Длинные периоды, высокие пороги',
            'forward_periods': 4,
            'threshold_pct': 0.5,
            'min_profit_pct': 0.5,
            'min_risk_reward_ratio': 2.0,
            'max_hold_periods': 48,
        },
        # Вариант 2: Средние периоды, средние пороги
        {
            'name': 'Вариант 2: Средние периоды, средние пороги',
            'forward_periods': 3,
            'threshold_pct': 0.4,
            'min_profit_pct': 0.4,
            'min_risk_reward_ratio': 1.8,
            'max_hold_periods': 36,
        },
        # Вариант 3: Короткие периоды, низкие пороги (текущий)
        {
            'name': 'Вариант 3: Короткие периоды, низкие пороги (текущий)',
            'forward_periods': 2,
            'threshold_pct': 0.3,
            'min_profit_pct': 0.3,
            'min_risk_reward_ratio': 1.5,
            'max_hold_periods': 24,
        },
        # Вариант 4: Очень длинные периоды, высокие пороги
        {
            'name': 'Вариант 4: Очень длинные периоды, высокие пороги',
            'forward_periods': 6,
            'threshold_pct': 0.6,
            'min_profit_pct': 0.6,
            'min_risk_reward_ratio': 2.5,
            'max_hold_periods': 72,
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
            print(f"\n📋 {config_name}:")
            print(f"   Средний % сигналов: {config_data['signal_pct'].mean():.2f}%")
            print(f"   Средний % LONG: {config_data['long_pct'].mean():.2f}%")
            print(f"   Средний % SHORT: {config_data['short_pct'].mean():.2f}%")
            print(f"   Средний баланс LONG/SHORT: {config_data['balance_ratio'].mean():.2f}")
        
        # Сохраняем результаты
        output_file = "1h_target_labeling_optimization.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Результаты сохранены в: {output_file}")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print(f"   Идеальный % сигналов: 15-25%")
        print(f"   Идеальный баланс LONG/SHORT: 0.8-1.2")
        print(f"   Выберите конфигурацию с лучшим балансом сигналов и качества")

if __name__ == "__main__":
    main()
