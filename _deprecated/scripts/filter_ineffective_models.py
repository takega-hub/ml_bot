#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для удаления неэффективных моделей из CSV файла сравнения
"""

import pandas as pd
import sys

def filter_ineffective_models(input_file, output_file=None):
    """
    Удаляет неэффективные модели из CSV файла.
    
    Критерии неэффективности:
    1. Модели с 0 сделок (total_trades == 0)
    2. Убыточные модели (total_pnl_pct < 0)
    3. Модели с очень низким win rate (< 30%) и убытком
    """
    # Читаем CSV
    df = pd.read_csv(input_file)
    
    print(f"Всего моделей в файле: {len(df)}")
    print(f"\nАнализ моделей...")
    
    # Критерии для удаления
    # 1. Модели с 0 сделок
    zero_trades = df['total_trades'] == 0
    print(f"\n1. Модели с 0 сделок: {zero_trades.sum()}")
    if zero_trades.sum() > 0:
        print("   Модели:")
        for idx, row in df[zero_trades].iterrows():
            print(f"   - {row['model_name']}")
    
    # 2. Убыточные модели (с хотя бы одной сделкой)
    losing_models = (df['total_trades'] > 0) & (df['total_pnl_pct'] < 0)
    print(f"\n2. Убыточные модели (PnL < 0): {losing_models.sum()}")
    if losing_models.sum() > 0:
        print("   Модели:")
        for idx, row in df[losing_models].iterrows():
            print(f"   - {row['model_name']}: {row['total_trades']} сделок, "
                  f"Win Rate: {row['win_rate_pct']:.1f}%, PnL: {row['total_pnl_pct']:.2f}%")
    
    # 3. Модели с очень низким win rate (< 30%) и убытком или очень низким PnL
    low_winrate = (df['total_trades'] > 0) & (df['win_rate_pct'] < 30) & (df['total_pnl_pct'] < 5)
    print(f"\n3. Модели с Win Rate < 30% и PnL < 5%: {low_winrate.sum()}")
    if low_winrate.sum() > 0:
        print("   Модели:")
        for idx, row in df[low_winrate].iterrows():
            print(f"   - {row['model_name']}: {row['total_trades']} сделок, "
                  f"Win Rate: {row['win_rate_pct']:.1f}%, PnL: {row['total_pnl_pct']:.2f}%")
    
    # Объединяем все критерии
    to_remove = zero_trades | losing_models | low_winrate
    
    print(f"\n{'='*60}")
    print(f"Всего моделей к удалению: {to_remove.sum()}")
    print(f"Останется моделей: {len(df) - to_remove.sum()}")
    print(f"{'='*60}")
    
    # Фильтруем эффективные модели
    effective_models = df[~to_remove].copy()
    
    # Сохраняем результат
    if output_file is None:
        output_file = input_file.replace('.csv', '_filtered.csv')
    
    effective_models.to_csv(output_file, index=False)
    print(f"\n✅ Эффективные модели сохранены в: {output_file}")
    
    # Показываем статистику оставшихся моделей
    if len(effective_models) > 0:
        print(f"\n📊 Статистика эффективных моделей:")
        print(f"   Средний PnL: {effective_models['total_pnl_pct'].mean():.2f}%")
        print(f"   Средний Win Rate: {effective_models['win_rate_pct'].mean():.2f}%")
        print(f"   Среднее количество сделок: {effective_models['total_trades'].mean():.1f}")
        print(f"   Лучшая модель по PnL: {effective_models.loc[effective_models['total_pnl_pct'].idxmax(), 'model_name']} "
              f"({effective_models['total_pnl_pct'].max():.2f}%)")
    
    return effective_models

if __name__ == "__main__":
    input_file = "ml_models_comparison_20260217_163101.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        filter_ineffective_models(input_file, output_file)
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
