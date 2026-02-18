#!/usr/bin/env python3
"""Анализ результатов TP/SL из CSV файла."""
import pandas as pd
import sys

csv_file = "ml_models_comparison_20260208_214447.csv"

try:
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print("АНАЛИЗ TP/SL РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    print(f"\n📊 Общая статистика:")
    print(f"   Всего моделей: {len(df)}")
    print(f"   Моделей с 100% TP/SL: {len(df[df['signals_with_tp_sl_pct'] == 100])}")
    print(f"   Моделей с <100% TP/SL: {len(df[df['signals_with_tp_sl_pct'] < 100])}")
    print(f"   Средний % TP/SL: {df['signals_with_tp_sl_pct'].mean():.1f}%")
    print(f"   Минимальный % TP/SL: {df['signals_with_tp_sl_pct'].min():.1f}%")
    print(f"   Максимальный % TP/SL: {df['signals_with_tp_sl_pct'].max():.1f}%")
    
    # Модели с <100%
    models_below_100 = df[df['signals_with_tp_sl_pct'] < 100]
    if len(models_below_100) > 0:
        print(f"\n⚠️  Модели с TP/SL < 100% ({len(models_below_100)} моделей):")
        print("-" * 80)
        for idx, row in models_below_100.iterrows():
            print(f"   {row['symbol']:10s} | {row['model_name']:40s} | {row['signals_with_tp_sl_pct']:5.1f}% | "
                  f"LONG: {row['long_signal_pct']:5.1f}% | SHORT: {row['short_signal_pct']:5.1f}%")
    else:
        print("\n✅ Все модели имеют 100% TP/SL!")
    
    # Статистика по символам
    print(f"\n📈 Статистика по символам:")
    print("-" * 80)
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        avg_tp_sl = symbol_df['signals_with_tp_sl_pct'].mean()
        min_tp_sl = symbol_df['signals_with_tp_sl_pct'].min()
        max_tp_sl = symbol_df['signals_with_tp_sl_pct'].max()
        below_100 = len(symbol_df[symbol_df['signals_with_tp_sl_pct'] < 100])
        print(f"   {symbol:10s} | Средний: {avg_tp_sl:5.1f}% | Мин: {min_tp_sl:5.1f}% | "
              f"Макс: {max_tp_sl:5.1f}% | <100%: {below_100}")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
