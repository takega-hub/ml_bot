import pandas as pd

# Читаем CSV
df = pd.read_csv('ml_models_comparison_20260217_163101.csv')

print(f"Всего моделей: {len(df)}")

# Критерии для удаления
# 1. Модели с 0 сделок
zero_trades = df['total_trades'] == 0
print(f"\nМодели с 0 сделок: {zero_trades.sum()}")

# 2. Убыточные модели
losing = (df['total_trades'] > 0) & (df['total_pnl_pct'] < 0)
print(f"Убыточные модели: {losing.sum()}")

# 3. Модели с низким win rate и низким PnL
low_perf = (df['total_trades'] > 0) & (df['win_rate_pct'] < 30) & (df['total_pnl_pct'] < 5)
print(f"Модели с Win Rate < 30% и PnL < 5%: {low_perf.sum()}")

# Объединяем критерии
to_remove = zero_trades | losing | low_perf
print(f"\nВсего к удалению: {to_remove.sum()}")

# Удаляем неэффективные модели
df_filtered = df[~to_remove].copy()

print(f"Останется: {len(df_filtered)}")

# Сохраняем обратно в исходный файл
df_filtered.to_csv('ml_models_comparison_20260217_163101.csv', index=False)
print("\nФайл обновлен!")
