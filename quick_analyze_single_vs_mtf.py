"""
Быстрый анализ результатов одиночных моделей vs MTF комбинаций.
Работает с текущими данными (1h модели) и показывает предварительные результаты.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_data():
    """Загружает все данные"""
    # MTF результаты
    mtf_btc = pd.read_csv("mtf_combinations_BTCUSDT_20260212_194950.csv")
    mtf_eth = pd.read_csv("mtf_combinations_ETHUSDT_20260214_020745.csv")
    df_mtf = pd.concat([mtf_btc, mtf_eth], ignore_index=True)
    
    # Одиночные модели (только 1h пока)
    df_single = pd.read_csv("ml_models_comparison_20260214_111828.csv")
    
    return df_single, df_mtf


def extract_model_type(model_name: str) -> str:
    """Извлекает тип модели из имени"""
    parts = model_name.split('_')
    if len(parts) > 0:
        return parts[0]  # rf, xgb, ensemble, etc.
    return "unknown"


def analyze_symbol(symbol: str, df_single: pd.DataFrame, df_mtf: pd.DataFrame):
    """Анализирует один символ"""
    print("=" * 100)
    print(f"🎯 АНАЛИЗ ДЛЯ {symbol}")
    print("=" * 100)
    
    # Фильтруем данные по символу
    single_symbol = df_single[df_single['symbol'] == symbol].copy()
    mtf_symbol = df_mtf[df_mtf['symbol'] == symbol].copy()
    
    if single_symbol.empty:
        print(f"⚠️  Нет данных одиночных моделей для {symbol}")
        return
    
    if mtf_symbol.empty:
        print(f"⚠️  Нет данных MTF комбинаций для {symbol}")
        return
    
    # Сортируем по PnL
    single_symbol = single_symbol.sort_values('total_pnl_pct', ascending=False)
    mtf_symbol = mtf_symbol.sort_values('total_pnl_pct', ascending=False)
    
    print("\n📊 ЛУЧШИЕ ОДИНОЧНЫЕ 1H МОДЕЛИ:")
    print("-" * 100)
    for i, row in single_symbol.head(5).iterrows():
        model_name = row['model_filename'].replace('.pkl', '')
        print(f"   {i+1}. {model_name}")
        print(f"      PnL: {row['total_pnl_pct']:.2f}% | WR: {row.get('win_rate_pct', 0):.1f}% | "
              f"PF: {row['profit_factor']:.2f} | Sharpe: {row['sharpe_ratio']:.2f}")
    
    print("\n🏆 ЛУЧШИЕ MTF КОМБИНАЦИИ:")
    print("-" * 100)
    for i, row in mtf_symbol.head(5).iterrows():
        print(f"   {i+1}. {row['model_1h']} + {row['model_15m']}")
        print(f"      PnL: {row['total_pnl_pct']:.2f}% | WR: {row['win_rate']:.1f}% | "
              f"PF: {row['profit_factor']:.2f} | Sharpe: {row['sharpe_ratio']:.2f}")
    
    # Анализ: какие модели из лучших одиночных попадают в лучшие MTF
    print("\n🔍 АНАЛИЗ СОВПАДЕНИЙ:")
    print("-" * 100)
    
    best_single_models = single_symbol.head(5)
    best_mtf = mtf_symbol.head(10)
    
    # Извлекаем типы моделей из лучших одиночных
    best_single_types = set()
    for _, row in best_single_models.iterrows():
        model_name = row['model_filename'].replace('.pkl', '')
        model_type = extract_model_type(model_name)
        best_single_types.add(model_type)
    
    print(f"   Типы лучших одиночных моделей: {', '.join(sorted(best_single_types))}")
    
    # Проверяем, какие из лучших одиночных моделей используются в лучших MTF
    matches_1h = []
    for _, mtf_row in best_mtf.iterrows():
        model_1h = mtf_row['model_1h']
        model_1h_type = extract_model_type(model_1h)
        
        # Проверяем, есть ли такая модель в лучших одиночных
        for _, single_row in best_single_models.iterrows():
            single_model = single_row['model_filename'].replace('.pkl', '')
            if model_1h_type in single_model or single_model in model_1h:
                matches_1h.append({
                    'mtf_rank': len(matches_1h) + 1,
                    'mtf_pnl': mtf_row['total_pnl_pct'],
                    'single_model': single_model,
                    'single_pnl': single_row['total_pnl_pct'],
                    'mtf_combo': f"{model_1h} + {mtf_row['model_15m']}"
                })
                break
    
    if matches_1h:
        print(f"\n   ✅ Найдено {len(matches_1h)} совпадений лучших одиночных моделей в топ-10 MTF:")
        for match in matches_1h[:5]:
            print(f"      - {match['single_model']} (одиночный PnL: {match['single_pnl']:.2f}%)")
            print(f"        → Используется в MTF: {match['mtf_combo']}")
            print(f"        → MTF PnL: {match['mtf_pnl']:.2f}%")
    else:
        print("   ⚠️  Лучшие одиночные модели НЕ совпадают с лучшими MTF комбинациями")
    
    # Статистика
    print("\n📈 СТАТИСТИКА:")
    print("-" * 100)
    best_single_pnl = single_symbol.iloc[0]['total_pnl_pct']
    best_mtf_pnl = mtf_symbol.iloc[0]['total_pnl_pct']
    improvement = best_mtf_pnl - best_single_pnl
    
    print(f"   Лучший одиночный PnL: {best_single_pnl:.2f}%")
    print(f"   Лучший MTF PnL: {best_mtf_pnl:.2f}%")
    print(f"   Улучшение MTF: {improvement:.2f}% ({improvement/best_single_pnl*100:.1f}% относительно одиночного)")
    
    # Средние значения
    avg_single_pnl = single_symbol['total_pnl_pct'].mean()
    avg_mtf_pnl = mtf_symbol['total_pnl_pct'].mean()
    print(f"\n   Средний одиночный PnL: {avg_single_pnl:.2f}%")
    print(f"   Средний MTF PnL: {avg_mtf_pnl:.2f}%")
    
    print()


def main():
    """Основная функция"""
    print("=" * 100)
    print("📊 БЫСТРЫЙ АНАЛИЗ: ОДИНОЧНЫЕ МОДЕЛИ VS MTF КОМБИНАЦИИ")
    print("=" * 100)
    print()
    print("⚠️  ВНИМАНИЕ: Анализ выполняется только на основе 1h моделей.")
    print("   Для полного анализа нужны также результаты 15m моделей.")
    print()
    
    # Загружаем данные
    print("📥 Загрузка данных...")
    df_single, df_mtf = load_data()
    print(f"✅ Загружено {len(df_single)} одиночных моделей, {len(df_mtf)} MTF комбинаций")
    print()
    
    # Анализируем каждый символ
    symbols = ['BTCUSDT', 'ETHUSDT']
    for symbol in symbols:
        analyze_symbol(symbol, df_single, df_mtf)
    
    # Общие выводы
    print("=" * 100)
    print("💡 ВЫВОДЫ:")
    print("=" * 100)
    print()
    print("1. Для полного анализа необходимо:")
    print("   - Результаты тестирования 15m моделей")
    print("   - Сравнение лучших одиночных 15m с лучшими MTF комбинациями")
    print()
    print("2. Предварительные наблюдения:")
    print("   - Проверьте, используются ли лучшие одиночные 1h модели в лучших MTF")
    print("   - Оцените, насколько MTF превосходит одиночные модели")
    print()
    print("3. Рекомендации по оптимизации:")
    print("   - Если лучшие одиночные модели попадают в лучшие MTF, можно:")
    print("     * Тестировать только топ-5 моделей каждого таймфрейма (25 комбинаций)")
    print("     * Вместо всех комбинаций (может быть 100+ комбинаций)")
    print("   - Это ускорит процесс тестирования в 4-5 раз")
    print()


if __name__ == "__main__":
    main()
