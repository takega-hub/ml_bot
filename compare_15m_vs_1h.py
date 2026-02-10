"""
Скрипт для сравнения результатов 15m и 1h моделей.
"""
import pandas as pd
from pathlib import Path

def load_and_compare():
    # Загружаем данные
    df_15m = pd.read_csv("ml_models_comparison_20260210_084726.csv")
    df_1h = pd.read_csv("ml_models_comparison_20260210_174157.csv")
    
    # Фильтруем только рабочие модели (с сделками > 0)
    df_15m_working = df_15m[df_15m['total_trades'] > 0].copy()
    df_1h_working = df_1h[df_1h['total_trades'] > 0].copy()
    
    print("=" * 100)
    print("📊 СРАВНЕНИЕ 15m И 1h МОДЕЛЕЙ")
    print("=" * 100)
    
    symbols = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    comparison_results = []
    
    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"📈 {symbol}")
        print(f"{'='*100}")
        
        # Лучшие модели для каждого таймфрейма
        symbol_15m = df_15m_working[df_15m_working['symbol'] == symbol].copy()
        symbol_1h = df_1h_working[df_1h_working['symbol'] == symbol].copy()
        
        if symbol_15m.empty:
            print(f"⚠️  Нет данных для 15m моделей {symbol}")
            continue
        if symbol_1h.empty:
            print(f"⚠️  Нет данных для 1h моделей {symbol}")
            continue
        
        # Лучшая 15m модель (по PnL%)
        best_15m = symbol_15m.loc[symbol_15m['total_pnl_pct'].idxmax()]
        
        # Лучшая 1h модель (по PnL%)
        best_1h = symbol_1h.loc[symbol_1h['total_pnl_pct'].idxmax()]
        
        print(f"\n🕐 ЛУЧШАЯ 15m МОДЕЛЬ:")
        print(f"   Модель: {best_15m['model_name']}")
        print(f"   Сделок: {int(best_15m['total_trades'])}")
        print(f"   PnL%: {best_15m['total_pnl_pct']:.2f}%")
        print(f"   Win Rate: {best_15m['win_rate_pct']:.2f}%")
        print(f"   Profit Factor: {best_15m['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {best_15m['sharpe_ratio']:.2f}")
        print(f"   Сделок/день: {best_15m['trades_per_day']:.2f}")
        print(f"   Max Drawdown: {best_15m['max_drawdown_pct']:.2f}%")
        
        print(f"\n🕐 ЛУЧШАЯ 1h МОДЕЛЬ:")
        print(f"   Модель: {best_1h['model_name']}")
        print(f"   Сделок: {int(best_1h['total_trades'])}")
        print(f"   PnL%: {best_1h['total_pnl_pct']:.2f}%")
        print(f"   Win Rate: {best_1h['win_rate_pct']:.2f}%")
        print(f"   Profit Factor: {best_1h['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {best_1h['sharpe_ratio']:.2f}")
        print(f"   Сделок/день: {best_1h['trades_per_day']:.2f}")
        print(f"   Max Drawdown: {best_1h['max_drawdown_pct']:.2f}%")
        
        # Сравнение
        print(f"\n📊 СРАВНЕНИЕ:")
        pnl_diff = best_1h['total_pnl_pct'] - best_15m['total_pnl_pct']
        trades_diff = best_1h['total_trades'] - best_15m['total_trades']
        wr_diff = best_1h['win_rate_pct'] - best_15m['win_rate_pct']
        
        print(f"   PnL% разница: {pnl_diff:+.2f}% ({'1h лучше' if pnl_diff > 0 else '15m лучше'})")
        print(f"   Сделок разница: {trades_diff:+.0f} ({'1h больше' if trades_diff > 0 else '15m больше'})")
        print(f"   Win Rate разница: {wr_diff:+.2f}% ({'1h лучше' if wr_diff > 0 else '15m лучше'})")
        
        # Рекомендация
        print(f"\n💡 РЕКОМЕНДАЦИЯ:")
        if best_1h['total_pnl_pct'] > best_15m['total_pnl_pct']:
            print(f"   ✅ Использовать 1h модель: {best_1h['model_name']}")
            print(f"      Преимущества: выше PnL ({best_1h['total_pnl_pct']:.2f}% vs {best_15m['total_pnl_pct']:.2f}%)")
        else:
            print(f"   ✅ Использовать 15m модель: {best_15m['model_name']}")
            print(f"      Преимущества: выше PnL ({best_15m['total_pnl_pct']:.2f}% vs {best_1h['total_pnl_pct']:.2f}%)")
            if best_15m['total_trades'] > best_1h['total_trades'] * 2:
                print(f"      Больше сделок ({int(best_15m['total_trades'])} vs {int(best_1h['total_trades'])})")
        
        comparison_results.append({
            'symbol': symbol,
            'best_15m_model': best_15m['model_name'],
            'best_15m_pnl': best_15m['total_pnl_pct'],
            'best_15m_trades': int(best_15m['total_trades']),
            'best_15m_wr': best_15m['win_rate_pct'],
            'best_1h_model': best_1h['model_name'],
            'best_1h_pnl': best_1h['total_pnl_pct'],
            'best_1h_trades': int(best_1h['total_trades']),
            'best_1h_wr': best_1h['win_rate_pct'],
            'pnl_diff': pnl_diff,
            'recommended': '1h' if best_1h['total_pnl_pct'] > best_15m['total_pnl_pct'] else '15m',
        })
    
    # Общая статистика
    print(f"\n{'='*100}")
    print("📊 ОБЩАЯ СТАТИСТИКА")
    print(f"{'='*100}")
    
    if comparison_results:
        df_comp = pd.DataFrame(comparison_results)
        
        print(f"\n📈 Средний PnL%:")
        print(f"   15m модели: {df_comp['best_15m_pnl'].mean():.2f}%")
        print(f"   1h модели: {df_comp['best_1h_pnl'].mean():.2f}%")
        
        print(f"\n📊 Среднее количество сделок:")
        print(f"   15m модели: {df_comp['best_15m_trades'].mean():.1f}")
        print(f"   1h модели: {df_comp['best_1h_trades'].mean():.1f}")
        
        print(f"\n🎯 Средний Win Rate:")
        print(f"   15m модели: {df_comp['best_15m_wr'].mean():.2f}%")
        print(f"   1h модели: {df_comp['best_1h_wr'].mean():.2f}%")
        
        print(f"\n🏆 Рекомендации по символам:")
        for _, row in df_comp.iterrows():
            recommended = row['recommended']
            model_col = f'best_{recommended}_model'
            print(f"   {row['symbol']}: {recommended} ({row[model_col]})")
        
        # Сохраняем результаты
        output_file = "comparison_15m_vs_1h.csv"
        df_comp.to_csv(output_file, index=False)
        print(f"\n💾 Результаты сохранены в: {output_file}")

if __name__ == "__main__":
    load_and_compare()
