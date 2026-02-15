"""
Сравнение предсказанных MTF комбинаций с реальными результатами бэктестов
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def load_predicted_results(json_path: str) -> Dict:
    """Загружает предсказанные результаты из JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_actual_results(csv_path: str) -> pd.DataFrame:
    """Загружает реальные результаты из CSV"""
    df = pd.read_csv(csv_path)
    # Нормализуем имена моделей (убираем .pkl)
    df['model_1h'] = df['model_1h'].str.replace('.pkl', '')
    df['model_15m'] = df['model_15m'].str.replace('.pkl', '')
    return df

def find_matching_combination(
    predicted: Dict,
    actual_df: pd.DataFrame,
    symbol: str
) -> Tuple[pd.Series, float]:
    """Находит соответствующую комбинацию в реальных результатах"""
    model_1h = predicted['model_1h']
    model_15m = predicted['model_15m']
    
    # Ищем точное совпадение
    match = actual_df[
        (actual_df['model_1h'] == model_1h) &
        (actual_df['model_15m'] == model_15m)
    ]
    
    if not match.empty:
        return match.iloc[0], 1.0
    
    # Если точного совпадения нет, ищем по части имени
    match = actual_df[
        (actual_df['model_1h'].str.contains(model_1h.split('_')[0], na=False)) &
        (actual_df['model_15m'].str.contains(model_15m.split('_')[0], na=False))
    ]
    
    if not match.empty:
        return match.iloc[0], 0.5  # Частичное совпадение
    
    return None, 0.0

def compare_results():
    """Сравнивает предсказанные и реальные результаты"""
    
    # Загружаем предсказанные результаты
    predicted_file = "mtf_predictions/best_predicted_mtf_20260215_141752.json"
    predicted_data = load_predicted_results(predicted_file)
    
    # Маппинг символов к CSV файлам
    csv_files = {
        "BTCUSDT": "mtf_combinations_BTCUSDT_20260212_194950.csv",
        "ETHUSDT": "mtf_combinations_ETHUSDT_20260214_020745.csv",
        "SOLUSDT": "mtf_combinations_SOLUSDT_20260214_232406.csv"
    }
    
    print("=" * 100)
    print("📊 СРАВНЕНИЕ ПРЕДСКАЗАННЫХ И РЕАЛЬНЫХ РЕЗУЛЬТАТОВ MTF КОМБИНАЦИЙ")
    print("=" * 100)
    print()
    
    all_comparisons = []
    
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if symbol not in predicted_data['best_combinations']:
            continue
        
        print(f"\n{'='*100}")
        print(f"🔍 {symbol}")
        print(f"{'='*100}\n")
        
        # Загружаем реальные результаты
        if symbol not in csv_files:
            print(f"⚠️  CSV файл для {symbol} не найден")
            continue
        
        actual_df = load_actual_results(csv_files[symbol])
        
        # Лучшая предсказанная комбинация
        predicted_best = predicted_data['best_combinations'][symbol]['best_combination']
        
        print(f"📈 ЛУЧШАЯ ПРЕДСКАЗАННАЯ КОМБИНАЦИЯ:")
        print(f"   1h: {predicted_best['model_1h']}")
        print(f"   15m: {predicted_best['model_15m']}")
        print(f"   Предсказанный PnL: {predicted_best['predicted_pnl_pct']:.2f}%")
        print(f"   Предсказанный WR: {predicted_best['predicted_wr']:.2f}%")
        print(f"   Предсказанный Score: {predicted_best['predicted_score']:.2f}")
        print(f"   Предсказанный PF: {predicted_best['predicted_profit_factor']:.2f}")
        print(f"   Предсказанный Sharpe: {predicted_best['predicted_sharpe']:.2f}")
        print()
        
        # Ищем соответствующую реальную комбинацию
        actual_match, match_quality = find_matching_combination(
            predicted_best, actual_df, symbol
        )
        
        if actual_match is not None:
            print(f"✅ НАЙДЕНА РЕАЛЬНАЯ КОМБИНАЦИЯ (качество совпадения: {match_quality*100:.0f}%):")
            print(f"   Реальный PnL: {actual_match['total_pnl_pct']:.2f}%")
            print(f"   Реальный WR: {actual_match['win_rate']*100:.2f}%")
            print(f"   Реальный PF: {actual_match['profit_factor']:.2f}")
            print(f"   Реальный Sharpe: {actual_match['sharpe_ratio']:.2f}")
            print(f"   Реальный Max DD: {actual_match['max_drawdown_pct']:.2f}%")
            print()
            
            # Вычисляем ошибки предсказания
            pnl_error = abs(predicted_best['predicted_pnl_pct'] - actual_match['total_pnl_pct'])
            pnl_error_pct = (pnl_error / actual_match['total_pnl_pct']) * 100 if actual_match['total_pnl_pct'] > 0 else 0
            
            wr_error = abs(predicted_best['predicted_wr'] - actual_match['win_rate']*100)
            pf_error = abs(predicted_best['predicted_profit_factor'] - actual_match['profit_factor'])
            sharpe_error = abs(predicted_best['predicted_sharpe'] - actual_match['sharpe_ratio'])
            
            print(f"📊 ОШИБКИ ПРЕДСКАЗАНИЯ:")
            print(f"   PnL: {pnl_error:.2f}% (относительная ошибка: {pnl_error_pct:.1f}%)")
            print(f"   WR: {wr_error:.2f}%")
            print(f"   PF: {pf_error:.2f}")
            print(f"   Sharpe: {sharpe_error:.2f}")
            print()
            
            all_comparisons.append({
                'symbol': symbol,
                'model_1h': predicted_best['model_1h'],
                'model_15m': predicted_best['model_15m'],
                'predicted_pnl': predicted_best['predicted_pnl_pct'],
                'actual_pnl': actual_match['total_pnl_pct'],
                'pnl_error_pct': pnl_error_pct,
                'predicted_wr': predicted_best['predicted_wr'],
                'actual_wr': actual_match['win_rate']*100,
                'wr_error': wr_error,
                'predicted_pf': predicted_best['predicted_profit_factor'],
                'actual_pf': actual_match['profit_factor'],
                'pf_error': pf_error,
                'match_quality': match_quality
            })
        else:
            print(f"❌ РЕАЛЬНАЯ КОМБИНАЦИЯ НЕ НАЙДЕНА")
            print(f"   (Эта комбинация не тестировалась в бэктесте)")
            print()
        
        # Лучшая реальная комбинация
        actual_best = actual_df.loc[actual_df['total_pnl_pct'].idxmax()]
        print(f"🏆 ЛУЧШАЯ РЕАЛЬНАЯ КОМБИНАЦИЯ (из бэктеста):")
        print(f"   1h: {actual_best['model_1h']}")
        print(f"   15m: {actual_best['model_15m']}")
        print(f"   Реальный PnL: {actual_best['total_pnl_pct']:.2f}%")
        print(f"   Реальный WR: {actual_best['win_rate']*100:.2f}%")
        print(f"   Реальный PF: {actual_best['profit_factor']:.2f}")
        print(f"   Реальный Sharpe: {actual_best['sharpe_ratio']:.2f}")
        print()
        
        # Проверяем, есть ли предсказание для лучшей реальной комбинации
        predicted_match = None
        for combo in predicted_data['best_combinations'][symbol]['top_combinations']:
            if (combo['model_1h'] == actual_best['model_1h'] and 
                combo['model_15m'] == actual_best['model_15m']):
                predicted_match = combo
                break
        
        if predicted_match:
            print(f"✅ ПРЕДСКАЗАНИЕ ДЛЯ ЛУЧШЕЙ РЕАЛЬНОЙ КОМБИНАЦИИ:")
            print(f"   Предсказанный PnL: {predicted_match['predicted_pnl_pct']:.2f}%")
            print(f"   Реальный PnL: {actual_best['total_pnl_pct']:.2f}%")
            print(f"   Позиция в топе предсказаний: {predicted_data['best_combinations'][symbol]['top_combinations'].index(predicted_match) + 1}")
        else:
            print(f"❌ ПРЕДСКАЗАНИЕ ДЛЯ ЛУЧШЕЙ РЕАЛЬНОЙ КОМБИНАЦИИ НЕ НАЙДЕНО")
        print()
    
    # Общая статистика
    if all_comparisons:
        print(f"\n{'='*100}")
        print("📊 ОБЩАЯ СТАТИСТИКА")
        print(f"{'='*100}\n")
        
        df_comp = pd.DataFrame(all_comparisons)
        
        print(f"Средняя относительная ошибка PnL: {df_comp['pnl_error_pct'].mean():.1f}%")
        print(f"Медианная относительная ошибка PnL: {df_comp['pnl_error_pct'].median():.1f}%")
        print(f"Средняя ошибка WR: {df_comp['wr_error'].mean():.2f}%")
        print(f"Средняя ошибка PF: {df_comp['pf_error'].mean():.2f}")
        print()
        
        print("Детальное сравнение:")
        print(df_comp[['symbol', 'predicted_pnl', 'actual_pnl', 'pnl_error_pct', 
                       'predicted_wr', 'actual_wr', 'wr_error']].to_string(index=False))

if __name__ == "__main__":
    compare_results()
