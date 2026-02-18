"""
Анализ результатов одиночных моделей vs MTF комбинаций.
Проверяет, можно ли предсказать лучшие MTF комбинации на основе результатов одиночных моделей.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


def load_mtf_results(mtf_file: Path) -> pd.DataFrame:
    """Загружает результаты MTF комбинаций"""
    df = pd.read_csv(mtf_file)
    return df


def load_single_results(single_file: Path) -> pd.DataFrame:
    """Загружает результаты одиночных моделей"""
    df = pd.read_csv(single_file)
    return df


def extract_model_name(filename: str) -> str:
    """Извлекает имя модели из имени файла"""
    return filename.replace('.pkl', '')


def analyze_single_models(df_single: pd.DataFrame, symbol: str) -> Dict[str, pd.Series]:
    """Анализирует одиночные модели для символа, разделяя по таймфреймам"""
    symbol_data = df_single[df_single['symbol'] == symbol].copy()
    
    # Разделяем на 1h и 15m модели
    models_1h = symbol_data[
        (symbol_data['mode_suffix'] == '1h') |
        (symbol_data['model_filename'].str.contains('_60_|_1h', na=False))
    ].copy()
    
    models_15m = symbol_data[
        (symbol_data['mode_suffix'] == '15m') |
        (symbol_data['model_filename'].str.contains('_15_|_15m', na=False))
    ].copy()
    
    # Сортируем по total_pnl_pct
    if not models_1h.empty:
        models_1h = models_1h.sort_values('total_pnl_pct', ascending=False)
    if not models_15m.empty:
        models_15m = models_15m.sort_values('total_pnl_pct', ascending=False)
    
    return {
        '1h': models_1h,
        '15m': models_15m
    }


def analyze_mtf_combinations(df_mtf: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Анализирует MTF комбинации для символа"""
    symbol_mtf = df_mtf[df_mtf['symbol'] == symbol].copy()
    symbol_mtf = symbol_mtf.sort_values('total_pnl_pct', ascending=False)
    return symbol_mtf


def calculate_model_score(row: pd.Series) -> float:
    """Вычисляет composite score для модели"""
    pnl = row.get('total_pnl_pct', 0)
    wr = row.get('win_rate', row.get('win_rate_pct', 0))
    pf = row.get('profit_factor', 0)
    sharpe = row.get('sharpe_ratio', 0)
    dd = row.get('max_drawdown_pct', 100)
    
    score = (
        pnl * 0.4 +
        wr * 0.2 +
        pf * 20.0 * 0.2 +
        sharpe * 0.1 +
        (100 - dd) * 0.1
    )
    return score


def predict_mtf_from_single(
    models_1h: pd.DataFrame,
    models_15m: pd.DataFrame,
    df_mtf: pd.DataFrame,
    symbol: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Предсказывает лучшие MTF комбинации на основе результатов одиночных моделей.
    Возвращает DataFrame с предсказаниями и реальными результатами.
    """
    predictions = []
    
    # Берем топ-N моделей каждого таймфрейма
    top_1h = models_1h.head(top_n) if not models_1h.empty else pd.DataFrame()
    top_15m = models_15m.head(top_n) if not models_15m.empty else pd.DataFrame()
    
    if top_1h.empty or top_15m.empty:
        return pd.DataFrame()
    
    # Генерируем все комбинации топ моделей
    for _, row_1h in top_1h.iterrows():
        for _, row_15m in top_15m.iterrows():
            model_1h_name = extract_model_name(row_1h['model_filename'])
            model_15m_name = extract_model_name(row_15m['model_filename'])
            
            # Вычисляем предсказанный score как среднее или сумму scores одиночных моделей
            score_1h = calculate_model_score(row_1h)
            score_15m = calculate_model_score(row_15m)
            predicted_score = (score_1h + score_15m) / 2  # Среднее
            
            # Ищем реальный результат MTF комбинации
            mtf_match = df_mtf[
                (df_mtf['model_1h'].str.contains(model_1h_name.split('_')[0], na=False)) &
                (df_mtf['model_15m'].str.contains(model_15m_name.split('_')[0], na=False))
            ]
            
            if not mtf_match.empty:
                mtf_result = mtf_match.iloc[0]
                actual_pnl = mtf_result['total_pnl_pct']
                actual_wr = mtf_result['win_rate']
                actual_score = calculate_model_score(mtf_result)
            else:
                actual_pnl = None
                actual_wr = None
                actual_score = None
            
            predictions.append({
                'model_1h': model_1h_name,
                'model_15m': model_15m_name,
                'single_1h_pnl': row_1h['total_pnl_pct'],
                'single_15m_pnl': row_15m['total_pnl_pct'],
                'single_1h_wr': row_1h.get('win_rate_pct', 0),
                'single_15m_wr': row_15m.get('win_rate_pct', 0),
                'predicted_score': predicted_score,
                'actual_mtf_pnl': actual_pnl,
                'actual_mtf_wr': actual_wr,
                'actual_mtf_score': actual_score,
                'prediction_error': abs(predicted_score - actual_score) if actual_score is not None else None,
            })
    
    df_predictions = pd.DataFrame(predictions)
    if not df_predictions.empty:
        df_predictions = df_predictions.sort_values('predicted_score', ascending=False)
    
    return df_predictions


def calculate_correlation(df_predictions: pd.DataFrame) -> Dict[str, float]:
    """Вычисляет корреляции между одиночными моделями и MTF результатами"""
    correlations = {}
    
    # Фильтруем только те, где есть реальные MTF результаты
    df_valid = df_predictions[df_predictions['actual_mtf_pnl'].notna()].copy()
    
    if df_valid.empty:
        return correlations
    
    # Корреляция между предсказанным и реальным score
    if 'predicted_score' in df_valid.columns and 'actual_mtf_score' in df_valid.columns:
        corr = df_valid['predicted_score'].corr(df_valid['actual_mtf_score'])
        correlations['score_correlation'] = corr
    
    # Корреляция между суммой PnL одиночных и реальным MTF PnL
    df_valid['sum_single_pnl'] = df_valid['single_1h_pnl'] + df_valid['single_15m_pnl']
    corr = df_valid['sum_single_pnl'].corr(df_valid['actual_mtf_pnl'])
    correlations['pnl_sum_correlation'] = corr
    
    # Корреляция между средним PnL одиночных и реальным MTF PnL
    df_valid['avg_single_pnl'] = (df_valid['single_1h_pnl'] + df_valid['single_15m_pnl']) / 2
    corr = df_valid['avg_single_pnl'].corr(df_valid['actual_mtf_pnl'])
    correlations['pnl_avg_correlation'] = corr
    
    # Корреляция между средним WR одиночных и реальным MTF WR
    df_valid['avg_single_wr'] = (df_valid['single_1h_wr'] + df_valid['single_15m_wr']) / 2
    corr = df_valid['avg_single_wr'].corr(df_valid['actual_mtf_wr'])
    correlations['wr_avg_correlation'] = corr
    
    return correlations


def generate_recommendations(
    df_single: pd.DataFrame,
    df_mtf: pd.DataFrame,
    symbols: List[str]
) -> Dict[str, any]:
    """Генерирует рекомендации по выбору моделей для MTF"""
    recommendations = {}
    
    for symbol in symbols:
        # Анализируем одиночные модели
        single_models = analyze_single_models(df_single, symbol)
        models_1h = single_models['1h']
        models_15m = single_models['15m']
        
        # Анализируем MTF комбинации
        mtf_combinations = analyze_mtf_combinations(df_mtf, symbol)
        
        # Если нет 15m моделей, пытаемся извлечь информацию из MTF результатов
        if models_15m.empty and not mtf_combinations.empty:
            # Извлекаем уникальные имена 15m моделей из MTF результатов
            unique_15m_models = mtf_combinations['model_15m'].unique()
            print(f"   ⚠️  Не найдено 15m моделей в файле сравнения для {symbol}")
            print(f"   Найдено {len(unique_15m_models)} уникальных 15m моделей в MTF результатах")
            print(f"   Будет выполнен частичный анализ на основе доступных данных")
        
        if models_1h.empty:
            recommendations[symbol] = {
                'status': 'insufficient_data',
                'message': f'Не найдено 1h моделей для {symbol}. Запустите: python compare_ml_models.py --symbols {symbol} --only-1h'
            }
            continue
        
        if models_15m.empty:
            # Выполняем частичный анализ только на основе 1h моделей
            recommendations[symbol] = {
                'status': 'partial_data',
                'message': f'Найдены только 1h модели для {symbol}. Для полного анализа нужны 15m модели.',
                'best_single_1h': {
                    'model': extract_model_name(models_1h.iloc[0]['model_filename']) if not models_1h.empty else None,
                    'pnl': models_1h.iloc[0]['total_pnl_pct'] if not models_1h.empty else None,
                    'wr': models_1h.iloc[0].get('win_rate_pct', 0) if not models_1h.empty else None,
                },
                'best_mtf_actual': {
                    'model_1h': mtf_combinations.iloc[0]['model_1h'] if not mtf_combinations.empty else None,
                    'model_15m': mtf_combinations.iloc[0]['model_15m'] if not mtf_combinations.empty else None,
                    'pnl': mtf_combinations.iloc[0]['total_pnl_pct'] if not mtf_combinations.empty else None,
                    'wr': mtf_combinations.iloc[0]['win_rate'] if not mtf_combinations.empty else None,
                },
            }
            continue
        
        # Предсказываем лучшие комбинации
        predictions = predict_mtf_from_single(models_1h, models_15m, mtf_combinations, symbol, top_n=5)
        
        if predictions.empty:
            recommendations[symbol] = {
                'status': 'no_predictions',
                'message': f'Не удалось сгенерировать предсказания для {symbol}'
            }
            continue
        
        # Вычисляем корреляции
        correlations = calculate_correlation(predictions)
        
        # Лучшие одиночные модели
        best_1h = models_1h.iloc[0] if not models_1h.empty else None
        best_15m = models_15m.iloc[0] if not models_15m.empty else None
        
        # Лучшая реальная MTF комбинация
        best_mtf = mtf_combinations.iloc[0] if not mtf_combinations.empty else None
        
        # Проверяем, есть ли лучшая MTF комбинация в предсказаниях
        top_predicted = predictions.head(10)
        best_predicted_match = None
        if best_mtf is not None:
            best_1h_name = extract_model_name(best_mtf['model_1h'])
            best_15m_name = extract_model_name(best_mtf['model_15m'])
            
            for _, pred in top_predicted.iterrows():
                if (best_1h_name in pred['model_1h'] and 
                    best_15m_name in pred['model_15m']):
                    best_predicted_match = pred
                    break
        
        recommendations[symbol] = {
            'status': 'success',
            'best_single_1h': {
                'model': extract_model_name(best_1h['model_filename']) if best_1h is not None else None,
                'pnl': best_1h['total_pnl_pct'] if best_1h is not None else None,
                'wr': best_1h.get('win_rate_pct', 0) if best_1h is not None else None,
            },
            'best_single_15m': {
                'model': extract_model_name(best_15m['model_filename']) if best_15m is not None else None,
                'pnl': best_15m['total_pnl_pct'] if best_15m is not None else None,
                'wr': best_15m.get('win_rate_pct', 0) if best_15m is not None else None,
            },
            'best_mtf_actual': {
                'model_1h': best_mtf['model_1h'] if best_mtf is not None else None,
                'model_15m': best_mtf['model_15m'] if best_mtf is not None else None,
                'pnl': best_mtf['total_pnl_pct'] if best_mtf is not None else None,
                'wr': best_mtf['win_rate'] if best_mtf is not None else None,
            },
            'correlations': correlations,
            'top_predictions': top_predicted.to_dict('records')[:5],
            'best_predicted_match_rank': None if best_predicted_match is None else 
                (top_predicted.index.get_loc(best_predicted_match.name) + 1 if best_predicted_match.name in top_predicted.index else None),
        }
    
    return recommendations


def print_analysis_report(recommendations: Dict, symbols: List[str]):
    """Печатает отчет об анализе"""
    print("=" * 100)
    print("📊 АНАЛИЗ ОДИНОЧНЫХ МОДЕЛЕЙ VS MTF КОМБИНАЦИЙ")
    print("=" * 100)
    print()
    
    for symbol in symbols:
        if symbol not in recommendations:
            continue
        
        rec = recommendations[symbol]
        
        if rec['status'] == 'insufficient_data':
            print(f"⚠️  {symbol}: {rec.get('message', 'Ошибка анализа')}")
            print()
            continue
        elif rec['status'] == 'partial_data':
            print(f"⚠️  {symbol}: {rec.get('message', 'Частичные данные')}")
            print("-" * 100)
            
            # Показываем доступную информацию
            if rec.get('best_single_1h', {}).get('model'):
                print("\n📈 ЛУЧШАЯ ОДИНОЧНАЯ 1H МОДЕЛЬ:")
                print(f"   {rec['best_single_1h']['model']}")
                print(f"   PnL: {rec['best_single_1h']['pnl']:.2f}%, WR: {rec['best_single_1h']['wr']:.1f}%")
            
            if rec.get('best_mtf_actual', {}).get('model_1h'):
                print("\n🏆 ЛУЧШАЯ РЕАЛЬНАЯ MTF КОМБИНАЦИЯ:")
                print(f"   {rec['best_mtf_actual']['model_1h']} + {rec['best_mtf_actual']['model_15m']}")
                print(f"   PnL: {rec['best_mtf_actual']['pnl']:.2f}%, WR: {rec['best_mtf_actual']['wr']:.1f}%")
            
            print("\n💡 Для полного анализа необходимо:")
            print(f"   python compare_ml_models.py --symbols {symbol} --interval 15m --days 30")
            print()
            continue
        elif rec['status'] != 'success':
            print(f"⚠️  {symbol}: {rec.get('message', 'Ошибка анализа')}")
            print()
            continue
        
        print(f"🎯 {symbol}")
        print("-" * 100)
        
        # Лучшие одиночные модели
        print("\n📈 ЛУЧШИЕ ОДИНОЧНЫЕ МОДЕЛИ:")
        if rec['best_single_1h']['model']:
            print(f"   1h: {rec['best_single_1h']['model']}")
            print(f"      PnL: {rec['best_single_1h']['pnl']:.2f}%, WR: {rec['best_single_1h']['wr']:.1f}%")
        if rec['best_single_15m']['model']:
            print(f"   15m: {rec['best_single_15m']['model']}")
            print(f"      PnL: {rec['best_single_15m']['pnl']:.2f}%, WR: {rec['best_single_15m']['wr']:.1f}%")
        
        # Лучшая реальная MTF комбинация
        print("\n🏆 ЛУЧШАЯ РЕАЛЬНАЯ MTF КОМБИНАЦИЯ:")
        if rec['best_mtf_actual']['model_1h']:
            print(f"   {rec['best_mtf_actual']['model_1h']} + {rec['best_mtf_actual']['model_15m']}")
            print(f"   PnL: {rec['best_mtf_actual']['pnl']:.2f}%, WR: {rec['best_mtf_actual']['wr']:.1f}%")
        
        # Корреляции
        print("\n🔗 КОРРЕЛЯЦИИ:")
        corr = rec['correlations']
        if corr:
            print(f"   Score correlation: {corr.get('score_correlation', 0):.3f}")
            print(f"   PnL sum correlation: {corr.get('pnl_sum_correlation', 0):.3f}")
            print(f"   PnL avg correlation: {corr.get('pnl_avg_correlation', 0):.3f}")
            print(f"   WR avg correlation: {corr.get('wr_avg_correlation', 0):.3f}")
        else:
            print("   Недостаточно данных для вычисления корреляций")
        
        # Топ предсказания
        print("\n🔮 ТОП-5 ПРЕДСКАЗАННЫХ КОМБИНАЦИЙ:")
        for i, pred in enumerate(rec['top_predictions'][:5], 1):
            print(f"   {i}. {pred['model_1h']} + {pred['model_15m']}")
            print(f"      Предсказанный score: {pred['predicted_score']:.2f}")
            if pred['actual_mtf_pnl'] is not None:
                print(f"      Реальный MTF PnL: {pred['actual_mtf_pnl']:.2f}%")
                print(f"      Ошибка предсказания: {pred.get('prediction_error', 0):.2f}")
            else:
                print(f"      ⚠️  Нет реальных данных MTF")
        
        # Проверка, попала ли лучшая реальная комбинация в топ предсказаний
        if rec['best_predicted_match_rank']:
            print(f"\n✅ Лучшая реальная MTF комбинация попала в топ предсказаний (ранг #{rec['best_predicted_match_rank']})")
        else:
            print(f"\n⚠️  Лучшая реальная MTF комбинация НЕ попала в топ-10 предсказаний")
        
        print()
    
    # Общие выводы
    print("=" * 100)
    print("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
    print("=" * 100)
    
    all_correlations = []
    for symbol in symbols:
        if symbol in recommendations and recommendations[symbol]['status'] == 'success':
            corr = recommendations[symbol]['correlations']
            if corr:
                all_correlations.append(corr)
    
    if all_correlations:
        avg_score_corr = np.mean([c.get('score_correlation', 0) for c in all_correlations])
        avg_pnl_corr = np.mean([c.get('pnl_avg_correlation', 0) for c in all_correlations])
        
        print(f"\n📊 Средние корреляции:")
        print(f"   Score correlation: {avg_score_corr:.3f}")
        print(f"   PnL avg correlation: {avg_pnl_corr:.3f}")
        
        if avg_score_corr > 0.5:
            print("\n✅ ВЫСОКАЯ КОРРЕЛЯЦИЯ: Можно использовать результаты одиночных моделей для предсказания MTF")
        elif avg_score_corr > 0.3:
            print("\n⚠️  СРЕДНЯЯ КОРРЕЛЯЦИЯ: Результаты одиночных моделей частично предсказывают MTF")
        else:
            print("\n❌ НИЗКАЯ КОРРЕЛЯЦИЯ: Результаты одиночных моделей слабо предсказывают MTF")
    
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("   1. Тестируйте топ-5 одиночных моделей каждого таймфрейма в комбинациях (25 комбинаций вместо всех)")
    print("   2. Если корреляция высокая (>0.5), можно пропустить тестирование худших одиночных моделей")
    print("   3. Обратите внимание на комбинации, где обе модели показывают хорошие результаты отдельно")
    print()


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ одиночных моделей vs MTF комбинаций")
    parser.add_argument("--mtf-btc", type=str, default="mtf_combinations_BTCUSDT_20260212_194950.csv",
                       help="Файл с результатами MTF для BTC")
    parser.add_argument("--mtf-eth", type=str, default="mtf_combinations_ETHUSDT_20260214_020745.csv",
                       help="Файл с результатами MTF для ETH")
    parser.add_argument("--single", type=str, default="ml_models_comparison_20260214_173015.csv",
                       help="Файл с результатами одиночных моделей")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT",
                       help="Символы для анализа (через запятую)")
    parser.add_argument("--output", type=str, default=None,
                       help="Файл для сохранения результатов (JSON)")
    
    args = parser.parse_args()
    
    # Загружаем данные
    print("📥 Загрузка данных...")
    df_single = load_single_results(Path(args.single))
    
    # Загружаем MTF результаты для каждого символа
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    mtf_files = {
        'BTCUSDT': Path(args.mtf_btc),
        'ETHUSDT': Path(args.mtf_eth),
    }
    
    df_mtf_all = []
    for symbol in symbols:
        if symbol in mtf_files and mtf_files[symbol].exists():
            df_mtf = load_mtf_results(mtf_files[symbol])
            df_mtf_all.append(df_mtf)
        else:
            print(f"⚠️  Файл MTF для {symbol} не найден: {mtf_files.get(symbol, 'N/A')}")
    
    if not df_mtf_all:
        print("❌ Не найдено файлов MTF результатов")
        return
    
    df_mtf = pd.concat(df_mtf_all, ignore_index=True)
    
    print(f"✅ Загружено {len(df_single)} одиночных моделей, {len(df_mtf)} MTF комбинаций")
    print()
    
    # Генерируем рекомендации
    print("🔍 Анализ данных...")
    recommendations = generate_recommendations(df_single, df_mtf, symbols)
    
    # Печатаем отчет
    print_analysis_report(recommendations, symbols)
    
    # Сохраняем результаты
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'recommendations': recommendations
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Результаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
