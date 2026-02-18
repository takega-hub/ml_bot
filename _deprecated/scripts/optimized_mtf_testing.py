"""
Оптимизированное тестирование MTF комбинаций.
Вместо тестирования всех комбинаций, выбирает топ-N моделей каждого таймфрейма
на основе результатов одиночных тестов и тестирует только их комбинации.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse

from backtest_mtf_strategy import run_mtf_backtest_all_combinations, find_all_models_for_symbol


def calculate_composite_score(row: pd.Series) -> float:
    """Вычисляет composite score для модели"""
    pnl = row.get('total_pnl_pct', 0)
    wr = row.get('win_rate_pct', row.get('win_rate', 0))
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


def load_single_results(comparison_file: Path) -> pd.DataFrame:
    """Загружает результаты одиночных моделей"""
    df = pd.read_csv(comparison_file)
    return df


def select_top_models(
    df_single: pd.DataFrame,
    symbol: str,
    timeframe: str,  # '1h' or '15m'
    top_n: int = 5
) -> List[str]:
    """
    Выбирает топ-N моделей для символа и таймфрейма на основе composite score.
    
    Returns:
        Список имен моделей (без .pkl)
    """
    symbol_data = df_single[df_single['symbol'] == symbol].copy()
    
    # Фильтруем по таймфрейму
    if timeframe == '1h':
        filtered = symbol_data[
            (symbol_data['mode_suffix'] == '1h') |
            (symbol_data['model_filename'].str.contains('_60_|_1h', na=False))
        ].copy()
    else:  # 15m
        filtered = symbol_data[
            (symbol_data['mode_suffix'] == '15m') |
            (symbol_data['model_filename'].str.contains('_15_|_15m', na=False))
        ].copy()
    
    if filtered.empty:
        return []
    
    # Вычисляем composite score
    filtered['composite_score'] = filtered.apply(calculate_composite_score, axis=1)
    
    # Сортируем по score
    filtered = filtered.sort_values('composite_score', ascending=False)
    
    # Берем топ-N
    top_models = filtered.head(top_n)
    
    # Извлекаем имена моделей (без .pkl)
    model_names = [name.replace('.pkl', '') for name in top_models['model_filename'].tolist()]
    
    return model_names


def find_model_paths(symbol: str, model_names: List[str], models_dir: Path = Path("ml_models")) -> List[str]:
    """Находит пути к моделям по их именам"""
    model_paths = []
    
    for model_name in model_names:
        # Пробуем разные варианты имени
        possible_names = [
            f"{model_name}.pkl",
            model_name,
        ]
        
        for name in possible_names:
            model_path = models_dir / name
            if model_path.exists():
                model_paths.append(str(model_path))
                break
    
    return model_paths


def test_optimized_mtf_combinations(
    symbol: str,
    comparison_file: Path,
    top_n: int = 5,
    days_back: int = 30,
    initial_balance: float = 100.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    confidence_threshold_1h: float = 0.50,
    confidence_threshold_15m: float = 0.35,
) -> pd.DataFrame:
    """
    Тестирует только топ-N комбинаций моделей на основе результатов одиночных тестов.
    """
    print("=" * 100)
    print(f"🚀 ОПТИМИЗИРОВАННОЕ ТЕСТИРОВАНИЕ MTF ДЛЯ {symbol}")
    print("=" * 100)
    print()
    
    # Загружаем результаты одиночных моделей
    print("📥 Загрузка результатов одиночных моделей...")
    df_single = load_single_results(comparison_file)
    print(f"✅ Загружено {len(df_single)} результатов")
    print()
    
    # Выбираем топ-модели
    print(f"🔍 Выбор топ-{top_n} моделей каждого таймфрейма...")
    top_1h = select_top_models(df_single, symbol, '1h', top_n)
    top_15m = select_top_models(df_single, symbol, '15m', top_n)
    
    if not top_1h:
        print(f"⚠️  Не найдено 1h моделей для {symbol}")
        print("   Попробуйте запустить тестирование 1h моделей:")
        print(f"   python compare_ml_models.py --symbols {symbol} --only-1h --interval 15m")
        return pd.DataFrame()
    
    if not top_15m:
        print(f"⚠️  Не найдено 15m моделей для {symbol}")
        print("   Попробуйте запустить тестирование 15m моделей:")
        print(f"   python compare_ml_models.py --symbols {symbol} --interval 15m")
        return pd.DataFrame()
    
    print(f"✅ Выбрано {len(top_1h)} 1h моделей и {len(top_15m)} 15m моделей")
    print(f"   Всего комбинаций для тестирования: {len(top_1h) * len(top_15m)}")
    print()
    
    print("📋 Топ-модели:")
    print("   1h модели:")
    for i, model in enumerate(top_1h, 1):
        print(f"      {i}. {model}")
    print("   15m модели:")
    for i, model in enumerate(top_15m, 1):
        print(f"      {i}. {model}")
    print()
    
    # Находим все модели для символа
    all_models_1h, all_models_15m = find_all_models_for_symbol(symbol)
    
    # Фильтруем только выбранные модели
    selected_1h = []
    selected_15m = []
    
    for model_path in all_models_1h:
        model_name = Path(model_path).stem
        # Проверяем, входит ли модель в топ
        for top_model in top_1h:
            if top_model in model_name or model_name in top_model:
                selected_1h.append(model_path)
                break
    
    for model_path in all_models_15m:
        model_name = Path(model_path).stem
        # Проверяем, входит ли модель в топ
        for top_model in top_15m:
            if top_model in model_name or model_name in top_model:
                selected_15m.append(model_path)
                break
    
    if not selected_1h or not selected_15m:
        print("❌ Не удалось найти файлы выбранных моделей")
        print(f"   Найдено 1h моделей: {len(selected_1h)}")
        print(f"   Найдено 15m моделей: {len(selected_15m)}")
        return pd.DataFrame()
    
    print(f"✅ Найдено файлов: {len(selected_1h)} 1h моделей, {len(selected_15m)} 15m моделей")
    print()
    
    # Тестируем комбинации
    print("🧪 Начало тестирования комбинаций...")
    print()
    
    results = []
    
    for model_1h_path in selected_1h:
        for model_15m_path in selected_15m:
            model_1h_name = Path(model_1h_path).name
            model_15m_name = Path(model_15m_path).name
            
            print(f"   Тестирование: {model_1h_name} + {model_15m_name}")
            
            try:
                from backtest_mtf_strategy import run_mtf_backtest
                
                metrics = run_mtf_backtest(
                    symbol=symbol,
                    days_back=days_back,
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                    leverage=leverage,
                    model_1h_path=str(model_1h_path),
                    model_15m_path=str(model_15m_path),
                    confidence_threshold_1h=confidence_threshold_1h,
                    confidence_threshold_15m=confidence_threshold_15m,
                    alignment_mode="strict",
                    require_alignment=True,
                )
                
                if metrics:
                    results.append({
                        'model_1h': model_1h_name,
                        'model_15m': model_15m_name,
                        'symbol': symbol,
                        'total_trades': metrics.total_trades,
                        'winning_trades': metrics.winning_trades,
                        'losing_trades': metrics.losing_trades,
                        'win_rate': metrics.win_rate,
                        'total_pnl': metrics.total_pnl,
                        'total_pnl_pct': metrics.total_pnl_pct,
                        'avg_win': metrics.avg_win,
                        'avg_loss': metrics.avg_loss,
                        'profit_factor': metrics.profit_factor,
                        'max_drawdown_pct': metrics.max_drawdown_pct,
                        'sharpe_ratio': metrics.sharpe_ratio,
                    })
                    print(f"      ✅ PnL: {metrics.total_pnl_pct:.2f}%, WR: {metrics.win_rate:.1f}%")
                else:
                    print(f"      ⚠️  Нет результатов")
                    
            except Exception as e:
                print(f"      ❌ Ошибка: {str(e)[:100]}")
    
    if not results:
        print("❌ Нет результатов тестирования")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('total_pnl_pct', ascending=False)
    
    print()
    print("=" * 100)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 100)
    print(f"Протестировано комбинаций: {len(results)}")
    print(f"Лучшая комбинация: {df_results.iloc[0]['model_1h']} + {df_results.iloc[0]['model_15m']}")
    print(f"   PnL: {df_results.iloc[0]['total_pnl_pct']:.2f}%, WR: {df_results.iloc[0]['win_rate']:.1f}%")
    print()
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Оптимизированное тестирование MTF комбинаций"
    )
    parser.add_argument("--symbols", type=str, required=True,
                       help="Символы для тестирования (через запятую)")
    parser.add_argument("--comparison-file", type=str,
                       default="ml_models_comparison_20260214_111828.csv",
                       help="Файл с результатами сравнения одиночных моделей")
    parser.add_argument("--top-n", type=int, default=5,
                       help="Количество топ-моделей каждого таймфрейма (по умолчанию 5)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для бэктеста")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                       help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    comparison_file = Path(args.comparison_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = []
    
    for symbol in symbols:
        print()
        df_results = test_optimized_mtf_combinations(
            symbol=symbol,
            comparison_file=comparison_file,
            top_n=args.top_n,
            days_back=args.days,
        )
        
        if not df_results.empty:
            # Сохраняем результаты
            filename = output_dir / f"optimized_mtf_{symbol}_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"💾 Результаты сохранены в {filename}")
            print()
            
            all_results.append(df_results)
    
    # Сводный отчет
    if all_results:
        print("=" * 100)
        print("📊 СВОДНЫЙ ОТЧЕТ")
        print("=" * 100)
        print()
        
        for symbol in symbols:
            symbol_results = [df for df in all_results if not df.empty and df.iloc[0]['symbol'] == symbol]
            if symbol_results:
                df = symbol_results[0]
                print(f"{symbol}:")
                print(f"   Лучшая комбинация: {df.iloc[0]['model_1h']} + {df.iloc[0]['model_15m']}")
                print(f"   PnL: {df.iloc[0]['total_pnl_pct']:.2f}%, WR: {df.iloc[0]['win_rate']:.1f}%")
                print()


if __name__ == "__main__":
    main()
