"""
Скрипт для массового тестирования ВСЕХ ML моделей по каждому символу.

Запускает бэктест (через backtest_ml_strategy.run_ml_backtest) для всех моделей в
директории ml_models и формирует сводную таблицу с результатами.

Улучшения:
1. Параллельное выполнение тестов для ускорения
2. Прогресс-бар для отслеживания выполнения
3. Дополнительные метрики анализа
4. Визуализация результатов
5. Проверка на переобучение

Использование:
    python compare_ml_models.py

Опции:
    --days 30           # Сколько дней тестировать (по умолчанию 30)
    --symbols BTCUSDT,ETHUSDT,SOLUSDT  # Ограничить список символов
    --models-dir ml_models             # Путь к директории с моделями
    --output all                        # Сохранить результаты (csv, plots, all)
    --workers 4                         # Количество параллельных процессов
    --check-overfitting                 # Проверить модели на переобучение
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from functools import partial
import concurrent.futures
import traceback
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Настройка для лучшего отображения графиков
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-v0_8-darkgrid')

# Устанавливаем backend для matplotlib чтобы избежать GUI проблем в multiprocessing
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend

try:
    from backtest_ml_strategy import run_exact_backtest as run_ml_backtest, BacktestMetrics
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что модуль backtest_ml_strategy доступен для импорта")
    sys.exit(1)


def find_models_for_symbol(models_dir: Path, symbol: str) -> List[Path]:
    """
    Ищет все ML модели для указанного символа.
    
    Ожидаемый формат имени файла:
        {model_type}_{SYMBOL}_{INTERVAL}.pkl
        {model_type}_{SYMBOL}_{INTERVAL}_{mode_suffix}.pkl  # mtf / 15m
    
    Примеры:
        ensemble_BTCUSDT_15.pkl
        ensemble_BTCUSDT_15_mtf.pkl
        quad_ensemble_ETHUSDT_15_15m.pkl
    """
    if not models_dir.exists():
        print(f"⚠️  Директория {models_dir} не существует")
        return []
    
    patterns = [
        f"*_{symbol}_*.pkl",
        f"*{symbol}*.pkl",  # Более широкий паттерн
    ]
    
    results: List[Path] = []
    for pattern in patterns:
        for f in models_dir.glob(pattern):
            if f.is_file() and f not in results:
                results.append(f)
    
    # Убираем дубликаты и сортируем по имени
    results = sorted(list({f.resolve() for f in results}))
    return results


def metrics_to_dict(m: BacktestMetrics, model_path: Path) -> Dict[str, Any]:
    """Преобразует BacktestMetrics в словарь для удобного сохранения/анализа."""
    if m is None:
        return {}
    
    # Извлекаем информацию из имени файла
    filename = model_path.name
    name_no_ext = filename.replace(".pkl", "")
    parts = name_no_ext.split("_")
    
    model_type = parts[0] if parts else "unknown"
    mode_suffix = None
    if len(parts) >= 4:
        mode_suffix = parts[-1]  # mtf / 15m / др.
    
    # Базовые метрики
    result = {
        "symbol": m.symbol,
        "model_name": m.model_name,
        "model_filename": filename,
        "model_path": str(model_path),
        "model_type": model_type,
        "mode_suffix": mode_suffix or "",
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "win_rate_pct": m.win_rate,
        "total_pnl_usd": m.total_pnl,
        "total_pnl_pct": m.total_pnl_pct,
        "profit_factor": m.profit_factor,
        "max_drawdown_usd": m.max_drawdown,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "long_trades": m.long_signals,
        "short_trades": m.short_signals,
        "avg_trade_duration_hours": m.avg_trade_duration_hours,
        "avg_win_usd": m.avg_win,
        "avg_loss_usd": m.avg_loss,
        "best_trade_usd": m.best_trade_pnl,
        "worst_trade_usd": m.worst_trade_pnl,
        "largest_win_usd": m.largest_win,
        "largest_loss_usd": m.largest_loss,
        "consecutive_wins": m.consecutive_wins,
        "consecutive_losses": m.consecutive_losses,
        "avg_confidence": m.avg_confidence,
    }
    
    # Рассчитываем дополнительные метрики
    if m.total_trades > 0:
        result["trades_per_day"] = m.total_trades / (args.days if 'args' in globals() else 30)
        result["expectancy_usd"] = (m.win_rate/100 * m.avg_win) - ((1 - m.win_rate/100) * abs(m.avg_loss))
    
    return result


def test_single_model(args_tuple: Tuple) -> Optional[Dict[str, Any]]:
    """
    Функция для тестирования одной модели.
    Используется для параллельного выполнения.
    """
    model_path, symbol, days, interval, initial_balance, risk_per_trade, leverage = args_tuple
    
    try:
        # Импортируем необходимые модули внутри функции для избежания проблем с pickling
        import sys
        import os
        # Устанавливаем backend для matplotlib чтобы избежать GUI проблем
        import matplotlib
        matplotlib.use('Agg')  # Используем non-interactive backend
        
        # Импортируем функции локально
        from backtest_ml_strategy import run_ml_backtest, BacktestMetrics
        
        metrics = run_ml_backtest(
            model_path=str(model_path),
            symbol=symbol,
            days_back=days,
            interval=interval,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            leverage=leverage,
        )
        
        if metrics is None:
            return None
        
        # Преобразуем метрики в словарь
        def metrics_to_dict_local(m, model_path) -> Dict[str, Any]:
            """Преобразует BacktestMetrics в словарь для удобного сохранения/анализа."""
            if m is None:
                return {}
            
            # Извлекаем информацию из имени файла
            filename = model_path.name
            name_no_ext = filename.replace(".pkl", "")
            parts = name_no_ext.split("_")
            
            model_type = parts[0] if parts else "unknown"
            mode_suffix = None
            if len(parts) >= 4:
                mode_suffix = parts[-1]  # mtf / 15m / др.
            
            # Базовые метрики
            result = {
                "symbol": getattr(m, 'symbol', ''),
                "model_name": getattr(m, 'model_name', ''),
                "model_filename": filename,
                "model_path": str(model_path),
                "model_type": model_type,
                "mode_suffix": mode_suffix or "",
                "total_trades": getattr(m, 'total_trades', 0),
                "winning_trades": getattr(m, 'winning_trades', 0),
                "losing_trades": getattr(m, 'losing_trades', 0),
                "win_rate_pct": getattr(m, 'win_rate', 0.0),
                "total_pnl_usd": getattr(m, 'total_pnl', 0.0),
                "total_pnl_pct": getattr(m, 'total_pnl_pct', 0.0),
                "profit_factor": getattr(m, 'profit_factor', 0.0),
                "max_drawdown_usd": getattr(m, 'max_drawdown', 0.0),
                "max_drawdown_pct": getattr(m, 'max_drawdown_pct', 0.0),
                "sharpe_ratio": getattr(m, 'sharpe_ratio', 0.0),
                "long_trades": getattr(m, 'long_signals', 0),
                "short_trades": getattr(m, 'short_signals', 0),
                "avg_trade_duration_hours": getattr(m, 'avg_trade_duration_hours', 0.0),
                "avg_win_usd": getattr(m, 'avg_win', 0.0),
                "avg_loss_usd": getattr(m, 'avg_loss', 0.0),
                "best_trade_usd": getattr(m, 'best_trade_pnl', 0.0),
                "worst_trade_usd": getattr(m, 'worst_trade_pnl', 0.0),
                "largest_win_usd": getattr(m, 'largest_win', 0.0),
                "largest_loss_usd": getattr(m, 'largest_loss', 0.0),
                "consecutive_wins": getattr(m, 'consecutive_wins', 0),
                "consecutive_losses": getattr(m, 'consecutive_losses', 0),
                "avg_confidence": getattr(m, 'avg_confidence', 0.0),
            }
            
            return result
        
        return metrics_to_dict_local(metrics, model_path)
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании {model_path.name}: {str(e)[:100]}")
        # Возвращаем специальный маркер ошибки
        return {"error": True, "model": model_path.name, "message": str(e)[:100]}


def compare_models(
    symbols: List[str],
    models_dir: Path,
    days: int = 30,
    interval: str = "15m",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    workers: int = 4,
    check_overfitting: bool = False,
) -> pd.DataFrame:
    """
    Запускает бэктест для всех моделей и возвращает DataFrame с результатами.
    Поддерживает параллельное выполнение.
    """
    all_results: List[Dict[str, Any]] = []
    
    print("=" * 80)
    print("🚀 ML MODELS COMPARISON BACKTEST (PARALLEL)")
    print("=" * 80)
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"📁 Models dir: {models_dir}")
    print(f"⚙️  Days: {days}, Interval: {interval}")
    print(f"💰 Initial balance: ${initial_balance:.2f}")
    print(f"🎯 Risk per trade: {risk_per_trade*100:.1f}%, Leverage: {leverage}x")
    print(f"⚡ Workers: {workers}")
    print("=" * 80)
    
    # Подготовка аргументов для всех моделей
    test_args = []
    total_models = 0
    
    for symbol in symbols:
        models = find_models_for_symbol(models_dir, symbol)
        if not models:
            print(f"⚠️  No models found for {symbol}")
            continue
        
        total_models += len(models)
        print(f"📦 Found {len(models)} models for {symbol}")
        
        for model_path in models:
            test_args.append((
                model_path, symbol, days, interval, 
                initial_balance, risk_per_trade, leverage
            ))
    
    if not test_args:
        print("❌ No models to test")
        return pd.DataFrame()
    
    print(f"\n🎯 Total models to test: {total_models}")
    
    # Параллельное выполнение тестов
    print("\n⚡ Running parallel backtests...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Используем tqdm для отображения прогресса
            results = list(tqdm(
                executor.map(test_single_model, test_args),
                total=len(test_args),
                desc="Testing models",
                unit="model",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            ))
    except concurrent.futures.process.BrokenProcessPool as e:
        print(f"❌ Multiprocessing error: {e}")
        print("🔄 Retrying with sequential execution...")
        # Аварийный переход на последовательное выполнение
        results = []
        for args in tqdm(test_args, desc="Testing models (sequential)", unit="model"):
            results.append(test_single_model(args))
    
    # Собираем результаты
    successful = 0
    errors = 0
    for result in results:
        if result is not None and len(result) > 0:
            # Проверяем, является ли результат ошибкой
            if isinstance(result, dict) and result.get("error"):
                print(f"⚠️  Model test failed: {result.get('model', 'Unknown')} - {result.get('message', 'Unknown error')}")
                errors += 1
            else:
                all_results.append(result)
                successful += 1
        else:
            # Пустой результат
            errors += 1
    
    print(f"\n✅ Successfully tested: {successful}/{total_models} models")
    if errors > 0:
        print(f"⚠️  Errors: {errors}/{total_models} models")
    
    if not all_results:
        print("❌ No results collected.")
        return pd.DataFrame()
    
    # Создаем DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Добавляем дополнительные метрики
    df_results = calculate_additional_metrics(df_results, days)
    
    # Проверка на переобучение
    if check_overfitting and len(df_results) > 0:
        df_results = add_overfitting_check(df_results, models_dir, days, interval, 
                                          initial_balance, risk_per_trade, leverage)
    
    # Сортировка: по символу, затем по total_pnl_pct (убывание)
    df_results.sort_values(
        by=["symbol", "total_pnl_pct", "win_rate_pct"],
        ascending=[True, False, False],
        inplace=True,
    )
    
    # Сброс индекса
    df_results.reset_index(drop=True, inplace=True)
    
    return df_results


def calculate_additional_metrics(df_results: pd.DataFrame, days: int) -> pd.DataFrame:
    """Добавляет дополнительные метрики для анализа."""
    if df_results.empty:
        return df_results
    
    # Коэффициент Кальмара
    df_results['calmar_ratio'] = df_results['total_pnl_pct'] / abs(df_results['max_drawdown_pct']).replace(0, 0.001)
    
    # Ожидание (expectancy) если не было рассчитано
    if 'expectancy_usd' not in df_results.columns:
        df_results['expectancy_usd'] = (
            (df_results['win_rate_pct']/100 * df_results['avg_win_usd']) - 
            ((1 - df_results['win_rate_pct']/100) * abs(df_results['avg_loss_usd']))
        )
    
    # Годовая доходность (аппроксимация)
    df_results['annualized_return_pct'] = df_results['total_pnl_pct'] * (365 / days)
    
    # Скорость сделок
    df_results['trades_per_day'] = df_results['total_trades'] / days
    
    # Коэффициент восстановления
    df_results['recovery_factor'] = df_results['total_pnl_usd'] / abs(df_results['max_drawdown_usd']).replace(0, 0.001)
    
    # Рейтинг модели (композитный показатель)
    df_results['composite_score'] = (
        df_results['win_rate_pct'].fillna(0) * 0.2 +
        df_results['profit_factor'].fillna(0) * 0.3 +
        df_results['sharpe_ratio'].fillna(0) * 0.2 +
        df_results['calmar_ratio'].fillna(0) * 0.3
    )
    
    # Категория риска
    def risk_category(row):
        if row['max_drawdown_pct'] < 5:
            return 'Low'
        elif row['max_drawdown_pct'] < 15:
            return 'Medium'
        else:
            return 'High'
    
    df_results['risk_category'] = df_results.apply(risk_category, axis=1)
    
    return df_results


def add_overfitting_check(df_results: pd.DataFrame, models_dir: Path, days: int,
                         interval: str, initial_balance: float, 
                         risk_per_trade: float, leverage: int) -> pd.DataFrame:
    """
    Проверяет модели на переобучение, сравнивая результаты
    на первой и второй половине периода тестирования.
    """
    print("\n🔍 Checking for overfitting...")
    
    overfitting_results = []
    
    # Используем последовательное выполнение для проверки переобучения
    # чтобы избежать проблем с multiprocessing
    for _, row in tqdm(df_results.iterrows(), total=len(df_results), desc="Overfitting check"):
        try:
            # Тест на первой половине периода
            metrics_first = run_ml_backtest(
                model_path=row['model_path'],
                symbol=row['symbol'],
                days_back=days // 2,
                interval=interval,
                initial_balance=initial_balance,
                risk_per_trade=risk_per_trade,
                leverage=leverage,
            )
            
            # Тест на второй половине периода
            metrics_second = run_ml_backtest(
                model_path=row['model_path'],
                symbol=row['symbol'],
                days_back=days,
                interval=interval,
                initial_balance=initial_balance,
                risk_per_trade=risk_per_trade,
                leverage=leverage,
                start_offset=days // 2,
            )
            
            if metrics_first and metrics_second:
                pnl_diff = abs(metrics_first.total_pnl_pct - metrics_second.total_pnl_pct)
                winrate_diff = abs(metrics_first.win_rate - metrics_second.win_rate)
                
                overfitting_results.append({
                    'model_name': row['model_name'],
                    'symbol': row['symbol'],
                    'pnl_first_half': metrics_first.total_pnl_pct,
                    'pnl_second_half': metrics_second.total_pnl_pct,
                    'pnl_difference': pnl_diff,
                    'winrate_difference': winrate_diff,
                    'is_overfit': pnl_diff > 20 or winrate_diff > 15,  # Пороговые значения
                })
                
        except Exception as e:
            print(f"⚠️  Overfitting check failed for {row['model_name']}: {str(e)[:50]}...")
            continue
    
    if overfitting_results:
        overfit_df = pd.DataFrame(overfitting_results)
        # Сохраняем отдельно
        overfit_output = f"overfitting_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        overfit_df.to_csv(overfit_output, index=False)
        print(f"💾 Overfitting check saved to: {overfit_output}")
        
        # Добавляем флаг переобучения в основной DataFrame
        overfit_dict = {row['model_name']: row['is_overfit'] for _, row in overfit_df.iterrows()}
        df_results['is_overfit'] = df_results['model_name'].map(overfit_dict).fillna(False)
    
    return df_results


def print_summary_table(df_results: pd.DataFrame) -> None:
    """Печатает компактную сводную таблицу по каждому символу."""
    if df_results.empty:
        print("❌ No results to display.")
        return
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY: BEST MODELS PER SYMBOL")
    print("=" * 80)
    
    for symbol, group in df_results.groupby("symbol"):
        print(f"\n📈 {symbol} (Top 5 by PnL%):")
        print("-" * 80)
        
        # Берём top-5 по PnL%
        top = group.head(5).copy()
        
        # Форматируем вывод
        display_cols = [
            "model_name", "model_type", "mode_suffix",
            "total_trades", "win_rate_pct", "total_pnl_pct",
            "profit_factor", "max_drawdown_pct", "sharpe_ratio",
            "composite_score", "risk_category"
        ]
        
        # Фильтруем только существующие колонки
        existing_cols = [col for col in display_cols if col in top.columns]
        
        display_df = top[existing_cols].copy()
        
        # Форматирование чисел
        formatters = {
            'win_rate_pct': '{:.1f}%'.format,
            'total_pnl_pct': '{:+.1f}%'.format,
            'profit_factor': '{:.2f}'.format,
            'max_drawdown_pct': '{:.1f}%'.format,
            'sharpe_ratio': '{:.2f}'.format,
            'composite_score': '{:.2f}'.format,
        }
        
        # Применяем форматирование
        for col, fmt in formatters.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(fmt)
        
        print(display_df.to_string(index=False))
        
        # Показываем статистику по всем моделям символа
        print(f"\n📊 Statistics for {symbol}:")
        print(f"   Models tested: {len(group)}")
        print(f"   Avg PnL%: {group['total_pnl_pct'].mean():.1f}%")
        print(f"   Best PnL%: {group['total_pnl_pct'].max():.1f}% ({group.loc[group['total_pnl_pct'].idxmax(), 'model_name']})")
        print(f"   Avg Win Rate: {group['win_rate_pct'].mean():.1f}%")
        print(f"   Profitable models: {(group['total_pnl_pct'] > 0).sum()}/{len(group)}")


def create_visualizations(df_results: pd.DataFrame, output_dir: str = "comparison_plots") -> None:
    """Создает визуализации для анализа результатов."""
    if df_results.empty:
        print("⚠️  No data for visualizations")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n🎨 Creating visualizations in '{output_dir}'...")
    
    # 1. Heatmap доходности по символам и моделям
    try:
        plt.figure(figsize=(14, 10))
        pivot_table = df_results.pivot_table(
            index='model_name', 
            columns='symbol', 
            values='total_pnl_pct',
            aggfunc='first'
        )
        
        # Ограничиваем количество моделей для лучшей читаемости
        if len(pivot_table) > 20:
            # Берем топ и худшие модели
            model_scores = df_results.groupby('model_name')['total_pnl_pct'].mean()
            top_models = model_scores.nlargest(15).index.tolist()
            bottom_models = model_scores.nsmallest(5).index.tolist()
            selected_models = top_models + bottom_models
            pivot_table = pivot_table.loc[selected_models]
        
        sns.heatmap(pivot_table, 
                   annot=True, 
                   fmt=".1f", 
                   cmap="RdYlGn", 
                   center=0,
                   linewidths=0.5,
                   cbar_kws={'label': 'PnL %'})
        
        plt.title("Model Performance by Symbol (PnL %)", fontsize=16, pad=20)
        plt.xlabel("Symbol", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_pnl.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: heatmap_pnl.png")
    except Exception as e:
        print(f"⚠️  Could not create heatmap: {e}")
    
    # 2. Scatter plot: риск vs доходность
    try:
        plt.figure(figsize=(12, 8))
        
        # Разные цвета для разных символов
        symbols = df_results['symbol'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
        
        for symbol, color in zip(symbols, colors):
            subset = df_results[df_results['symbol'] == symbol]
            plt.scatter(
                subset['max_drawdown_pct'], 
                subset['total_pnl_pct'],
                label=symbol,
                alpha=0.7,
                s=100,
                color=color,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.xlabel('Max Drawdown (%)', fontsize=12)
        plt.ylabel('Total PnL (%)', fontsize=12)
        plt.title('Risk-Return Profile', fontsize=16, pad=20)
        plt.legend(title='Symbol')
        plt.grid(True, alpha=0.3)
        
        # Добавляем лучшие модели аннотациями
        top_models = df_results.nlargest(5, 'total_pnl_pct')
        for _, row in top_models.iterrows():
            plt.annotate(
                f"{row['model_name'][:15]}...",
                xy=(row['max_drawdown_pct'], row['total_pnl_pct']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_return.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: risk_return.png")
    except Exception as e:
        print(f"⚠️  Could not create risk-return plot: {e}")
    
    # 3. Bar plot: лучшие модели по символу
    try:
        fig, axes = plt.subplots(len(symbols), 1, figsize=(14, 5*len(symbols)))
        if len(symbols) == 1:
            axes = [axes]
        
        for idx, symbol in enumerate(symbols):
            subset = df_results[df_results['symbol'] == symbol].head(10)
            ax = axes[idx]
            
            bars = ax.barh(
                range(len(subset)),
                subset['total_pnl_pct'],
                color=['green' if x > 0 else 'red' for x in subset['total_pnl_pct']],
                edgecolor='black'
            )
            
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels([f"{row['model_type']} ({row['mode_suffix']})" 
                               for _, row in subset.iterrows()])
            ax.set_xlabel('PnL %')
            ax.set_title(f'Top 10 Models for {symbol}', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Добавляем значения на бары
            for bar, pnl in zip(bars, subset['total_pnl_pct']):
                width = bar.get_width()
                ax.text(width + (0.5 if width >= 0 else -2), 
                       bar.get_y() + bar.get_height()/2,
                       f'{pnl:.1f}%',
                       ha='left' if width >= 0 else 'right',
                       va='center',
                       fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_models.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: top_models.png")
    except Exception as e:
        print(f"⚠️  Could not create top models plot: {e}")
    
    # 4. Корреляционная матрица метрик
    try:
        plt.figure(figsize=(12, 10))
        
        # Выбираем числовые колонки для корреляции
        numeric_cols = df_results.select_dtypes(include=[np.number]).columns
        # Ограничиваем до важных метрик
        important_metrics = ['total_pnl_pct', 'win_rate_pct', 'profit_factor', 
                           'max_drawdown_pct', 'sharpe_ratio', 'calmar_ratio',
                           'total_trades', 'avg_confidence']
        corr_cols = [col for col in important_metrics if col in numeric_cols]
        
        if len(corr_cols) > 1:
            corr_matrix = df_results[corr_cols].corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt=".2f", 
                       cmap="coolwarm", 
                       center=0,
                       square=True,
                       cbar_kws={'label': 'Correlation'},
                       linewidths=0.5)
            
            plt.title("Correlation Matrix of Performance Metrics", fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Created: correlation_matrix.png")
    except Exception as e:
        print(f"⚠️  Could not create correlation matrix: {e}")
    
    print(f"\n🎨 All visualizations saved to '{output_dir}/' directory")


def save_detailed_report(df_results: pd.DataFrame, args, output_dir: str = "reports") -> None:
    """Сохраняет детальный отчет в JSON формате."""
    if df_results.empty:
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сохраняем параметры запуска
    report = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "days": args.days,
            "symbols": args.symbols,
            "models_dir": str(args.models_dir),
            "interval": args.interval,
            "initial_balance": args.balance,
            "risk_per_trade": args.risk,
            "leverage": args.leverage,
            "workers": args.workers,
            "check_overfitting": args.check_overfitting,
        },
        "summary_statistics": {
            "total_models_tested": len(df_results),
            "profitable_models": int((df_results['total_pnl_pct'] > 0).sum()),
            "avg_pnl_pct": float(df_results['total_pnl_pct'].mean()),
            "avg_win_rate": float(df_results['win_rate_pct'].mean()),
            "best_model": df_results.iloc[0].to_dict() if len(df_results) > 0 else None,
        },
        "best_models_per_symbol": {},
        "recommendations": []
    }
    
    # Лучшие модели по каждому символу
    for symbol in df_results['symbol'].unique():
        symbol_df = df_results[df_results['symbol'] == symbol]
        best_model = symbol_df.iloc[0].to_dict()
        report["best_models_per_symbol"][symbol] = best_model
    
    # Формируем рекомендации
    profitable_df = df_results[df_results['total_pnl_pct'] > 0]
    if len(profitable_df) > 0:
        # Рекомендуем модели с хорошим балансом доходности и риска
        good_models = profitable_df[
            (profitable_df['max_drawdown_pct'] < 15) &
            (profitable_df['win_rate_pct'] > 50) &
            (profitable_df['profit_factor'] > 1.2)
        ]
        
        if len(good_models) > 0:
            for _, row in good_models.head(5).iterrows():
                report["recommendations"].append({
                    "model": row['model_name'],
                    "symbol": row['symbol'],
                    "pnl_pct": float(row['total_pnl_pct']),
                    "win_rate": float(row['win_rate_pct']),
                    "max_dd": float(row['max_drawdown_pct']),
                    "reason": "Good balance of profitability and risk management"
                })
    
    # Сохраняем отчет
    report_file = f"{output_dir}/detailed_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📋 Detailed report saved to: {report_file}")


def main():
    global args  # Делаем args глобальной для использования в других функциях
    
    parser = argparse.ArgumentParser(
        description="Compare all ML models via parallel backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Базовая команда
  python compare_ml_models.py
  
  # Расширенное тестирование
  python compare_ml_models.py --days 60 --symbols BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT 
  
  # С проверкой переобучения и 8 процессами
  python compare_ml_models.py --check-overfitting --workers 8 --output all
  
  # Тестирование с низким риском
  python compare_ml_models.py --risk 0.01 --leverage 5 --balance 5000
        """
    )
    
    parser.add_argument("--days", type=int, default=30, 
                       help="Days to backtest (default: 30)")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="Comma-separated list of symbols (default: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="ml_models",
        help="Directory with ML models (default: ml_models)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Timeframe interval (default: 15m)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Initial balance (default: 1000.0)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.02,
        help="Risk per trade fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=10,
        help="Leverage (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--check-overfitting",
        action="store_true",
        help="Check models for overfitting (slower but more thorough)",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["none", "csv", "plots", "all"],
        default="csv",
        help="Output options: none, csv, plots, all (default: csv)",
    )
    
    args = parser.parse_args()
    
    # Преобразуем аргументы
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    models_dir = Path(args.models_dir)
    
    # Проверяем существование директории с моделями
    if not models_dir.exists():
        print(f"❌ Директория с моделями не существует: {models_dir}")
        print(f"   Текущая рабочая директория: {Path.cwd()}")
        return
    
    # Запускаем сравнение моделей
    try:
        df_results = compare_models(
            symbols=symbols,
            models_dir=models_dir,
            days=args.days,
            interval=args.interval,
            initial_balance=args.balance,
            risk_per_trade=args.risk,
            leverage=args.leverage,
            workers=args.workers,
            check_overfitting=args.check_overfitting,
        )
    except Exception as e:
        print(f"❌ Fatal error during model comparison: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if df_results.empty:
        print("❌ No results to analyze")
        return
    
    # Печатаем сводку
    print_summary_table(df_results)
    
    # Сохраняем результаты в зависимости от опций
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output in ["csv", "all"]:
        csv_name = f"ml_models_comparison_{timestamp}.csv"
        df_results.to_csv(csv_name, index=False, encoding='utf-8')
        print(f"\n💾 Full comparison table saved to: {csv_name}")
        print(f"   Rows: {len(df_results)}, Columns: {len(df_results.columns)}")
    
    if args.output in ["plots", "all"]:
        plots_dir = f"comparison_plots_{timestamp}"
        create_visualizations(df_results, plots_dir)
    
    if args.output == "all":
        save_detailed_report(df_results, args, "reports")
    
    # Выводим итоговую статистику
    print("\n" + "=" * 80)
    print("🎯 FINAL STATISTICS")
    print("=" * 80)
    print(f"📈 Total models tested: {len(df_results)}")
    print(f"✅ Profitable models: {(df_results['total_pnl_pct'] > 0).sum()} ({df_results['total_pnl_pct'].gt(0).mean()*100:.1f}%)")
    print(f"📊 Average PnL%: {df_results['total_pnl_pct'].mean():.2f}%")
    print(f"🎯 Average Win Rate: {df_results['win_rate_pct'].mean():.2f}%")
    
    # Лучшая модель
    best_model = df_results.iloc[0]
    print(f"\n🏆 BEST OVERALL MODEL:")
    print(f"   Name: {best_model['model_name']}")
    print(f"   Symbol: {best_model['symbol']}")
    print(f"   PnL%: {best_model['total_pnl_pct']:.2f}%")
    print(f"   Win Rate: {best_model['win_rate_pct']:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()