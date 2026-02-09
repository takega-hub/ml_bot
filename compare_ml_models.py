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
6. Сравнение с предыдущими результатами (до/после переобучения)
7. Детальный анализ сигналов, качества TP/SL, конверсии
8. Анализ распределения сигналов (LONG/SHORT/HOLD)

Использование:
    # Базовое использование
    python compare_ml_models.py
    
    # После переобучения - сравнение с предыдущими результатами
    python compare_ml_models.py --compare-with ml_models_comparison_20260205_120000.csv --detailed-analysis
    
    # Полный анализ с визуализацией
    python compare_ml_models.py --output all --detailed-analysis --check-overfitting

Опции:
    --days 30                           # Сколько дней тестировать (по умолчанию 30)
    --symbols auto                      # Автоматически найти все символы из моделей (по умолчанию)
    --symbols BTCUSDT,ETHUSDT,SOLUSDT   # Или указать конкретные символы
    --models-dir ml_models              # Путь к директории с моделями
    --output all                        # Сохранить результаты (csv, plots, all)
    --workers 4                         # Количество параллельных процессов
    --check-overfitting                 # Проверить модели на переобучение
    --compare-with <file.csv>          # Сравнить с предыдущими результатами
    --detailed-analysis                 # Запустить детальный анализ
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


def find_all_symbols(models_dir: Path) -> List[str]:
    """
    Автоматически находит все символы из имен файлов моделей.
    
    Ожидаемый формат имени файла:
        {model_type}_{SYMBOL}_{INTERVAL}.pkl
        {model_type}_{SYMBOL}_{INTERVAL}_{mode_suffix}.pkl  # mtf / 15m
    
    Примеры:
        ensemble_BTCUSDT_15.pkl
        ensemble_BTCUSDT_15_mtf.pkl
        quad_ensemble_ETHUSDT_15_15m.pkl
    """
    if not models_dir.exists():
        return []
    
    symbols = set()
    
    # Ищем все .pkl файлы
    for model_file in models_dir.glob("*.pkl"):
        name = model_file.stem  # Имя без расширения
        
        # Пытаемся извлечь символ из имени файла
        # Формат: {model_type}_{SYMBOL}_{...}
        parts = name.split("_")
        
        if len(parts) >= 2:
            # Пробуем найти известные символы в частях имени
            for part in parts:
                part_upper = part.upper()
                # Проверяем, является ли часть известным символом (заканчивается на USDT)
                if part_upper.endswith("USDT") and len(part_upper) >= 6:
                    symbols.add(part_upper)
                    break
    
    return sorted(list(symbols))


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
    
    # Добавляем все доступные метрики из BacktestMetrics
    additional_metrics = {
        "sortino_ratio": getattr(m, 'sortino_ratio', 0.0),
        "calmar_ratio": getattr(m, 'calmar_ratio', 0.0),
        "total_signals": getattr(m, 'total_signals', 0),
        "avg_mfe": getattr(m, 'avg_mfe', 0.0),
        "avg_mae": getattr(m, 'avg_mae', 0.0),
        "mfe_mae_ratio": getattr(m, 'mfe_mae_ratio', 0.0),
        "var_95": getattr(m, 'var_95', 0.0),
        "cvar_95": getattr(m, 'cvar_95', 0.0),
        "recovery_factor": getattr(m, 'recovery_factor', 0.0),
        "expectancy_usd": getattr(m, 'expectancy_usd', 0.0),
        "risk_reward_ratio": getattr(m, 'risk_reward_ratio', 0.0),
        "trade_frequency_per_day": getattr(m, 'trade_frequency_per_day', 0.0),
        "profitable_days_pct": getattr(m, 'profitable_days_pct', 0.0),
        "ulcer_index": getattr(m, 'ulcer_index', 0.0),
        "kelly_criterion": getattr(m, 'kelly_criterion', 0.0),
        "avg_tp_distance_pct": getattr(m, 'avg_tp_distance_pct', 0.0),
        "avg_sl_distance_pct": getattr(m, 'avg_sl_distance_pct', 0.0),
        "avg_rr_ratio": getattr(m, 'avg_rr_ratio', 0.0),
        "signal_quality_score": getattr(m, 'signal_quality_score', 0.0),
        "signals_with_tp_sl_pct": getattr(m, 'signals_with_tp_sl_pct', 100.0),
        "signals_with_correct_sl_pct": getattr(m, 'signals_with_correct_sl_pct', 100.0),
        "avg_position_size_usd": getattr(m, 'avg_position_size_usd', 0.0),
    }
    result.update(additional_metrics)
    
    return result


def extract_interval_from_model(model_path: Path) -> str:
    """
    Извлекает интервал из имени файла модели.
    
    Форматы:
        rf_BTCUSDT_15_15m.pkl -> "15"
        rf_BTCUSDT_60_1h.pkl -> "60"
        ensemble_BTCUSDT_15_mtf.pkl -> "15"
        ensemble_BTCUSDT_60_mtf_1h.pkl -> "60"
    """
    name = model_path.stem
    parts = name.split("_")
    
    # Ищем интервал в частях имени
    for part in parts:
        if part in ["15", "60", "240", "D"]:
            return part
    
    # По умолчанию 15 минут
    return "15"


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
        from backtest_ml_strategy import run_exact_backtest, BacktestMetrics
        
        # Определяем интервал из имени модели, если не указан явно
        model_interval = extract_interval_from_model(model_path)
        if interval == "15m" and model_interval != "15":
            # Используем интервал из имени модели
            test_interval = model_interval
        else:
            # Используем указанный интервал или извлекаем из имени
            test_interval = interval.replace("m", "") if interval.endswith("m") else interval
            if test_interval == "15" and model_interval != "15":
                test_interval = model_interval
        
        metrics = run_exact_backtest(
            model_path=str(model_path),
            symbol=symbol,
            days_back=days,
            interval=test_interval,
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
            
            # Добавляем все доступные метрики
            additional_metrics = {
                "sortino_ratio": getattr(m, 'sortino_ratio', 0.0),
                "calmar_ratio": getattr(m, 'calmar_ratio', 0.0),
                "total_signals": getattr(m, 'total_signals', 0),
                "avg_mfe": getattr(m, 'avg_mfe', 0.0),
                "avg_mae": getattr(m, 'avg_mae', 0.0),
                "mfe_mae_ratio": getattr(m, 'mfe_mae_ratio', 0.0),
                "var_95": getattr(m, 'var_95', 0.0),
                "cvar_95": getattr(m, 'cvar_95', 0.0),
                "recovery_factor": getattr(m, 'recovery_factor', 0.0),
                "expectancy_usd": getattr(m, 'expectancy_usd', 0.0),
                "risk_reward_ratio": getattr(m, 'risk_reward_ratio', 0.0),
                "trade_frequency_per_day": getattr(m, 'trade_frequency_per_day', 0.0),
                "profitable_days_pct": getattr(m, 'profitable_days_pct', 0.0),
                "ulcer_index": getattr(m, 'ulcer_index', 0.0),
                "kelly_criterion": getattr(m, 'kelly_criterion', 0.0),
                "avg_tp_distance_pct": getattr(m, 'avg_tp_distance_pct', 0.0),
                "avg_sl_distance_pct": getattr(m, 'avg_sl_distance_pct', 0.0),
                "avg_rr_ratio": getattr(m, 'avg_rr_ratio', 0.0),
                "signal_quality_score": getattr(m, 'signal_quality_score', 0.0),
                "signals_with_tp_sl_pct": getattr(m, 'signals_with_tp_sl_pct', 100.0),
                "signals_with_correct_sl_pct": getattr(m, 'signals_with_correct_sl_pct', 100.0),
                "avg_position_size_usd": getattr(m, 'avg_position_size_usd', 0.0),
            }
            result.update(additional_metrics)
            
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
    initial_balance: float = 100.0,
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
    
    # Добавляем расширенный анализ
    df_results = add_advanced_analysis(df_results, days)
    
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


def add_advanced_analysis(df_results: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Добавляет расширенный анализ результатов:
    - Анализ распределения сигналов (LONG/SHORT/HOLD)
    - Анализ качества сигналов
    - Анализ эффективности TP/SL
    """
    if df_results.empty:
        return df_results
    
    # Анализ распределения сигналов
    if 'total_signals' in df_results.columns and 'long_trades' in df_results.columns and 'short_trades' in df_results.columns:
        df_results['hold_signals'] = df_results['total_signals'] - df_results['long_trades'] - df_results['short_trades']
        df_results['long_signal_pct'] = (df_results['long_trades'] / df_results['total_signals'].replace(0, 1) * 100).fillna(0)
        df_results['short_signal_pct'] = (df_results['short_trades'] / df_results['total_signals'].replace(0, 1) * 100).fillna(0)
        df_results['hold_signal_pct'] = (df_results['hold_signals'] / df_results['total_signals'].replace(0, 1) * 100).fillna(0)
        df_results['signal_utilization_pct'] = ((df_results['long_trades'] + df_results['short_trades']) / df_results['total_signals'].replace(0, 1) * 100).fillna(0)
        df_results['long_short_balance'] = (df_results['long_trades'] / df_results['short_trades'].replace(0, 1)).fillna(1.0)
    
    # Анализ качества сигналов
    if 'signals_with_tp_sl_pct' in df_results.columns:
        df_results['signal_quality'] = pd.cut(
            df_results['signals_with_tp_sl_pct'],
            bins=[0, 50, 80, 95, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
    
    # Анализ эффективности TP/SL
    if 'avg_tp_distance_pct' in df_results.columns and 'avg_sl_distance_pct' in df_results.columns:
        df_results['tp_sl_ratio'] = (df_results['avg_tp_distance_pct'] / df_results['avg_sl_distance_pct'].replace(0, 0.001)).fillna(0)
        df_results['risk_reward_efficiency'] = df_results['tp_sl_ratio'] * df_results['win_rate_pct'] / 100
    
    # Анализ конверсии сигналов в сделки
    if 'total_trades' in df_results.columns and 'total_signals' in df_results.columns:
        df_results['signal_to_trade_ratio'] = (df_results['total_trades'] / df_results['total_signals'].replace(0, 1)).fillna(0)
    
    # Анализ стабильности (на основе MFE/MAE)
    if 'avg_mfe' in df_results.columns and 'avg_mae' in df_results.columns:
        df_results['trade_control'] = (df_results['avg_mfe'] / df_results['avg_mae'].replace(0, 0.001)).fillna(0)
        df_results['trade_control_category'] = pd.cut(
            df_results['trade_control'],
            bins=[0, 0.5, 1.0, 2.0, float('inf')],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
    
    # Дополнительные метрики для диагностики
    # Эффективность использования сигналов (сколько сделок на 100 сигналов)
    if 'total_trades' in df_results.columns and 'total_signals' in df_results.columns:
        df_results['trades_per_100_signals'] = (df_results['total_trades'] / df_results['total_signals'].replace(0, 1) * 100).fillna(0)
    
    # Потенциальная прибыльность (если бы все сигналы использовались)
    if 'total_pnl_pct' in df_results.columns and 'signal_utilization_pct' in df_results.columns:
        df_results['potential_pnl_if_all_signals'] = (
            df_results['total_pnl_pct'] / (df_results['signal_utilization_pct'].replace(0, 1) / 100)
        ).fillna(0)
    
    # Эффективность модели (PnL на сделку)
    if 'total_pnl_pct' in df_results.columns and 'total_trades' in df_results.columns:
        df_results['pnl_per_trade'] = (
            df_results['total_pnl_pct'] / df_results['total_trades'].replace(0, 1)
        ).fillna(0)
    
    return df_results


def compare_with_previous_results(df_results: pd.DataFrame, previous_csv: str = None) -> pd.DataFrame:
    """
    Сравнивает текущие результаты с предыдущими (до/после переобучения).
    """
    if previous_csv is None or not Path(previous_csv).exists():
        return df_results
    
    try:
        df_previous = pd.read_csv(previous_csv)
        
        # Переименовываем колонки в предыдущих результатах
        df_previous_renamed = df_previous[['model_name', 'symbol', 'total_pnl_pct', 'win_rate_pct', 
                                           'profit_factor', 'max_drawdown_pct']].copy()
        df_previous_renamed = df_previous_renamed.rename(columns={
            'total_pnl_pct': 'total_pnl_pct_previous',
            'win_rate_pct': 'win_rate_pct_previous',
            'profit_factor': 'profit_factor_previous',
            'max_drawdown_pct': 'max_drawdown_pct_previous',
        })
        
        # Объединяем по model_name и symbol
        df_merged = df_results.merge(
            df_previous_renamed,
            on=['model_name', 'symbol'],
            how='left'
        )
        
        # Вычисляем изменения
        df_merged['pnl_change_pct'] = df_merged['total_pnl_pct'] - df_merged['total_pnl_pct_previous'].fillna(0)
        df_merged['winrate_change_pct'] = df_merged['win_rate_pct'] - df_merged['win_rate_pct_previous'].fillna(0)
        df_merged['profit_factor_change'] = df_merged['profit_factor'] - df_merged['profit_factor_previous'].fillna(0)
        df_merged['dd_change_pct'] = df_merged['max_drawdown_pct'] - df_merged['max_drawdown_pct_previous'].fillna(0)
        
        # Флаг улучшения
        df_merged['is_improved'] = (
            (df_merged['pnl_change_pct'] > 0) &
            (df_merged['winrate_change_pct'] >= -2) &  # Допускаем небольшое снижение win rate
            (df_merged['dd_change_pct'] <= 2)  # Допускаем небольшое увеличение DD
        )
        
        print(f"\n📊 Сравнение с предыдущими результатами:")
        print(f"   Улучшено: {df_merged['is_improved'].sum()}/{len(df_merged)} моделей")
        print(f"   Среднее изменение PnL%: {df_merged['pnl_change_pct'].mean():.2f}%")
        print(f"   Среднее изменение Win Rate: {df_merged['winrate_change_pct'].mean():.2f}%")
        
        return df_merged
        
    except Exception as e:
        print(f"⚠️  Ошибка при сравнении с предыдущими результатами: {e}")
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


def print_detailed_analysis(df_results: pd.DataFrame) -> None:
    """Выводит детальный анализ результатов."""
    if df_results.empty:
        return
    
    print("\n" + "=" * 80)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # 1. Анализ распределения сигналов
    if 'long_signal_pct' in df_results.columns:
        print("\n📈 Анализ распределения сигналов:")
        print("-" * 80)
        for symbol in df_results['symbol'].unique():
            symbol_df = df_results[df_results['symbol'] == symbol]
            print(f"\n{symbol}:")
            print(f"   Средний % LONG сигналов: {symbol_df['long_signal_pct'].mean():.1f}%")
            print(f"   Средний % SHORT сигналов: {symbol_df['short_signal_pct'].mean():.1f}%")
            print(f"   Средний % HOLD сигналов: {symbol_df['hold_signal_pct'].mean():.1f}%")
            print(f"   Использование сигналов: {symbol_df['signal_utilization_pct'].mean():.1f}%")
            print(f"   Баланс LONG/SHORT: {symbol_df['long_short_balance'].mean():.2f}:1")
    
    # 2. Анализ качества сигналов
    if 'signals_with_tp_sl_pct' in df_results.columns:
        print("\n🎯 Анализ качества сигналов:")
        print("-" * 80)
        print(f"   Средний % сигналов с TP/SL: {df_results['signals_with_tp_sl_pct'].mean():.1f}%")
        if 'signals_with_correct_sl_pct' in df_results.columns:
            print(f"   Средний % сигналов с правильным SL (1%): {df_results['signals_with_correct_sl_pct'].mean():.1f}%")
        if 'signal_quality' in df_results.columns:
            quality_dist = df_results['signal_quality'].value_counts()
            print(f"   Распределение качества:")
            for quality, count in quality_dist.items():
                print(f"      {quality}: {count} моделей ({count/len(df_results)*100:.1f}%)")
    
    # 3. Анализ эффективности TP/SL
    if 'avg_tp_distance_pct' in df_results.columns and 'avg_sl_distance_pct' in df_results.columns:
        print("\n💰 Анализ эффективности TP/SL:")
        print("-" * 80)
        print(f"   Среднее расстояние до TP: {df_results['avg_tp_distance_pct'].mean():.2f}%")
        print(f"   Среднее расстояние до SL: {df_results['avg_sl_distance_pct'].mean():.2f}%")
        if 'tp_sl_ratio' in df_results.columns:
            print(f"   Средний TP/SL ratio: {df_results['tp_sl_ratio'].mean():.2f}")
        if 'risk_reward_efficiency' in df_results.columns:
            print(f"   Эффективность риск/прибыль: {df_results['risk_reward_efficiency'].mean():.2f}")
    
    # 4. Анализ конверсии сигналов
    if 'signal_to_trade_ratio' in df_results.columns:
        print("\n🔄 Анализ конверсии сигналов в сделки:")
        print("-" * 80)
        print(f"   Средняя конверсия сигнал→сделка: {df_results['signal_to_trade_ratio'].mean():.2%}")
        for symbol in df_results['symbol'].unique():
            symbol_df = df_results[df_results['symbol'] == symbol]
            print(f"   {symbol}: {symbol_df['signal_to_trade_ratio'].mean():.2%}")
    
    # 5. Анализ контроля сделок (MFE/MAE)
    if 'trade_control' in df_results.columns:
        print("\n📊 Анализ контроля сделок (MFE/MAE):")
        print("-" * 80)
        print(f"   Средний MFE/MAE ratio: {df_results['trade_control'].mean():.2f}")
        if 'trade_control_category' in df_results.columns:
            control_dist = df_results['trade_control_category'].value_counts()
            print(f"   Распределение контроля:")
            for category, count in control_dist.items():
                print(f"      {category}: {count} моделей ({count/len(df_results)*100:.1f}%)")
    
    # 6. Топ-5 моделей по различным метрикам
    print("\n🏆 ТОП-5 МОДЕЛЕЙ ПО РАЗЛИЧНЫМ МЕТРИКАМ:")
    print("-" * 80)
    
    metrics_to_show = [
        ('total_pnl_pct', 'PnL %'),
        ('win_rate_pct', 'Win Rate %'),
        ('profit_factor', 'Profit Factor'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('composite_score', 'Composite Score'),
    ]
    
    for metric_col, metric_name in metrics_to_show:
        if metric_col in df_results.columns:
            top5 = df_results.nlargest(5, metric_col)[['model_name', 'symbol', metric_col]]
            print(f"\n{metric_name}:")
            for idx, (_, row) in enumerate(top5.iterrows(), 1):
                print(f"   {idx}. {row['model_name']} ({row['symbol']}): {row[metric_col]:.2f}")
    
    print("\n" + "=" * 80)
    
    # 7. Анализ проблем и рекомендации
    print_problems_and_recommendations(df_results)


def print_problems_and_recommendations(df_results: pd.DataFrame) -> None:
    """
    Анализирует результаты и выдает рекомендации по улучшению.
    Фокус на проблемах: низкое использование сигналов, низкая конверсия, качество сигналов.
    """
    if df_results.empty:
        return
    
    print("\n" + "=" * 80)
    print("🔍 АНАЛИЗ ПРОБЛЕМ И РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("=" * 80)
    
    # 1. Анализ использования сигналов
    if 'signal_utilization_pct' in df_results.columns:
        avg_utilization = df_results['signal_utilization_pct'].mean()
        print(f"\n📊 ПРОБЛЕМА 1: Использование сигналов")
        print("-" * 80)
        print(f"   Текущее использование: {avg_utilization:.1f}% (цель: 30-40%)")
        
        if avg_utilization < 25:
            print(f"   ⚠️  КРИТИЧНО: Использование сигналов слишком низкое!")
            print(f"   📉 Средний % HOLD: {df_results['hold_signal_pct'].mean():.1f}%")
            print(f"   📉 Средний % LONG: {df_results['long_signal_pct'].mean():.1f}%")
            print(f"   📉 Средний % SHORT: {df_results['short_signal_pct'].mean():.1f}%")
            
            print(f"\n   💡 РЕКОМЕНДАЦИИ:")
            print(f"      1. Уменьшить threshold_pct в target labeling с 0.5% до 0.3%")
            print(f"      2. Уменьшить min_profit_pct с 0.5% до 0.3%")
            print(f"      3. Снизить базовый confidence_threshold на 5-10%")
            print(f"      4. Проверить фильтры в strategy_ml.py - возможно слишком строгие")
            print(f"      5. Увеличить forward_periods с 5 до 6-7 для большего охвата")
            
            # Анализ по символам
            print(f"\n   📈 По символам:")
            for symbol in df_results['symbol'].unique():
                symbol_df = df_results[df_results['symbol'] == symbol]
                util = symbol_df['signal_utilization_pct'].mean()
                hold = symbol_df['hold_signal_pct'].mean()
                print(f"      {symbol}: {util:.1f}% использование ({hold:.1f}% HOLD)")
        else:
            print(f"   ✅ Использование сигналов в норме")
    
    # 2. Анализ конверсии сигналов в сделки
    if 'signal_to_trade_ratio' in df_results.columns:
        avg_conversion = df_results['signal_to_trade_ratio'].mean() * 100
        print(f"\n🔄 ПРОБЛЕМА 2: Конверсия сигналов в сделки")
        print("-" * 80)
        print(f"   Текущая конверсия: {avg_conversion:.2f}% (цель: 10-15%)")
        
        if avg_conversion < 5:
            print(f"   ⚠️  КРИТИЧНО: Конверсия слишком низкая!")
            
            # Анализ возможных причин
            if 'avg_confidence' in df_results.columns:
                avg_conf = df_results['avg_confidence'].mean()
                print(f"   📊 Средняя уверенность модели: {avg_conf:.1%}")
                if avg_conf < 0.5:
                    print(f"      ⚠️  Низкая уверенность - возможно модель не уверена в сигналах")
            
            if 'signals_with_tp_sl_pct' in df_results.columns:
                tp_sl_pct = df_results['signals_with_tp_sl_pct'].mean()
                if tp_sl_pct < 100:
                    print(f"   ⚠️  Только {tp_sl_pct:.1f}% сигналов имеют TP/SL")
                    print(f"      Это может блокировать открытие позиций")
            
            print(f"\n   💡 РЕКОМЕНДАЦИИ:")
            print(f"      1. Проверить фильтры в strategy_ml.py:")
            print(f"         - max_signals_per_day (сейчас 10) - возможно слишком низкий")
            print(f"         - Фильтры по RSI (экстремальные зоны)")
            print(f"         - Фильтры по объему (низкий объем)")
            print(f"         - Фильтры стабильности (stability_filter)")
            print(f"      2. Упростить условия открытия позиций")
            print(f"      3. Проверить баланс - возможно недостаточно средств")
            print(f"      4. Добавить логирование причин отклонения сигналов")
            
            # Анализ по символам
            print(f"\n   📈 По символам:")
            for symbol in df_results['symbol'].unique():
                symbol_df = df_results[df_results['symbol'] == symbol]
                conv = symbol_df['signal_to_trade_ratio'].mean() * 100
                trades = symbol_df['total_trades'].mean()
                signals = symbol_df['total_signals'].mean()
                print(f"      {symbol}: {conv:.2f}% ({trades:.0f} сделок из {signals:.0f} сигналов)")
        else:
            print(f"   ✅ Конверсия в норме")
    
    # 3. Сравнение MTF vs без MTF
    if 'mode_suffix' in df_results.columns:
        mtf_models = df_results[df_results['mode_suffix'] == 'mtf']
        no_mtf_models = df_results[df_results['mode_suffix'] == '15m']
        
        if len(mtf_models) > 0 and len(no_mtf_models) > 0:
            print(f"\n📊 СРАВНЕНИЕ: MTF vs БЕЗ MTF")
            print("-" * 80)
            
            mtf_pnl = mtf_models['total_pnl_pct'].mean()
            no_mtf_pnl = no_mtf_models['total_pnl_pct'].mean()
            mtf_util = mtf_models['signal_utilization_pct'].mean() if 'signal_utilization_pct' in mtf_models.columns else 0
            no_mtf_util = no_mtf_models['signal_utilization_pct'].mean() if 'signal_utilization_pct' in no_mtf_models.columns else 0
            
            print(f"   MTF модели:")
            print(f"      Средний PnL%: {mtf_pnl:.2f}%")
            print(f"      Использование сигналов: {mtf_util:.1f}%")
            print(f"      Моделей: {len(mtf_models)}")
            
            print(f"\n   БЕЗ MTF модели:")
            print(f"      Средний PnL%: {no_mtf_pnl:.2f}%")
            print(f"      Использование сигналов: {no_mtf_util:.1f}%")
            print(f"      Моделей: {len(no_mtf_models)}")
            
            if no_mtf_pnl > mtf_pnl:
                print(f"\n   ✅ Модели БЕЗ MTF показывают лучшие результаты!")
                print(f"      Разница: {no_mtf_pnl - mtf_pnl:.2f}%")
            elif mtf_pnl > no_mtf_pnl:
                print(f"\n   ✅ MTF модели показывают лучшие результаты!")
                print(f"      Разница: {mtf_pnl - no_mtf_pnl:.2f}%")
            else:
                print(f"\n   ⚖️  Результаты сопоставимы")
    
    # 4. Анализ качества сигналов
    if 'signals_with_tp_sl_pct' in df_results.columns:
        avg_tp_sl = df_results['signals_with_tp_sl_pct'].mean()
        print(f"\n🎯 ПРОБЛЕМА 3: Качество сигналов")
        print("-" * 80)
        print(f"   Сигналов с TP/SL: {avg_tp_sl:.1f}% (цель: 100%)")
        
        if avg_tp_sl < 100:
            print(f"   ⚠️  Не все сигналы имеют TP/SL!")
            print(f"   💡 РЕКОМЕНДАЦИЯ: Проверить генерацию TP/SL в strategy_ml.py")
        else:
            print(f"   ✅ Все сигналы имеют TP/SL")
        
        if 'signals_with_correct_sl_pct' in df_results.columns:
            correct_sl = df_results['signals_with_correct_sl_pct'].mean()
            print(f"   Сигналов с правильным SL (1%): {correct_sl:.1f}% (цель: 100%)")
            if correct_sl < 95:
                print(f"   ⚠️  Много сигналов с неправильным SL!")
                print(f"   💡 РЕКОМЕНДАЦИЯ: Проверить расчет SL в strategy_ml.py")
    
    # 5. Анализ баланса LONG/SHORT
    if 'long_short_balance' in df_results.columns:
        avg_balance = df_results['long_short_balance'].mean()
        print(f"\n⚖️  ПРОБЛЕМА 4: Баланс LONG/SHORT")
        print("-" * 80)
        print(f"   Соотношение LONG/SHORT: {avg_balance:.2f}:1 (цель: ~1:1)")
        
        if avg_balance > 2.0 or avg_balance < 0.5:
            print(f"   ⚠️  Дисбаланс LONG/SHORT!")
            print(f"   💡 РЕКОМЕНДАЦИЯ: Проверить class weights в обучении")
            print(f"      Убедиться, что minority class получает достаточный вес")
        else:
            print(f"   ✅ Баланс в норме")
    
    # 6. Анализ MFE/MAE (контроль сделок)
    if 'trade_control' in df_results.columns:
        avg_control = df_results['trade_control'].mean()
        print(f"\n📊 ПРОБЛЕМА 5: Контроль сделок (MFE/MAE)")
        print("-" * 80)
        print(f"   Средний MFE/MAE ratio: {avg_control:.2f} (цель: > 1.0)")
        
        if avg_control == 0 or avg_control < 0.5:
            print(f"   ⚠️  КРИТИЧНО: MFE/MAE не рассчитывается или очень низкий!")
            print(f"   💡 РЕКОМЕНДАЦИЯ: Исправить расчет MFE/MAE в backtest_ml_strategy.py")
        elif avg_control < 1.0:
            print(f"   ⚠️  Низкий контроль - сделки уходят в убыток быстрее, чем в прибыль")
            print(f"   💡 РЕКОМЕНДАЦИЯ: Улучшить timing входа (возможно, слишком ранние входы)")
        else:
            print(f"   ✅ Контроль в норме")
    
    # 7. Общие рекомендации
    print(f"\n💡 ОБЩИЕ РЕКОМЕНДАЦИИ:")
    print("-" * 80)
    
    profitable = df_results[df_results['total_pnl_pct'] > 0]
    if len(profitable) > 0:
        profitable_pct = len(profitable) / len(df_results) * 100
        print(f"   ✅ Прибыльных моделей: {profitable_pct:.1f}% ({len(profitable)}/{len(df_results)})")
        
        if profitable_pct < 50:
            print(f"   ⚠️  Меньше половины моделей прибыльны!")
            print(f"      Рекомендуется пересмотреть параметры обучения")
        
        # Лучшие модели
        best = profitable.nlargest(3, 'total_pnl_pct')
        print(f"\n   🏆 Топ-3 прибыльных модели:")
        for idx, (_, row) in enumerate(best.iterrows(), 1):
            print(f"      {idx}. {row['model_name']} ({row['symbol']}): {row['total_pnl_pct']:.2f}% PnL")
    else:
        print(f"   ❌ КРИТИЧНО: Нет прибыльных моделей!")
        print(f"      Необходимо пересмотреть:")
        print(f"      1. Параметры target labeling")
        print(f"      2. Гиперпараметры моделей")
        print(f"      3. Фильтры в strategy_ml.py")
        print(f"      4. Параметры TP/SL")
    
    # 8. Приоритетные задачи
    print(f"\n🎯 ПРИОРИТЕТНЫЕ ЗАДАЧИ (по результатам анализа):")
    print("-" * 80)
    
    priorities = []
    
    if avg_utilization < 25 if 'signal_utilization_pct' in df_results.columns else False:
        priorities.append("1. УВЕЛИЧИТЬ использование сигналов (сейчас слишком много HOLD)")
    
    if avg_conversion < 5 if 'signal_to_trade_ratio' in df_results.columns else False:
        priorities.append("2. УЛУЧШИТЬ конверсию сигналов в сделки (проверить фильтры)")
    
    if avg_tp_sl < 100 if 'signals_with_tp_sl_pct' in df_results.columns else False:
        priorities.append("3. ИСПРАВИТЬ генерацию TP/SL (не все сигналы имеют TP/SL)")
    
    if avg_control == 0 or avg_control < 0.5 if 'trade_control' in df_results.columns else False:
        priorities.append("4. ИСПРАВИТЬ расчет MFE/MAE (сейчас не работает)")
    
    if len(profitable) == 0 if 'total_pnl_pct' in df_results.columns else False:
        priorities.append("5. КРИТИЧНО: Пересмотреть параметры обучения (нет прибыльных моделей)")
    
    if priorities:
        for priority in priorities:
            print(f"   {priority}")
    else:
        print(f"   ✅ Все основные метрики в норме!")
        print(f"   Можно переходить к оптимизации гиперпараметров")
    
    print("\n" + "=" * 80)


def print_best_models_per_symbol(df_results: pd.DataFrame) -> None:
    """
    Выводит лучшие модели для каждого символа с краткой статистикой.
    """
    if df_results.empty:
        return
    
    print("\n" + "=" * 80)
    print("🏆 ЛУЧШИЕ МОДЕЛИ ПО КАЖДОМУ СИМВОЛУ")
    print("=" * 80)
    
    # Группируем по символам и выбираем лучшую модель для каждого
    for symbol in sorted(df_results['symbol'].unique()):
        symbol_df = df_results[df_results['symbol'] == symbol].copy()
        
        # Сортируем по PnL% (убывание)
        symbol_df = symbol_df.sort_values('total_pnl_pct', ascending=False)
        
        # Берем лучшую модель
        best = symbol_df.iloc[0]
        
        print(f"\n📈 {symbol}:")
        print("-" * 80)
        print(f"   Модель: {best['model_name']}")
        print(f"   Тип: {best.get('model_type', 'N/A')} ({best.get('mode_suffix', 'N/A')})")
        print(f"   📊 Статистика:")
        print(f"      • Сделок: {int(best['total_trades'])}")
        print(f"      • PnL%: {best['total_pnl_pct']:+.2f}%")
        print(f"      • PnL USD: ${best['total_pnl_usd']:+.2f}")
        print(f"      • Win Rate: {best['win_rate_pct']:.1f}% ({int(best['winning_trades'])}/{int(best['total_trades'])})")
        print(f"      • Profit Factor: {best['profit_factor']:.2f}")
        print(f"      • Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        print(f"      • Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        
        # Дополнительные метрики, если доступны
        if 'trades_per_day' in best and pd.notna(best['trades_per_day']):
            print(f"      • Сделок в день: {best['trades_per_day']:.2f}")
        
        if 'avg_win_usd' in best and pd.notna(best['avg_win_usd']):
            print(f"      • Средняя прибыль: ${best['avg_win_usd']:.2f}")
        
        if 'avg_loss_usd' in best and pd.notna(best['avg_loss_usd']):
            print(f"      • Средний убыток: ${best['avg_loss_usd']:.2f}")
        
        if 'long_trades' in best and 'short_trades' in best:
            long_count = int(best['long_trades']) if pd.notna(best['long_trades']) else 0
            short_count = int(best['short_trades']) if pd.notna(best['short_trades']) else 0
            if long_count + short_count > 0:
                print(f"      • LONG/SHORT: {long_count}/{short_count}")
        
        if 'avg_trade_duration_hours' in best and pd.notna(best['avg_trade_duration_hours']):
            print(f"      • Средняя длительность сделки: {best['avg_trade_duration_hours']:.1f} ч")
        
        if 'avg_confidence' in best and pd.notna(best['avg_confidence']):
            print(f"      • Средняя уверенность: {best['avg_confidence']*100:.1f}%")
        
        # Показываем топ-3 модели для этого символа
        top3 = symbol_df.head(3)
        if len(top3) > 1:
            print(f"\n   📊 Топ-3 модели для {symbol}:")
            for idx, (_, row) in enumerate(top3.iterrows(), 1):
                pnl_sign = "+" if row['total_pnl_pct'] >= 0 else ""
                print(f"      {idx}. {row['model_name']}: {pnl_sign}{row['total_pnl_pct']:.2f}% PnL, "
                      f"{row['win_rate_pct']:.1f}% WR, {int(row['total_trades'])} сделок")
    
    print("\n" + "=" * 80)


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
  
  # После переобучения - сравнение с предыдущими результатами
  python compare_ml_models.py --compare-with ml_models_comparison_20260205_120000.csv --detailed-analysis
  
  # Расширенное тестирование с детальным анализом
  python compare_ml_models.py --days 60 --symbols BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT --detailed-analysis
  
  # С проверкой переобучения и 8 процессами
  python compare_ml_models.py --check-overfitting --workers 8 --output all --detailed-analysis
  
  # Тестирование с низким риском
  python compare_ml_models.py --risk 0.01 --leverage 5 --balance 5000
  
  # Полный анализ после переобучения
  python compare_ml_models.py --compare-with previous_results.csv --output all --detailed-analysis --check-overfitting
        """
    )
    
    parser.add_argument("--days", type=int, default=30, 
                       help="Days to backtest (default: 30)")
    parser.add_argument(
        "--symbols",
        type=str,
        default="auto",
        help="Comma-separated list of symbols or 'auto' to auto-detect from models (default: auto)",
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
        default=100.0,
        help="Initial balance (default: 100.0)",
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
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Path to previous CSV results for comparison (before/after retraining)",
    )
    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        help="Run detailed analysis (signal distribution, quality metrics, etc.)",
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    # Проверяем существование директории с моделями
    if not models_dir.exists():
        print(f"❌ Директория с моделями не существует: {models_dir}")
        print(f"   Текущая рабочая директория: {Path.cwd()}")
        return
    
    # Определяем список символов
    if args.symbols.lower() == "auto" or args.symbols.strip() == "":
        # Автоматическое обнаружение символов из моделей
        print(f"🔍 Автоматическое обнаружение символов из моделей...")
        symbols = find_all_symbols(models_dir)
        if not symbols:
            print(f"⚠️  Не удалось найти символы в моделях. Используем дефолтные: BTCUSDT,ETHUSDT,SOLUSDT")
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        else:
            print(f"✅ Найдено символов: {', '.join(symbols)} ({len(symbols)} символов)")
    else:
        # Используем указанные символы
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    if not symbols:
        print(f"❌ Не указаны символы для тестирования")
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
    
    # Сравнение с предыдущими результатами
    if args.compare_with:
        df_results = compare_with_previous_results(df_results, args.compare_with)
    
    if args.output in ["csv", "all"]:
        csv_name = f"ml_models_comparison_{timestamp}.csv"
        df_results.to_csv(csv_name, index=False, encoding='utf-8')
        print(f"\n💾 Full comparison table saved to: {csv_name}")
        print(f"   Rows: {len(df_results)}, Columns: {len(df_results.columns)}")
    
    # Детальный анализ (включает анализ проблем и рекомендации)
    if args.detailed_analysis:
        print_detailed_analysis(df_results)
    else:
        # Даже без --detailed-analysis показываем краткий анализ проблем
        print_problems_and_recommendations(df_results)
    
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
    
    # Выводим лучшие модели для каждого символа
    print_best_models_per_symbol(df_results)


if __name__ == "__main__":
    main()