"""
Бэктест комбинированной MTF стратегии (1h + 15m).

Использует:
- 1h модель для фильтрации направления тренда
- 15m модель для точного входа
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

warnings.filterwarnings('ignore')

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, ApiSettings
from bot.ml.data_collector import DataCollector
from bot.ml.mtf_strategy import MultiTimeframeMLStrategy
from bot.ml.strategy_ml import MLStrategy
from bot.strategy import Action, Signal, Bias
from backtest_ml_strategy import (
    MLBacktestSimulator,
    BacktestMetrics,
    ExitReason,
)


def find_best_models_from_comparison(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Находит лучшие модели из результатов сравнения.
    Сначала ищет в comparison_15m_vs_1h.csv, затем в ml_models_comparison_*.csv.
    
    Returns:
        (model_1h_path, model_15m_path) или (None, None) если не найдены
    """
    models_dir = Path("ml_models")
    symbol_upper = symbol.upper()
    
    # 1. Сначала ищем в comparison_15m_vs_1h.csv (если есть)
    comparison_15m_1h = Path("comparison_15m_vs_1h.csv")
    if comparison_15m_1h.exists():
        try:
            df = pd.read_csv(comparison_15m_1h)
            symbol_data = df[df['symbol'] == symbol_upper]
            if not symbol_data.empty:
                best_row = symbol_data.iloc[0]
                best_15m_name = best_row.get('best_15m_model', '')
                best_1h_name = best_row.get('best_1h_model', '')
                
                if best_15m_name and best_1h_name:
                    model_15m_path = models_dir / f"{best_15m_name}.pkl"
                    model_1h_path = models_dir / f"{best_1h_name}.pkl"
                    
                    if model_15m_path.exists() and model_1h_path.exists():
                        return str(model_1h_path), str(model_15m_path)
        except Exception as e:
            print(f"⚠️  Ошибка загрузки из comparison_15m_vs_1h.csv: {e}")
    
    # 2. Ищем в ml_models_comparison_*.csv
    comparison_files = sorted(
        Path(".").glob("ml_models_comparison_*.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    if not comparison_files:
        return None, None
    
    try:
        # Загружаем последний файл сравнения
        df = pd.read_csv(comparison_files[0])
        
        # Ищем лучшие модели для символа, раздельно для 1h и 15m
        symbol_data = df[df['symbol'] == symbol_upper]
        if symbol_data.empty:
            return None, None
        
        # Лучшая 1h модель (mode_suffix == '1h')
        symbol_1h = symbol_data[symbol_data.get('mode_suffix', '') == '1h']
        if symbol_1h.empty:
            # Пробуем найти по имени файла
            symbol_1h = symbol_data[symbol_data['model_filename'].str.contains('_60_|_1h', na=False)]
        
        # Лучшая 15m модель (mode_suffix == '15m')
        symbol_15m = symbol_data[symbol_data.get('mode_suffix', '') == '15m']
        if symbol_15m.empty:
            # Пробуем найти по имени файла
            symbol_15m = symbol_data[symbol_data['model_filename'].str.contains('_15_|_15m', na=False)]
        
        if symbol_1h.empty or symbol_15m.empty:
            return None, None
        
        # Сортируем по total_pnl_pct и берем лучшие
        best_1h = symbol_1h.sort_values('total_pnl_pct', ascending=False).iloc[0]
        best_15m = symbol_15m.sort_values('total_pnl_pct', ascending=False).iloc[0]
        
        # Извлекаем имена моделей
        best_1h_name = best_1h.get('model_name', '') or best_1h.get('model_filename', '').replace('.pkl', '')
        best_15m_name = best_15m.get('model_name', '') or best_15m.get('model_filename', '').replace('.pkl', '')
        
        if not best_1h_name or not best_15m_name:
            return None, None
        
        # Ищем файлы моделей
        model_1h_path = models_dir / f"{best_1h_name}.pkl"
        model_15m_path = models_dir / f"{best_15m_name}.pkl"
        
        if model_1h_path.exists() and model_15m_path.exists():
            return str(model_1h_path), str(model_15m_path)
    except Exception as e:
        print(f"⚠️  Ошибка загрузки лучших моделей из сравнения: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None


def get_effective_models_from_comparison(symbol: str) -> Tuple[set, set]:
    """
    Загружает список эффективных моделей из CSV файла сравнения.
    
    Returns:
        (set_1h_model_names, set_15m_model_names) - множества имен эффективных моделей
    """
    effective_1h = set()
    effective_15m = set()
    
    # Ищем последний файл сравнения
    comparison_files = sorted(
        Path(".").glob("ml_models_comparison_*.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    if not comparison_files:
        return effective_1h, effective_15m
    
    try:
        df = pd.read_csv(comparison_files[0])
        symbol_upper = symbol.upper()
        
        # Фильтруем по символу и исключаем модели с 0 сделок или отрицательным PnL
        symbol_data = df[
            (df['symbol'] == symbol_upper) & 
            (df['total_trades'] > 0) & 
            (df['total_pnl_pct'] > 0)
        ]
        
        if symbol_data.empty:
            return effective_1h, effective_15m
        
        # Разделяем на 1h и 15m модели
        for _, row in symbol_data.iterrows():
            # Используем model_filename как основной источник имени (без .pkl)
            model_filename = row.get('model_filename', '')
            if model_filename:
                model_name = model_filename.replace('.pkl', '')
            else:
                model_name = row.get('model_name', '')
            
            if not model_name:
                continue
            
            mode_suffix = row.get('mode_suffix', '')
            
            # Определяем тип модели по mode_suffix или имени файла
            if mode_suffix == '1h' or '_60_' in model_name or '_1h' in model_name:
                effective_1h.add(model_name)
            elif mode_suffix == '15m' or '_15_' in model_name or '_15m' in model_name:
                effective_15m.add(model_name)
    except Exception as e:
        print(f"⚠️  Ошибка загрузки эффективных моделей из CSV: {e}")
    
    return effective_1h, effective_15m


def find_all_models_for_symbol(symbol: str) -> Tuple[List[str], List[str]]:
    """
    Находит ВСЕ эффективные модели 1h и 15m для символа.
    Фильтрует модели на основе CSV файла сравнения (исключает неэффективные).
    
    Returns:
        (list_1h_models, list_15m_models)
    """
    models_dir = Path("ml_models")
    if not models_dir.exists():
        return [], []
    
    # Загружаем список эффективных моделей из CSV
    effective_1h_names, effective_15m_names = get_effective_models_from_comparison(symbol)
    
    # Ищем 1h модели
    models_1h = list(models_dir.glob(f"*_{symbol}_60_*.pkl"))
    if not models_1h:
        models_1h = list(models_dir.glob(f"*_{symbol}_*1h*.pkl"))
    
    # Ищем 15m модели
    models_15m = list(models_dir.glob(f"*_{symbol}_15_*.pkl"))
    if not models_15m:
        models_15m = list(models_dir.glob(f"*_{symbol}_*15m*.pkl"))
    
    # Фильтруем модели: оставляем только эффективные (если список не пустой)
    if effective_1h_names:
        models_1h = [m for m in models_1h if m.stem in effective_1h_names]
    
    if effective_15m_names:
        models_15m = [m for m in models_15m if m.stem in effective_15m_names]
    
    # Сортируем по имени (для стабильности)
    models_1h = sorted([str(m) for m in models_1h])
    models_15m = sorted([str(m) for m in models_15m])
    
    return models_1h, models_15m


def find_models_for_symbol(
    symbol: str, 
    use_best_from_comparison: bool = True,
    model_1h_name: Optional[str] = None,
    model_15m_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Находит модели 1h и 15m для символа.
    
    Args:
        symbol: Торговая пара
        use_best_from_comparison: Использовать лучшие модели из результатов сравнения
        model_1h_name: Конкретное имя 1h модели (если указано, используется оно)
        model_15m_name: Конкретное имя 15m модели (если указано, используется оно)
    
    Returns:
        (model_1h_path, model_15m_path) или (None, None) если не найдены
    """
    models_dir = Path("ml_models")
    if not models_dir.exists():
        print(f"❌ Директория {models_dir} не существует")
        return None, None
    
    # Если указаны конкретные модели, используем их
    if model_1h_name:
        model_1h_path = models_dir / f"{model_1h_name}.pkl"
        if model_1h_path.exists():
            model_1h = str(model_1h_path)
        else:
            print(f"⚠️  Указанная 1h модель не найдена: {model_1h_name}")
            model_1h = None
    else:
        model_1h = None
    
    if model_15m_name:
        model_15m_path = models_dir / f"{model_15m_name}.pkl"
        if model_15m_path.exists():
            model_15m = str(model_15m_path)
        else:
            print(f"⚠️  Указанная 15m модель не найдена: {model_15m_name}")
            model_15m = None
    else:
        model_15m = None
    
    # Если обе модели найдены, возвращаем их
    if model_1h and model_15m:
        return model_1h, model_15m
    
    # Если не указаны конкретные модели, ищем лучшие из сравнения
    if use_best_from_comparison and (not model_1h or not model_15m):
        best_1h, best_15m = find_best_models_from_comparison(symbol)
        if best_1h and best_15m:
            print(f"✅ Используются лучшие модели из сравнения:")
            print(f"   1h: {Path(best_1h).name}")
            print(f"   15m: {Path(best_15m).name}")
            return best_1h, best_15m
    
    # Если не нашли лучшие, ищем любые доступные
    models_1h_list, models_15m_list = find_all_models_for_symbol(symbol)
    
    if not model_1h and models_1h_list:
        model_1h = models_1h_list[0]
        print(f"📦 Используется 1h модель: {Path(model_1h).name}")
    
    if not model_15m and models_15m_list:
        model_15m = models_15m_list[0]
        print(f"📦 Используется 15m модель: {Path(model_15m).name}")
    
    if not model_1h:
        print(f"⚠️  1h модель для {symbol} не найдена")
        if models_1h_list:
            print(f"   Доступные 1h модели: {[Path(m).name for m in models_1h_list]}")
    if not model_15m:
        print(f"⚠️  15m модель для {symbol} не найдена")
        if models_15m_list:
            print(f"   Доступные 15m модели: {[Path(m).name for m in models_15m_list]}")
    
    return model_1h, model_15m


def run_mtf_backtest_all_combinations(
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    initial_balance: float = 100.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    confidence_threshold_1h: float = 0.50,
    confidence_threshold_15m: float = 0.35,
    alignment_mode: str = "strict",
    require_alignment: bool = True,
) -> pd.DataFrame:
    """
    Запускает бэктест для ВСЕХ комбинаций моделей 1h и 15m.
    
    Returns:
        DataFrame с результатами всех комбинаций
    """
    print("=" * 80)
    print("🚀 БЭКТЕСТ ВСЕХ КОМБИНАЦИЙ MTF СТРАТЕГИИ")
    print("=" * 80)
    print(f"Символ: {symbol}")
    print(f"Период: {days_back} дней")
    print()
    
    # Находим все модели
    models_1h, models_15m = find_all_models_for_symbol(symbol)
    
    if not models_1h:
        print(f"❌ Не найдено 1h моделей для {symbol}")
        return pd.DataFrame()
    if not models_15m:
        print(f"❌ Не найдено 15m моделей для {symbol}")
        return pd.DataFrame()
    
    print(f"📦 Найдено моделей:")
    print(f"   1h: {len(models_1h)}")
    for m in models_1h:
        print(f"      - {Path(m).name}")
    print(f"   15m: {len(models_15m)}")
    for m in models_15m:
        print(f"      - {Path(m).name}")
    print()
    print(f"🎯 Всего комбинаций: {len(models_1h) * len(models_15m)}")
    print()
    
    # Результаты
    results = []
    
    # Тестируем все комбинации
    for i, model_1h in enumerate(models_1h, 1):
        for j, model_15m in enumerate(models_15m, 1):
            combo_num = (i - 1) * len(models_15m) + j
            total_combos = len(models_1h) * len(models_15m)
            
            print("=" * 80)
            print(f"📊 Комбинация {combo_num}/{total_combos}:")
            print(f"   1h: {Path(model_1h).name}")
            print(f"   15m: {Path(model_15m).name}")
            print("-" * 80)
            
            try:
                metrics = run_mtf_backtest(
                    symbol=symbol,
                    days_back=days_back,
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                    leverage=leverage,
                    model_1h_path=model_1h,
                    model_15m_path=model_15m,
                    confidence_threshold_1h=confidence_threshold_1h,
                    confidence_threshold_15m=confidence_threshold_15m,
                    alignment_mode=alignment_mode,
                    require_alignment=require_alignment,
                )
                
                if metrics:
                    results.append({
                        "model_1h": Path(model_1h).name,
                        "model_15m": Path(model_15m).name,
                        "symbol": symbol,
                        "total_trades": metrics.total_trades,
                        "winning_trades": metrics.winning_trades,
                        "losing_trades": metrics.losing_trades,
                        "win_rate": metrics.win_rate,
                        "total_pnl": metrics.total_pnl,
                        "total_pnl_pct": metrics.total_pnl_pct,
                        "avg_win": metrics.avg_win,
                        "avg_loss": metrics.avg_loss,
                        "profit_factor": metrics.profit_factor,
                        "max_drawdown_pct": metrics.max_drawdown_pct,
                        "sharpe_ratio": metrics.sharpe_ratio,
                    })
                    print(f"✅ Результат: {metrics.total_trades} сделок, PnL: {metrics.total_pnl_pct:.2f}%, WR: {metrics.win_rate:.1f}%")
                else:
                    print(f"❌ Ошибка при тестировании комбинации")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
            
            print()
    
    # Создаем DataFrame с результатами
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_pnl_pct', ascending=False)
        
        print("=" * 80)
        print("🏆 ЛУЧШИЕ КОМБИНАЦИИ")
        print("=" * 80)
        print(df_results.head(10).to_string(index=False))
        print()
        
        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mtf_combinations_{symbol}_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        print(f"✅ Результаты сохранены в {filename}")
        
        return df_results
    else:
        print("❌ Нет результатов для отображения")
        return pd.DataFrame()


def run_mtf_backtest(
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    initial_balance: float = 100.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    model_1h_path: Optional[str] = None,
    model_15m_path: Optional[str] = None,
    confidence_threshold_1h: float = 0.50,
    confidence_threshold_15m: float = 0.35,
    alignment_mode: str = "strict",
    require_alignment: bool = True,
) -> Optional[BacktestMetrics]:
    """
    Запускает бэктест комбинированной MTF стратегии.
    
    Args:
        symbol: Торговая пара
        days_back: Количество дней назад для тестирования
        initial_balance: Начальный баланс
        risk_per_trade: Риск на сделку (2% = 0.02)
        leverage: Плечо
        model_1h_path: Путь к 1h модели (если None - ищется автоматически)
        model_15m_path: Путь к 15m модели (если None - ищется автоматически)
        confidence_threshold_1h: Порог уверенности для 1h модели
        confidence_threshold_15m: Порог уверенности для 15m модели
        alignment_mode: Режим выравнивания ("strict" или "weighted")
        require_alignment: Требовать совпадение направлений
    
    Returns:
        BacktestMetrics или None при ошибке
    """
    print("=" * 80)
    print("🚀 БЭКТЕСТ КОМБИНИРОВАННОЙ MTF СТРАТЕГИИ")
    print("=" * 80)
    print(f"Символ: {symbol}")
    print(f"Период: {days_back} дней")
    print(f"Начальный баланс: ${initial_balance:.2f}")
    print(f"Риск на сделку: {risk_per_trade*100:.1f}%")
    print(f"Плечо: {leverage}x")
    print()
    
    # Находим модели (если не указаны явно)
    if model_1h_path is None or model_15m_path is None:
        print("🔍 Поиск моделей...")
        # Извлекаем имена моделей из путей, если они указаны
        model_1h_name_param = Path(model_1h_path).stem if model_1h_path and Path(model_1h_path).exists() else None
        model_15m_name_param = Path(model_15m_path).stem if model_15m_path and Path(model_15m_path).exists() else None
        
        found_1h, found_15m = find_models_for_symbol(
            symbol,
            use_best_from_comparison=True,  # Использовать лучшие из сравнения
            model_1h_name=model_1h_name_param,
            model_15m_name=model_15m_name_param,
        )
        if found_1h is None or found_15m is None:
            print("❌ Не удалось найти обе модели")
            # Показываем доступные модели
            models_1h_list, models_15m_list = find_all_models_for_symbol(symbol)
            if models_1h_list:
                print(f"   Доступные 1h модели: {[Path(m).name for m in models_1h_list]}")
            if models_15m_list:
                print(f"   Доступные 15m модели: {[Path(m).name for m in models_15m_list]}")
            return None
        model_1h_path = found_1h
        model_15m_path = found_15m
    
    # Показываем, какие модели используются
    print(f"✅ 1h модель: {Path(model_1h_path).name}")
    print(f"✅ 15m модель: {Path(model_15m_path).name}")
    
    print(f"✅ 1h модель: {Path(model_1h_path).name}")
    print(f"✅ 15m модель: {Path(model_15m_path).name}")
    print()
    
    # Загружаем настройки
    settings = load_settings()
    
    # Собираем данные
    print("📥 Сбор данных...")
    collector = DataCollector(settings.api)
    
    # Собираем 15m данные (основной таймфрейм)
    start_date = datetime.now() - timedelta(days=days_back)
    df_15m = collector.collect_klines(
        symbol=symbol,
        interval="15",
        start_date=start_date,
        end_date=None,
        limit=days_back * 96,  # 96 свечей в день для 15m
    )
    
    if df_15m.empty:
        print("❌ Не удалось собрать данные")
        return None
    
    print(f"✅ Собрано {len(df_15m)} свечей 15m")
    print(f"   Период: {df_15m['timestamp'].min()} - {df_15m['timestamp'].max()}")
    print()
    
    # Создаем MTF стратегию
    print("🤖 Создание MTF стратегии...")
    try:
        strategy = MultiTimeframeMLStrategy(
            model_1h_path=model_1h_path,
            model_15m_path=model_15m_path,
            confidence_threshold_1h=confidence_threshold_1h,
            confidence_threshold_15m=confidence_threshold_15m,
            require_alignment=require_alignment,
            alignment_mode=alignment_mode,
        )
        print("✅ MTF стратегия создана")
        print()
    except Exception as e:
        print(f"❌ Ошибка создания стратегии: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Подготавливаем данные (ТОЧНО как в основном бэктесте)
    print("🔧 Подготовка данных...")
    
    # ОПТИМИЗАЦИЯ: Создаем фичи один раз для всего DataFrame (как в основном бэктесте)
    # ВАЖНО: prepare_with_indicators требует колонку 'timestamp', поэтому вызываем ДО установки индекса
    print("🔧 Создание технических индикаторов и фичей...")
    try:
        from bot.indicators import prepare_with_indicators
        
        # Подготавливаем индикаторы (нужна колонка timestamp, не индекс)
        # df_15m уже имеет колонку timestamp из collector.collect_klines
        df_with_indicators = prepare_with_indicators(df_15m.copy())
        
        # Теперь устанавливаем timestamp как индекс (как в основном бэктесте)
        df_work = df_with_indicators.copy()
        if "timestamp" in df_work.columns:
            df_work = df_work.set_index("timestamp")
        
        # Убеждаемся, что индекс - DatetimeIndex
        if not isinstance(df_work.index, pd.DatetimeIndex):
            df_work.index = pd.to_datetime(df_work.index, errors='coerce')
        
        # Сортируем по времени
        df_work = df_work.sort_index()
        
        # Создаем технические индикаторы через feature_engineer 15m модели
        df_with_features = strategy.strategy_15m.feature_engineer.create_technical_indicators(df_work)
        
        print(f"✅ Фичи созданы: {len(df_with_features)} строк 15m, {len(df_with_features.columns)} колонок")
        
        # Диагностика: проверяем, сколько будет 1h свечей после агрегации
        try:
            df_1h_test = df_with_features.resample("60min").agg({
                "open": "first",
                "high": "max", 
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            print(f"   После агрегации будет ~{len(df_1h_test)} свечей 1h")
            
            if len(df_1h_test) < 100:
                print(f"   ⚠️  ВНИМАНИЕ: Мало 1h свечей ({len(df_1h_test)}), 1h модель может не давать сигналов")
                print(f"   💡 Рекомендация: увеличьте период тестирования или снизьте порог 1h модели")
        except:
            pass
    except Exception as e:
        print(f"⚠️  Ошибка создания фичей: {e}")
        import traceback
        traceback.print_exc()
        # Продолжаем без оптимизации - устанавливаем индекс вручную
        if "timestamp" in df_15m.columns:
            df_15m = df_15m.set_index("timestamp")
        if not isinstance(df_15m.index, pd.DatetimeIndex):
            df_15m.index = pd.to_datetime(df_15m.index, errors='coerce')
        df_15m = df_15m.sort_index()
        df_with_features = df_15m
        print("⚠️  Продолжаем без оптимизации фичей (будет медленнее)")
    
    print(f"✅ Данные подготовлены: {len(df_with_features)} строк")
    print()
    
    # Запускаем бэктест
    print("📊 Запуск бэктеста...")
    print("-" * 80)
    
    simulator = MLBacktestSimulator(
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        max_position_hours=48.0,
    )
    
    # Передаем настройки размера позиции в симулятор (как в реальном боте)
    simulator._margin_pct_balance = settings.risk.margin_pct_balance  # 20% от баланса
    # Используем фиксированную сумму $100 с учетом плеча
    simulator._base_order_usd = 100.0  # Фиксированная сумма позиции $100
    
    # Генерируем сигналы для каждой свечи (ТОЧНО как в основном бэктесте)
    signals_generated = 0
    trades_executed = 0
    processed_bars = 0
    
    min_bars_required = 200  # Минимум баров для расчета индикаторов
    min_window_size = min_bars_required
    
    # Прогресс-бар
    import time
    start_time_loop = time.time()
    total_bars = len(df_with_features) - min_window_size
    
    for idx in range(min_window_size, len(df_with_features)):
        try:
            # Получаем текущие данные (как в основном бэктесте)
            current_time = df_with_features.index[idx]
            row = df_with_features.iloc[idx]
            current_price = row['close']
            high = row['high']
            low = row['low']
            
            # ВАЖНО: Используем ВСЕ данные до текущего момента ВКЛЮЧИТЕЛЬНО (как в основном бэктесте)
            # Это критично для правильной работы индикаторов и ML модели
            df_window = df_with_features.iloc[:idx+1]  # ВСЕ данные до текущего момента ВКЛЮЧИТЕЛЬНО
            
            # ВАЖНО: СНАЧАЛА проверяем выход из позиции (как реальный бот)
            # Это важно, так как может быть сигнал на закрытие текущей позиции
            if simulator.current_position is not None:
                try:
                    exited = simulator.check_exit(current_time, current_price, high, low)
                except Exception as e:
                    print(f"⚠️  Ошибка в check_exit() на свече {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Если позиция закрыта, не открываем новую на этой же итерации
                if exited:
                    processed_bars += 1
                    # Логируем прогресс
                    if processed_bars % 500 == 0:
                        elapsed = time.time() - start_time_loop if start_time_loop else 0
                        bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                        remaining = total_bars - processed_bars
                        eta_seconds = remaining / bars_per_sec if bars_per_sec > 0 else 0
                        eta_minutes = eta_seconds / 60
                        print(f"📊 Прогресс: {processed_bars}/{total_bars} баров ({processed_bars*100/total_bars:.1f}%), "
                              f"сигналов: {signals_generated}, сделок: {len(simulator.trades)}, "
                              f"скорость: {bars_per_sec:.1f} бар/сек, ETA: {eta_minutes:.1f} мин")
                    continue
            
            # Определяем текущую позицию (как в основном бэктесте)
            has_position = None
            if simulator.current_position is not None:
                has_position = Bias.LONG if simulator.current_position.action == Action.LONG else Bias.SHORT
            
            # ПОТОМ генерируем сигнал (ТОЧНО как реальный бот)
            try:
                # ОПТИМИЗАЦИЯ: Используем skip_feature_creation=True, так как фичи уже созданы
                # Это значительно ускоряет бэктест (с ~0.6 сек на бар до ~0.01 сек)
                signal = strategy.generate_signal(
                    row=row,  # Текущая свеча (как в основном бэктесте)
                    df_15m=df_window,  # Все данные до текущего момента ВКЛЮЧИТЕЛЬНО
                    df_1h=None,  # Будет агрегировано внутри
                    has_position=has_position,
                    current_price=current_price,
                    leverage=leverage,
                    target_profit_pct_margin=settings.ml_strategy.target_profit_pct_margin,
                    max_loss_pct_margin=settings.ml_strategy.max_loss_pct_margin,
                    skip_feature_creation=True,  # ОПТИМИЗАЦИЯ: фичи уже созданы
                )
                
                # ВАЛИДАЦИЯ: Проверяем, что сигнал имеет правильный тип
                if not isinstance(signal, Signal):
                    print(f"⚠️  Сигнал должен быть типа Signal, получен {type(signal)}")
                    signal = Signal(
                        timestamp=current_time,
                        action=Action.HOLD,
                        reason=f"mtf_invalid_signal_type",
                        price=current_price
                    )
                
                # ОТЛАДКА: Логируем первые несколько сигналов для диагностики
                if processed_bars < 10:
                    indicators_info = signal.indicators_info if signal.indicators_info else {}
                    reason = signal.reason if signal.reason else "unknown"
                    print(f"   🔍 Бар {idx}: {signal.action.value} | {reason[:60]}")
                    if indicators_info:
                        print(f"      1h: pred={indicators_info.get('1h_pred')}, conf={indicators_info.get('1h_conf', 0):.2f}")
                        print(f"      15m: pred={indicators_info.get('15m_pred')}, conf={indicators_info.get('15m_conf', 0):.2f}")
                        print(f"      mtf_reason: {indicators_info.get('mtf_reason', 'N/A')}")
                
            except Exception as e:
                # Если ошибка при генерации сигнала, логируем и пропускаем
                if idx < 10 or processed_bars % 1000 == 0:
                    print(f"⚠️  Ошибка генерации сигнала на {current_time} (бар {idx}): {e}")
                    if idx < 10:
                        import traceback
                        traceback.print_exc()
                signal = Signal(
                    timestamp=current_time,
                    action=Action.HOLD,
                    reason=f"mtf_error_{str(e)[:30]}",
                    price=current_price
                )
            
            # Анализируем сигнал (только статистика, без изменений)
            try:
                simulator.analyze_signal(signal, current_price)
            except Exception as e:
                print(f"⚠️  Ошибка в analyze_signal(): {e}")
            
            # Открываем новую позицию, если есть сигнал
            if signal and signal.action != Action.HOLD:
                signals_generated += 1
                trade_opened = simulator.open_position(signal, current_time, symbol)
                if trade_opened:
                    trades_executed += 1
            
            processed_bars += 1
            
            # Логируем прогресс
            if processed_bars % 500 == 0:
                elapsed = time.time() - start_time_loop if start_time_loop else 0
                bars_per_sec = processed_bars / elapsed if elapsed > 0 else 0
                remaining = total_bars - processed_bars
                eta_seconds = remaining / bars_per_sec if bars_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                print(f"📊 Прогресс: {processed_bars}/{total_bars} баров ({processed_bars*100/total_bars:.1f}%), "
                      f"сигналов: {signals_generated}, сделок: {len(simulator.trades)}, "
                      f"скорость: {bars_per_sec:.1f} бар/сек, ETA: {eta_minutes:.1f} мин")
        
        except Exception as e:
            print(f"⚠️  Ошибка на свече {idx}: {e}")
            import traceback
            if idx < 10:
                traceback.print_exc()
            continue
    
    # Закрываем открытые позиции
    if simulator.current_position is not None:
        last_row = df_with_features.iloc[-1]
        last_time = df_with_features.index[-1]
        simulator.close_position(
            exit_time=last_time,
            exit_price=last_row['close'],
            exit_reason=ExitReason.END_OF_BACKTEST
        )
    
    # Вычисляем метрики
    print("-" * 80)
    print("📈 Вычисление метрики...")
    
    metrics = simulator.calculate_metrics(
        symbol=symbol,
        model_name=f"MTF_{Path(model_1h_path).stem}_{Path(model_15m_path).stem}",
    )
    
    # Выводим результаты
    print()
    print("=" * 80)
    print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("=" * 80)
    print(f"Символ: {symbol}")
    print(f"Модель: MTF (1h + 15m)")
    print(f"Период: {days_back} дней")
    print()
    print(f"Сделок: {metrics.total_trades}")
    print(f"Прибыльных: {metrics.winning_trades}")
    print(f"Убыточных: {metrics.losing_trades}")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print()
    print(f"Общий PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:.2f}%)")
    print(f"Средний выигрыш: ${metrics.avg_win:.2f}")
    print(f"Средний проигрыш: ${metrics.avg_loss:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print()
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print()
    print(f"Сигналов сгенерировано: {signals_generated}")
    print(f"Сделок выполнено: {trades_executed}")
    print("=" * 80)
    
    return metrics


def main():
    """CLI для бэктеста MTF стратегии."""
    parser = argparse.ArgumentParser(
        description="Бэктест комбинированной MTF стратегии (1h + 15m)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Бэктест с лучшими моделями из сравнения (по умолчанию)
  python backtest_mtf_strategy.py --symbol BTCUSDT --days 30
  
  # Бэктест с конкретными моделями
  python backtest_mtf_strategy.py --symbol BTCUSDT --days 30 \\
      --model-1h quad_ensemble_BTCUSDT_60_1h \\
      --model-15m quad_ensemble_BTCUSDT_15_15m
  
  # Тестировать ВСЕ комбинации моделей
  python backtest_mtf_strategy.py --symbol BTCUSDT --days 30 --test-all-combinations
  
  # Бэктест с кастомными порогами
  python backtest_mtf_strategy.py --symbol ETHUSDT --days 60 --conf-1h 0.60 --conf-15m 0.40
  
  # Бэктест в режиме взвешенного голосования
  python backtest_mtf_strategy.py --symbol SOLUSDT --days 30 --alignment-mode weighted
  
  # Использовать первые найденные модели (не лучшие из сравнения)
  python backtest_mtf_strategy.py --symbol BTCUSDT --days 30 --no-use-best
        """
    )
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Торговая пара")
    parser.add_argument("--days", type=int, default=30, help="Количество дней для тестирования")
    parser.add_argument("--balance", type=float, default=100.0, help="Начальный баланс")
    parser.add_argument("--risk", type=float, default=0.02, help="Риск на сделку (0.02 = 2%%)")
    parser.add_argument("--leverage", type=int, default=10, help="Плечо")
    
    parser.add_argument("--model-1h", type=str, default=None, help="Путь к 1h модели или имя файла (без .pkl)")
    parser.add_argument("--model-15m", type=str, default=None, help="Путь к 15m модели или имя файла (без .pkl)")
    
    parser.add_argument("--test-all-combinations", action="store_true",
                       help="Тестировать ВСЕ комбинации моделей 1h и 15m")
    parser.add_argument("--use-best-from-comparison", action="store_true", default=True,
                       help="Использовать лучшие модели из результатов сравнения (по умолчанию: True)")
    parser.add_argument("--no-use-best", action="store_true",
                       help="НЕ использовать лучшие модели из сравнения (использовать первые найденные)")
    
    parser.add_argument("--conf-1h", type=float, default=0.50, help="Порог уверенности для 1h модели")
    parser.add_argument("--conf-15m", type=float, default=0.35, help="Порог уверенности для 15m модели")
    
    parser.add_argument("--alignment-mode", type=str, default="strict", choices=["strict", "weighted"],
                       help="Режим выравнивания: strict (строгое совпадение) или weighted (взвешенное голосование)")
    parser.add_argument("--no-require-alignment", action="store_true",
                       help="Не требовать совпадение направлений (только для weighted режима)")
    
    parser.add_argument("--save", action="store_true", help="Сохранить результаты в файл")
    parser.add_argument("--plot", action="store_true", help="Построить графики")
    
    args = parser.parse_args()
    
    # Если указан тест всех комбинаций
    if args.test_all_combinations:
        df_results = run_mtf_backtest_all_combinations(
            symbol=args.symbol,
            days_back=args.days,
            initial_balance=args.balance,
            risk_per_trade=args.risk,
            leverage=args.leverage,
            confidence_threshold_1h=args.conf_1h,
            confidence_threshold_15m=args.conf_15m,
            alignment_mode=args.alignment_mode,
            require_alignment=not args.no_require_alignment,
        )
        return
    
    # Если указаны имена моделей (без пути), ищем их
    model_1h_path = args.model_1h
    model_15m_path = args.model_15m
    
    if model_1h_path and not Path(model_1h_path).exists():
        # Возможно, это имя файла без .pkl
        if not model_1h_path.endswith('.pkl'):
            model_1h_path = f"{model_1h_path}.pkl"
        # Ищем в ml_models
        model_1h_full = Path("ml_models") / model_1h_path
        if model_1h_full.exists():
            model_1h_path = str(model_1h_full)
        else:
            print(f"⚠️  Модель 1h не найдена: {args.model_1h}")
            model_1h_path = None
    
    if model_15m_path and not Path(model_15m_path).exists():
        # Возможно, это имя файла без .pkl
        if not model_15m_path.endswith('.pkl'):
            model_15m_path = f"{model_15m_path}.pkl"
        # Ищем в ml_models
        model_15m_full = Path("ml_models") / model_15m_path
        if model_15m_full.exists():
            model_15m_path = str(model_15m_full)
        else:
            print(f"⚠️  Модель 15m не найдена: {args.model_15m}")
            model_15m_path = None
    
    # Извлекаем имена моделей для find_models_for_symbol (если указаны)
    model_1h_name = Path(model_1h_path).stem if model_1h_path and Path(model_1h_path).exists() else None
    model_15m_name = Path(model_15m_path).stem if model_15m_path and Path(model_15m_path).exists() else None
    
    # Запускаем бэктест
    metrics = run_mtf_backtest(
        symbol=args.symbol,
        days_back=args.days,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        model_1h_path=model_1h_path,
        model_15m_path=model_15m_path,
        confidence_threshold_1h=args.conf_1h,
        confidence_threshold_15m=args.conf_15m,
        alignment_mode=args.alignment_mode,
        require_alignment=not args.no_require_alignment,
    )
    
    if metrics and args.save:
        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_mtf_{args.symbol}_{timestamp}.json"
        
        # Сохраняем метрики в JSON
        results_dir = Path("backtest_reports")
        results_dir.mkdir(exist_ok=True)
        filepath = results_dir / filename
        
        import json
        from dataclasses import asdict
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
        
        print(f"\n✅ Результаты сохранены в {filepath}")
    
    if metrics and args.plot:
        # Строим графики (базовая реализация)
        print("\n📊 Построение графиков...")
        print("⚠️  Функция построения графиков будет добавлена в следующей версии")
        print("   Используйте backtest_ml_strategy.py --plot для полных графиков")


if __name__ == "__main__":
    main()
