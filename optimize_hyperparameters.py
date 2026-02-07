"""
Скрипт для оптимизации гиперпараметров лучших моделей через Grid Search.

Использование:
    python optimize_hyperparameters.py --model rf_BTCUSDT_15_mtf --symbol BTCUSDT
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer
from bot.config import load_settings


def optimize_rf_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]] = None
) -> Dict[str, Any]:
    """Оптимизация гиперпараметров Random Forest."""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [8, 10, 12, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None],
        }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Используем TimeSeriesSplit для временных рядов
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Используем f1_macro для лучшей оценки на несбалансированных данных
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring='f1_macro',  # Лучше для несбалансированных классов
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }


def optimize_xgb_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]] = None
) -> Dict[str, Any]:
    """Оптимизация гиперпараметров XGBoost."""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
        }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Используем f1_macro для лучшей оценки на несбалансированных данных
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=tscv,
        scoring='f1_macro',  # Лучше для несбалансированных классов
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }


def optimize_lgb_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]] = None
) -> Dict[str, Any]:
    """Оптимизация гиперпараметров LightGBM."""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'num_leaves': [31, 50, 70, 100],
            'subsample': [0.8, 0.9, 1.0],
            'min_child_samples': [20, 30, 50],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5],
        }
    
    lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Используем f1_macro для лучшей оценки на несбалансированных данных
    grid_search = GridSearchCV(
        lgb_model,
        param_grid,
        cv=tscv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for ML models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model type: rf, xgb, lgb')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Symbol (e.g., BTCUSDT). If not specified, optimizes for all symbols')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT,SOLUSDT)')
    parser.add_argument('--interval', type=str, default='15',
                       help='Interval (default: 15)')
    parser.add_argument('--mtf', action='store_true',
                       help='Use MTF features')
    
    args = parser.parse_args()
    
    # Определяем список символов
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # По умолчанию все 6 торговых пар
        symbols = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT"]
    
    print("=" * 80)
    print("🔧 ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
    print("=" * 80)
    print(f"Модель: {args.model}")
    print(f"Символы: {', '.join(symbols)} ({len(symbols)} символов)")
    print(f"Интервал: {args.interval}")
    print(f"MTF: {args.mtf}")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    all_results = []
    
    # Оптимизируем для каждого символа
    for symbol in symbols:
        print("\n" + "=" * 80)
        print(f"📊 ОПТИМИЗАЦИЯ ДЛЯ {symbol}")
        print("=" * 80)
        
        # Собираем данные
        print("\n📥 Сбор данных...")
        collector = DataCollector(settings.api)
        df_raw = collector.collect_klines(
            symbol=symbol,
            interval=args.interval,
            start_date=None,
            end_date=None,
            limit=5000,
            save_to_file=False,
        )
        
        if df_raw.empty:
            print(f"❌ Нет данных для {symbol}, пропускаем")
            continue
        
        print(f"✅ Собрано {len(df_raw)} свечей")
        
        # Создаем фичи
        print("\n🔧 Создание фичей...")
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_technical_indicators(df_raw)
        
        # MTF фичи
        if args.mtf:
            print("   Добавление MTF фичей...")
            # Собираем данные для высших ТФ
            df_1h = collector.collect_klines(
                symbol=symbol,
                interval="60",
                start_date=None,
                end_date=None,
                limit=2000,
                save_to_file=False,
            )
            df_4h = collector.collect_klines(
                symbol=symbol,
                interval="240",
                start_date=None,
                end_date=None,
                limit=1000,
                save_to_file=False,
            )
            
            higher_timeframes = {}
            if not df_1h.empty:
                higher_timeframes["60"] = df_1h
            if not df_4h.empty:
                higher_timeframes["240"] = df_4h
            
            if higher_timeframes:
                df_features = feature_engineer.add_mtf_features(df_features, higher_timeframes)
        
        # Создаем target
        print("\n🎯 Создание target variable...")
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=5,
            threshold_pct=0.5,
            use_atr_threshold=True,
            use_risk_adjusted=True,
            min_risk_reward_ratio=1.5,
            max_hold_periods=96,
            min_profit_pct=0.5,
            use_adaptive_params=True,
        )
        
        # Подготавливаем данные
        print("\n📊 Подготовка данных для ML...")
        X, y = feature_engineer.prepare_features_for_ml(df_with_target)
        
        print(f"   X shape: {X.shape}, y shape: {y.shape}")
        print(f"   Распределение классов: {np.bincount(y + 1)}")  # +1 для индексации
        
        # Оптимизация
        print(f"\n🔍 Оптимизация гиперпараметров для {args.model}...")
        
        try:
            if args.model == 'rf':
                results = optimize_rf_hyperparameters(X, y)
            elif args.model == 'xgb':
                results = optimize_xgb_hyperparameters(X, y)
            elif args.model == 'lgb':
                results = optimize_lgb_hyperparameters(X, y)
            else:
                print(f"❌ Неизвестный тип модели: {args.model}")
                continue
            
            # Сохраняем результаты для этого символа
            result_dict = {
                'model': args.model,
                'symbol': symbol,
                'interval': args.interval,
                'mtf': args.mtf,
                'best_params': results['best_params'],
                'best_score': float(results['best_score']),
                'timestamp': datetime.now().isoformat(),
            }
            
            all_results.append(result_dict)
            
            print(f"\n✅ {symbol} - Лучшие параметры:")
            for param, value in results['best_params'].items():
                print(f"   {param}: {value}")
            print(f"   Лучший score: {results['best_score']:.4f}")
            
        except Exception as e:
            print(f"❌ Ошибка при оптимизации {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сохраняем все результаты в один файл
    if all_results:
        output_file = f"hyperparams_{args.model}_all_{args.interval}{'_mtf' if args.mtf else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ДЛЯ ВСЕХ СИМВОЛОВ")
        print("=" * 80)
        print(f"Обработано символов: {len(all_results)}/{len(symbols)}")
        print(f"\n💾 Все результаты сохранены в: {output_file}")
        
        # Выводим сводку
        print("\n📊 СВОДКА:")
        for result in all_results:
            print(f"   {result['symbol']}: score = {result['best_score']:.4f}")
    else:
        print("\n❌ Не удалось оптимизировать ни один символ")


if __name__ == "__main__":
    main()
