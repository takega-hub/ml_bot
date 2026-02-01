"""
Скрипт для обучения QuadEnsemble модели (RF + XGBoost + LightGBM + LSTM).
Использование: python train_quad_ensemble.py --symbol BTCUSDT --days 180
"""
import warnings
import os
import argparse
import sys
from pathlib import Path

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

# Настройка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is not installed!")
    print("   Install with: pip install torch>=2.0.0")

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE, LSTM_AVAILABLE


def safe_print(*args, **kwargs):
    """Безопасный вывод для Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Заменяем эмодзи на текстовые метки
        text = ' '.join(str(arg) for arg in args)
        text = text.replace('🚀', '[START]')
        text = text.replace('📊', '[INFO]')
        text = text.replace('✅', '[OK]')
        text = text.replace('❌', '[ERROR]')
        text = text.replace('⚠️', '[WARNING]')
        text = text.replace('🎉', '[SUCCESS]')
        text = text.replace('💡', '[TIP]')
        print(text, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Train QuadEnsemble ML model (RF+XGB+LGB+LSTM)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=180,
                       help='Number of days of historical data (default: 180)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Timeframe interval (default: 15m)')
    
    # RandomForest параметры
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                       help='Number of RF estimators (default: 100)')
    parser.add_argument('--rf_max_depth', type=int, default=10,
                       help='RF max depth (default: 10)')
    
    # XGBoost параметры
    parser.add_argument('--xgb_n_estimators', type=int, default=100,
                       help='Number of XGB estimators (default: 100)')
    parser.add_argument('--xgb_max_depth', type=int, default=6,
                       help='XGB max depth (default: 6)')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1,
                       help='XGB learning rate (default: 0.1)')
    
    # LightGBM параметры
    parser.add_argument('--lgb_n_estimators', type=int, default=100,
                       help='Number of LGB estimators (default: 100)')
    parser.add_argument('--lgb_max_depth', type=int, default=6,
                       help='LGB max depth (default: 6)')
    parser.add_argument('--lgb_learning_rate', type=float, default=0.1,
                       help='LGB learning rate (default: 0.1)')
    
    # LSTM параметры
    parser.add_argument('--lstm_sequence_length', type=int, default=60,
                       help='LSTM sequence length (default: 60)')
    parser.add_argument('--lstm_hidden_size', type=int, default=64,
                       help='LSTM hidden size (default: 64)')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='LSTM number of layers (default: 2)')
    parser.add_argument('--lstm_epochs', type=int, default=50,
                       help='LSTM training epochs (default: 50)')
    parser.add_argument('--lstm_batch_size', type=int, default=32,
                       help='LSTM batch size (default: 32)')
    parser.add_argument('--lstm_learning_rate', type=float, default=0.001,
                       help='LSTM learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Проверяем зависимости
    if not TORCH_AVAILABLE:
        return
    
    if not LIGHTGBM_AVAILABLE:
        safe_print("ERROR: LightGBM is not installed!")
        safe_print("   Install with: pip install lightgbm>=4.0.0")
        return
    
    if not LSTM_AVAILABLE:
        safe_print("ERROR: LSTM module is not available!")
        safe_print("   Check that bot.ml.lstm_model can be imported")
        return
    
    safe_print("=" * 80)
    safe_print("QuadEnsemble ML Model Training (RF + XGBoost + LightGBM + LSTM)")
    safe_print("=" * 80)
    safe_print(f"Symbol: {args.symbol}")
    safe_print(f"Days: {args.days}")
    safe_print(f"Interval: {args.interval}")
    safe_print(f"\nModel Parameters:")
    safe_print(f"  RandomForest: {args.rf_n_estimators} trees, max_depth={args.rf_max_depth}")
    safe_print(f"  XGBoost: {args.xgb_n_estimators} trees, max_depth={args.xgb_max_depth}, lr={args.xgb_learning_rate}")
    safe_print(f"  LightGBM: {args.lgb_n_estimators} trees, max_depth={args.lgb_max_depth}, lr={args.lgb_learning_rate}")
    safe_print(f"  LSTM: seq_len={args.lstm_sequence_length}, hidden={args.lstm_hidden_size}, layers={args.lstm_num_layers}, epochs={args.lstm_epochs}")
    safe_print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Определяем, использовать ли MTF-режим при обучении (читаем из окружения)
    ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "1")
    ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
    mode_suffix = "mtf" if ml_mtf_enabled else "15m"
    
    # === Шаг 1: Сбор данных ===
    if ml_mtf_enabled:
        safe_print(f"\n[Step 1] Collecting historical data (15m, 1h, 4h) for {args.symbol}...")
    else:
        safe_print(f"\n[Step 1] Collecting historical data (15m only) for {args.symbol}...")
    collector = DataCollector(settings.api)
    
    # Базовый интервал (ожидаем '15m')
    base_interval = args.interval.replace('m', '')
    
    if ml_mtf_enabled:
        # Собираем данные сразу для нескольких таймфреймов
        mtf_data = collector.collect_multiple_timeframes(
            symbol=args.symbol,
            intervals=[base_interval, "60", "240"],  # 15m, 1h, 4h
            start_date=None,
            end_date=None,
        )
        
        df_raw_15m = mtf_data.get(base_interval)
        df_raw_1h = mtf_data.get("60")
        df_raw_4h = mtf_data.get("240")
        
        if df_raw_15m is None or df_raw_15m.empty:
            safe_print(f"ERROR: No 15m data collected for {args.symbol}. Skipping.")
            return
        
        safe_print(f"OK: Collected {len(df_raw_15m)} candles on 15m timeframe")
    else:
        # Старый режим: собираем только 15m данные
        df_raw_15m = collector.collect_klines(
            symbol=args.symbol,
            interval=base_interval,
            start_date=None,
            end_date=None,
            limit=200,
        )
        if df_raw_15m.empty:
            safe_print(f"ERROR: No 15m data collected for {args.symbol}. Skipping.")
            return
        safe_print(f"OK: Collected {len(df_raw_15m)} candles on 15m timeframe (no higher TF)")
    
    # === Шаг 2: Feature Engineering (включая MTF при необходимости) ===
    safe_print(f"\n[Step 2] Creating features{' (including higher timeframes)' if ml_mtf_enabled else ' (15m only)'}...")
    feature_engineer = FeatureEngineer()
    
    # Создаем технические индикаторы на базовом ТФ (15m)
    df_features = feature_engineer.create_technical_indicators(df_raw_15m)
    
    if ml_mtf_enabled:
        # Добавляем мульти‑таймфреймовые признаки (1h, 4h), если данные есть
        higher_timeframes = {}
        df_raw_1h = mtf_data.get("60")
        df_raw_4h = mtf_data.get("240")
        if df_raw_1h is not None and not df_raw_1h.empty:
            higher_timeframes["60"] = df_raw_1h
        if df_raw_4h is not None and not df_raw_4h.empty:
            higher_timeframes["240"] = df_raw_4h
        
        if higher_timeframes:
            df_features = feature_engineer.add_mtf_features(df_features, higher_timeframes)
            safe_print(f"OK: Created {len(feature_engineer.get_feature_names())} features (with MTF)")
        else:
            safe_print("WARNING: Could not collect 1h/4h data — training on 15m features only.")
            safe_print(f"OK: Created {len(feature_engineer.get_feature_names())} features")
    else:
        safe_print(f"OK: Created {len(feature_engineer.get_feature_names())} features (15m only)")
    
    # Создаем целевую переменную
    safe_print(f"\n[Step 3] Creating target variable...")
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=5,  # 5 * 15m = 75 минут
        threshold_pct=1.0,  # 1.0% порог
        use_atr_threshold=True,
        use_risk_adjusted=True,
        min_risk_reward_ratio=2.0,
        max_hold_periods=48,
        min_profit_pct=1.0,
    )
    
    target_dist = df_with_target['target'].value_counts().to_dict()
    safe_print(f"OK: Target distribution:")
    for target_val, count in sorted(target_dist.items()):
        pct = (count / len(df_with_target)) * 100
        target_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(
            target_val, f"UNKNOWN({target_val})")
        safe_print(f"    {target_name:6s}: {count:5d} ({pct:5.1f}%)")
    
    # === Шаг 4: Подготовка данных для ML ===
    safe_print(f"\n[Step 4] Preparing data for ML...")
    X, y = feature_engineer.prepare_features_for_ml(df_with_target)
    safe_print(f"OK: Prepared data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Проверяем, что данных достаточно для LSTM
    if len(df_with_target) < args.lstm_sequence_length + 100:
        safe_print(f"WARNING: Not enough data for LSTM (need at least {args.lstm_sequence_length + 100} rows, got {len(df_with_target)})")
        safe_print(f"   Consider reducing --lstm_sequence_length or collecting more data")
    
    # === Шаг 5: Обучение QuadEnsemble ===
    safe_print(f"\n[Step 5] Training QuadEnsemble...")
    safe_print(f"   This will train 4 models sequentially:")
    safe_print(f"   1. RandomForest")
    safe_print(f"   2. XGBoost")
    safe_print(f"   3. LightGBM")
    safe_print(f"   4. LSTM (this may take longer)")
    safe_print()
    
    # Вычисляем веса классов для фокуса на прибыльных сделках
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(y)
    base_weights = compute_class_weight('balanced', classes=classes, y=y)
    
    # УСИЛЕННЫЕ веса для LONG/SHORT, МИНИМИЗИРУЕМ HOLD
    class_weight_dict = {}
    for i, cls in enumerate(classes):
        if cls == 0:  # HOLD
            class_weight_dict[cls] = base_weights[i] * 0.1  # Сильно уменьшаем вес HOLD
        else:  # LONG or SHORT
            class_weight_dict[cls] = base_weights[i] * 3.0  # Увеличиваем вес LONG/SHORT
    
    safe_print(f"   Class weights:")
    for cls, weight in class_weight_dict.items():
        label_name = "LONG" if cls == 1 else ("SHORT" if cls == -1 else "HOLD")
        safe_print(f"      {label_name}: {weight:.3f}")
    safe_print()
    
    trainer = ModelTrainer()
    
    try:
        model, metrics = trainer.train_quad_ensemble(
            X=X,
            y=y,
            df=df_with_target,  # Полный DataFrame для LSTM
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            lgb_n_estimators=args.lgb_n_estimators,
            lgb_max_depth=args.lgb_max_depth,
            lgb_learning_rate=args.lgb_learning_rate,
            lstm_sequence_length=args.lstm_sequence_length,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_epochs=args.lstm_epochs,
            class_weight=class_weight_dict,  # Улучшенные веса классов
        )
        
        safe_print(f"\nQuadEnsemble Results:")
        safe_print(f"  RandomForest CV Accuracy: {metrics['rf_metrics']['cv_mean']:.4f} (+/- {metrics['rf_metrics']['cv_std'] * 2:.4f})")
        safe_print(f"  XGBoost CV Accuracy: {metrics['xgb_metrics']['cv_mean']:.4f} (+/- {metrics['xgb_metrics']['cv_std'] * 2:.4f})")
        safe_print(f"  LightGBM CV Accuracy: {metrics['lgb_metrics']['cv_mean']:.4f} (+/- {metrics['lgb_metrics']['cv_std'] * 2:.4f})")
        safe_print(f"  LSTM Accuracy: {metrics['lstm_metrics'].get('accuracy', 0):.4f}")
        safe_print(f"\n  Ensemble Weights:")
        safe_print(f"    RF:   {metrics['rf_weight']:.3f}")
        safe_print(f"    XGB:  {metrics['xgb_weight']:.3f}")
        safe_print(f"    LGB:  {metrics['lgb_weight']:.3f}")
        safe_print(f"    LSTM: {metrics['lstm_weight']:.3f}")
        
    except Exception as e:
        safe_print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === Шаг 6: Сохранение модели ===
    safe_print(f"\n[Step 6] Saving model...")
    model_filename = f"quad_ensemble_{args.symbol}_{args.interval.replace('m', '')}_{mode_suffix}.pkl"
    
    try:
        trainer.save_model(
            model,
            trainer.scaler,
            feature_engineer.get_feature_names(),
            metrics,
            model_filename,
            symbol=args.symbol,
            interval=args.interval.replace('m', ''),
            model_type="quad_ensemble",
        )
        
        safe_print(f"OK: Model saved: {model_filename}")
        safe_print(f"\nSUCCESS: Training completed successfully!")
        safe_print(f"\nNext steps:")
        safe_print(f"   1. Test the model: python -m bot.ml.diagnose_model ml_models/{model_filename}")
        safe_print(f"   2. Backtest: python backtest_ml_strategy.py --model ml_models/{model_filename} --symbol {args.symbol} --days 30")
        safe_print(f"   3. Use in live trading: Enable ML strategy in config")
        safe_print(f"   4. Compare with other models: Check backtest results")
        
    except Exception as e:
        safe_print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()