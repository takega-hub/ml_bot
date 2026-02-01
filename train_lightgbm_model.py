"""
Скрипт для обучения LightGBM модели (третья ML стратегия).
Использование: python train_lightgbm_model.py --symbol BTCUSDT --days 180
"""
import warnings
import os
import argparse
import sys
from pathlib import Path

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM ML model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=180,
                       help='Number of days of historical data (default: 180)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Timeframe interval (default: 15m)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Train triple ensemble (RF+XGB+LGB) instead of single LightGBM')
    parser.add_argument('--n_estimators', type=int, default=150,
                       help='Number of estimators for LightGBM (default: 150)')
    parser.add_argument('--max_depth', type=int, default=7,
                       help='Max depth for LightGBM (default: 7)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='Learning rate for LightGBM (default: 0.05)')
    
    args = parser.parse_args()
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ ERROR: LightGBM is not installed!")
        print("   Install with: pip install lightgbm>=4.0.0")
        return
    
    print("=" * 70)
    print("🚀 LightGBM ML Model Training")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print(f"Mode: {'Triple Ensemble (RF+XGB+LGB)' if args.ensemble else 'Single LightGBM'}")
    print("=" * 70)
    
    # Загружаем настройки
    settings = load_settings()
    
    # === Шаг 1: Сбор данных ===
    print(f"\n[Step 1] Collecting historical data for {args.symbol}...")
    collector = DataCollector(settings.api)
    
    # Собираем данные
    df_raw = collector.collect_klines(
        symbol=args.symbol,
        interval=args.interval.replace('m', ''),
        start_date=None,
        end_date=None,
        limit=200,
    )
    
    if df_raw.empty:
        print(f"❌ No data collected for {args.symbol}. Skipping.")
        return
    
    print(f"✅ Collected {len(df_raw)} candles")
    
    # === Шаг 2: Feature Engineering ===
    print(f"\n[Step 2] Creating features...")
    feature_engineer = FeatureEngineer()
    
    # Создаем технические индикаторы
    df_features = feature_engineer.create_technical_indicators(df_raw)
    print(f"✅ Created {len(feature_engineer.get_feature_names())} features")
    
    # Создаем целевую переменную
    print(f"\n[Step 3] Creating target variable...")
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=5,  # 5 * 15m = 75 минут
        threshold_pct=1.0,  # 1.0% порог
        use_atr_threshold=True,
        use_risk_adjusted=True,
        min_risk_reward_ratio=2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам TP=25%, SL=10%)
        max_hold_periods=48,  # Максимум 48 * 15m = 12 часов для качественных сделок (смягчено: было 32)
        min_profit_pct=1.0,  # Минимальная прибыль 1.0% для классификации как LONG/SHORT (смягчено: было 1.5%)
    )
    
    target_dist = df_with_target['target'].value_counts().to_dict()
    print(f"✅ Target distribution:")
    for target_val, count in sorted(target_dist.items()):
        pct = (count / len(df_with_target)) * 100
        target_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(
            target_val, f"UNKNOWN({target_val})")
        print(f"    {target_name:6s}: {count:5d} ({pct:5.1f}%)")
    
    # === Шаг 4: Подготовка данных для ML ===
    print(f"\n[Step 4] Preparing data for ML...")
    X, y = feature_engineer.prepare_features_for_ml(df_with_target)
    print(f"✅ Prepared data: X.shape={X.shape}, y.shape={y.shape}")
    
    # === Шаг 5: Обучение модели ===
    print(f"\n[Step 5] Training model...")
    trainer = ModelTrainer()
    
    if args.ensemble:
        # Обучаем тройной ансамбль
        print(f"\n🎯 Training Triple Ensemble (RF + XGBoost + LightGBM)...")
        model, metrics = trainer.train_ensemble(
            X, y,
            ensemble_method="triple",
            include_lightgbm=True,
            rf_n_estimators=100,
            rf_max_depth=10,
            xgb_n_estimators=100,
            xgb_max_depth=6,
            xgb_learning_rate=0.1,
            lgb_n_estimators=args.n_estimators,
            lgb_max_depth=args.max_depth,
            lgb_learning_rate=args.learning_rate,
        )
        
        model_filename = f"triple_ensemble_{args.symbol}_{args.interval.replace('m', '')}.pkl"
        model_type = "triple_ensemble"
        
        print(f"\n📊 Triple Ensemble Results:")
        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        print(f"  F1-Score: {metrics.get('cv_f1_mean', 0):.4f}")
        print(f"  Weights: RF={metrics['rf_weight']:.3f}, "
              f"XGB={metrics['xgb_weight']:.3f}, "
              f"LGB={metrics['lgb_weight']:.3f}")
    else:
        # Обучаем отдельную LightGBM модель
        print(f"\n🎯 Training LightGBM Classifier...")
        model, metrics = trainer.train_lightgbm_classifier(
            X, y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
        
        model_filename = f"lgb_{args.symbol}_{args.interval.replace('m', '')}.pkl"
        model_type = "lightgbm"
        
        print(f"\n📊 LightGBM Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
    
    # === Шаг 6: Сохранение модели ===
    print(f"\n[Step 6] Saving model...")
    trainer.save_model(
        model,
        trainer.scaler,
        feature_engineer.get_feature_names(),
        metrics,
        model_filename,
        symbol=args.symbol,
        interval=args.interval.replace('m', ''),
        model_type=model_type,
    )
    
    print(f"✅ Model saved: {model_filename}")
    print(f"\n🎉 Training completed successfully!")
    print(f"\n💡 Next steps:")
    print(f"   1. Test the model: python -m bot.ml.diagnose_model {model_filename}")
    print(f"   2. Use in live trading: Enable ML strategy in config")
    print(f"   3. Compare with other models: Check backtest results")


if __name__ == "__main__":
    main()
