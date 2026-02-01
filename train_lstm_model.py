"""
Скрипт для обучения LSTM модели (четвертая ML стратегия).
Использование: python train_lstm_model.py --symbol BTCUSDT --days 180
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
from bot.ml.lstm_model import LSTMTrainer


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
    parser = argparse.ArgumentParser(description='Train LSTM ML model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=180,
                       help='Number of days of historical data (default: 180)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Timeframe interval (default: 15m)')
    parser.add_argument('--sequence_length', type=int, default=90,
                       help='Sequence length in candles (default: 90, improved from 60)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size (default: 128, improved from 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100, improved from 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        return
    
    safe_print("=" * 70)
    safe_print("LSTM ML Model Training")
    safe_print("=" * 70)
    safe_print(f"Symbol: {args.symbol}")
    safe_print(f"Days: {args.days}")
    safe_print(f"Interval: {args.interval}")
    safe_print(f"Sequence Length: {args.sequence_length} candles")
    safe_print(f"Hidden Size: {args.hidden_size}")
    safe_print(f"Layers: {args.num_layers}")
    safe_print(f"Epochs: {args.epochs}")
    safe_print("=" * 70)
    
    # Загружаем настройки
    settings = load_settings()
    
    # === Шаг 1: Сбор данных ===
    safe_print(f"\n[Step 1] Collecting historical data for {args.symbol}...")
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
        safe_print(f"ERROR: No data collected for {args.symbol}. Skipping.")
        return
    
    safe_print(f"OK: Collected {len(df_raw)} candles")
    
    # === Шаг 2: Feature Engineering ===
    safe_print(f"\n[Step 2] Creating features...")
    feature_engineer = FeatureEngineer()
    
    # Создаем технические индикаторы
    df_features = feature_engineer.create_technical_indicators(df_raw)
    safe_print(f"OK: Created {len(feature_engineer.get_feature_names())} features")
    
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
    
    # Проверяем, достаточно ли данных для последовательностей
    if len(df_with_target) < args.sequence_length + 100:
        safe_print(f"ERROR: Not enough data. Need at least {args.sequence_length + 100} candles, got {len(df_with_target)}")
        return
    
    # === Шаг 4: Обучение LSTM модели ===
    safe_print(f"\n[Step 4] Training LSTM model...")
    
    trainer = LSTMTrainer(
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )
    
    model, metrics = trainer.train(df_with_target, validation_split=0.2)
    
    # === Шаг 5: Сохранение модели ===
    safe_print(f"\n[Step 5] Saving model...")
    
    model_filename = f"lstm_{args.symbol}_{args.interval.replace('m', '')}.pkl"
    model_path = Path("ml_models") / model_filename
    
    trainer.save_model(
        str(model_path),
        feature_engineer.get_feature_names(),
        metrics,
        symbol=args.symbol,
        interval=args.interval.replace('m', ''),
    )
    
    safe_print(f"OK: Model saved: {model_path}")
    safe_print(f"\nSUCCESS: Training completed successfully!")
    safe_print(f"\nResults:")
    safe_print(f"   Best Validation Accuracy: {metrics['best_val_acc']:.4f}")
    safe_print(f"   Final Validation Accuracy: {metrics['final_val_acc']:.4f}")
    safe_print(f"   Epochs Trained: {metrics['num_epochs_trained']}")
    safe_print(f"\nNext steps:")
    safe_print(f"   1. Test the model on historical data")
    safe_print(f"   2. Compare with other ML models")
    safe_print(f"   3. Integrate into live trading")


if __name__ == "__main__":
    main()