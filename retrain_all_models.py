"""
Мастер-скрипт для обучения всех 6 типов моделей для всех символов.
"""
import warnings
import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

# Настройки
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

sys.path.insert(0, str(Path(__file__).parent))

# Конфигурация
SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
MODEL_CONFIGS = [
    {
        "name": "rf",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "15"  # Базовый интервал
    },
    {
        "name": "xgb", 
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "15"
    },
    {
        "name": "ensemble",
        "script": "retrain_ml_optimized.py", 
        "args": ["--days", "180"],
        "suffix": "15"
    },
    {
        "name": "triple_ensemble",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "15"
    },
    {
        "name": "quad_ensemble",
        "script": "train_quad_ensemble.py",
        "args": ["--days", "180", "--interval", "15m"],
        "suffix": "15"
    },
    {
        "name": "lstm",
        "script": "train_lstm_model.py",
        "args": ["--days", "180", "--interval", "15m"],
        "suffix": "15"
    }
]

# MTF варианты (если нужны)
MTF_MODEL_CONFIGS = [
    {
        "name": "rf_mtf",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "mtf",
        "env": {"ML_MTF_ENABLED": "1"}
    },
    {
        "name": "xgb_mtf",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "mtf",
        "env": {"ML_MTF_ENABLED": "1"}
    },
    {
        "name": "ensemble_mtf",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "mtf", 
        "env": {"ML_MTF_ENABLED": "1"}
    },
    {
        "name": "triple_ensemble_mtf",
        "script": "retrain_ml_optimized.py",
        "args": ["--days", "180"],
        "suffix": "mtf",
        "env": {"ML_MTF_ENABLED": "1"}
    },
    {
        "name": "quad_ensemble_mtf",
        "script": "train_quad_ensemble.py",
        "args": ["--days", "180", "--interval", "15m"],
        "suffix": "mtf",
        "env": {"ML_MTF_ENABLED": "1"}
    }
]

def run_training(config, symbol, use_mtf=False):
    """Запускает обучение одной модели."""
    
    print(f"\n{'='*80}")
    print(f"🚀 Обучение: {config['name']} для {symbol}")
    print(f"{'='*80}")
    
    # Формируем команду
    cmd = [sys.executable, config['script'], "--symbol", symbol]
    cmd.extend(config['args'])
    
    # Добавляем MTF параметры если нужно
    if use_mtf and 'env' in config:
        env = os.environ.copy()
        env.update(config['env'])
    else:
        env = os.environ.copy()
        # Для non-MTF явно выключаем MTF
        env['ML_MTF_ENABLED'] = '0'
    
    print(f"Команда: {' '.join(cmd)}")
    print(f"MTF: {'Да' if use_mtf else 'Нет'}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=Path(__file__).parent,
            env=env
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Успешно завершено за {elapsed_time:.1f} сек")
            return True, result.stdout[:500]  # Первые 500 символов вывода
        else:
            print(f"❌ Ошибка (код: {result.returncode})")
            error_msg = result.stderr or result.stdout
            print(f"Ошибка: {error_msg[:500]}...")
            return False, error_msg[:500]
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Исключение: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Обучение всех моделей')
    parser.add_argument('--symbol', type=str, help='Обучить только для этого символа')
    parser.add_argument('--model-type', type=str, help='Обучить только этот тип модели')
    parser.add_argument('--mtf', action='store_true', help='Обучить MTF модели')
    parser.add_argument('--no-mtf', action='store_true', help='Обучить только non-MTF модели')
    parser.add_argument('--dry-run', action='store_true', help='Показать команды без выполнения')
    parser.add_argument('--skip-existing', action='store_true', help='Пропустить существующие модели')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 МАСТЕР-СКРИПТ ОБУЧЕНИЯ ВСЕХ ML МОДЕЛЕЙ")
    print("=" * 80)
    
    # Определяем какие символы обучать
    symbols = [args.symbol] if args.symbol else SYMBOLS
    
    # Определяем какие модели обучать
    if args.mtf:
        configs = MTF_MODEL_CONFIGS
        print("📊 Режим: Только MTF модели")
    elif args.no_mtf:
        configs = MODEL_CONFIGS
        print("📊 Режим: Только non-MTF модели")
    else:
        # Обучаем обе группы
        configs = MODEL_CONFIGS + MTF_MODEL_CONFIGS
        print("📊 Режим: Все модели (MTF + non-MTF)")
    
    # Фильтр по типу модели если указан
    if args.model_type:
        configs = [c for c in configs if args.model_type in c['name']]
        print(f"📊 Фильтр по типу: {args.model_type}")
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - команды не будут выполнены")
    
    if args.skip_existing:
        print("⏭️  Пропуск существующих моделей")
    
    print(f"📊 Символы: {', '.join(symbols)}")
    print(f"📊 Моделей для обучения: {len(configs)} типов")
    print("=" * 80)
    
    results = []
    total_models = len(symbols) * len(configs)
    completed = 0
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    for symbol in symbols:
        print(f"\n📈 СИМВОЛ: {symbol}")
        print("-" * 40)
        
        for config in configs:
            completed += 1
            
            # Проверяем существующую модель если нужно пропустить
            if args.skip_existing:
                model_name = f"{config['name']}_{symbol}_{config['suffix']}.pkl"
                model_path = Path("ml_models") / model_name
                if model_path.exists():
                    print(f"⏭️  Пропускаем {model_name} (уже существует)")
                    skipped += 1
                    results.append({
                        "symbol": symbol,
                        "model": config['name'],
                        "status": "skipped",
                        "message": "Already exists"
                    })
                    continue
            
            # Dry run режим
            if args.dry_run:
                print(f"[DRY RUN] {config['name']} для {symbol}")
                results.append({
                    "symbol": symbol,
                    "model": config['name'],
                    "status": "dry_run",
                    "message": "Command shown but not executed"
                })
                continue
            
            # Запускаем обучение
            use_mtf = "mtf" in config['name'] or ("env" in config and config["env"].get("ML_MTF_ENABLED") == "1")
            success, message = run_training(config, symbol, use_mtf)
            
            if success:
                successful += 1
                status = "success"
            else:
                failed += 1
                status = "failed"
            
            results.append({
                "symbol": symbol,
                "model": config['name'],
                "status": status,
                "message": message
            })
            
            # Небольшая пауза между моделями
            if completed < total_models:
                print(f"⏳ Пауза 2 секунды...")
                time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Выводим итоги
    print(f"\n{'='*80}")
    print("📊 ИТОГИ ОБУЧЕНИЯ")
    print(f"{'='*80}")
    print(f"Всего моделей запланировано: {total_models}")
    print(f"✅ Успешно: {successful}")
    print(f"❌ Ошибок: {failed}")
    print(f"⏭️  Пропущено: {skipped}")
    if args.dry_run:
        print(f"📝 Dry run: {len([r for r in results if r['status'] == 'dry_run'])}")
    print(f"⏱️  Время: {total_time/60:.1f} минут")
    print(f"{'='*80}")
    
    # Подробный отчет
    if failed > 0:
        print(f"\n❌ Модели с ошибками:")
        for result in results:
            if result['status'] == 'failed':
                print(f"   - {result['model']}_{result['symbol']}: {result['message']}")
    
    # Проверяем созданные модели
    print(f"\n🔍 ПРОВЕРКА СОЗДАННЫХ МОДЕЛЕЙ:")
    models_dir = Path("ml_models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        print(f"   Всего .pkl файлов: {len(model_files)}")
        
        # Группируем по символам
        symbols_found = {}
        for model_file in model_files:
            name = model_file.name
            parts = name.replace(".pkl", "").split("_")
            if len(parts) >= 2:
                symbol = parts[1] if parts[0] not in ['triple', 'quad'] else parts[2]
                if symbol not in symbols_found:
                    symbols_found[symbol] = []
                symbols_found[symbol].append(name)
        
        for symbol, models in symbols_found.items():
            print(f"   {symbol}: {len(models)} моделей")
            for model in sorted(models)[:5]:  # Показываем первые 5
                print(f"     • {model}")
            if len(models) > 5:
                print(f"     ... и еще {len(models) - 5}")
    else:
        print("   Папка ml_models не существует!")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if successful > 0:
        print("   1. Протестируйте модели:")
        print("      python test_ml_strategy.py --symbol SOLUSDT --days 7")
        print("   2. Сравните результаты:")
        print("      python compare_models.py")
    else:
        print("   1. Проверьте ошибки выше")
        print("   2. Убедитесь что все зависимости установлены")
        print("   3. Попробуйте обучить модели по одной")
    
    print(f"\n📂 Создайте файл отчета:")
    report_file = f"training_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛЕЙ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Успешно: {successful}\n")
        f.write(f"Ошибки: {failed}\n")
        f.write(f"Пропущено: {skipped}\n")
        f.write(f"Время: {total_time/60:.1f} минут\n\n")
        
        f.write("Детальные результаты:\n")
        for result in results:
            f.write(f"{result['symbol']}_{result['model']}: {result['status']} - {result['message'][:100]}\n")
    
    print(f"   Отчет сохранен в: {report_file}")

if __name__ == "__main__":
    main()