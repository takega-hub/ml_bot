#!/usr/bin/env python3
"""
Безопасный запуск эксперимента с предотвращением зависаний LSTM.
"""

import subprocess
import sys
import uuid
import time
import json
import os
from datetime import datetime
from pathlib import Path

def create_safe_config():
    """Создает конфигурацию для безопасного обучения без LSTM"""
    config = {
        "ml_settings": {
            "confidence_threshold": 0.35,
            "use_mtf_strategy": False,
            "model_type": "ensemble",  # Без LSTM
            "mtf_enabled": False,
            "no_mtf": True
        },
        "training_settings": {
            "max_training_time_per_model": 300,  # 5 минут max
            "skip_lstm": True,
            "use_ensemble_only": True,
            "sequence_length": 30,  # Меньшая длина для LSTM если включен
            "early_stopping": True
        }
    }
    
    config_file = Path("safe_training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Создан файл безопасной конфигурации: {config_file}")
    return config_file

def start_safe_experiment(symbol="BTCUSDT", experiment_type="aggressive"):
    """Запускает безопасный эксперимент"""
    
    # Создаем уникальный ID для нового эксперимента
    experiment_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    print(f"🚀 Запуск безопасного эксперимента: {experiment_id}")
    print(f"📊 Символ: {symbol}, Тип: {experiment_type}")
    
    # Создаем безопасную конфигурацию
    config_file = create_safe_config()
    
    # Формируем команду для запуска
    cmd = [
        sys.executable,
        "run_research.py",
        "--symbol", symbol,
        "--type", experiment_type,
        "--experiment-id", experiment_id,
        "--interval", "15m",  # Используем 15m для aggressive
        "--no-mtf"  # Отключаем MTF для предотвращения сложностей
    ]
    
    print(f"📝 Команда: {' '.join(cmd)}")
    
    try:
        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Процесс запущен с PID: {process.pid}")
        print(f"📝 Логи будут сохранены в research.log")
        print(f"📊 Для мониторинга используйте: tail -f research.log")
        
        # Сохраняем информацию о запуске
        run_info = {
            "experiment_id": experiment_id,
            "pid": process.pid,
            "start_time": datetime.now().isoformat(),
            "symbol": symbol,
            "type": experiment_type,
            "config_file": str(config_file),
            "status": "running"
        }
        
        info_file = Path(f"experiment_{experiment_id}_info.json")
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"ℹ️  Информация о запуске сохранена в: {info_file}")
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "pid": process.pid,
            "process": process
        }
        
    except Exception as e:
        print(f"❌ Ошибка при запуске эксперимента: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("🔬 Безопасный запуск эксперимента ML обучения")
    print("=" * 50)
    
    # Запрашиваем параметры у пользователя
    symbol = input("Введите торговую пару [BTCUSDT]: ").strip() or "BTCUSDT"
    experiment_type = input("Введите тип эксперимента [aggressive]: ").strip() or "aggressive"
    
    print()
    result = start_safe_experiment(symbol, experiment_type)
    
    if result["success"]:
        print(f"\n🎉 Эксперимент успешно запущен!")
        print(f"📋 ID эксперимента: {result['experiment_id']}")
        print(f"🔍 PID процесса: {result['pid']}")
        print(f"\n📊 Для мониторинга выполните:")
        print(f"   tail -f research.log | grep {result['experiment_id']}")
        print(f"\n🛑 Для остановки выполните:")
        print(f"   kill {result['pid']}")
    else:
        print(f"\n❌ Не удалось запустить эксперимент: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())