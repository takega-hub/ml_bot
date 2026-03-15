#!/usr/bin/env python3
"""
Скрипт для ручного обновления статуса эксперимента и предотвращения зависаний LSTM.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def update_experiment_status(experiment_id: str, status: str, error: str = None):
    """Обновляет статус эксперимента в experiments.json"""
    try:
        experiments_file = Path("experiments.json")
        
        if not experiments_file.exists():
            print(f"❌ Файл {experiments_file} не найден")
            return False
            
        # Читаем текущие данные
        with open(experiments_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if experiment_id not in data:
            print(f"❌ Эксперимент {experiment_id} не найден")
            return False
            
        # Обновляем статус
        experiment = data[experiment_id]
        experiment["status"] = status
        experiment["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        if error:
            experiment["error"] = error
            experiment["failed_at"] = datetime.now(timezone.utc).isoformat()
            
        # Добавляем информацию о причине
        if status == "failed":
            experiment["failure_reason"] = "LSTM training timeout - process killed"
            experiment["recommendation"] = "Consider disabling LSTM in aggressive mode or reducing sequence length"
            
        # Сохраняем обратно
        with open(experiments_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"✅ Статус эксперимента {experiment_id} обновлен на: {status}")
        if error:
            print(f"📝 Ошибка: {error}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обновлении статуса: {e}")
        return False

def create_lstm_safe_config():
    """Создает конфигурацию для предотвращения зависаний LSTM"""
    config = {
        "lstm_training_timeout_seconds": 300,  # 5 минут максимум
        "lstm_max_sequence_length": 30,       # Меньшая длина последовательности
        "lstm_early_stopping_patience": 5,   # Раннее прекращение
        "lstm_max_epochs": 50,              # Меньше эпох
        "disable_lstm_for_aggressive": True, # Отключить LSTM для aggressive режима
        "lstm_device": "cpu",                 # Использовать CPU (более стабильно)
        "lstm_batch_size": 32,               # Меньший batch size
    }
    
    config_file = Path("lstm_safe_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Создан файл безопасной конфигурации LSTM: {config_file}")
    return config_file

def main():
    experiment_id = "exp_10010691_17cb064b"
    
    print(f"🔧 Обновление статуса эксперимента {experiment_id}")
    
    # Обновляем статус как failed
    success = update_experiment_status(
        experiment_id, 
        "failed", 
        "LSTM training timeout - process was hanging for 30+ minutes"
    )
    
    if success:
        print("\n📋 Создание конфигурации для предотвращения будущих зависаний:")
        config_file = create_lstm_safe_config()
        
        print(f"\n🎯 Рекомендации:")
        print(f"1. Используйте параметры из {config_file} для LSTM обучения")
        print(f"2. Рассмотрите отключение LSTM для aggressive режима")
        print(f"3. Уменьшите длину последовательности с 60 до 30")
        print(f"4. Добавьте таймауты для обучения (5 минут максимум)")
        
        print(f"\n💡 Для перезапуска эксперимента без LSTM:")
        print(f"   python run_research.py --symbol BTCUSDT --type aggressive --experiment-id exp_$(date +%s)_fixed --no-lstm")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())