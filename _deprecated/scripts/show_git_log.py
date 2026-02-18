#!/usr/bin/env python3
"""Показать историю коммитов git."""
import subprocess
import sys

try:
    # Получаем последние 20 коммитов
    result = subprocess.run(
        ['git', 'log', '--oneline', '-20'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode == 0:
        print("📋 Последние 20 коммитов:")
        print("=" * 80)
        print(result.stdout)
        
        # Также покажем статус
        print("\n📊 Текущий статус:")
        print("=" * 80)
        status_result = subprocess.run(
            ['git', 'status', '--short'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if status_result.returncode == 0:
            if status_result.stdout.strip():
                print(status_result.stdout)
            else:
                print("Нет изменений в рабочей директории")
        else:
            print("Не удалось получить статус")
    else:
        print(f"Ошибка: {result.stderr}")
        sys.exit(1)
        
except FileNotFoundError:
    print("❌ Git не найден. Убедитесь, что Git установлен и доступен в PATH.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)
