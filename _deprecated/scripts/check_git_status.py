#!/usr/bin/env python3
"""Проверить статус git и найти unmerged файлы."""
import subprocess
import sys

try:
    # Проверяем unmerged файлы
    print("🔍 Проверка unmerged файлов...")
    print("=" * 80)
    unmerged = subprocess.run(
        ['git', 'ls-files', '-u'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if unmerged.returncode == 0 and unmerged.stdout.strip():
        print("❌ Найдены unmerged файлы:")
        print(unmerged.stdout)
        files = set()
        for line in unmerged.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    files.add(parts[-1])
        
        print(f"\n📋 Файлы для разрешения: {', '.join(files)}")
        return files
    else:
        print("✅ Нет unmerged файлов в индексе")
    
    # Проверяем общий статус
    print("\n📊 Общий статус git:")
    print("=" * 80)
    status = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if status.returncode == 0:
        if status.stdout.strip():
            print(status.stdout)
        else:
            print("Нет изменений")
    
    # Проверяем, есть ли незавершенный merge
    print("\n🔍 Проверка merge состояния...")
    print("=" * 80)
    merge_head = subprocess.run(
        ['git', 'rev-parse', '--verify', 'MERGE_HEAD'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if merge_head.returncode == 0:
        print(f"⚠️  Обнаружен незавершенный merge: {merge_head.stdout.strip()}")
        return True
    else:
        print("✅ Нет незавершенного merge")
        return False
        
except FileNotFoundError:
    print("❌ Git не найден")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
