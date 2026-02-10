#!/usr/bin/env python3
"""Разрешить merge конфликт для bot/config.py и завершить merge."""
import subprocess
import sys

try:
    # Проверяем текущий статус
    print("🔍 Проверка статуса git...")
    print("=" * 80)
    status = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if status.returncode == 0:
        print(status.stdout)
    
    # Проверяем unmerged файлы
    print("\n🔍 Проверка unmerged файлов...")
    print("=" * 80)
    unmerged = subprocess.run(
        ['git', 'ls-files', '-u'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if unmerged.returncode == 0 and unmerged.stdout.strip():
        print("Найдены unmerged файлы:")
        print(unmerged.stdout)
        files = set()
        for line in unmerged.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    files.add(parts[-1])
        
        print(f"\n📋 Добавляем файлы в индекс: {', '.join(files)}")
        for file in files:
            result = subprocess.run(
                ['git', 'add', file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                print(f"✅ Добавлен: {file}")
            else:
                print(f"❌ Ошибка при добавлении {file}: {result.stderr}")
                sys.exit(1)
    else:
        print("✅ Нет unmerged файлов")
    
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
        print("\n💾 Завершаем merge коммитом...")
        print("=" * 80)
        
        commit_result = subprocess.run(
            ['git', 'commit', '-m', 'Resolve merge conflicts: keep local changes for bot/config.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if commit_result.returncode == 0:
            print("✅ Merge успешно завершен!")
            print(commit_result.stdout)
        else:
            print(f"❌ Ошибка при коммите: {commit_result.stderr}")
            sys.exit(1)
    else:
        print("✅ Нет незавершенного merge")
    
    # Финальный статус
    print("\n📊 Финальный статус:")
    print("=" * 80)
    final_status = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if final_status.returncode == 0:
        if final_status.stdout.strip():
            print(final_status.stdout)
        else:
            print("✅ Рабочая директория чистая")
    
    print("\n✅ Готово! Теперь можно выполнить git pull")
        
except FileNotFoundError:
    print("❌ Git не найден")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
