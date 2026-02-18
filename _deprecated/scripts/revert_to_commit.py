#!/usr/bin/env python3
"""Откатить изменения к коммиту e3b75c9."""
import subprocess
import sys

commit_hash = "e3b75c9"

try:
    # Проверяем, существует ли коммит
    check_result = subprocess.run(
        ['git', 'show', '--oneline', '-s', commit_hash],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if check_result.returncode != 0:
        print(f"❌ Коммит {commit_hash} не найден")
        print(check_result.stderr)
        sys.exit(1)
    
    print(f"✅ Найден коммит: {check_result.stdout.strip()}")
    
    # Показываем, какие файлы были изменены в этом коммите
    print(f"\n📋 Файлы, измененные в коммите {commit_hash}:")
    print("=" * 80)
    files_result = subprocess.run(
        ['git', 'show', '--name-only', '--pretty=format:', commit_hash],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if files_result.returncode == 0:
        files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]
        for f in files:
            print(f"  - {f}")
    
    # Откатываем файлы к этому коммиту
    print(f"\n🔄 Откат файлов к коммиту {commit_hash}...")
    print("=" * 80)
    
    # Откатываем все файлы из коммита
    for file in files:
        result = subprocess.run(
            ['git', 'checkout', commit_hash, '--', file],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            print(f"✅ Откачен: {file}")
        else:
            print(f"⚠️  Не удалось откатить {file}: {result.stderr}")
    
    print(f"\n✅ Готово! Файлы откачены к коммиту {commit_hash}")
    print("\n📊 Текущий статус:")
    status_result = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if status_result.returncode == 0:
        print(status_result.stdout)
        
except FileNotFoundError:
    print("❌ Git не найден. Убедитесь, что Git установлен и доступен в PATH.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
