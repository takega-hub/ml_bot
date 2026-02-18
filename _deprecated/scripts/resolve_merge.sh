#!/bin/bash
# Разрешить merge конфликт и завершить merge

echo "🔍 Проверка статуса git..."
git status --short

echo ""
echo "📋 Добавляем bot/config.py в индекс..."
git add bot/config.py

echo ""
echo "💾 Завершаем merge коммитом..."
git commit -m "Resolve merge conflicts: keep local changes for bot/config.py"

echo ""
echo "📊 Финальный статус:"
git status --short

echo ""
echo "✅ Готово! Теперь можно выполнить git pull"
