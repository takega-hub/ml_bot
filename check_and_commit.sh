#!/bin/bash
# Проверить изменения и закоммитить перед pull

echo "🔍 Проверка изменений в bot/config.py..."
echo "=========================================="
git diff bot/config.py | head -50

echo ""
echo "📋 Показать все измененные файлы..."
git status --short

echo ""
read -p "Продолжить и закоммитить изменения? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Отменено"
    exit 1
fi

echo ""
echo "📋 Добавляем изменения..."
git add bot/config.py

echo ""
echo "💾 Создаем коммит..."
git commit -m "Keep local config changes: confidence_threshold=0.35, max_signals_per_day=20"

echo ""
echo "📥 Выполняем git pull..."
git pull origin main

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Возникли конфликты при pull. Нужно разрешить вручную."
    echo "Проверьте: git status"
    exit 1
fi

echo ""
echo "📊 Финальный статус:"
git status --short

echo ""
echo "✅ Готово!"
