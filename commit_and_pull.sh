#!/bin/bash
# Закоммитить локальные изменения и выполнить pull

echo "🔍 Проверка статуса git..."
git status --short

echo ""
echo "📋 Добавляем изменения в bot/config.py..."
git add bot/config.py

echo ""
echo "💾 Создаем коммит с локальными изменениями..."
git commit -m "Keep local config changes: confidence_threshold=0.35, max_signals_per_day=20"

echo ""
echo "📥 Выполняем git pull..."
git pull origin main

echo ""
echo "📊 Финальный статус:"
git status --short

echo ""
echo "✅ Готово!"
