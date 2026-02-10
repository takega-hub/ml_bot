# Автоматическая оптимизация стратегий

## Обзор

Система автоматической оптимизации стратегий выполняет полный цикл:
1. **Обучение моделей** (15m и 1h) для всех активных символов
2. **Сравнение моделей** для выявления лучших
3. **Тестирование MTF комбинаций** всех возможных пар моделей
4. **Автоматический выбор лучших стратегий** на основе метрик
5. **Сохранение результатов** в `best_strategies_*.json` для автоматического применения в боте

## Компоненты системы

### 1. auto_strategy_optimizer.py
Основная утилита для полного цикла оптимизации.

**Использование:**
```bash
# Полный цикл оптимизации
python auto_strategy_optimizer.py

# Пропустить обучение (использовать существующие модели)
python auto_strategy_optimizer.py --skip-training

# Только сравнение и MTF тестирование
python auto_strategy_optimizer.py --skip-training

# Для конкретных символов
python auto_strategy_optimizer.py --symbols BTCUSDT,ETHUSDT

# Кастомное количество дней для бэктеста
python auto_strategy_optimizer.py --days 60
```

**Параметры:**
- `--symbols` - Список символов через запятую (по умолчанию из `state.active_symbols`)
- `--days` - Количество дней для бэктеста (по умолчанию 30)
- `--output-dir` - Директория для сохранения результатов (по умолчанию `optimization_results`)
- `--skip-training` - Пропустить обучение моделей
- `--skip-comparison` - Пропустить сравнение моделей
- `--skip-mtf-testing` - Пропустить тестирование MTF комбинаций

### 2. optimize_strategies.py
Удобная обертка для ручного запуска оптимизации.

**Использование:**
```bash
# Быстрый запуск (без обучения)
python optimize_strategies.py --quick

# Полный цикл
python optimize_strategies.py --full

# Запуск немедленно
python optimize_strategies.py --now
```

### 3. schedule_strategy_optimizer.py
Планировщик для автоматического запуска оптимизации по расписанию.

**Использование:**
```bash
# Запуск планировщика (по умолчанию: воскресенье в 3:00)
python schedule_strategy_optimizer.py

# Кастомное расписание (понедельник в 2:00)
python schedule_strategy_optimizer.py --day monday --hour 2

# Тестовый запуск (один раз)
python schedule_strategy_optimizer.py --run-once
```

**Требования:**
```bash
pip install schedule
```

### 4. bot/ml/model_selector.py
Утилита для автоматического выбора лучших моделей в боте.

**Приоритет выбора:**
1. `best_strategies_*.json` (последний файл) - **высший приоритет**
2. `comparison_15m_vs_1h.csv`
3. `ml_models_comparison_*.csv` (последний файл)
4. Fallback на первые найденные модели

## Формат best_strategies_*.json

Файл содержит информацию о лучших стратегиях для каждого символа:

```json
{
  "timestamp": "2024-02-10T03:00:00Z",
  "optimization_version": "1.0",
  "backtest_days": 30,
  "symbols": {
    "BTCUSDT": {
      "strategy_type": "mtf",
      "model_1h": "quad_ensemble_BTCUSDT_60_1h.pkl",
      "model_15m": "quad_ensemble_BTCUSDT_15_15m.pkl",
      "confidence_threshold_1h": 0.50,
      "confidence_threshold_15m": 0.35,
      "alignment_mode": "strict",
      "require_alignment": true,
      "metrics": {
        "total_pnl_pct": 65.2,
        "win_rate": 58.5,
        "profit_factor": 2.1,
        "sharpe_ratio": 8.5,
        "total_trades": 45,
        "max_drawdown_pct": 5.2
      },
      "source": "mtf_combinations_test"
    },
    "ETHUSDT": {
      "strategy_type": "single",
      "model": "quad_ensemble_ETHUSDT_15_15m.pkl",
      "confidence_threshold": 0.40,
      "metrics": {
        "total_pnl_pct": 52.3,
        "win_rate": 55.2,
        "profit_factor": 1.8,
        "sharpe_ratio": 7.2
      },
      "source": "model_comparison"
    }
  }
}
```

## Логика выбора лучшей стратегии

Система использует composite score для выбора лучшей стратегии:

```
composite_score = (
    total_pnl_pct * 0.4 +
    win_rate * 0.2 +
    profit_factor * 20.0 * 0.2 +
    sharpe_ratio * 0.1 +
    (100 - max_drawdown_pct) * 0.1
)
```

**Правила выбора:**
1. Для каждого символа анализируются результаты MTF комбинаций
2. Выбирается комбинация с максимальным composite_score
3. Если лучшая MTF стратегия хуже лучшей single стратегии на >20% - используется single
4. Все метрики сохраняются для мониторинга

## Настройка расписания

### Linux (cron)

Добавьте в crontab:
```bash
crontab -e
```

Добавьте строку для запуска каждое воскресенье в 3:00:
```
0 3 * * 0 cd /path/to/ml_bot && /path/to/python schedule_strategy_optimizer.py --run-once >> /path/to/scheduler.log 2>&1
```

Или используйте скрипт-обертку:
```
0 3 * * 0 cd /path/to/ml_bot && /path/to/run_optimizer.sh >> /path/to/scheduler.log 2>&1
```

### Windows (Task Scheduler)

1. Откройте Task Scheduler
2. Создайте новую задачу
3. Настройте:
   - **Триггер**: Еженедельно, воскресенье, 3:00
   - **Действие**: Запустить программу
   - **Программа**: `python.exe`
   - **Аргументы**: `schedule_strategy_optimizer.py --run-once`
   - **Рабочая папка**: Путь к директории проекта

Или используйте скрипт-обертку:
- **Программа**: `run_optimizer.bat`

## Интеграция с ботом

Бот автоматически использует лучшие стратегии из `best_strategies_*.json` при запуске:

1. При инициализации `TradingLoop` проверяются MTF модели для активных символов
2. Используется `model_selector.select_best_models()` для выбора моделей
3. Если найден `best_strategies_*.json`, используются модели и параметры из него
4. Логируется источник выбора моделей и ожидаемые метрики

**Включение MTF стратегии:**
```python
# В конфигурационном файле или через Telegram бота
use_mtf_strategy = True
```

## Мониторинг и уведомления

После каждой оптимизации отправляется отчет в Telegram (если настроен):
- Список обработанных символов
- Лучшие стратегии для каждого символа
- Метрики (PnL%, Win Rate, Profit Factor)
- Время выполнения
- Количество ошибок

## Обработка ошибок

- **Обучение модели не удалось** → Используются существующие модели
- **Сравнение не удалось** → Символ пропускается с предупреждением
- **MTF тестирование не удалось** → Используются лучшие модели из сравнения
- Все ошибки логируются в `optimization_errors_{timestamp}.log`

## Примеры использования

### Полная оптимизация раз в неделю
```bash
# Запустить планировщик
python schedule_strategy_optimizer.py --day sunday --hour 3
```

### Быстрая оптимизация существующих моделей
```bash
python optimize_strategies.py --quick
```

### Оптимизация для конкретных символов
```bash
python auto_strategy_optimizer.py --symbols BTCUSDT,ETHUSDT --skip-training
```

### Тестирование перед продакшн
```bash
# Полный цикл на тестовых данных
python auto_strategy_optimizer.py --days 14 --symbols BTCUSDT
```

## Troubleshooting

### Проблема: Оптимизация не запускается по расписанию
**Решение:**
- Проверьте логи в `scheduler.log`
- Убедитесь, что планировщик запущен
- Проверьте права доступа к файлам

### Проблема: Модели не найдены
**Решение:**
- Убедитесь, что модели обучены: `python retrain_ml_optimized.py`
- Проверьте формат имен моделей: `*_{SYMBOL}_60_*.pkl` и `*_{SYMBOL}_15_*.pkl`

### Проблема: Бот не использует лучшие стратегии
**Решение:**
- Проверьте наличие файла `best_strategies_*.json`
- Убедитесь, что `use_mtf_strategy = True` в конфигурации
- Проверьте логи бота на наличие ошибок загрузки моделей

### Проблема: Оптимизация занимает слишком много времени
**Решение:**
- Используйте `--skip-training` для пропуска обучения
- Уменьшите количество дней бэктеста: `--days 14`
- Оптимизируйте для конкретных символов: `--symbols BTCUSDT`

## Производительность

**Ожидаемое время выполнения:**
- Обучение моделей: ~10-20 минут на символ
- Сравнение моделей: ~30-60 минут для всех символов
- MTF тестирование: ~5-10 минут на символ (зависит от количества комбинаций)
- **Общее время**: ~2-4 часа для 5 символов (полный цикл)

**Рекомендации:**
- Запускать ночью (низкая нагрузка на систему)
- Использовать `--skip-training` для быстрой оптимизации существующих моделей
- Запускать раз в неделю (достаточно для стабильных результатов)

## Следующие шаги

1. ✅ Настройте расписание запуска оптимизации
2. ✅ Включите MTF стратегию в боте (`use_mtf_strategy = True`)
3. ✅ Мониторьте результаты через Telegram уведомления
4. ✅ Анализируйте файлы `best_strategies_*.json` для понимания выбранных стратегий
5. ✅ При необходимости корректируйте параметры оптимизации
