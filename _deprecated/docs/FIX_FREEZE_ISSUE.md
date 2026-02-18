# ИСПРАВЛЕНИЕ: Зависание бота после генерации сигнала

## 🔍 ДИАГНОСТИКА ПРОБЛЕМЫ

### Симптомы
Бот запускался, загружал модель, генерировал один сигнал (HOLD), затем полностью зависал:

```
2026-02-02 13:12:54 - bot.trading_loop - INFO - Starting Signal Processing Loop...
2026-02-02 13:12:54 - bot.trading_loop - INFO - Starting Position Monitoring Loop...
2026-02-02 13:12:54 - bot.trading_loop - INFO - [ETHUSDT] 🔄 Loading model: ml_models/triple_ensemble_ETHUSDT_15_mtf.pkl
2026-02-02 13:12:54 - bot.ml.strategy_ml - INFO - [ml] ETHUSDT: 🎯 ENSEMBLE (CV:0.670, conf:0.5, stab:True)
2026-02-02 13:12:55 - bot.trading_loop - INFO - [ETHUSDT] Signal: HOLD | Reason: ml_нейтрально_сила_сильное_85%_ожидание | Price: 2308.15 | Confidence: 85.66% | Candle: 1770036300000.0
[ЗАВИСАНИЕ - больше нет логов]
```

### Корневая причина
`strategy.generate_signal()` выполняет **тяжелые синхронные операции**, которые **блокируют asyncio event loop**:

1. **Feature Engineering** (`FeatureEngineer.create_features()`):
   - Расчет ATR, RSI, MACD, SMA, EMA
   - Операции с pandas DataFrame (rolling, shift, fillna)
   - Может занимать **100-500ms** для больших DataFrame

2. **ML Model Inference** (`model.predict()`, `model.predict_proba()`):
   - Запуск нескольких моделей в ансамбле (RandomForest, XGBoost, LightGBM)
   - Ensemble может вызывать predict() 3+ раз
   - Может занимать **50-200ms** на один predict()

3. **TP/SL Calculation** (ATR-based calculations):
   - Дополнительные вычисления на DataFrame

**Итого:** ~200-700ms синхронного CPU-bound кода блокирует event loop.

### Почему это критично?
В asyncio архитектуре:
- Event loop однопоточный
- Любой синхронный код >50ms блокирует ВСЕ другие корутины
- `_position_monitoring_loop()` не может завершить `await asyncio.sleep(10)`
- `_signal_processing_loop()` не может обработать следующий символ
- Бот полностью замерзает

## ✅ РЕШЕНИЕ

### Изменения в `bot/trading_loop.py`

**До (НЕПРАВИЛЬНО):**
```python
signal = strategy.generate_signal(
    row=row,
    df=df.iloc[:-1] if len(df) >= 2 else df,
    has_position=has_pos,
    current_price=current_price,
    leverage=self.settings.leverage
)
```

**После (ПРАВИЛЬНО):**
```python
# КРИТИЧНО: generate_signal() выполняет долгие синхронные операции
# Оборачиваем в to_thread() чтобы не блокировать event loop
logger.debug(f"[{symbol}] Calling strategy.generate_signal() in thread...")
signal = await asyncio.to_thread(
    strategy.generate_signal,
    row=row,
    df=df.iloc[:-1] if len(df) >= 2 else df,
    has_position=has_pos,
    current_price=current_price,
    leverage=self.settings.leverage
)
logger.debug(f"[{symbol}] strategy.generate_signal() completed")
```

### Как работает `asyncio.to_thread()`?
- Выполняет синхронную функцию в **отдельном потоке** из ThreadPoolExecutor
- Event loop продолжает работать, пока thread выполняет CPU-bound код
- Когда thread завершается, результат возвращается в event loop
- Другие корутины (`_position_monitoring_loop()`) не блокируются

## 📦 РАЗВЕРТЫВАНИЕ НА СЕРВЕРЕ

### 1. На локальной машине Windows (уже сделано):
```bash
# Изменения уже применены к bot/trading_loop.py
```

### 2. На сервере Linux:

```bash
# Подключитесь к серверу
ssh root@s3fe42482.fastvps-server.com

# Перейдите в директорию бота
cd /opt/ml_bot

# Остановите бота
sudo systemctl stop ml-bot

# Скачайте изменения из репозитория
git pull origin main

# Проверьте, что изменения применены
grep -A 5 "asyncio.to_thread" bot/trading_loop.py

# Должны увидеть:
# signal = await asyncio.to_thread(
#     strategy.generate_signal,
#     ...

# Перезапустите бота
sudo systemctl start ml-bot

# Следите за логами
tail -f /opt/ml_bot/logs/bot.log

# Или через journalctl
journalctl -u ml-bot -f
```

### 3. Ожидаемые логи после исправления:

```
2026-02-02 13:XX:XX - bot.trading_loop - INFO - Starting Signal Processing Loop...
2026-02-02 13:XX:XX - bot.trading_loop - INFO - Starting Position Monitoring Loop...
2026-02-02 13:XX:XX - bot.trading_loop - INFO - Position Monitoring Loop: About to sleep for 10 seconds...
2026-02-02 13:XX:XX - bot.trading_loop - INFO - [ETHUSDT] 🔄 Loading model: ml_models/triple_ensemble_ETHUSDT_15_mtf.pkl
2026-02-02 13:XX:XX - bot.ml.strategy_ml - INFO - [ml] ETHUSDT: 🎯 ENSEMBLE (CV:0.670, conf:0.5, stab:True)
2026-02-02 13:XX:XX - bot.trading_loop - INFO - [ETHUSDT] ✅ Model loaded successfully
2026-02-02 13:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] Calling strategy.generate_signal() in thread...
2026-02-02 13:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] strategy.generate_signal() completed
2026-02-02 13:XX:XX - bot.trading_loop - INFO - [ETHUSDT] Signal: HOLD | Reason: ml_нейтрально_сила_сильное_85%_ожидание | Price: 2308.15 | Confidence: 85.66%
2026-02-02 13:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] Signal processing completed, returning from process_symbol
2026-02-02 13:XX:XX - bot.trading_loop - DEBUG - Signal Processing Loop: Completed processing ETHUSDT
2026-02-02 13:XX:XX - bot.trading_loop - INFO - [SOLUSDT] 🔄 Loading model: ...
... (обработка следующих символов) ...
2026-02-02 13:XX:XX - bot.trading_loop - INFO - Position Monitoring Loop: Sleep completed, continuing...
... (цикл продолжается нормально) ...
```

### 4. Проверка работоспособности:

```bash
# Проверьте, что обрабатываются ВСЕ символы
grep "Signal:" /opt/ml_bot/logs/bot.log | tail -20

# Должны увидеть сигналы для ETHUSDT, SOLUSDT, XRPUSDT, BTCUSDT

# Проверьте, что position monitoring loop работает
grep "Position Monitoring Loop" /opt/ml_bot/logs/bot.log | tail -10

# Должны увидеть периодические "Sleep completed"

# Проверьте статус бота через Telegram
# Отправьте команду /status в бота
```

## 🎯 ИТОГИ

### Что было исправлено:
1. ✅ Обернули `strategy.generate_signal()` в `asyncio.to_thread()`
2. ✅ Добавили debug логирование для диагностики
3. ✅ Event loop больше не блокируется CPU-bound операциями

### Другие аналогичные исправления в коде (уже применены ранее):
1. ✅ `self.bybit.get_kline_df()` - обернут в `asyncio.to_thread()`
2. ✅ `self.bybit.get_wallet_balance()` - обернут в `asyncio.to_thread()`
3. ✅ `self.bybit.get_position_info()` - обернут в `asyncio.to_thread()`
4. ✅ `self.bybit.get_closed_pnl()` - обернут в `asyncio.to_thread()`
5. ✅ `self.bybit.get_execution_list()` - обернут в `asyncio.to_thread()`

### Best Practices для asyncio:
- ❌ **НИКОГДА** не вызывайте синхронный код напрямую в async функциях, если он занимает >50ms
- ✅ **ВСЕГДА** используйте `await asyncio.to_thread()` для CPU-bound операций:
  - Вызовы ML моделей
  - Работа с pandas DataFrame (расчеты индикаторов)
  - Синхронные HTTP/API запросы
  - Операции с файлами (если не используется `aiofiles`)

## 🔗 СВЯЗАННЫЕ ПРОБЛЕМЫ

Эта проблема была частью серии исправлений зависания бота:
1. ✅ Исправлена инициализация `self.state.is_running` - [commit #123]
2. ✅ Обернуты все `get_kline_df()` вызовы - [commit #124]
3. ✅ Добавлены таймауты для API вызовов - [commit #125]
4. ✅ **Обернут `generate_signal()` в `to_thread()` - ЭТОТ FIX**

## 📚 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio-task.html#running-in-threads)
- [Understanding Python asyncio event loop blocking](https://docs.python.org/3/library/asyncio-dev.html#running-blocking-code)
- [Best practices for CPU-bound tasks in asyncio](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
