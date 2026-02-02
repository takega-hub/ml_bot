import asyncio
import logging
import signal
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from bot.config import load_settings
from bot.state import BotState
from bot.exchange.bybit_client import BybitClient
from bot.model_manager import ModelManager
from bot.telegram_bot import TelegramBot
from bot.trading_loop import TradingLoop
from bot.health_monitor import HealthMonitor

# Создаем директорию для логов
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Настройка расширенного логирования
# Основной лог с ротацией
main_handler = RotatingFileHandler(
    'logs/bot.log', 
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
main_handler.setLevel(logging.INFO)

# Лог сделок
trade_handler = RotatingFileHandler(
    'logs/trades.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
trade_handler.setLevel(logging.INFO)

# Лог сигналов
signal_handler = RotatingFileHandler(
    'logs/signals.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
signal_handler.setLevel(logging.INFO)

# Лог ошибок
error_handler = RotatingFileHandler(
    'logs/errors.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)

# Консольный вывод
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Форматтер
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
main_handler.setFormatter(formatter)
trade_handler.setFormatter(formatter)
signal_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Настраиваем root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Очищаем существующие обработчики, чтобы избежать дублирования
root_logger.handlers.clear()
root_logger.addHandler(main_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(error_handler)
# Отключаем propagate для root logger, чтобы избежать двойного логирования
root_logger.propagate = False

# Настраиваем специализированные логгеры
trade_logger = logging.getLogger("trades")
trade_logger.handlers.clear()  # Очищаем существующие обработчики
trade_logger.addHandler(trade_handler)
trade_logger.propagate = False  # Отключаем propagate

signal_logger = logging.getLogger("signals")
signal_logger.handlers.clear()  # Очищаем существующие обработчики
signal_logger.addHandler(signal_handler)
signal_logger.propagate = False  # Отключаем propagate

logger = logging.getLogger("main")

async def main():
    try:
        logger.info("Initializing ML Trading Bot Terminal...")
        
        # 1. Загрузка настроек
        try:
            settings = load_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}", exc_info=True)
            raise
        
        if not settings.telegram_token:
            logger.error("TELEGRAM_TOKEN not found in .env file!")
            sys.exit(1)

        # 2. Инициализация состояния
        try:
            state = BotState()
        except Exception as e:
            logger.error(f"Failed to initialize BotState: {e}", exc_info=True)
            raise
        
        # 3. Инициализация клиента биржи
        try:
            bybit = BybitClient(settings.api)
        except Exception as e:
            logger.error(f"Failed to initialize BybitClient: {e}", exc_info=True)
            raise
        
        # 4. Инициализация менеджера моделей
        try:
            model_manager = ModelManager(settings, state)
        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {e}", exc_info=True)
            raise
        
        # 5. Инициализация Telegram бота (передаем bybit для получения позиций)
        try:
            tg_bot = TelegramBot(settings, state, model_manager, bybit)
        except Exception as e:
            logger.error(f"Failed to initialize TelegramBot: {e}", exc_info=True)
            raise
        
        # 6. Инициализация торгового цикла
        try:
            trading_loop = TradingLoop(settings, state, bybit, tg_bot)
        except Exception as e:
            logger.error(f"Failed to initialize TradingLoop: {e}", exc_info=True)
            raise
        
        # 7. Инициализация Health Monitor
        try:
            health_monitor = HealthMonitor(settings, state, bybit, tg_bot)
        except Exception as e:
            logger.error(f"Failed to initialize HealthMonitor: {e}", exc_info=True)
            raise
    
        # 8. Запуск компонентов
        try:
            # Запускаем все компоненты параллельно
            await asyncio.gather(
                tg_bot.start(),
                trading_loop.run(),
                health_monitor.run()
            )
        except asyncio.CancelledError:
            logger.info("Bot execution cancelled.")
        except Exception as e:
            logger.error(f"Fatal error during execution: {e}", exc_info=True)
            raise
        finally:
            logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Обработка прерываний (Ctrl+C)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Manual shutdown.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)
