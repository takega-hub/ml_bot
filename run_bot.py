import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from bot.config import load_settings, log_server_config
from bot.state import BotState
from bot.exchange.bybit_client import BybitClient
from bot.model_manager import ModelManager
from bot.telegram_bot import TelegramBot
from bot.trading_loop import TradingLoop
from bot.health_monitor import HealthMonitor
from bot.api_server import run_api_server
try:
    from telegram.error import Conflict
except ImportError:
    Conflict = None

# Флаг для защиты от повторной настройки логирования
_logging_configured = False

def setup_logging():
    """Настройка логирования (вызывается один раз)"""
    global _logging_configured
    
    # Если логирование уже настроено, пропускаем
    if _logging_configured:
        return logging.getLogger("main")
    
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
    main_handler.setLevel(logging.DEBUG)  # Временно включаем DEBUG для диагностики

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

    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main_handler.setFormatter(formatter)
    trade_handler.setFormatter(formatter)
    signal_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Настраиваем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # INFO уровень для production (было DEBUG для диагностики)

    # Очищаем существующие обработчики (на всякий случай)
    root_logger.handlers.clear()

    # Добавляем только file handlers (без console_handler, чтобы избежать дублирования)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)

    # Настраиваем специализированные логгеры
    trade_logger = logging.getLogger("trades")
    trade_logger.handlers.clear()
    trade_logger.addHandler(trade_handler)
    trade_logger.propagate = False

    signal_logger = logging.getLogger("signals")
    signal_logger.handlers.clear()
    signal_logger.addHandler(signal_handler)
    signal_logger.propagate = False
    
    # Отключаем шумные логи библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    logging.getLogger("pybit").setLevel(logging.INFO)  # Только важные сообщения от pybit
    
    _logging_configured = True
    return logging.getLogger("main")

async def main():
    tg_bot = None
    trading_loop = None
    health_monitor = None
    logger = None
    
    try:
        # Настраиваем логирование (защищено от повторного вызова)
        logger = setup_logging()
        logger.info("Initializing ML Trading Bot Terminal...")
        
        # 1. Загрузка настроек
        try:
            settings = load_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}", exc_info=True)
            raise

        # Сводка конфигурации для проверки деплоя (файлы настроек, включённые улучшения)
        try:
            log_server_config(settings, log=logger)
        except Exception as e:
            logger.warning(f"Could not log server config: {e}")
        
        if not settings.telegram_token:
            logger.error("TELEGRAM_TOKEN not found in .env file!")
            sys.exit(1)

        # 2. Инициализация состояния
        try:
            state = BotState()
            state.ensure_known_symbols(settings.symbols)
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
            # Передаем trading_loop в TelegramBot для обновления стратегий при изменении настроек
            tg_bot.trading_loop = trading_loop
        except Exception as e:
            logger.error(f"Failed to initialize TradingLoop: {e}", exc_info=True)
            raise
        
        # 7. Инициализация Health Monitor
        try:
            health_monitor = HealthMonitor(settings, state, bybit, tg_bot)
        except Exception as e:
            logger.error(f"Failed to initialize HealthMonitor: {e}", exc_info=True)
            raise
    
        # 8. Запуск компонентов (включая REST API для мобильного приложения)
        tasks = [
            tg_bot.start(),
            trading_loop.run(),
            health_monitor.run(),
        ]
        mobile_api_key = (os.getenv("MOBILE_API_KEY") or os.getenv("ALLOWED_USER_ID") or "").strip()
        if mobile_api_key:
            try:
                api_port = int(os.getenv("MOBILE_API_PORT", "8765"))
                api_host = "0.0.0.0"
                tasks.append(run_api_server(state, bybit, settings, trading_loop, model_manager, host=api_host, port=api_port))
                logger.info(
                    f"[Mobile API] Задача добавлена. Слушаем {api_host}:{api_port}. "
                    f"Снаружи: http://5.101.179.47:{api_port}/api/health (без ключа). "
                    f"Убедитесь, что порт {api_port} открыт в ufw/security group."
                )
            except Exception as e:
                logger.warning(f"[Mobile API] Не удалось добавить задачу: {e}", exc_info=True)
        else:
            logger.info(
                "[Mobile API] Не запущен: в .env нет MOBILE_API_KEY и нет ALLOWED_USER_ID. "
                "Добавьте MOBILE_API_KEY=ваш_ключ и перезапустите бота."
            )
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot execution cancelled.")
        except Exception as e:
            logger.error(f"Fatal error during execution: {e}", exc_info=True)
            raise
    except KeyboardInterrupt:
        if logger:
            logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        # Handle Telegram Conflict error specifically
        if Conflict and isinstance(e, Conflict):
            if logger:
                logger.error(
                    "Telegram bot conflict detected. Another instance is already running.\n"
                    "Please stop the other instance before starting this one.\n"
                    "You can check for running instances with: ps aux | grep run_bot.py"
                )
            else:
                print(
                    "ERROR: Another Telegram bot instance is already running.\n"
                    "Please stop the other instance before starting this one.",
                    file=sys.stderr
                )
            sys.exit(1)
        elif logger:
            logger.error(f"Fatal error: {e}", exc_info=True)
        else:
            print(f"Fatal error: {e}", file=sys.stderr)
        raise
    finally:
        # Cleanup in reverse order
        if logger:
            logger.info("Shutting down components...")
        
        # Shutdown Telegram bot first (most important for conflict resolution)
        if tg_bot:
            try:
                await tg_bot.shutdown()
            except Exception as e:
                if logger:
                    logger.error(f"Error shutting down Telegram bot: {e}", exc_info=True)
        
        # Note: trading_loop and health_monitor should handle their own cleanup
        # when their run() methods are cancelled
        
        if logger:
            logger.info("Shutdown complete.")

if __name__ == "__main__":
    # Обработка прерываний (Ctrl+C)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Manual shutdown.")
        sys.exit(0)
    except Exception as e:
        # Try to log if logger is available, otherwise print
        try:
            logger = logging.getLogger("main")
            logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        except:
            print(f"Unhandled exception in main: {e}", file=sys.stderr)
        sys.exit(1)
