"""
Планировщик для автоматического запуска оптимизации стратегий.

Запускает auto_strategy_optimizer.py по расписанию (раз в неделю, ночью).

Использование:
    # Запуск планировщика
    python schedule_strategy_optimizer.py
    
    # Запуск с кастомным расписанием
    python schedule_strategy_optimizer.py --day sunday --hour 3
    
    # Запуск один раз для тестирования
    python schedule_strategy_optimizer.py --run-once
"""
import argparse
import logging
import sys
import subprocess
from datetime import datetime, time
from pathlib import Path

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("⚠️  Библиотека 'schedule' не установлена. Установите: pip install schedule")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_optimization():
    """Запускает оптимизацию стратегий"""
    logger.info("=" * 80)
    logger.info("🚀 ЗАПУСК АВТОМАТИЧЕСКОЙ ОПТИМИЗАЦИИ СТРАТЕГИЙ")
    logger.info("=" * 80)
    logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Проверяем, включено ли автообновление в настройках
    try:
        from bot.config import load_settings
        settings = load_settings()
        if not settings.ml_strategy.auto_optimize_strategies:
            logger.info("⏸️  Автообновление стратегий выключено в настройках. Пропускаем оптимизацию.")
            logger.info("💡 Для включения используйте Telegram бота: ML НАСТРОЙКИ → Автообновление")
            return
    except Exception as e:
        logger.warning(f"⚠️  Не удалось проверить настройки автообновления: {e}. Продолжаем оптимизацию.")
    
    try:
        python_exe = sys.executable
        cmd = [python_exe, "auto_strategy_optimizer.py"]
        
        # Запускаем оптимизацию
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 часа таймаут
        )
        
        if result.returncode == 0:
            logger.info("✅ Оптимизация завершена успешно")
            logger.info(f"STDOUT (последние 500 символов):\n{result.stdout[-500:]}")
        else:
            logger.error("❌ Оптимизация завершилась с ошибкой")
            logger.error(f"STDERR:\n{result.stderr[-500:]}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Таймаут при выполнении оптимизации (превышен лимит 2 часа)")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при запуске оптимизации: {e}", exc_info=True)


def setup_scheduler(day: str = "sunday", hour: int = 3):
    """Настраивает расписание запуска оптимизации"""
    if not SCHEDULE_AVAILABLE:
        logger.error("❌ Библиотека 'schedule' не установлена. Установите: pip install schedule")
        return None
    
    # Маппинг дней недели
    day_map = {
        "monday": schedule.every().monday,
        "tuesday": schedule.every().tuesday,
        "wednesday": schedule.every().wednesday,
        "thursday": schedule.every().thursday,
        "friday": schedule.every().friday,
        "saturday": schedule.every().saturday,
        "sunday": schedule.every().sunday,
    }
    
    if day.lower() not in day_map:
        logger.error(f"❌ Неверный день недели: {day}. Используйте: monday, tuesday, ..., sunday")
        return None
    
    # Настраиваем расписание
    day_func = day_map[day.lower()]
    day_func.at(f"{hour:02d}:00").do(run_optimization)
    
    logger.info(f"✅ Расписание настроено: каждое {day} в {hour:02d}:00")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Планировщик автоматической оптимизации стратегий",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Запуск с настройками по умолчанию (воскресенье в 3:00)
  python schedule_strategy_optimizer.py
  
  # Запуск в понедельник в 2:00
  python schedule_strategy_optimizer.py --day monday --hour 2
  
  # Запуск один раз для тестирования
  python schedule_strategy_optimizer.py --run-once
        """
    )
    
    parser.add_argument("--day", type=str, default="sunday",
                       choices=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                       help="День недели для запуска (по умолчанию: sunday)")
    parser.add_argument("--hour", type=int, default=3,
                       help="Час для запуска (0-23, по умолчанию: 3)")
    parser.add_argument("--run-once", action="store_true",
                       help="Запустить оптимизацию один раз и выйти (для тестирования)")
    
    args = parser.parse_args()
    
    # Валидация часа
    if not 0 <= args.hour <= 23:
        logger.error(f"❌ Неверный час: {args.hour}. Используйте значение от 0 до 23")
        sys.exit(1)
    
    # Если --run-once, запускаем один раз
    if args.run_once:
        logger.info("🔧 Режим тестирования: запуск один раз")
        run_optimization()
        return
    
    # Настраиваем расписание
    if not setup_scheduler(args.day, args.hour):
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("📅 ПЛАНИРОВЩИК ЗАПУЩЕН")
    logger.info("=" * 80)
    logger.info(f"Расписание: каждое {args.day} в {args.hour:02d}:00")
    logger.info("Планировщик работает. Нажмите Ctrl+C для остановки.")
    logger.info("=" * 80)
    
    # Основной цикл планировщика
    try:
        while True:
            schedule.run_pending()
            import time
            time.sleep(60)  # Проверяем каждую минуту
    except KeyboardInterrupt:
        logger.info("\n⚠️  Планировщик остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в планировщике: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
