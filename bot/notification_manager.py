import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class NotificationLevel(Enum):
    """Уровни важности уведомлений"""
    CRITICAL = 1  # Ошибки API, критические убытки
    HIGH = 2      # Открытие/закрытие позиций, достижение TP/SL
    MEDIUM = 3    # Сигналы высокой уверенности, предупреждения
    LOW = 4       # Все сигналы, общая статистика

class NotificationManager:
    """Управление уведомлениями с уровнями важности"""
    
    def __init__(self, tg_bot, settings):
        self.tg_bot = tg_bot
        self.settings = settings
        
        # Преобразуем строку в enum
        level_map = {
            "CRITICAL": NotificationLevel.CRITICAL,
            "HIGH": NotificationLevel.HIGH,
            "MEDIUM": NotificationLevel.MEDIUM,
            "LOW": NotificationLevel.LOW
        }
        notification_level = getattr(settings, 'notification_level', 'HIGH')
        if notification_level:
            notification_level = str(notification_level).upper()
        else:
            notification_level = 'HIGH'
        self.current_level = level_map.get(notification_level, NotificationLevel.HIGH)
    
    async def send(self, message: str, level: NotificationLevel = NotificationLevel.MEDIUM):
        """Отправляет уведомление если уровень соответствует настройкам"""
        try:
            # Проверяем, нужно ли отправлять уведомление
            if level.value <= self.current_level.value:
                if self.tg_bot:
                    await self.tg_bot.send_notification(message)
                else:
                    logger.info(f"[Notification] {message}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def critical(self, message: str):
        """Критичное уведомление"""
        await self.send(f"🚨 КРИТИЧНО\n{message}", NotificationLevel.CRITICAL)
    
    async def high(self, message: str):
        """Важное уведомление"""
        await self.send(message, NotificationLevel.HIGH)
    
    async def medium(self, message: str):
        """Среднее уведомление"""
        await self.send(message, NotificationLevel.MEDIUM)
    
    async def low(self, message: str):
        """Низкоприоритетное уведомление"""
        await self.send(message, NotificationLevel.LOW)
