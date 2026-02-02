import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, memory monitoring will be disabled")

class HealthMonitor:
    """Мониторинг здоровья бота"""
    
    def __init__(self, settings, state, bybit_client, tg_bot=None):
        self.settings = settings
        self.state = state
        self.bybit = bybit_client
        self.tg_bot = tg_bot
        
        # Статус последних проверок
        self.last_api_check = None
        self.last_api_success = False
        self.last_notification = datetime.now()
        
        # Пороги для уведомлений (из настроек или по умолчанию)
        self.memory_threshold_mb = getattr(settings, 'memory_threshold_mb', 1000.0)  # МБ - порог для предупреждений
        self.memory_critical_mb = getattr(settings, 'memory_critical_mb', 2000.0)  # МБ - критический порог
        self.health_check_interval = getattr(settings, 'health_check_interval_seconds', 300)  # секунды
        self.notification_cooldown = timedelta(minutes=15)  # Не чаще раза в 15 минут
    
    async def run(self):
        """Основной цикл мониторинга здоровья"""
        logger.info("Starting Health Monitor...")
        
        while True:
            try:
                # Проверяем здоровье с интервалом из настроек
                await asyncio.sleep(self.health_check_interval)
                
                if not self.state.is_running:
                    continue
                
                health_status = await self.check_health()
                
                # Если есть проблемы, уведомляем
                if not health_status["healthy"] and self.tg_bot:
                    # Проверяем cooldown уведомлений
                    if datetime.now() - self.last_notification > self.notification_cooldown:
                        await self.send_health_alert(health_status)
                        self.last_notification = datetime.now()
            
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def check_health(self) -> Dict[str, Any]:
        """Проверяет здоровье всех компонентов"""
        health = {
            "healthy": True,
            "issues": [],
            "api": False,
            "models": False,
            "memory": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Проверка API
        api_ok = await self._check_api_connection()
        health["api"] = api_ok
        if not api_ok:
            health["healthy"] = False
            health["issues"].append("❌ API connection failed")
        
        # 2. Проверка моделей
        models_ok = self._check_models()
        health["models"] = models_ok
        if not models_ok:
            health["healthy"] = False
            health["issues"].append("❌ Model files missing or invalid")
        
        # 3. Проверка памяти
        memory_ok, memory_info = self._check_memory()
        health["memory"] = memory_ok
        health["memory_usage_mb"] = memory_info["used_mb"]
        if not memory_ok:
            # Различаем предупреждение и критическую ситуацию
            if memory_info["used_mb"] >= self.memory_critical_mb:
                health["healthy"] = False
                health["issues"].append(f"🔴 CRITICAL memory usage: {memory_info['used_mb']:.1f} MB")
            else:
                # Высокое использование, но не критическое (например, во время обучения)
                # Не помечаем как unhealthy, но добавляем в issues для информации
                health["issues"].append(f"⚠️ High memory usage: {memory_info['used_mb']:.1f} MB (normal during model training)")
        
        return health
    
    async def _check_api_connection(self) -> bool:
        """Проверяет подключение к Bybit API"""
        try:
            # Пробуем получить баланс
            response = await asyncio.to_thread(
                self.bybit.get_wallet_balance
            )
            
            if response and response.get("retCode") == 0:
                self.last_api_check = datetime.now()
                self.last_api_success = True
                return True
            else:
                self.last_api_success = False
                logger.warning(f"API check failed: {response}")
                return False
        
        except Exception as e:
            self.last_api_success = False
            logger.error(f"API check error: {e}")
            return False
    
    def _check_models(self) -> bool:
        """Проверяет доступность моделей"""
        try:
            models_dir = Path("ml_models")
            if not models_dir.exists():
                logger.warning("Models directory not found")
                return False
            
            # Проверяем, есть ли модели для активных символов
            missing_models = []
            for symbol in self.state.active_symbols:
                model_path = self.state.symbol_models.get(symbol)
                
                if not model_path or not Path(model_path).exists():
                    # Пытаемся найти любую модель для символа
                    models = list(models_dir.glob(f"*_{symbol}_*.pkl"))
                    if not models:
                        missing_models.append(symbol)
            
            if missing_models:
                logger.warning(f"Missing models for: {', '.join(missing_models)}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Model check error: {e}")
            return False
    
    def _check_memory(self) -> tuple[bool, Dict[str, float]]:
        """Проверяет использование памяти"""
        if not PSUTIL_AVAILABLE:
            # Если psutil недоступен, считаем что все ОК
            return True, {"used_mb": 0, "threshold_mb": self.memory_threshold_mb, "note": "psutil not available"}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            used_mb = memory_info.rss / 1024 / 1024  # Байты в МБ
            
            info = {
                "used_mb": used_mb,
                "threshold_mb": self.memory_threshold_mb,
                "critical_threshold_mb": self.memory_critical_mb
            }
            
            # Считаем проблемой только критическое использование памяти
            is_ok = used_mb < self.memory_critical_mb
            
            # Логируем предупреждение только при превышении порога предупреждений
            if used_mb >= self.memory_threshold_mb:
                if used_mb >= self.memory_critical_mb:
                    logger.warning(f"🔴 CRITICAL memory usage: {used_mb:.1f} MB (threshold: {self.memory_critical_mb:.1f} MB)")
                else:
                    # Высокое использование, но не критическое (может быть нормально при обучении моделей)
                    logger.info(f"⚠️ High memory usage: {used_mb:.1f} MB (threshold: {self.memory_threshold_mb:.1f} MB, critical: {self.memory_critical_mb:.1f} MB)")
            
            return is_ok, info
        
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return True, {"used_mb": 0, "threshold_mb": self.memory_threshold_mb}
    
    async def send_health_alert(self, health_status: Dict[str, Any]):
        """Отправляет уведомление о проблемах"""
        if not self.tg_bot:
            return
        
        message = "⚠️ ПРОБЛЕМЫ СО ЗДОРОВЬЕМ БОТА\n\n"
        
        for issue in health_status["issues"]:
            message += f"{issue}\n"
        
        message += f"\n🕐 Время: {datetime.now().strftime('%H:%M:%S')}"
        
        try:
            await self.tg_bot.send_notification(message)
        except Exception as e:
            logger.error(f"Error sending health alert: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус здоровья"""
        return {
            "api_status": "🟢 OK" if self.last_api_success else "🔴 FAILED",
            "last_api_check": self.last_api_check.isoformat() if self.last_api_check else None
        }
