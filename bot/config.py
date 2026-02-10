from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ApiSettings:
    """Настройки API для подключения к бирже"""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.bybit.com"  # По умолчанию продакшн
    
    def __post_init__(self):
        """Загружаем значения из окружения при создании объекта"""
        # Автоматически загружаем из переменных окружения если не заданы явно
        if not self.api_key:
            self.api_key = os.getenv("BYBIT_API_KEY", "").strip()
        if not self.api_secret:
            self.api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
        if not self.base_url or self.base_url == "https://api.bybit.com":
            env_url = os.getenv("BYBIT_BASE_URL", "").strip()
            if env_url:
                self.base_url = env_url


@dataclass
class StrategyParams:  # БЫЛО: MLStrategyParams
    """Параметры ML-стратегии"""
    
    # Основные параметры стратегии
    confidence_threshold: float = 0.35  # Минимальная уверенность модели для открытия позиции (35% по умолчанию, настроено для ~5 сделок в день)
    min_signal_strength: str = "слабое"  # Минимальная сила сигнала
    stability_filter: bool = True  # Фильтр стабильности сигналов (минимальная защита от частой смены направления)
    
    # Параметры риск-менеджмента
    target_profit_pct_margin: float = 18.0  # Целевая прибыль от маржи в %
    max_loss_pct_margin: float = 10.0  # Максимальный убыток от маржи в %
    
    # Целевые показатели стратегии (для мониторинга)
    min_signals_per_day: int = 1  # Минимум сигналов в день
    max_signals_per_day: int = 20  # Максимум сигналов в день (не используется для ограничения, только для мониторинга)
    
    # Настройки моделей
    model_type: Optional[str] = None  # Тип модели: "rf", "gb", "ensemble", "triple_ensemble", "quad_ensemble"
    mtf_enabled: bool = False  # Использовать мультитаймфреймовые модели
    feature_engineering_enabled: bool = True  # Использовать фичи инжиниринг
    
    # Параметры комбинированной MTF стратегии (1h + 15m)
    # ВАЖНО: По умолчанию False для безопасности. Включите через конфигурацию после настройки моделей.
    # Для включения установите use_mtf_strategy = True в конфигурационном файле или через Telegram бота.
    use_mtf_strategy: bool = False  # Использовать комбинированную стратегию (1h фильтр + 15m вход)
    mtf_confidence_threshold_1h: float = 0.50  # Порог уверенности для 1h модели (фильтр)
    mtf_confidence_threshold_15m: float = 0.35  # Порог уверенности для 15m модели (вход)
    mtf_alignment_mode: str = "strict"  # Режим выравнивания: "strict" или "weighted"
    mtf_require_alignment: bool = True  # Требовать совпадение направлений обеих моделей
    
    # Параметры для ретрейна
    retrain_days: int = 7  # Количество дней данных для ретрейна
    retrain_interval_hours: int = 24  # Интервал переобучения в часах
    
    def __post_init__(self):
        """Валидация значений"""
        # Убедимся, что confidence_threshold в пределах [0, 1]
        if not 0 <= self.confidence_threshold <= 1:
            self.confidence_threshold = 0.75
        
        # Убедимся, что min_signal_strength валиден
        valid_strengths = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
        if self.min_signal_strength not in valid_strengths:
            self.min_signal_strength = "слабое"
        
        # Убедимся, что целевая прибыль положительная
        if self.target_profit_pct_margin <= 0:
            self.target_profit_pct_margin = 18.0
            
        # Убедимся, что максимальный убыток положительный
        if self.max_loss_pct_margin <= 0:
            self.max_loss_pct_margin = 10.0
        
        # Валидация параметров MTF стратегии
        if not 0 <= self.mtf_confidence_threshold_1h <= 1:
            self.mtf_confidence_threshold_1h = 0.50
        if not 0 <= self.mtf_confidence_threshold_15m <= 1:
            self.mtf_confidence_threshold_15m = 0.35
        if self.mtf_alignment_mode not in ["strict", "weighted"]:
            self.mtf_alignment_mode = "strict"


@dataclass
class RiskParams:
    """Параметры управления рисками"""
    
    # Размеры позиций
    max_position_usd: float = 200.0
    position_size_mode: str = "percentage"  # УСТАРЕЛО: больше не используется, оставлено для обратной совместимости
    base_order_usd: float = 50.0  # Фиксированная сумма маржи в USD
    add_order_usd: float = 50.0
    margin_pct_balance: float = 0.20  # Маржа как процент от баланса (20%)
    # ИСПОЛЬЗУЕТСЯ: минимум из base_order_usd и margin_pct_balance% от баланса
    
    # Параметры стоп-лосса и тейк-профита
    stop_loss_pct: float = 0.01  # 1% от входа
    take_profit_pct: float = 0.02  # 2% от входа
    
    # Трейлинг стоп
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.003  # Активировать при прибыли 0.3%
    trailing_stop_distance_pct: float = 0.002  # Расстояние 0.2% от максимума
    
    # Частичное закрытие
    enable_partial_close: bool = True
    partial_close_pct: float = 0.5  # Закрывать 50% позиции
    partial_close_at_tp_pct: float = 0.5  # При достижении 50% пути к TP
    partial_close_levels: List[tuple] = field(default_factory=lambda: [(0.5, 0.5), (0.75, 0.25)])  # (прогресс к TP, % закрытия)
    
    # Защита от убытков
    enable_loss_cooldown: bool = True
    loss_cooldown_minutes: int = 120  # Охлаждение 2 часа после убытка
    max_consecutive_losses: int = 3  # Максимум 3 убытка подряд
    
    # Защита прибыли
    enable_profit_protection: bool = True
    profit_protection_activation_pct: float = 0.01  # Активировать при прибыли 1%
    profit_protection_retreat_pct: float = 0.003  # Закрыть при откате 0.3%
    
    # Безубыток
    enable_breakeven: bool = True
    breakeven_activation_pct: float = 0.005  # Активировать при прибыли 0.5%

    # Комиссия биржи (per side). Например 0.0006 = 0.06% за вход или выход
    fee_rate: float = 0.0006

    # Категоризация горизонта позиции по TP/SL
    mid_term_tp_pct: float = 0.025  # 2.5% от цены
    long_term_tp_pct: float = 0.04  # 4% от цены
    long_term_sl_pct: float = 0.02  # 2% от цены
    long_term_ignore_reverse: bool = True

    # Усреднение (DCA)
    dca_enabled: bool = True
    dca_drawdown_pct: float = 0.003  # Усреднять при просадке 0.3%
    dca_max_adds: int = 2
    dca_min_confidence: float = 0.6

    # Реверс при сильном обратном сигнале
    reverse_on_strong_signal: bool = True
    reverse_min_confidence: float = 0.75
    reverse_min_strength: str = "сильное"  # слабое/умеренное/среднее/сильное/очень_сильное
    
    # Динамическое изменение размера позиции
    enable_dynamic_position_sizing: bool = True
    volatility_reduction_factor: float = 0.5  # Уменьшать до 50% при высокой волатильности
    high_volatility_atr_multiplier: float = 1.5  # Высокая волатильность = ATR > 1.5x от среднего
    
    def __post_init__(self):
        """Валидация значений риска"""
        # Преобразуем проценты если нужно
        if self.stop_loss_pct >= 1:
            self.stop_loss_pct /= 100.0
        if self.take_profit_pct >= 1:
            self.take_profit_pct /= 100.0
        if self.trailing_stop_activation_pct >= 1:
            self.trailing_stop_activation_pct /= 100.0
        if self.trailing_stop_distance_pct >= 1:
            self.trailing_stop_distance_pct /= 100.0
        if self.profit_protection_activation_pct >= 1:
            self.profit_protection_activation_pct /= 100.0
        if self.profit_protection_retreat_pct >= 1:
            self.profit_protection_retreat_pct /= 100.0
        if self.breakeven_activation_pct >= 1:
            self.breakeven_activation_pct /= 100.0
        if self.fee_rate >= 1:
            self.fee_rate /= 100.0
        if self.mid_term_tp_pct >= 1:
            self.mid_term_tp_pct /= 100.0
        if self.long_term_tp_pct >= 1:
            self.long_term_tp_pct /= 100.0
        if self.long_term_sl_pct >= 1:
            self.long_term_sl_pct /= 100.0
        if self.dca_drawdown_pct >= 1:
            self.dca_drawdown_pct /= 100.0
        if self.reverse_min_confidence >= 1:
            self.reverse_min_confidence /= 100.0


@dataclass
class SymbolMLSettings:
    """Настройки ML для конкретной торговой пары"""
    
    # Включение/выключение стратегии для пары
    enabled: bool = True
    
    # Настройки моделей
    model_type: Optional[str] = None  # Тип модели для этой пары
    mtf_enabled: Optional[bool] = None  # Использовать MTF для этой пары
    model_path: Optional[str] = None  # Путь к конкретной модели
    
    # Параметры для этой пары
    confidence_threshold: Optional[float] = None
    min_signal_strength: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Преобразует настройки в словарь"""
        result = {"enabled": self.enabled}
        if self.model_type is not None:
            result["model_type"] = self.model_type
        if self.mtf_enabled is not None:
            result["mtf_enabled"] = self.mtf_enabled
        if self.model_path is not None:
            result["model_path"] = self.model_path
        if self.confidence_threshold is not None:
            result["confidence_threshold"] = self.confidence_threshold
        if self.min_signal_strength is not None:
            result["min_signal_strength"] = self.min_signal_strength
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolMLSettings':
        """Создает настройки из словаря"""
        return cls(
            enabled=data.get("enabled", True),
            model_type=data.get("model_type", None),
            mtf_enabled=data.get("mtf_enabled", None),
            model_path=data.get("model_path", None),
            confidence_threshold=data.get("confidence_threshold", None),
            min_signal_strength=data.get("min_signal_strength", None),
        )


@dataclass
class AppSettings:
    """Основные настройки приложения"""
    
    # Telegram settings
    telegram_token: str = ""
    allowed_user_id: Optional[int] = None
    notification_level: str = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW
    
    # Торговые пары
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    active_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    primary_symbol: str = "BTCUSDT"  # Основная пара для UI
    
    # Настройки API
    api: ApiSettings = field(default_factory=ApiSettings)
    
    # ML стратегия (единственная доступная)
    ml_strategy: StrategyParams = field(default_factory=StrategyParams)  # БЫЛО: MLStrategyParams
    
    # Управление рисками
    risk: RiskParams = field(default_factory=RiskParams)
    
    # Общие настройки
    timeframe: str = "15m"  # Таймфрейм для торговли
    leverage: int = 10  # Плечо
    live_poll_seconds: int = 120  # Пауза между циклами (увеличено со 60 до 120 для снижения rate limit)
    kline_limit: int = 1000  # Количество свечей для анализа
    
    # Настройки мониторинга здоровья
    health_check_interval_seconds: int = 300  # Интервал проверки здоровья (5 минут)
    memory_threshold_mb: float = 1000.0  # Порог использования памяти для предупреждений (МБ)
    memory_critical_mb: float = 2000.0  # Критический порог использования памяти (МБ)
    
    # Настройки моделей по парам
    symbol_ml_settings: Dict[str, SymbolMLSettings] = field(default_factory=dict)
    
    def __post_init__(self):
        """Инициализация после создания"""
        # Валидация символов
        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        
        if not self.symbols:
            self.symbols = available_symbols
        
        # Фильтруем активные символы
        self.active_symbols = [s for s in self.active_symbols if s in available_symbols]
        if not self.active_symbols:
            self.active_symbols = [available_symbols[0]]
        
        # Проверяем primary_symbol
        if self.primary_symbol not in available_symbols:
            self.primary_symbol = self.active_symbols[0]
    
    def get_ml_settings_for_symbol(self, symbol: str) -> SymbolMLSettings:
        """Получает настройки ML для конкретной пары"""
        symbol = symbol.upper()
        if symbol in self.symbol_ml_settings:
            return self.symbol_ml_settings[symbol]
        
        # Возвращаем настройки по умолчанию
        return SymbolMLSettings(
            enabled=True,
            model_type=self.ml_strategy.model_type,
            mtf_enabled=self.ml_strategy.mtf_enabled,
            confidence_threshold=self.ml_strategy.confidence_threshold,
            min_signal_strength=self.ml_strategy.min_signal_strength,
        )
    
    def set_ml_settings_for_symbol(self, symbol: str, settings: SymbolMLSettings) -> None:
        """Устанавливает настройки ML для конкретной пары"""
        symbol = symbol.upper()
        self.symbol_ml_settings[symbol] = settings


def load_settings() -> AppSettings:
    """
    Загружает настройки из .env файла и переменных окружения
    """
    # Определяем путь к .env файлу
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    # Загружаем .env файл
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    
    # Создаем базовые настройки
    settings = AppSettings()
    
    # Загружаем настройки API
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").strip()
    
    if api_key:
        settings.api.api_key = api_key
    if api_secret:
        settings.api.api_secret = api_secret
    if base_url:
        settings.api.base_url = base_url
    
    # Загружаем торговые символы
    trading_symbol = os.getenv("TRADING_SYMBOL", "").strip().upper()
    if trading_symbol and trading_symbol in settings.symbols:
        settings.active_symbols = [trading_symbol]
        settings.primary_symbol = trading_symbol
    
    # Загружаем активные символы (для многопарной торговли)
    active_symbols_env = os.getenv("ACTIVE_SYMBOLS", "").strip()
    if active_symbols_env:
        symbols_list = [s.strip().upper() for s in active_symbols_env.split(",") if s.strip()]
        valid_symbols = [s for s in symbols_list if s in settings.symbols]
        if valid_symbols:
            settings.active_symbols = valid_symbols
            if not trading_symbol:
                settings.primary_symbol = valid_symbols[0]
    
    # Загружаем основной символ
    primary_symbol_env = os.getenv("PRIMARY_SYMBOL", "").strip().upper()
    if primary_symbol_env and primary_symbol_env in settings.symbols:
        settings.primary_symbol = primary_symbol_env
    
    # Загружаем настройки ML стратегии
    ml_conf_threshold = os.getenv("ML_CONFIDENCE_THRESHOLD", "").strip()
    if ml_conf_threshold:
        try:
            settings.ml_strategy.confidence_threshold = float(ml_conf_threshold)
        except ValueError:
            pass
    
    ml_min_strength = os.getenv("ML_MIN_SIGNAL_STRENGTH", "").strip()
    if ml_min_strength:
        valid_strengths = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
        if ml_min_strength.lower() in valid_strengths:
            settings.ml_strategy.min_signal_strength = ml_min_strength.lower()
    
    # Загружаем сохраненные ML настройки из файла (если есть)
    # Это перезапишет значения по умолчанию и переменные окружения
    ml_settings_file = project_root / "ml_settings.json"
    if ml_settings_file.exists():
        try:
            with open(ml_settings_file, 'r', encoding='utf-8') as f:
                ml_dict = json.load(f)
                if "confidence_threshold" in ml_dict:
                    settings.ml_strategy.confidence_threshold = float(ml_dict["confidence_threshold"])
                    logger.info(f"Loaded confidence_threshold from ml_settings.json: {settings.ml_strategy.confidence_threshold}")
        except Exception as e:
            logger.warning(f"Failed to load ml_settings.json: {e}")
    
    ml_stability = os.getenv("ML_STABILITY_FILTER", "").strip()
    if ml_stability:
        settings.ml_strategy.stability_filter = ml_stability.lower() in ("true", "1", "yes")
    
    ml_target_profit = os.getenv("ML_TARGET_PROFIT_PCT_MARGIN", "").strip()
    if ml_target_profit:
        try:
            settings.ml_strategy.target_profit_pct_margin = float(ml_target_profit)
        except ValueError:
            pass
    
    ml_max_loss = os.getenv("ML_MAX_LOSS_PCT_MARGIN", "").strip()
    if ml_max_loss:
        try:
            settings.ml_strategy.max_loss_pct_margin = float(ml_max_loss)
        except ValueError:
            pass
    
    ml_model_type = os.getenv("ML_MODEL_TYPE", "").strip().lower()
    if ml_model_type and ml_model_type in ["rf", "gb", "ensemble", "triple_ensemble", "quad_ensemble"]:
        settings.ml_strategy.model_type = ml_model_type
    
    ml_mtf_enabled = os.getenv("ML_MTF_ENABLED", "").strip()
    if ml_mtf_enabled:
        settings.ml_strategy.mtf_enabled = ml_mtf_enabled.lower() not in ("0", "false", "no", "off")
    
    # Загружаем путь к модели
    ml_model_path = os.getenv("ML_MODEL_PATH", "").strip()
    if ml_model_path:
        settings.ml_strategy.model_type = None  # Если задан путь, тип модели берем из пути
    
    # Автоматический поиск модели
    if not ml_model_path:
        _auto_find_ml_model(settings)
    
    # Загружаем настройки рисков
    max_position = os.getenv("MAX_POSITION_USD", "").strip()
    if max_position:
        try:
            settings.risk.max_position_usd = float(max_position)
        except ValueError:
            pass
    
    base_order = os.getenv("BASE_ORDER_USD", "").strip()
    if base_order:
        try:
            settings.risk.base_order_usd = float(base_order)
        except ValueError:
            pass
    
    stop_loss = os.getenv("STOP_LOSS_PCT", "").strip()
    if stop_loss:
        try:
            value = float(stop_loss)
            if value >= 1:
                value /= 100.0
            settings.risk.stop_loss_pct = value
        except ValueError:
            pass
    
    take_profit = os.getenv("TAKE_PROFIT_PCT", "").strip()
    if take_profit:
        try:
            value = float(take_profit)
            if value >= 1:
                value /= 100.0
            settings.risk.take_profit_pct = value
        except ValueError:
            pass

    fee_rate = os.getenv("FEE_RATE", "").strip()
    if fee_rate:
        try:
            value = float(fee_rate)
            if value >= 1:
                value /= 100.0
            settings.risk.fee_rate = value
        except ValueError:
            pass

    dca_drawdown = os.getenv("DCA_DRAWDOWN_PCT", "").strip()
    if dca_drawdown:
        try:
            value = float(dca_drawdown)
            if value >= 1:
                value /= 100.0
            settings.risk.dca_drawdown_pct = value
        except ValueError:
            pass

    reverse_min_conf = os.getenv("REVERSE_MIN_CONFIDENCE", "").strip()
    if reverse_min_conf:
        try:
            value = float(reverse_min_conf)
            if value >= 1:
                value /= 100.0
            settings.risk.reverse_min_confidence = value
        except ValueError:
            pass
    
    # Загружаем общие настройки
    timeframe = os.getenv("TIMEFRAME", "").strip()
    if timeframe:
        settings.timeframe = timeframe
    
    # Telegram settings from env
    settings.telegram_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    tg_user_id = os.getenv("ALLOWED_USER_ID", "").strip()
    if tg_user_id:
        try:
            settings.allowed_user_id = int(tg_user_id)
        except ValueError:
            pass
            
    leverage_env = os.getenv("LEVERAGE", "").strip()
    if leverage_env:
        try:
            settings.leverage = int(leverage_env)
        except ValueError:
            pass
    
    poll_seconds = os.getenv("LIVE_POLL_SECONDS", "").strip()
    if poll_seconds:
        try:
            settings.live_poll_seconds = int(poll_seconds)
        except ValueError:
            pass
    
    kline_limit_env = os.getenv("KLINE_LIMIT", "").strip()
    if kline_limit_env:
        try:
            settings.kline_limit = int(kline_limit_env)
        except ValueError:
            pass
    
    # Загружаем настройки мониторинга здоровья
    health_check_interval = os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "").strip()
    if health_check_interval:
        try:
            settings.health_check_interval_seconds = int(health_check_interval)
        except ValueError:
            pass
    
    memory_threshold = os.getenv("MEMORY_THRESHOLD_MB", "").strip()
    if memory_threshold:
        try:
            settings.memory_threshold_mb = float(memory_threshold)
        except ValueError:
            pass
    
    memory_critical = os.getenv("MEMORY_CRITICAL_MB", "").strip()
    if memory_critical:
        try:
            settings.memory_critical_mb = float(memory_critical)
        except ValueError:
            pass
    
    # Загружаем настройки по символам из JSON
    _load_symbol_ml_settings(settings)
    
    # Загружаем настройки риска из JSON
    _load_risk_settings(settings)
    
    return settings


def _auto_find_ml_model(settings: AppSettings) -> None:
    """Автоматически ищет ML модель для активных символов"""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "ml_models"
    
    if not models_dir.exists():
        return
    
    # Ищем модели для каждого активного символа
    for symbol in settings.active_symbols:
        found_model = None
        
        # Сначала пробуем найти по предпочитаемому типу модели
        if settings.ml_strategy.model_type:
            pattern = f"{settings.ml_strategy.model_type}_{symbol}_*.pkl"
            for model_file in sorted(models_dir.glob(pattern), reverse=True):
                if model_file.is_file():
                    found_model = str(model_file)
                    break
        
        # Если не нашли, пробуем другие типы
        if not found_model:
            for model_type in ["quad_ensemble", "triple_ensemble", "ensemble", "rf", "gb"]:
                pattern = f"{model_type}_{symbol}_*.pkl"
                for model_file in sorted(models_dir.glob(pattern), reverse=True):
                    if model_file.is_file():
                        found_model = str(model_file)
                        break
                if found_model:
                    break
        
        # Сохраняем найденную модель в настройки символа
        if found_model:
            symbol_settings = settings.get_ml_settings_for_symbol(symbol)
            symbol_settings.model_path = found_model
            settings.set_ml_settings_for_symbol(symbol, symbol_settings)
            
            # Если это primary_symbol, устанавливаем глобальный путь
            if symbol == settings.primary_symbol:
                # Определяем тип модели из имени файла
                model_name = Path(found_model).stem
                if "_" in model_name:
                    model_type_from_name = model_name.split("_")[0]
                    if model_type_from_name in ["rf", "gb", "ensemble", "triple_ensemble", "quad_ensemble"]:
                        settings.ml_strategy.model_type = model_type_from_name


def _get_ml_settings_file() -> Path:
    """Возвращает путь к файлу с настройками ML по символам"""
    config_dir = Path(__file__).parent.parent
    return config_dir / "symbol_ml_settings.json"


def _load_symbol_ml_settings(settings: AppSettings) -> None:
    """Загружает настройки ML для каждого символа из JSON файла"""
    settings_file = _get_ml_settings_file()
    if not settings_file.exists():
        return
    
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for symbol, symbol_data in data.items():
            symbol = symbol.upper()
            if symbol in settings.symbols:
                ml_settings = SymbolMLSettings.from_dict(symbol_data)
                settings.symbol_ml_settings[symbol] = ml_settings
    except Exception as e:
        print(f"[config] Warning: Error loading symbol ML settings: {e}")


def save_symbol_ml_settings(settings: AppSettings) -> None:
    """Сохраняет настройки ML для каждого символа в JSON файл"""
    settings_file = _get_ml_settings_file()
    try:
        data = {}
        for symbol, symbol_settings in settings.symbol_ml_settings.items():
            data[symbol] = symbol_settings.to_dict()
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[config] Warning: Error saving symbol ML settings: {e}")


def _get_risk_settings_file() -> Path:
    """Возвращает путь к файлу с настройками риска"""
    config_dir = Path(__file__).parent.parent
    return config_dir / "risk_settings.json"


def _load_risk_settings(settings: AppSettings) -> None:
    """Загружает настройки риска из JSON файла"""
    settings_file = _get_risk_settings_file()
    if not settings_file.exists():
        return
    
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        risk = settings.risk
        
        # Загружаем значения если они есть
        if "margin_pct_balance" in data:
            risk.margin_pct_balance = float(data["margin_pct_balance"])
        if "base_order_usd" in data:
            risk.base_order_usd = float(data["base_order_usd"])
        if "stop_loss_pct" in data:
            risk.stop_loss_pct = float(data["stop_loss_pct"])
        if "take_profit_pct" in data:
            risk.take_profit_pct = float(data["take_profit_pct"])
        if "enable_trailing_stop" in data:
            risk.enable_trailing_stop = bool(data["enable_trailing_stop"])
        if "trailing_stop_activation_pct" in data:
            risk.trailing_stop_activation_pct = float(data["trailing_stop_activation_pct"])
        if "trailing_stop_distance_pct" in data:
            risk.trailing_stop_distance_pct = float(data["trailing_stop_distance_pct"])
        if "enable_partial_close" in data:
            risk.enable_partial_close = bool(data["enable_partial_close"])
        if "enable_breakeven" in data:
            risk.enable_breakeven = bool(data["enable_breakeven"])
        if "breakeven_activation_pct" in data:
            risk.breakeven_activation_pct = float(data["breakeven_activation_pct"])
        if "enable_loss_cooldown" in data:
            risk.enable_loss_cooldown = bool(data["enable_loss_cooldown"])
        if "fee_rate" in data:
            value = float(data["fee_rate"])
            if value >= 1:
                value /= 100.0
            risk.fee_rate = value
        if "mid_term_tp_pct" in data:
            value = float(data["mid_term_tp_pct"])
            if value >= 1:
                value /= 100.0
            risk.mid_term_tp_pct = value
        if "long_term_tp_pct" in data:
            value = float(data["long_term_tp_pct"])
            if value >= 1:
                value /= 100.0
            risk.long_term_tp_pct = value
        if "long_term_sl_pct" in data:
            value = float(data["long_term_sl_pct"])
            if value >= 1:
                value /= 100.0
            risk.long_term_sl_pct = value
        if "long_term_ignore_reverse" in data:
            risk.long_term_ignore_reverse = bool(data["long_term_ignore_reverse"])
        if "dca_enabled" in data:
            risk.dca_enabled = bool(data["dca_enabled"])
        if "dca_drawdown_pct" in data:
            value = float(data["dca_drawdown_pct"])
            if value >= 1:
                value /= 100.0
            risk.dca_drawdown_pct = value
        if "dca_max_adds" in data:
            risk.dca_max_adds = int(data["dca_max_adds"])
        if "dca_min_confidence" in data:
            risk.dca_min_confidence = float(data["dca_min_confidence"])
        if "reverse_on_strong_signal" in data:
            risk.reverse_on_strong_signal = bool(data["reverse_on_strong_signal"])
        if "reverse_min_confidence" in data:
            value = float(data["reverse_min_confidence"])
            if value >= 1:
                value /= 100.0
            risk.reverse_min_confidence = value
        if "reverse_min_strength" in data:
            risk.reverse_min_strength = str(data["reverse_min_strength"])
        
        print(f"[config] Risk settings loaded from {settings_file}")
    except Exception as e:
        print(f"[config] Warning: Error loading risk settings: {e}")