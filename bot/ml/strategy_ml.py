"""
ML-стратегия для торгового бота.
Использует обученную ML-модель для генерации торговых сигналов.
"""
import warnings
import os

# Подавляем предупреждения scikit-learn ДО импорта библиотек
# Устанавливаем переменную окружения ПЕРВОЙ
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['SKLEARN_WARNINGS'] = 'ignore'

# Фильтруем все предупреждения sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*propagate the scikit-learn configuration.*')
# Специфичное предупреждение из терминала
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
# Подавляем предупреждения XGBoost про pickle и версии
warnings.filterwarnings('ignore', message='.*loading a serialized model.*')
warnings.filterwarnings('ignore', message='.*XGBoost.*')
os.environ['XGB_SILENT'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from bot.strategy import Action, Bias, Signal
from bot.ml.feature_engineering import FeatureEngineer
from bot.config import StrategyParams
from bot.config import calculate_price_pct_from_margin_pct, DEFAULT_LEVERAGE
# Импортируем классы ансамбля для корректной десериализации pickle
from bot.ml.model_trainer import PreTrainedVotingEnsemble, WeightedEnsemble, TripleEnsemble

logger = logging.getLogger(__name__)


class MLStrategy:
    """
    ML-стратегия, использующая обученную модель для предсказания движения цены.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.35, min_signal_strength: str = "слабое", stability_filter: bool = True, use_dynamic_threshold: bool = True, min_signals_per_day: int = 1, max_signals_per_day: int = 20, use_dynamic_ensemble_weights: bool = False, adx_trend_threshold: float = 25.0, adx_flat_threshold: float = 20.0, trend_weights: Optional[Dict[str, float]] = None, flat_weights: Optional[Dict[str, float]] = None, use_adaptive_confidence_by_atr: bool = False, adaptive_confidence_k: float = 0.3, adaptive_confidence_min: float = 0.8, adaptive_confidence_max: float = 1.2, adaptive_confidence_atr_lookback: int = 500, use_fixed_sl_from_risk: bool = False):
        """
        Инициализирует ML-стратегию.
        
        Args:
            model_path: Путь к сохраненной модели (.pkl файл)
            confidence_threshold: Минимальная уверенность модели для открытия позиции (0-1)
            min_signal_strength: Минимальная сила сигнала ("слабое", "умеренное", "среднее", "сильное", "очень_сильное")
            stability_filter: Фильтр стабильности - требовать более высокую уверенность для смены направления
            use_dynamic_threshold: Использовать динамические пороги на основе рыночных условий
            min_signals_per_day: Минимальное количество сигналов в день (гарантирует хотя бы 1 сигнал)
            max_signals_per_day: Максимальное количество сигналов в день (ограничивает избыточную торговлю)
            use_dynamic_ensemble_weights: Динамические веса ансамбля по режиму рынка (тренд/флэт по ADX)
            adx_trend_threshold: ADX > этого значения = тренд
            adx_flat_threshold: ADX < этого значения = флэт
            trend_weights: Веса для тренда (rf_weight, xgb_weight, lgb_weight[, lstm_weight])
            flat_weights: Веса для флэта (rf_weight, xgb_weight, lgb_weight[, lstm_weight])
            use_adaptive_confidence_by_atr: Порог уверенности по формуле от ATR (выше волатильность — ниже порог)
            adaptive_confidence_k: Коэффициент k в формуле (1 + k * (atr_median - atr_current) / atr_median)
            adaptive_confidence_min: Минимальный множитель порога (относительно base)
            adaptive_confidence_max: Максимальный множитель порога (относительно base)
            adaptive_confidence_atr_lookback: Число баров для расчёта медианы ATR
            use_fixed_sl_from_risk: True = SL только из риска (stop_loss_pct / max_loss/leverage), False = от модели/ATR
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_signal_strength = min_signal_strength
        self.stability_filter = stability_filter
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_dynamic_ensemble_weights = use_dynamic_ensemble_weights
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_flat_threshold = adx_flat_threshold
        self.trend_weights = trend_weights or {}
        self.flat_weights = flat_weights or {}
        self.use_adaptive_confidence_by_atr = use_adaptive_confidence_by_atr
        self.adaptive_confidence_k = adaptive_confidence_k
        self.adaptive_confidence_min = adaptive_confidence_min
        self.adaptive_confidence_max = adaptive_confidence_max
        self.adaptive_confidence_atr_lookback = adaptive_confidence_atr_lookback
        self.use_fixed_sl_from_risk = use_fixed_sl_from_risk

        # Определяем минимальный порог уверенности на основе силы сигнала
        strength_thresholds = {
            "слабое": 0.0,
            "умеренное": 0.6,
            "среднее": 0.7,
            "сильное": 0.8,
            "очень_сильное": 0.9
        }
        self.min_strength_threshold = strength_thresholds.get(min_signal_strength, 0.6)
        
        # История уверенности для адаптивных порогов
        self.confidence_history = []
        self.max_history_size = 100
        
        # История последних сигналов для предотвращения противоречивых сигналов
        # Хранит последние N сигналов: [(timestamp, action, confidence), ...]
        self.signal_history = []
        self.max_signal_history = 20  # Храним последние 20 сигналов
        self.min_bars_between_opposite_signals = 4  # Минимум баров между противоположными сигналами
        self.min_confidence_difference = 0.15  # Минимальная разница уверенности между LONG и SHORT (15%)
        
        # Отслеживание сигналов в день для ограничения количества
        # Хранит количество сигналов по датам: {date_str: count}
        self.daily_signals_count = {}
        self.min_signals_per_day = min_signals_per_day
        self.max_signals_per_day = max_signals_per_day
        
        # Загружаем модель
        self.model_data = self._load_model()
        if "model" not in self.model_data:
            raise KeyError(f"Model data is missing 'model' key. Available keys: {list(self.model_data.keys())}")
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_names = self.model_data["feature_names"]
        self.is_ensemble = self.model_data.get("metadata", {}).get("model_type", "").startswith("ensemble")

        # Загружаем оптимальный порог и мета-фильтр
        self.optimal_threshold = self.model_data.get("optimal_threshold")
        if self.optimal_threshold:
            logger.info(f"[ml_strategy] Using optimal_threshold from model: {self.optimal_threshold:.4f}")
            self.confidence_threshold = self.optimal_threshold

        self.meta_model = self.model_data.get("meta_model")
        if self.meta_model:
            logger.info(f"[ml_strategy] Meta-filter (Signal Filter) is ACTIVE")

        # Если это QuadEnsemble, восстанавливаем feature_names в lstm_trainer
        if hasattr(self.model, 'lstm_trainer') and self.model.lstm_trainer is not None:
            # Если feature_names не установлены в lstm_trainer, пытаемся восстановить
            if not hasattr(self.model.lstm_trainer, 'feature_names') or self.model.lstm_trainer.feature_names is None:
                # Пытаемся определить из scaler (количество фичей)
                if hasattr(self.model.lstm_trainer, 'scaler') and self.model.lstm_trainer.scaler is not None:
                    expected_features = self.model.lstm_trainer.scaler.n_features_in_ if hasattr(self.model.lstm_trainer.scaler, 'n_features_in_') else None
                    if expected_features and self.feature_names:
                        # Используем первые expected_features фичей (как при обучении LSTM)
                        # LSTM обычно использует первые N фичей (например, 50)
                        self.model.lstm_trainer.feature_names = self.feature_names[:expected_features]
                        if not hasattr(self, '_lstm_feature_names_restored'):
                            logger.debug(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features")
                            self._lstm_feature_names_restored = True
                    elif self.feature_names:
                        # Если не можем определить из scaler, используем все feature_names
                        self.model.lstm_trainer.feature_names = self.feature_names
                        if not hasattr(self, '_lstm_feature_names_restored'):
                            logger.debug(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features (from all features)")
                            self._lstm_feature_names_restored = True
                elif self.feature_names:
                    # Если scaler недоступен, используем все feature_names
                    self.model.lstm_trainer.feature_names = self.feature_names
                    if not hasattr(self, '_lstm_feature_names_restored'):
                        logger.debug(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features (scaler unavailable)")
                        self._lstm_feature_names_restored = True
        
        # Инициализируем feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Извлекаем символ из пути к модели для логирования
        model_filename = Path(model_path).name
        symbol_from_model = "UNKNOWN"
        if "_" in model_filename:
            parts = model_filename.replace(".pkl", "").split("_")
            # Форматы:
            # - rf_ETHUSDT_15_15m.pkl -> ["rf","ETHUSDT","15","15m"]
            # - ensemble_BTCUSDT_15_mtf.pkl -> ["ensemble","BTCUSDT","15","mtf"]
            # - triple_ensemble_BTCUSDT_15_15m.pkl -> ["triple","ensemble","BTCUSDT","15","15m"]
            # - quad_ensemble_BTCUSDT_15_mtf.pkl -> ["quad","ensemble","BTCUSDT","15","mtf"]
            if len(parts) >= 3 and parts[0] in ("triple", "quad") and parts[1] == "ensemble":
                symbol_from_model = parts[2]
            elif len(parts) >= 2:
                symbol_from_model = parts[1]
        
        # Получаем метаданные модели
        model_metadata = self.model_data.get("metadata", {})
        model_type_str = model_metadata.get("model_type", "unknown")
        if "ensemble" in model_type_str.lower():
            self.is_ensemble = True
        
        # Компактный лог загрузки модели (только при первой загрузке)
        if not hasattr(self, '_model_loaded_logged'):
            model_type = '🎯 ENSEMBLE' if self.is_ensemble else 'Single'
            cv_acc = self.model_data.get("metrics", {}).get('cv_mean', 0) if self.is_ensemble else 0
            logger.info(f"[ml] {symbol_from_model}: {model_type} (CV:{cv_acc:.3f}, conf:{confidence_threshold}, stab:{stability_filter})")
            self._model_loaded_logged = True
    
    def _load_model(self) -> Dict[str, Any]:
        """Загружает модель из файла."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Проверяем, что загруженные данные являются словарем
            if not isinstance(model_data, dict):
                raise TypeError(f"Expected dict from model file, got {type(model_data)}")
            
            # Проверяем наличие необходимых ключей
            required_keys = ["model", "scaler", "feature_names"]
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                raise KeyError(f"Missing required keys in model data: {missing_keys}. Available keys: {list(model_data.keys())}")
            
            return model_data
        except Exception as e:
            raise Exception(f"Failed to load model from {self.model_path}: {str(e)}") from e
    
    def prepare_features(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> np.ndarray:
        """
        Подготавливает фичи из DataFrame для предсказания модели.
        
        Args:
            df: DataFrame с OHLCV данными и индикаторами (может уже содержать фичи)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            Массив фичей для модели
        """
        # Если фичи уже созданы (skip_feature_creation=True), используем их напрямую
        if skip_feature_creation:
            df_with_features = df.copy()
        else:
            # Создаем фичи заново (для обратной совместимости)
            # Проверяем, есть ли timestamp как колонка (нужно для feature_engineer)
            df_work = df.copy()
            if "timestamp" in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                df_work = df_work.set_index("timestamp")
            elif "timestamp" not in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                # Если нет timestamp, создаем его из индекса
                if isinstance(df_work.index, pd.DatetimeIndex):
                    pass  # Уже DatetimeIndex
                else:
                    # Пытаемся создать временной индекс
                    df_work.index = pd.to_datetime(df_work.index, errors='coerce')
            
            # Создаем все необходимые фичи через FeatureEngineer
            logger.debug(f"[ml_strategy] Preparing features: input DataFrame has {len(df_work)} rows")
            try:
                df_with_features = self.feature_engineer.create_technical_indicators(df_work)
                logger.debug(f"[ml_strategy] After create_technical_indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
            except TypeError as e:
                if "'>' not supported" in str(e) or "NoneType" in str(e):
                    logger.error(f"[ml_strategy] ❌ ERROR: Comparison with None detected in create_technical_indicators")
                    logger.error(f"[ml_strategy]   Error: {e}")
                    logger.error(f"[ml_strategy]   Checking for None values in DataFrame...")
                    # Проверяем наличие None в ключевых колонках
                    for col in ["open", "high", "low", "close", "volume", "atr", "atr_pct", "rsi"]:
                        if col in df_work.columns:
                            none_count = df_work[col].isna().sum() + (df_work[col] == None).sum()
                            if none_count > 0:
                                logger.error(f"[ml_strategy]   Column '{col}' has {none_count} None/NaN values")
                    raise
                raise
        
        # Проверяем, что есть хотя бы основные данные (OHLCV)
        key_columns = ["open", "high", "low", "close", "volume"]
        if all(col in df_with_features.columns for col in key_columns):
            # Сохраняем только строки, где хотя бы основные колонки присутствуют
            rows_before = len(df_with_features)
            df_with_features = df_with_features[df_with_features[key_columns].notna().any(axis=1)]
            rows_after = len(df_with_features)
            # Логируем только если количество строк изменилось И это не skip_feature_creation (чтобы не засорять логи)
            if not skip_feature_creation and rows_before != rows_after:
                logger.debug(f"[ml_strategy] After filtering key columns: {rows_before} -> {rows_after} rows")
        else:
            # Логируем предупреждение только если это не skip_feature_creation
            if not skip_feature_creation:
                missing_key_cols = [col for col in key_columns if col not in df_with_features.columns]
                logger.warning(f"[ml_strategy] ⚠️ WARNING: Missing key columns: {missing_key_cols}")
        
        # Проверяем, что есть данные после фильтрации основных колонок
        if len(df_with_features) == 0:
            logger.error(f"[ml_strategy] ❌ ERROR: No rows after filtering key columns")
            logger.error(f"[ml_strategy]   Input DataFrame shape: {df_work.shape}")
            logger.error(f"[ml_strategy]   After create_technical_indicators shape: {df_with_features.shape if 'df_with_features' in locals() else 'N/A'}")
            raise ValueError("No data available after creating features (all rows contain NaN in key columns)")
        
        # ВАЖНО: Заполняем NaN в фичах нулями ПЕРЕД любыми другими операциями
        # Это позволяет сохранить все строки, даже если некоторые индикаторы не вычислились
        # Сначала заполняем NaN в индикаторах (но не в основных колонках)
        feature_columns = [col for col in df_with_features.columns if col not in key_columns]
        if feature_columns:
            df_with_features[feature_columns] = df_with_features[feature_columns].fillna(0)
        
        # Удаляем только строки, где ВСЕ значения (включая основные колонки) NaN
        df_with_features = df_with_features.dropna(how='all')
        
        # Финальная проверка
        if len(df_with_features) == 0:
            raise ValueError("No data available after creating features (all rows contain NaN)")
        
        # Проверяем наличие всех необходимых фичей
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            # Выводим только один раз при первом обнаружении
            if not hasattr(self, "_missing_features_warned"):
                logger.warning(
                    f"[ml_strategy] ⚠️ WARNING: Missing {len(missing_features)} features: "
                    f"{missing_features[:10]}..."
                )
                logger.warning(
                    f"[ml_strategy]   Expected {len(self.feature_names)} features, "
                    f"got {len(df_with_features.columns)}"
                )
                self._missing_features_warned = True
            
            # Заполняем отсутствующие фичи нулями одним батчем, чтобы избежать фрагментации DataFrame
            zeros_df = pd.DataFrame(
                0.0,
                index=df_with_features.index,
                columns=missing_features,
            )
            df_with_features = pd.concat([df_with_features, zeros_df], axis=1)
        
        # Проверяем лишние фичи (которые есть в данных, но не ожидаются моделью)
        extra_features = [f for f in df_with_features.columns if f not in self.feature_names and f not in key_columns]
        # Убираем логи о лишних фичах - это нормальная ситуация (они просто игнорируются)
        if extra_features:
            self._extra_features_warned = True  # Устанавливаем флаг, но не логируем
        
        # Выбираем только нужные фичи в правильном порядке
        X = df_with_features[self.feature_names].values
        
        # Проверяем, что есть данные для нормализации
        if len(X) == 0:
            raise ValueError("No data available after feature selection")
        
        # Проверяем соответствие количества фичей с моделью
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature count mismatch: X has {X.shape[1]} features, but model expects {len(self.feature_names)}")
        
        # Нормализуем
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            if "features" in str(e).lower() or "n_features" in str(e).lower():
                # Пробуем исправить несоответствие количества фичей
                scaler_expected = getattr(self.scaler, 'n_features_in_', None)
                if scaler_expected is None:
                    # Старая версия sklearn - пробуем получить из shape
                    try:
                        scaler_expected = self.scaler.mean_.shape[0] if hasattr(self.scaler, 'mean_') else None
                    except:
                        pass
                
                if scaler_expected and X.shape[1] != scaler_expected:
                    # Автоматически исправляем несоответствие без логирования (это нормальная ситуация)
                    if not hasattr(self, '_feature_mismatch_warned'):
                        self._feature_mismatch_warned = True
                    
                    # Если scaler ожидает больше фичей, добавляем недостающие нулями
                    if X.shape[1] < scaler_expected:
                        missing_count = scaler_expected - X.shape[1]
                        if not hasattr(self, '_feature_adjustment_logged'):
                            self._feature_adjustment_logged = True
                        # Добавляем нулевые колонки
                        zeros = np.zeros((X.shape[0], missing_count))
                        X = np.hstack([X, zeros])
                    # Если scaler ожидает меньше фичей, обрезаем
                    elif X.shape[1] > scaler_expected:
                        X = X[:, :scaler_expected]
                
                # Пробуем снова после исправления
                try:
                    X_scaled = self.scaler.transform(X)
                except ValueError as e2:
                    logger.error(f"[ml_strategy] ❌ ERROR: Still cannot transform after adjustment")
                    logger.error(f"[ml_strategy]   Scaler expects: {scaler_expected} features")
                    logger.error(f"[ml_strategy]   X has: {X.shape[1]} features")
                    raise ValueError(f"Feature count mismatch: Scaler expects {scaler_expected} features, but got {X.shape[1]}. "
                                   f"Please retrain the model with the current feature set.") from e2
            else:
                raise
        
        return X_scaled
    
    def prepare_features_with_df(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Подготавливает фичи из DataFrame и возвращает как массив, так и DataFrame с фичами.
        
        Args:
            df: DataFrame с OHLCV данными и индикаторы (может уже содержать фичи)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            (X_scaled, df_with_features) где:
            - X_scaled: Нормализованный массив фичей для модели
            - df_with_features: DataFrame со всеми фичами (для передачи в QuadEnsemble)
        """
        # Если фичи уже созданы (skip_feature_creation=True), используем их напрямую
        if skip_feature_creation:
            df_with_features = df.copy()
        else:
            # Создаем фичи заново (для обратной совместимости)
            # Проверяем, есть ли timestamp как колонка (нужно для feature_engineer)
            df_work = df.copy()
            if "timestamp" in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                df_work = df_work.set_index("timestamp")
            elif "timestamp" not in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                # Если нет timestamp, создаем его из индекса
                if isinstance(df_work.index, pd.DatetimeIndex):
                    pass  # Уже DatetimeIndex
                else:
                    # Пытаемся создать временной индекс
                    df_work.index = pd.to_datetime(df_work.index, errors='coerce')
            
            # Создаем все необходимые фичи через FeatureEngineer
            if not skip_feature_creation:
                logger.debug(f"[ml_strategy] Preparing features: input DataFrame has {len(df_work)} rows")
            try:
                df_with_features = self.feature_engineer.create_technical_indicators(df_work)
                if not skip_feature_creation:
                    logger.debug(f"[ml_strategy] After create_technical_indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
            except Exception as e:
                logger.error(f"[ml_strategy] ❌ ERROR in create_technical_indicators: {e}")
                raise

        # Проверяем, что есть хотя бы основные данные (OHLCV)
        key_columns = ["open", "high", "low", "close", "volume"]
        if all(col in df_with_features.columns for col in key_columns):
            rows_before = len(df_with_features)
            df_with_features = df_with_features[df_with_features[key_columns].notna().any(axis=1)]
            rows_after = len(df_with_features)
        else:
            missing_key_cols = [col for col in key_columns if col not in df_with_features.columns]
            raise ValueError(f"Missing key columns: {missing_key_cols}")
        
        if len(df_with_features) == 0:
            raise ValueError("No data available after filtering key columns")
        
        # Заполняем NaN в фичах
        feature_columns = [col for col in df_with_features.columns if col not in key_columns]
        if feature_columns:
            df_with_features[feature_columns] = df_with_features[feature_columns].ffill().bfill().fillna(0.0)
        
        # Проверяем наличие всех необходимых фичей
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            # Заполняем отсутствующие фичи нулями
            zeros_df = pd.DataFrame(
                0.0,
                index=df_with_features.index,
                columns=missing_features,
            )
            df_with_features = pd.concat([df_with_features, zeros_df], axis=1)
        
        # Выбираем только нужные фичи в правильном порядке
        X = df_with_features[self.feature_names].values
        
        if len(X) == 0:
            raise ValueError("No data available after feature selection")
        
        # Нормализуем
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            if "features" in str(e).lower() or "n_features" in str(e).lower():
                scaler_expected = getattr(self.scaler, 'n_features_in_', None)
                if scaler_expected is None:
                    try:
                        scaler_expected = self.scaler.mean_.shape[0] if hasattr(self.scaler, 'mean_') else None
                    except:
                        pass
                
                if scaler_expected and X.shape[1] != scaler_expected:
                    if X.shape[1] < scaler_expected:
                        missing_count = scaler_expected - X.shape[1]
                        zeros = np.zeros((X.shape[0], missing_count))
                        X = np.hstack([X, zeros])
                    elif X.shape[1] > scaler_expected:
                        X = X[:, :scaler_expected]
                    
                    X_scaled = self.scaler.transform(X)
                else:
                    raise
            else:
                raise
        
        return X_scaled, df_with_features
    
    def predict(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> tuple[int, float]:
        """
        Делает предсказание на основе последнего бара.
        
        Args:
            df: DataFrame с данными (OHLCV, фичи будут созданы автоматически или уже присутствуют)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            (prediction, confidence) где:
            - prediction: 1 (LONG), -1 (SHORT), 0 (HOLD)
            - confidence: уверенность модели (0-1)
        """
        # Берем последний бар
        if len(df) == 0:
            return 0, 0.0
        
        try:
            # Логируем начало predict для отладки
            if hasattr(self, '_generate_signal_call_count') and self._generate_signal_call_count <= 3:
                logger.debug(f"[ml_strategy] predict() начат, df.shape={df.shape}, skip_feature_creation={skip_feature_creation}")
                import time
                prepare_start = time.time()
            
            # Подготавливаем фичи (создаст все необходимые индикаторы или использует уже созданные)
            # Нужно получить и X (массив фичей) и df_with_features (DataFrame с фичами) для QuadEnsemble
            X, df_with_features = self.prepare_features_with_df(df, skip_feature_creation=skip_feature_creation)
            
            if hasattr(self, '_generate_signal_call_count') and self._generate_signal_call_count <= 3:
                prepare_elapsed = time.time() - prepare_start
                logger.debug(f"[ml_strategy] prepare_features_with_df() завершен за {prepare_elapsed:.2f} сек, X.shape={X.shape}")
            
            # Берем последний образец
            X_last = X[-1:].reshape(1, -1)
            
            if hasattr(self, '_generate_signal_call_count') and self._generate_signal_call_count <= 3:
                logger.debug(f"[ml_strategy] X_last подготовлен: shape={X_last.shape}")
        except Exception as e:
            logger.error(f"[ml_strategy] Error preparing features: {e}")
            return 0, 0.0
        
        # Динамические веса ансамбля по режиму рынка (тренд/флэт по ADX)
        weights_override = None
        regime = None
        if self.is_ensemble:
            try:
                row = df_with_features.iloc[-1]
                adx = row.get("adx", np.nan)
                atr_pct = row.get("atr_pct", np.nan)

                # Определение режима для QuadEnsemble (авто-веса)
                if np.isfinite(adx):
                    if adx > self.adx_trend_threshold:
                        regime = "trend"
                    elif adx < self.adx_flat_threshold:
                        regime = "sideways"

                # ATR-фильтр волатильности (дополнительный режим)
                if np.isfinite(atr_pct) and atr_pct > 1.5:
                    regime = "high_vol"

                # Явное переопределение весов из конфига (имеет приоритет если включено)
                if self.use_dynamic_ensemble_weights and (self.trend_weights or self.flat_weights):
                    if regime == "trend" and self.trend_weights:
                        weights_override = self.trend_weights
                    elif regime == "sideways" and self.flat_weights:
                        weights_override = self.flat_weights
            except Exception as e:
                logger.debug(f"[ml_strategy] regime detection error: {e}")
        
        # Предсказание
        if hasattr(self.model, "predict_proba"):
            try:
                # Настройка параметров для ансамблей
                kw = {}
                if self.is_ensemble:
                    if weights_override is not None:
                        kw["weights_override"] = weights_override
                    if regime is not None and hasattr(self.model, 'lstm_trainer'):
                        kw["regime"] = regime

                # Для классификаторов с вероятностями (включая ансамбль)
                # Проверяем, является ли это QuadEnsemble (требует историю для LSTM)
                if hasattr(self.model, 'lstm_trainer') and hasattr(self.model, 'sequence_length'):
                    # QuadEnsemble: передаем историю данных для LSTM
                    proba = self.model.predict_proba(X_last, df_history=df_with_features, **kw)[0]
                else:
                    # Обычные модели и ансамбли (TripleEnsemble, WeightedEnsemble)
                    proba = self.model.predict_proba(X_last, **kw)[0]
            except Exception as e:
                logger.error(f"[ml_strategy] Ошибка при вызове predict_proba: {e}")
                logger.error(f"[ml_strategy] Тип модели: {type(self.model)}")
                logger.error(f"[ml_strategy] X_last shape: {X_last.shape if hasattr(X_last, 'shape') else 'N/A'}")
                import traceback
                logger.error(f"[ml_strategy] Traceback:\n{traceback.format_exc()}")
                # Возвращаем равномерное распределение при ошибке
                proba = np.array([0.33, 0.34, 0.33])  # SHORT, HOLD, LONG
                logger.warning(f"[ml_strategy] Используется равномерное распределение из-за ошибки")
            
            # Проверяем proba на NaN
            if np.any(np.isnan(proba)) or not np.all(np.isfinite(proba)):
                # Если proba содержит NaN, используем равномерное распределение
                proba = np.array([0.33, 0.34, 0.33])  # SHORT, HOLD, LONG
                logger.warning(f"[ml_strategy] Warning: proba contains NaN, using uniform distribution")
            
            # Для ансамбля proba уже в правильном формате [-1, 0, 1]
            if self.is_ensemble:
                # Ансамбль уже возвращает вероятности в формате [-1, 0, 1]
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
                # Проверяем на NaN
                if np.isnan(long_prob) or not np.isfinite(long_prob):
                    long_prob = 0.0
                if np.isnan(short_prob) or not np.isfinite(short_prob):
                    short_prob = 0.0
                if np.isnan(hold_prob) or not np.isfinite(hold_prob):
                    hold_prob = 0.0
                
                # ЛОГИКА ДЛЯ АНСАМБЛЕЙ
                # Используем confidence_threshold из настроек как минимальный порог
                # Это гарантирует единообразную обработку для всех типов моделей
                ensemble_min = self.confidence_threshold
                if ensemble_min is None:
                    ensemble_min = 0.35  # Значение по умолчанию
                
                # Вычисляем разницу между LONG и SHORT
                prob_diff = abs(long_prob - short_prob)
                
                # Определяем предсказание
                # Для ансамблей требуем:
                # 1. Вероятность >= confidence_threshold из настроек
                # 2. Минимальная разница между LONG и SHORT (15%)
                # Логируем для диагностики (первые несколько раз)
                if hasattr(self, '_predict_debug_count'):
                    self._predict_debug_count += 1
                else:
                    self._predict_debug_count = 1
                
                if self._predict_debug_count <= 5:
                    logger.debug(f"[ml_strategy] predict: long_prob={long_prob:.3f}, short_prob={short_prob:.3f}, hold_prob={hold_prob:.3f}, prob_diff={prob_diff:.3f}, ensemble_min={ensemble_min}, min_diff={self.min_confidence_difference}")
                
                if ensemble_min is not None and long_prob >= ensemble_min and long_prob > short_prob and prob_diff >= self.min_confidence_difference:
                    prediction = 1  # LONG
                    # Confidence = базовая вероятность (без искусственного увеличения)
                    confidence = long_prob
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = long_prob
                    if self._predict_debug_count <= 5:
                        logger.debug(f"[ml_strategy] predict: LONG selected (conf={confidence:.3f})")
                elif ensemble_min is not None and short_prob >= ensemble_min and short_prob > long_prob and prob_diff >= self.min_confidence_difference:
                    prediction = -1  # SHORT
                    # Confidence = базовая вероятность (без искусственного увеличения)
                    confidence = short_prob
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = short_prob
                    if self._predict_debug_count <= 5:
                        logger.debug(f"[ml_strategy] predict: SHORT selected (conf={confidence:.3f})")
                else:
                    prediction = 0
                    confidence = hold_prob
                    if self._predict_debug_count <= 5:
                        reason = []
                        if long_prob < ensemble_min:
                            reason.append(f"long_prob {long_prob:.3f} < threshold {ensemble_min}")
                        if short_prob < ensemble_min:
                            reason.append(f"short_prob {short_prob:.3f} < threshold {ensemble_min}")
                        if prob_diff < self.min_confidence_difference:
                            reason.append(f"prob_diff {prob_diff:.3f} < min_diff {self.min_confidence_difference}")
                        logger.info(f"[ml_strategy] predict: HOLD selected (conf={confidence:.3f}), reason: {', '.join(reason) if reason else 'no clear direction'}")
                        logger.info(f"[ml_strategy] predict: Probabilities - LONG: {long_prob:.3f}, SHORT: {short_prob:.3f}, HOLD: {hold_prob:.3f}, diff: {prob_diff:.3f}")
                
                # Fallback
                if prediction == 0:
                    prediction_idx = np.argmax(proba)
                    prediction = prediction_idx - 1
                    confidence = proba[prediction_idx]
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = hold_prob if np.isfinite(hold_prob) else 0.0
                    if self._predict_debug_count <= 5:
                        logger.info(f"[ml_strategy] predict: Fallback to argmax: pred={prediction}, conf={confidence:.3f}, max_prob_idx={prediction_idx}")
                
                # Обновляем историю уверенности
                if len(self.confidence_history) >= self.max_history_size:
                    self.confidence_history.pop(0)
                self.confidence_history.append(confidence)
            elif len(proba) == 3:
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1
                confidence = proba[prediction_idx]
                
                if np.isnan(confidence) or not np.isfinite(confidence):
                    confidence = 0.0
                
                # УЛУЧШЕНИЕ: Если модель предсказывает HOLD, но вероятность LONG или SHORT достаточно высока
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
                if np.isnan(long_prob) or not np.isfinite(long_prob):
                    long_prob = 0.0
                if np.isnan(short_prob) or not np.isfinite(short_prob):
                    short_prob = 0.0
                if np.isnan(hold_prob) or not np.isfinite(hold_prob):
                    hold_prob = 0.0
                
                # Динамический порог
                if self.use_dynamic_threshold and len(self.confidence_history) > 10:
                    recent_confidence_median = np.median(self.confidence_history[-20:])
                    adaptive_threshold = max(self.min_strength_threshold, recent_confidence_median * 0.9)
                else:
                    adaptive_threshold = self.min_strength_threshold
                
                if prediction == 0:
                    if long_prob >= adaptive_threshold and long_prob > short_prob:
                        prediction = 1
                        confidence = long_prob
                    elif short_prob >= adaptive_threshold and short_prob > long_prob:
                        prediction = -1
                        confidence = short_prob
                
                # Обновляем историю уверенности
                if len(self.confidence_history) >= self.max_history_size:
                    self.confidence_history.pop(0)
                self.confidence_history.append(confidence)
            else:
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1 if len(proba) == 3 else prediction_idx
                confidence = proba[prediction_idx]
                
                if np.isnan(prediction) or not np.isfinite(prediction):
                    prediction = 0
                if np.isnan(confidence) or not np.isfinite(confidence):
                    confidence = 0.0
        else:
            # Для моделей без predict_proba
            prediction_raw = self.model.predict(X_last)[0]
            if np.isnan(prediction_raw) or not np.isfinite(prediction_raw):
                prediction = 0
            else:
                if hasattr(self.model, 'classes_'):
                    classes = self.model.classes_
                    if len(classes) == 3:
                        prediction = int(prediction_raw) - 1
                    else:
                        prediction = int(prediction_raw)
                else:
                    prediction = int(prediction_raw)
            confidence = 1.0
        
        # Проверяем на NaN перед возвратом
        if np.isnan(prediction) or not np.isfinite(prediction):
            prediction = 0
        if np.isnan(confidence) or not np.isfinite(confidence):
            confidence = 0.0
        
        return int(prediction), float(confidence)
    
    def generate_signal(
        self,
        row: pd.Series,
        df: pd.DataFrame,
        has_position: Optional[Bias],
        current_price: float,
        leverage: int = 10,
        target_profit_pct_margin: float = 25.0,
        max_loss_pct_margin: float = 10.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        skip_feature_creation: bool = False,
        precalculated_pred: Optional[int] = None,
        precalculated_conf: Optional[float] = None,
    ) -> Signal:
        """
        Генерирует торговый сигнал на основе ML-предсказания.
        
        Args:
            ...
            precalculated_pred: Уже вычисленное предсказание (для ускорения batch-обработки)
            precalculated_conf: Уже вычисленная уверенность
        """
        try:
            # Логируем начало generate_signal для отладки (только первые несколько раз)
            if not hasattr(self, '_generate_signal_call_count'):
                self._generate_signal_call_count = 0
            self._generate_signal_call_count += 1

            # Если предсказания не переданы, вычисляем их
            if precalculated_pred is not None and precalculated_conf is not None:
                prediction, confidence = precalculated_pred, precalculated_conf
            else:
                prediction, confidence = self.predict(df, skip_feature_creation=skip_feature_creation)
            
            if self._generate_signal_call_count <= 3:
                predict_elapsed = time.time() - predict_start
                logger.debug(f"[ml_strategy] predict() завершен за {predict_elapsed:.2f} сек, prediction={prediction}, confidence={confidence:.4f}")
            
            # === ДИНАМИЧЕСКИЙ CONFIDENCE THRESHOLD ===
            # Адаптируем порог на основе рыночных условий
            effective_threshold = self.confidence_threshold
            
            if prediction != 0:
                # Режим: адаптивный порог по ATR (формула из роадмэпа)
                # Высокая волатильность -> ниже порог (больше сигналов), низкая -> выше порог (меньше шума)
                if self.use_adaptive_confidence_by_atr:
                    atr_pct = row.get("atr_pct", np.nan)
                    if np.isfinite(atr_pct) and "atr_pct" in df.columns and len(df) >= 2:
                        lookback = min(self.adaptive_confidence_atr_lookback, len(df) - 1)
                        atr_series = df["atr_pct"].dropna()
                        if len(atr_series) >= max(10, lookback // 2):
                            atr_median = float(atr_series.iloc[-lookback:].median())
                            if atr_median > 0:
                                multiplier = 1.0 + self.adaptive_confidence_k * (atr_median - atr_pct) / atr_median
                                multiplier = max(self.adaptive_confidence_min, min(self.adaptive_confidence_max, multiplier))
                                effective_threshold = float(np.clip(self.confidence_threshold * multiplier, 0.01, 0.99))
                elif self.use_dynamic_threshold:
                    # Существующая логика: порог по уровням ATR/ADX
                    atr_pct = row.get("atr_pct", np.nan)
                    adx = row.get("adx", np.nan)
                    if np.isfinite(atr_pct):
                        if atr_pct > 1.5:
                            effective_threshold = self.confidence_threshold * 1.2
                        elif atr_pct < 0.5:
                            effective_threshold = self.confidence_threshold * 0.9
                    if np.isfinite(adx):
                        if adx < 20:
                            effective_threshold = max(effective_threshold, self.confidence_threshold * 1.15)
                        elif adx > 25:
                            effective_threshold = min(effective_threshold, self.confidence_threshold * 0.95)
                    min_allowed = self.confidence_threshold * 0.8
                    max_allowed = min(0.95, self.confidence_threshold * 1.5)
                    effective_threshold = max(min_allowed, min(max_allowed, effective_threshold))
            
            # Применяем динамический порог
            if prediction != 0 and confidence < effective_threshold:
                # Сигнал отклонен из-за низкой уверенности (с учетом динамического порога)
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.HOLD,
                    reason=f"ml_низкая_уверенность_{int(confidence*100)}%_порог_{int(effective_threshold*100)}%",
                    price=current_price,
                    indicators_info={
                        "strategy": "ML",
                        "prediction": "HOLD",
                        "confidence": round(confidence, 4),
                        "threshold": round(effective_threshold, 4),
                        "rejected_reason": "dynamic_threshold"
                    }
                )

            # === META-LABELING FILTER (Signal Filter) ===
            if prediction != 0 and self.meta_model is not None:
                try:
                    # Извлекаем фичи для текущей строки безопасно
                    # КРИТИЧНО: Фикс KeyError если модель требует фичи, которых нет в текущем ряду (например, старые модели или ob_imbalance)
                    X_meta_list = []
                    for f in self.feature_names:
                        if f in row.index:
                            X_meta_list.append(row[f])
                        else:
                            X_meta_list.append(0.0)

                    X_meta = np.array(X_meta_list).reshape(1, -1)
                    X_meta_scaled = self.scaler.transform(X_meta)

                    # Предсказываем вероятность успеха (class 1)
                    meta_prob = self.meta_model.predict_proba(X_meta_scaled)[0, 1]

                    # Порог для мета-фильтра (обычно 0.5, но можно сделать настраиваемым)
                    meta_threshold = 0.5

                    if meta_prob < meta_threshold:
                        return Signal(
                            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                            action=Action.HOLD,
                            reason=f"meta_filter_rejected_{int(meta_prob*100)}%",
                            price=current_price,
                            indicators_info={
                                "strategy": "ML",
                                "prediction": "HOLD",
                                "confidence": round(confidence, 4),
                                "meta_prob": round(meta_prob, 4),
                                "rejected_reason": "meta_filter"
                            }
                        )
                    else:
                        logger.info(f"✅ Signal PASSED meta-filter (prob: {meta_prob:.2f})")
                except Exception as e:
                    logger.error(f"[ml_strategy] Error in meta-filter: {e}")

            # === РАСЧЕТ SL ОТ УРОВНЕЙ + TP ПО RR 2-3:1 ===
            sl_price = None
            tp_price = None
            sl_source = None
            sl_level = None

            def _is_finite_number(value: Any) -> bool:
                try:
                    return value is not None and np.isfinite(float(value))
                except Exception:
                    return False

            def _collect_level_candidates(side: str) -> list[tuple[str, float]]:
                candidates: list[tuple[str, float]] = []
                if df is None or len(df) == 0:
                    return candidates
                lookback = min(60, len(df))
                df_tail = df.iloc[-lookback:]

                recent_low = df_tail["low"].min() if "low" in df_tail.columns else None
                recent_high = df_tail["high"].max() if "high" in df_tail.columns else None

                def add_candidate(name: str, value: Any, compare: str):
                    if not _is_finite_number(value):
                        return
                    value_f = float(value)
                    if compare == "below" and value_f < current_price:
                        candidates.append((name, value_f))
                    elif compare == "above" and value_f > current_price:
                        candidates.append((name, value_f))

                if side == "LONG":
                    add_candidate("recent_low", recent_low, "below")
                    add_candidate("bb_lower", row.get("bb_lower"), "below")
                    add_candidate("sma_20", row.get("sma_20"), "below")
                    add_candidate("ema_26", row.get("ema_26"), "below")
                    add_candidate("ema_12", row.get("ema_12"), "below")
                else:
                    add_candidate("recent_high", recent_high, "above")
                    add_candidate("bb_upper", row.get("bb_upper"), "above")
                    add_candidate("sma_20", row.get("sma_20"), "above")
                    add_candidate("ema_26", row.get("ema_26"), "above")
                    add_candidate("ema_12", row.get("ema_12"), "above")

                return candidates

            def _calculate_sl_from_levels(side: str) -> tuple[Optional[float], Optional[str], Optional[float]]:
                candidates = _collect_level_candidates(side)
                if not candidates:
                    return None, None, None
                if side == "LONG":
                    # Ближайшая поддержка (самая высокая ниже цены)
                    selected = max(candidates, key=lambda x: x[1])
                else:
                    # Ближайшее сопротивление (самое низкое выше цены)
                    selected = min(candidates, key=lambda x: x[1])

                level_name, level_price = selected

                # Буфер за уровнем (ATR или минимум 0.1%)
                atr_value = row.get("atr")
                if _is_finite_number(atr_value) and float(atr_value) > 0:
                    buffer_value = max(current_price * 0.001, float(atr_value) * 0.2)
                else:
                    buffer_value = current_price * 0.001

                if side == "LONG":
                    sl = level_price - buffer_value
                else:
                    sl = level_price + buffer_value

                if side == "LONG" and sl >= current_price:
                    return None, None, None
                if side == "SHORT" and sl <= current_price:
                    return None, None, None

                return sl, level_name, level_price

            # === РАСЧЕТ SL и TP (НОВАЯ ЛОГИКА С ATR) ===
            # Всегда используем max_loss_pct_margin / target_profit_pct_margin с учетом плеча
            # Это гарантирует нужный % от маржи независимо от плеча
            if max_loss_pct_margin:
                sl_ratio = calculate_price_pct_from_margin_pct(max_loss_pct_margin, leverage)
            else:
                sl_ratio = (10.0 / leverage) / 100.0  # Дефолт: 10% от маржи

            if target_profit_pct_margin:
                tp_ratio = calculate_price_pct_from_margin_pct(target_profit_pct_margin, leverage)
            else:
                tp_ratio = (18.0 / leverage) / 100.0  # Дефолт: 18% от маржи

            # АДАПТАЦИЯ ПО ATR (если доступен) — только когда SL не фиксирован из риска
            # Если use_fixed_sl_from_risk: всегда используем sl_ratio/tp_ratio из риска (выше)
            atr_pct = row.get("atr_pct", np.nan)
            atr_value = row.get("atr", np.nan)
            use_dynamic_atr = not self.use_fixed_sl_from_risk  # При фикс. SL из риска не переопределяем ATR
            
            if use_dynamic_atr and np.isfinite(atr_value) and atr_value > 0:
                # Используем множители из конфига (по умолчанию 1.5 ATR для SL)
                # Базовый ATR multiplier: SL = 1.5 * ATR, TP = 2.5 * ATR
                # Это более надежно, чем фиксированный %
                
                sl_dist = atr_value * 1.5  # 1.5 ATR
                tp_dist = atr_value * 2.5  # 2.5 ATR
                
                # Конвертируем в % для совместимости
                sl_ratio = sl_dist / current_price
                tp_ratio = tp_dist / current_price
                
                # Ограничиваем разумными пределами (0.5% ... 5%)
                sl_ratio = max(0.005, min(0.05, sl_ratio))
                tp_ratio = max(0.0075, min(0.08, tp_ratio))
                
                if self._generate_signal_call_count <= 3:
                    logger.debug(f"[ml_strategy] Dynamic ATR risk: atr={atr_value:.2f}, sl_ratio={sl_ratio:.4f}, tp_ratio={tp_ratio:.4f}")
            
            elif not self.use_fixed_sl_from_risk and np.isfinite(atr_pct) and atr_pct > 0:
                # Fallback к старой логике адаптации процента
                # Нормализуем ATR к среднему (примерно 0.5% для 15m крипты)
                atr_factor = atr_pct / 0.5
                atr_factor = max(0.8, min(1.5, atr_factor))  # Ограничиваем влияние (0.8x ... 1.5x)
                
                sl_ratio *= atr_factor
                tp_ratio *= atr_factor
                
                if self._generate_signal_call_count <= 3:
                    logger.debug(f"[ml_strategy] ATR adaptation: atr_pct={atr_pct:.4f}, factor={atr_factor:.2f}, new_sl={sl_ratio:.4f}, new_tp={tp_ratio:.4f}")

            if prediction == 1:
                # LONG
                sl_price = current_price * (1 - sl_ratio)
                tp_price = current_price * (1 + tp_ratio)
                sl_source = "fixed_pct_atr"
                sl_level = None
            elif prediction == -1:
                # SHORT
                sl_price = current_price * (1 + sl_ratio)
                tp_price = current_price * (1 - tp_ratio)
                sl_source = "fixed_pct_atr"
                sl_level = None
            
            # Опционально: проверяем уровни S/R для валидации (информативно)
            if prediction != 0:
                sl_from_levels, _, _ = _calculate_sl_from_levels("LONG" if prediction == 1 else "SHORT")
                # Логику "pass" оставляем, уровни пока не влияют на SL

            # RR логику (951-962) УДАЛЯЕМ, так как она переписывает TP на основе SL
            # Мы хотим фиксированный TP/SL из конфига, который доказал эффективность в анализе

            
            # УЛУЧШЕНИЕ: Валидация TP/SL (с учетом динамических ratio)
            # Проверяем корректность TP/SL для LONG
            if prediction == 1 and tp_price is not None and sl_price is not None:
                if not (sl_price < current_price and tp_price > current_price):
                    sl_price = current_price * (1 - sl_ratio)
                    tp_price = current_price * (1 + tp_ratio)
            
            # Проверяем корректность TP/SL для SHORT
            if prediction == -1 and tp_price is not None and sl_price is not None:
                if not (sl_price > current_price and tp_price < current_price):
                    sl_price = current_price * (1 + sl_ratio)
                    tp_price = current_price * (1 - tp_ratio)
            
            # УЛУЧШЕНИЕ: Финальная проверка на валидность
            # ВАЖНО: Если TP/SL невалидны, мы их пересчитываем по ratio
            if prediction != 0:  # Только для LONG/SHORT сигналов
                # КРИТИЧНО: Для LONG/SHORT ВСЕГДА должны быть валидные TP/SL
                if tp_price is None or sl_price is None or \
                   not (np.isfinite(tp_price) and np.isfinite(sl_price)) or \
                   tp_price <= 0 or sl_price <= 0:
                    # Если TP/SL не установлены или некорректны, устанавливаем их принудительно
                    if prediction == 1:  # LONG
                        sl_price = current_price * (1 - sl_ratio)
                        tp_price = current_price * (1 + tp_ratio)
                    elif prediction == -1:  # SHORT
                        sl_price = current_price * (1 + sl_ratio)
                        tp_price = current_price * (1 - tp_ratio)

            
            # Определяем силу предсказания
            if confidence >= 0.9:
                strength = "очень_сильное"
            elif confidence >= 0.8:
                strength = "сильное"
            elif confidence >= 0.7:
                strength = "среднее"
            elif confidence >= 0.6:
                strength = "умеренное"
            else:
                strength = "слабое"
            
            # Формируем причину
            confidence_pct = int(confidence * 100) if np.isfinite(confidence) else 0
            tp_pct_display = (abs(tp_price - current_price) / current_price) * 100 if tp_price else 0.0
            sl_pct_display = (abs(current_price - sl_price) / current_price) * 100 if sl_price else 0.0
            
            # Проверяем количество сигналов за сегодня
            from datetime import datetime, timezone
            # Используем дату свечи для бэктеста, иначе текущую дату
            if hasattr(row, 'name') and isinstance(row.name, pd.Timestamp):
                current_date = row.name.date()
            else:
                current_date = datetime.now(timezone.utc).date()
            date_str = current_date.isoformat()
            signals_today = self.daily_signals_count.get(date_str, 0)
            
            # Минимальная сила сигнала (УПРОЩЕНО для улучшения конверсии)
            # Используем только базовый confidence_threshold, без дополнительных ограничений
            min_strength = self.confidence_threshold  # Используем базовый порог
            
            if prediction != 0 and confidence < min_strength:
                # Собираем информацию для ML (даже для отклоненных сигналов)
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "HOLD",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "leverage": leverage,
                    "has_position": has_position.value if has_position else None,
                    "rejected_reason": f"confidence_too_low_min_{int(min_strength*100)}%"
                }
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.HOLD, 
                    reason=f"ml_сила_слишком_слабая_{strength}_{confidence_pct}%_мин_{int(min_strength*100)}%", 
                    price=current_price,
                    indicators_info=indicators_info
                )
            
            # Фильтр стабильности: если есть противоположная позиция, требуем немного больше уверенности
            # УПРОЩЕНО: минимальные требования для естественного отбора качественных сигналов
            if self.stability_filter and prediction != 0:
                if has_position == Bias.SHORT and prediction == 1:
                    # Есть SHORT, хотим открыть LONG - требуем немного больше уверенности
                    stability_threshold = max(confidence * 1.02, min_strength * 1.02)  # УМЕНЬШЕНО с 1.05 до 1.02 (минимальная защита)
                    if confidence < stability_threshold:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_стабильность_требует_{int(stability_threshold*100)}%_против_SHORT", 
                            current_price
                        )
                elif has_position == Bias.LONG and prediction == -1:
                    # Есть LONG, хотим открыть SHORT - требуем немного больше уверенности
                    stability_threshold = max(confidence * 1.02, min_strength * 1.02)  # УМЕНЬШЕНО с 1.05 до 1.02 (минимальная защита)
                    if confidence < stability_threshold:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_стабильность_требует_{int(stability_threshold*100)}%_против_LONG", 
                            current_price
                        )
            
            # Дополнительные фильтры для волатильных рынков
            is_volatile_symbol = symbol in ("ETHUSDT", "SOLUSDT")
            
            # Фильтр по RSI для экстремальных зон (ОСЛАБЛЕН: блокируем только в крайних случаях)
            rsi = row.get("rsi", np.nan)
            if prediction != 0 and np.isfinite(rsi):
                # Блокируем только в самых экстремальных зонах (>98/<2)
                if (prediction == 1 and rsi > 98) or (prediction == -1 and rsi < 2):
                    # В экстремальных зонах требуем немного больше уверенности
                    extreme_threshold = confidence * 1.02  # УМЕНЬШЕНО с 1.05 до 1.02 (минимальная защита)
                    if confidence < extreme_threshold:
                        rsi_int = int(rsi) if np.isfinite(rsi) else 0
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_экстремальный_RSI_{rsi_int}_{strength}_{confidence_pct}%", 
                            current_price
                        )
            
            # Фильтр по объему (ОСЛАБЛЕН: применяется только для очень сильных сигналов и очень низкого объема)
            # Применяется только для очень сильных сигналов и с очень мягким порогом
            if confidence > 0.90:  # УВЕЛИЧЕНО с 0.85 до 0.90 (применяется только для очень сильных сигналов)
                volume = row.get("volume", np.nan)
                vol_sma = row.get("vol_sma", np.nan)
                if not np.isfinite(vol_sma):
                    vol_sma = row.get("volume_sma_20", np.nan)
                
                if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0:
                    volume_ratio = volume / vol_sma
                    if volume_ratio < 0.15:  # УМЕНЬШЕНО с 0.2 до 0.15 (блокируем только при очень низком объеме)
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_низкий_объем_{volume_ratio:.1f}x_{strength}_{confidence_pct}%", 
                            current_price
                        )
            
            # КРИТИЧНО: ГАРАНТИРУЕМ, что TP/SL установлены и валидны перед генерацией LONG/SHORT сигналов
            # ВАЖНО: Это ДОПОЛНИТЕЛЬНАЯ проверка после всех предыдущих
            if prediction != 0:
                # ГАРАНТИРУЕМ, что TP/SL установлены и валидны
                if tp_price is None or sl_price is None or not np.isfinite(tp_price) or not np.isfinite(sl_price) or tp_price <= 0 or sl_price <= 0:
                    # Если TP/SL не установлены или невалидны, пересчитываем их принудительно
                    logger.warning(f"TP/SL invalid for prediction={prediction}, recalculating. tp={tp_price}, sl={sl_price}, price={current_price}")
                    if prediction == 1:  # LONG
                        sl_price = current_price * 0.99  # 1% ниже
                        tp_price = current_price * 1.015 # 1.5% выше
                        sl_source = sl_source or "fallback_1pct"
                    elif prediction == -1:  # SHORT
                        sl_price = current_price * 1.01  # 1% выше
                        tp_price = current_price * 0.985 # 1.5% ниже
                        sl_source = sl_source or "fallback_1pct"
                
                # Рассчитываем проценты для отображения
                tp_pct_display = (abs(tp_price - current_price) / current_price) * 100 if tp_price else 0.0
                sl_pct_display = (abs(current_price - sl_price) / current_price) * 100 if sl_price else 0.0
                
                # ФИНАЛЬНАЯ ГАРАНТИЯ: Убеждаемся, что TP/SL валидны перед продолжением
                if tp_price is None or sl_price is None:
                    logger.error(f"CRITICAL: TP/SL still None after all checks! prediction={prediction}, price={current_price}")
                    # Принудительно устанавливаем
                    if prediction == 1:  # LONG
                        sl_price = current_price * 0.99
                        tp_price = current_price * 1.025
                    elif prediction == -1:  # SHORT
                        sl_price = current_price * 1.01
                        tp_price = current_price * 0.975
            
            # Генерируем сигналы
            if prediction == 1:  # LONG
                # КРИТИЧНО: Убеждаемся, что TP/SL установлены и валидны
                if tp_price is None or sl_price is None or not np.isfinite(tp_price) or not np.isfinite(sl_price) or tp_price <= 0 or sl_price <= 0:
                    # Принудительно устанавливаем TP/SL
                    sl_price = current_price * 0.99  # 1% ниже (строго 1.0%)
                    tp_price = current_price * 1.025  # 2.5% выше (базовый TP)
                    sl_pct_display = 1.0
                    tp_pct_display = 2.5
                
                # Дополнительная проверка: убеждаемся, что SL < цена < TP для LONG
                if sl_price >= current_price or tp_price <= current_price:
                    sl_price = current_price * 0.99
                    tp_price = current_price * 1.025
                
                reason = f"ml_LONG_сила_{strength}_{confidence_pct}%_TP_{tp_pct_display:.1f}%_SL_{sl_pct_display:.1f}%"
                
                # Обновляем историю сигналов
                self.signal_history.append((row.name, Action.LONG, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                # Обновляем счетчик сигналов за день
                self.daily_signals_count[date_str] = signals_today + 1
                # Очищаем старые даты (старше 7 дней)
                from datetime import timedelta
                cutoff_date = (current_date - timedelta(days=7)).isoformat()
                self.daily_signals_count = {k: v for k, v in self.daily_signals_count.items() if k >= cutoff_date}
                
                # Рассчитываем Risk-Reward Ratio
                rr_value = 0.0
                if sl_price and tp_price and abs(current_price - sl_price) > 0:
                    rr_value = abs(tp_price - current_price) / abs(current_price - sl_price)
                
                # Собираем информацию для ML (с улучшениями из успешного бэктеста)
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "LONG",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "tp_pct": round(tp_pct_display, 2),
                    "sl_pct": round(sl_pct_display, 2),
                    "target_profit_margin_pct": target_profit_pct_margin,
                    "max_loss_margin_pct": max_loss_pct_margin,
                    "leverage": leverage,
                    "has_position": has_position.value if has_position else None,
                    "stop_loss": sl_price,   # Цена SL
                    "take_profit": tp_price,  # Цена TP
                    "sl_source": sl_source,
                    "sl_level": sl_level,
                    "risk_reward": round(rr_value, 2),
                }
                
                # УЛУЧШЕНИЕ: Добавляем ATR в indicators_info если доступен (из успешного бэктеста)
                try:
                    if 'atr' in df.columns and len(df) > 0:
                        current_atr = df['atr'].iloc[-1]
                        if pd.notna(current_atr) and current_atr > 0:
                            indicators_info['atr'] = float(current_atr)
                            indicators_info['atr_pct'] = round((current_atr / current_price) * 100, 3)
                except Exception:
                    pass
                
                # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что TP/SL валидны перед возвратом
                if sl_price is None or tp_price is None or not np.isfinite(sl_price) or not np.isfinite(tp_price) or sl_price <= 0 or tp_price <= 0:
                    logger.error(f"CRITICAL: Invalid TP/SL for LONG signal! sl_price={sl_price}, tp_price={tp_price}, price={current_price}")
                    # Принудительно устанавливаем валидные значения
                    sl_price = current_price * 0.99
                    tp_price = current_price * 1.025
                
                # Проверяем логическую корректность для LONG
                if sl_price >= current_price or tp_price <= current_price:
                    logger.warning(f"Fixing invalid TP/SL for LONG: sl={sl_price}, tp={tp_price}, price={current_price}")
                    sl_price = current_price * 0.99
                    tp_price = current_price * 1.025
                
                # Обновляем indicators_info с финальными значениями TP/SL
                indicators_info['stop_loss'] = sl_price
                indicators_info['take_profit'] = tp_price
                
                # КРИТИЧНО: ФИНАЛЬНАЯ ГАРАНТИЯ перед возвратом - TP/SL ДОЛЖНЫ быть установлены
                if sl_price is None or tp_price is None:
                    logger.error(f"CRITICAL ERROR: TP/SL is None for LONG signal before return! sl={sl_price}, tp={tp_price}, price={current_price}")
                    sl_price = current_price * 0.99
                    tp_price = current_price * 1.025
                    indicators_info['stop_loss'] = sl_price
                    indicators_info['take_profit'] = tp_price
                
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.LONG,
                    reason=reason,
                    price=current_price,
                    stop_loss=sl_price,  # ВСЕГДА установлен и валиден
                    take_profit=tp_price,  # ВСЕГДА установлен и валиден
                    indicators_info=indicators_info
                )
            
            elif prediction == -1:  # SHORT
                # КРИТИЧНО: Убеждаемся, что TP/SL установлены и валидны
                if tp_price is None or sl_price is None or not np.isfinite(tp_price) or not np.isfinite(sl_price) or tp_price <= 0 or sl_price <= 0:
                    # Принудительно устанавливаем TP/SL
                    sl_price = current_price * 1.01  # 1% выше (строго 1.0%)
                    tp_price = current_price * 0.975  # 2.5% ниже (базовый TP)
                    sl_pct_display = 1.0
                    tp_pct_display = 2.5
                
                # Дополнительная проверка: убеждаемся, что TP < цена < SL для SHORT
                if tp_price >= current_price or sl_price <= current_price:
                    sl_price = current_price * 1.01
                    tp_price = current_price * 0.975
                
                reason = f"ml_SHORT_сила_{strength}_{confidence_pct}%_TP_{tp_pct_display:.1f}%_SL_{sl_pct_display:.1f}%"
                
                # Обновляем историю сигналов
                self.signal_history.append((row.name, Action.SHORT, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                # Обновляем счетчик сигналов за день
                self.daily_signals_count[date_str] = signals_today + 1
                # Очищаем старые даты (старше 7 дней)
                from datetime import timedelta
                cutoff_date = (current_date - timedelta(days=7)).isoformat()
                self.daily_signals_count = {k: v for k, v in self.daily_signals_count.items() if k >= cutoff_date}
                
                # Рассчитываем Risk-Reward Ratio
                rr_value = 0.0
                if sl_price and tp_price and abs(current_price - sl_price) > 0:
                    rr_value = abs(tp_price - current_price) / abs(current_price - sl_price)

                # Собираем информацию для ML (с улучшениями из успешного бэктеста)
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "SHORT",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "tp_pct": round(tp_pct_display, 2),
                    "sl_pct": round(sl_pct_display, 2),
                    "target_profit_margin_pct": target_profit_pct_margin,
                    "max_loss_margin_pct": max_loss_pct_margin,
                    "leverage": leverage,
                    "has_position": has_position.value if has_position else None,
                    "stop_loss": sl_price,   # Цена SL
                    "take_profit": tp_price,  # Цена TP
                    "sl_source": sl_source,
                    "sl_level": sl_level,
                    "risk_reward": round(rr_value, 2),
                }
                
                # УЛУЧШЕНИЕ: Добавляем ATR в indicators_info если доступен (из успешного бэктеста)
                try:
                    if 'atr' in df.columns and len(df) > 0:
                        current_atr = df['atr'].iloc[-1]
                        if pd.notna(current_atr) and current_atr > 0:
                            indicators_info['atr'] = float(current_atr)
                            indicators_info['atr_pct'] = round((current_atr / current_price) * 100, 3)
                except Exception:
                    pass
                
                # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что TP/SL валидны перед возвратом
                if sl_price is None or tp_price is None or not np.isfinite(sl_price) or not np.isfinite(tp_price) or sl_price <= 0 or tp_price <= 0:
                    logger.error(f"CRITICAL: Invalid TP/SL for SHORT signal! sl_price={sl_price}, tp_price={tp_price}, price={current_price}")
                    # Принудительно устанавливаем валидные значения
                    sl_price = current_price * 1.01
                    tp_price = current_price * 0.975
                
                # Проверяем логическую корректность для SHORT
                if tp_price >= current_price or sl_price <= current_price:
                    logger.warning(f"Fixing invalid TP/SL for SHORT: sl={sl_price}, tp={tp_price}, price={current_price}")
                    sl_price = current_price * 1.01
                    tp_price = current_price * 0.975
                
                # Обновляем indicators_info с финальными значениями TP/SL
                indicators_info['stop_loss'] = sl_price
                indicators_info['take_profit'] = tp_price
                
                # КРИТИЧНО: ФИНАЛЬНАЯ ГАРАНТИЯ перед возвратом - TP/SL ДОЛЖНЫ быть установлены
                if sl_price is None or tp_price is None:
                    logger.error(f"CRITICAL ERROR: TP/SL is None for SHORT signal before return! sl={sl_price}, tp={tp_price}, price={current_price}")
                    sl_price = current_price * 1.01
                    tp_price = current_price * 0.975
                    indicators_info['stop_loss'] = sl_price
                    indicators_info['take_profit'] = tp_price
                
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.SHORT,
                    reason=reason,
                    price=current_price,
                    stop_loss=sl_price,  # ВСЕГДА установлен и валиден
                    take_profit=tp_price,  # ВСЕГДА установлен и валиден
                    indicators_info=indicators_info
                )
            
            else:  # prediction == 0 (HOLD)
                reason = f"ml_нейтрально_сила_{strength}_{confidence_pct}%_ожидание"
                
                # Обновляем историю сигналов (HOLD тоже записываем)
                self.signal_history.append((row.name, Action.HOLD, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                # Собираем информацию для ML (даже для HOLD)
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "HOLD",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "leverage": leverage,
                    "has_position": has_position.value if has_position else None,
                }
                
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.HOLD,
                    reason=reason,
                    price=current_price,
                    indicators_info=indicators_info
                )
        
        except Exception as e:
            logger.error(f"[ml_strategy] Error generating signal: {e}")
            import traceback
            traceback.print_exc()
            return Signal(
                timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                action=Action.HOLD,
                reason=f"ml_ошибка_{str(e)[:20]}",
                price=current_price
            )


def build_ml_signals(
    df: pd.DataFrame,
    model_path: str,
    confidence_threshold: float = 0.35,
    min_signal_strength: str = "слабое",
    stability_filter: bool = True,
    leverage: int = 10,
    target_profit_pct_margin: float = 25.0,
    max_loss_pct_margin: float = 10.0,
    min_signals_per_day: int = 1,
    max_signals_per_day: int = 20,
) -> list[Signal]:
    """
    Оптимизированная пакетная генерация сигналов.
    """
    strategy = MLStrategy(
        model_path, 
        confidence_threshold, 
        min_signal_strength, 
        stability_filter,
        min_signals_per_day=min_signals_per_day,
        max_signals_per_day=max_signals_per_day
    )

    # 1. Подготовка данных и фичей (пакетно)
    df_work = df.copy()
    if "timestamp" in df_work.columns:
        df_work = df_work.set_index("timestamp")

    try:
        X_scaled, df_with_features = strategy.prepare_features_with_df(df_work)
    except Exception as e:
        logger.error(f"[build_ml_signals] Feature error: {e}")
        return []

    # 2. Пакетное предсказание (Batch Prediction)
    logger.info(f"[build_ml_signals] Batch predicting {len(X_scaled)} rows...")
    
    # Обработка разных типов моделей для пакетного режима
    if hasattr(strategy.model, "predict_proba") and not strategy.is_ensemble:
        # Для одиночных моделей (XGBoost, и т.д.)
        all_probas = strategy.model.predict_proba(X_scaled)
    else:
        # Для ансамблей или QuadEnsemble пока используем цикл, но БЕЗ пересчета фичей
        # (QuadEnsemble требует историю, пакетный режим там сложнее)
        all_probas = None

    signals: list[Signal] = []
    position_bias: Optional[Bias] = None
    
    # 3. Цикл генерации объектов Signal (теперь он быстрый)
    for i, (idx, row) in enumerate(df_with_features.iterrows()):
        # Пропускаем начало, пока не набралось достаточно данных для индикаторов
        if i < 50:
            signals.append(Signal(idx, Action.HOLD, "ml_warmup", row["close"]))
            continue

        pred, conf = None, None

        # Если есть пакетные вероятности, извлекаем их
        if all_probas is not None:
            proba = all_probas[i]
            pred_idx = np.argmax(proba)
            pred = int(pred_idx - 1) if len(proba) == 3 else int(pred_idx)
            conf = float(proba[pred_idx])

        # Генерируем сигнал (передаем уже готовые pred/conf если есть)
        signal = strategy.generate_signal(
            row=row,
            df=df_with_features.iloc[max(0, i-200):i+1], # Ограниченное окно истории
            has_position=position_bias,
            current_price=row["close"],
            leverage=leverage,
            target_profit_pct_margin=target_profit_pct_margin,
            max_loss_pct_margin=max_loss_pct_margin,
            skip_feature_creation=True,
            precalculated_pred=pred,
            precalculated_conf=conf
        )

        # Обновляем состояние позиции для следующей итерации (упрощенно для бэктеста)
        if signal.action == Action.LONG: position_bias = Bias.LONG
        elif signal.action == Action.SHORT: position_bias = Bias.SHORT

        signals.append(signal)
    
    return signals
