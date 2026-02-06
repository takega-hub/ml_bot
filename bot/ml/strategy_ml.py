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
# Импортируем классы ансамбля для корректной десериализации pickle
from bot.ml.model_trainer import PreTrainedVotingEnsemble, WeightedEnsemble, TripleEnsemble

logger = logging.getLogger(__name__)


class MLStrategy:
    """
    ML-стратегия, использующая обученную модель для предсказания движения цены.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, min_signal_strength: str = "слабое", stability_filter: bool = True, use_dynamic_threshold: bool = True, min_signals_per_day: int = 1, max_signals_per_day: int = 10):
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
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_signal_strength = min_signal_strength
        self.stability_filter = stability_filter
        self.use_dynamic_threshold = use_dynamic_threshold
        
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
            except TypeError as e:
                if "'>' not supported" in str(e) or "NoneType" in str(e):
                    logger.error(f"[ml_strategy] ❌ ERROR: Comparison with None detected in create_technical_indicators")
                    logger.error(f"[ml_strategy]   Error: {e}")
                    raise
                raise

            # Добавляем MTF фичи, если включено
            import os
            ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "0")
            ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
            if ml_mtf_enabled and isinstance(df_work.index, pd.DatetimeIndex):
                try:
                    ohlcv_agg = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    df_1h = df_work.resample("60min").agg(ohlcv_agg).dropna()
                    df_4h = df_work.resample("240min").agg(ohlcv_agg).dropna()
                    higher_timeframes = {}
                    if not df_1h.empty:
                        higher_timeframes["60"] = df_1h
                    if not df_4h.empty:
                        higher_timeframes["240"] = df_4h
                    if higher_timeframes:
                        df_with_features = self.feature_engineer.add_mtf_features(
                            df_with_features,
                            higher_timeframes,
                        )
                        logger.debug(f"[ml_strategy] MTF features enabled in prepare_features_with_df. Columns: {len(df_with_features.columns)}")
                except Exception as mtf_err:
                    logger.warning(f"[ml_strategy] Warning: failed to add MTF features in prepare_features_with_df: {mtf_err}")
        
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
            # Подготавливаем фичи (создаст все необходимые индикаторы или использует уже созданные)
            # Нужно получить и X (массив фичей) и df_with_features (DataFrame с фичами) для QuadEnsemble
            X, df_with_features = self.prepare_features_with_df(df, skip_feature_creation=skip_feature_creation)
            
            # Берем последний образец
            X_last = X[-1:].reshape(1, -1)
        except Exception as e:
            logger.error(f"[ml_strategy] Error preparing features: {e}")
            return 0, 0.0
        
        # Предсказание
        if hasattr(self.model, "predict_proba"):
            # Для классификаторов с вероятностями (включая ансамбль)
            # Проверяем, является ли это QuadEnsemble (требует историю для LSTM)
            if hasattr(self.model, 'lstm_trainer') and hasattr(self.model, 'sequence_length'):
                # QuadEnsemble: передаем историю данных для LSTM
                # Используем df_with_features, который уже содержит все фичи
                proba = self.model.predict_proba(X_last, df_history=df_with_features)[0]
            else:
                # Обычные модели и ансамбли (TripleEnsemble, etc.)
                proba = self.model.predict_proba(X_last)[0]
            
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
                ensemble_absolute_min = 0.003  # Минимальная уверенность 0.3%
                
                # Вычисляем разницу между LONG и SHORT
                prob_diff = abs(long_prob - short_prob)
                
                # Определяем предсказание
                if long_prob >= ensemble_absolute_min and long_prob > short_prob and prob_diff >= self.min_confidence_difference:
                    prediction = 1  # LONG
                    confidence = min(long_prob * (1 + prob_diff * 0.3), long_prob)
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = long_prob
                elif short_prob >= ensemble_absolute_min and short_prob > long_prob and prob_diff >= self.min_confidence_difference:
                    prediction = -1  # SHORT
                    confidence = min(short_prob * (1 + prob_diff * 0.3), short_prob)
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = short_prob
                else:
                    prediction = 0
                    confidence = hold_prob
                
                # Fallback
                if prediction == 0:
                    prediction_idx = np.argmax(proba)
                    prediction = prediction_idx - 1
                    confidence = proba[prediction_idx]
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = hold_prob if np.isfinite(hold_prob) else 0.0
                
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
    ) -> Signal:
        """
        Генерирует торговый сигнал на основе ML-предсказания.
        
        ВАЖНО: SL рассчитывается от значимых уровней (поддержка/сопротивление)
                TP рассчитывается по RR 2-3:1 от SL
        
        Args:
            row: Текущий бар (pd.Series)
            df: DataFrame со всеми данными
            has_position: Текущая позиция (None, Bias.LONG, Bias.SHORT)
            current_price: Текущая цена
            leverage: Плечо (default: 10)
            target_profit_pct_margin: Целевая прибыль от маржи в % (25%)
            max_loss_pct_margin: Максимальный убыток от маржи в % (10%)
        
        Returns:
            Signal объект с уровневым SL и RR TP
        """
        try:
            # Определяем символ
            symbol = getattr(self, '_symbol', None)
            if symbol is None:
                model_filename = Path(self.model_path).name
                if "_" in model_filename:
                    parts = model_filename.replace(".pkl", "").split("_")
                    if len(parts) >= 3 and parts[0] in ("triple", "quad") and parts[1] == "ensemble":
                        symbol = parts[2].upper()
                        self._symbol = symbol
                    elif len(parts) >= 2:
                        symbol = parts[1].upper()
                        self._symbol = symbol
                    else:
                        symbol = "UNKNOWN"
                else:
                    symbol = "UNKNOWN"
            
            # Делаем предсказание
            # ВАЖНО: НЕ пропускаем создание фичей, чтобы индикаторы обновлялись для новых свечей
            prediction, confidence = self.predict(df, skip_feature_creation=False)
            
            # === ДИНАМИЧЕСКИЙ CONFIDENCE THRESHOLD ===
            # Адаптируем порог на основе рыночных условий
            effective_threshold = self.confidence_threshold
            
            if self.use_dynamic_threshold and prediction != 0:
                # Получаем рыночные индикаторы
                atr_pct = row.get("atr_pct", np.nan)
                adx = row.get("adx", np.nan)
                
                # Адаптация на основе волатильности (ATR)
                if np.isfinite(atr_pct):
                    # Высокая волатильность = выше порог (больше шума)
                    # Низкая волатильность = ниже порог (меньше шума)
                    if atr_pct > 1.5:  # Высокая волатильность
                        effective_threshold = self.confidence_threshold * 1.2
                    elif atr_pct < 0.5:  # Низкая волатильность
                        effective_threshold = self.confidence_threshold * 0.9
                
                # Адаптация на основе силы тренда (ADX)
                if np.isfinite(adx):
                    # Слабый тренд (ADX < 20) = выше порог (меньше уверенности)
                    # Сильный тренд (ADX > 25) = ниже порог (больше уверенности)
                    if adx < 20:  # Слабый тренд
                        effective_threshold = max(effective_threshold, self.confidence_threshold * 1.15)
                    elif adx > 25:  # Сильный тренд
                        effective_threshold = min(effective_threshold, self.confidence_threshold * 0.95)
                
                # Ограничиваем диапазон (0.3 - 0.8)
                effective_threshold = max(0.3, min(0.8, effective_threshold))
            
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

            # КРИТИЧНО: ВСЕГДА используем SL=1% (строгое требование)
            # Уровни S/R используем только для информации, но не для расчета SL
            if prediction == 1:
                # LONG: SL = цена * 0.99 (строго 1% ниже)
                sl_price = current_price * 0.99
                sl_source = "fixed_1pct"
                sl_level = None
            elif prediction == -1:
                # SHORT: SL = цена * 1.01 (строго 1% выше)
                sl_price = current_price * 1.01
                sl_source = "fixed_1pct"
                sl_level = None
            
            # Опционально: проверяем уровни S/R для валидации (но не используем для SL)
            if prediction != 0:
                sl_from_levels, _, _ = _calculate_sl_from_levels("LONG" if prediction == 1 else "SHORT")
                if sl_from_levels is not None:
                    # Проверяем, близок ли SL от уровней к 1% (в пределах ±0.2%)
                    if prediction == 1:
                        sl_distance_from_levels = (current_price - sl_from_levels) / current_price
                        if 0.008 <= sl_distance_from_levels <= 0.012:  # 0.8% - 1.2%
                            # SL от уровней близок к 1%, можно использовать его (но это опционально)
                            # Для строгости оставляем фиксированный 1%
                            pass
                    else:  # SHORT
                        sl_distance_from_levels = (sl_from_levels - current_price) / current_price
                        if 0.008 <= sl_distance_from_levels <= 0.012:  # 0.8% - 1.2%
                            # SL от уровней близок к 1%, можно использовать его (но это опционально)
                            # Для строгости оставляем фиксированный 1%
                            pass

            # RR 2-3:1 (динамически от уверенности, но всегда в диапазоне)
            rr = 2.0
            if _is_finite_number(confidence):
                rr = 2.0 + min(1.0, max(0.0, (confidence - 0.5) / 0.4))
            rr = float(min(3.0, max(2.0, rr)))

            if prediction == 1 and sl_price is not None:
                risk = abs(current_price - sl_price)
                tp_price = current_price + (risk * rr)
            elif prediction == -1 and sl_price is not None:
                risk = abs(sl_price - current_price)
                tp_price = current_price - (risk * rr)
            
            # УЛУЧШЕНИЕ: Валидация TP/SL (из успешного бэктеста)
            # Проверяем корректность TP/SL для LONG
            if prediction == 1 and tp_price is not None and sl_price is not None:
                if not (sl_price < current_price and tp_price > current_price):
                    sl_price = current_price * 0.99
                    tp_price = current_price * 1.025
            
            # Проверяем корректность TP/SL для SHORT
            if prediction == -1 and tp_price is not None and sl_price is not None:
                if not (sl_price > current_price and tp_price < current_price):
                    sl_price = current_price * 1.01
                    tp_price = current_price * 0.975
            
            # УЛУЧШЕНИЕ: Финальная проверка на валидность (из успешного бэктеста)
            # ВАЖНО: Если TP/SL невалидны, мы их пересчитаем позже, но НЕ устанавливаем в None
            # для LONG/SHORT сигналов, так как они ВСЕГДА должны иметь TP/SL
            if tp_price is not None and sl_price is not None:
                # Проверяем, что цены не NaN и не бесконечны
                if not (np.isfinite(tp_price) and np.isfinite(sl_price)):
                    # Для LONG/SHORT пересчитываем, а не устанавливаем None
                    if prediction == 1:  # LONG
                        sl_price = current_price * 0.99
                        tp_price = current_price * 1.025
                    elif prediction == -1:  # SHORT
                        sl_price = current_price * 1.01
                        tp_price = current_price * 0.975
                    else:
                        tp_price = None
                        sl_price = None
                # Проверяем, что цены положительные
                elif tp_price <= 0 or sl_price <= 0:
                    # Для LONG/SHORT пересчитываем, а не устанавливаем None
                    if prediction == 1:  # LONG
                        sl_price = current_price * 0.99
                        tp_price = current_price * 1.025
                    elif prediction == -1:  # SHORT
                        sl_price = current_price * 1.01
                        tp_price = current_price * 0.975
                    else:
                        tp_price = None
                        sl_price = None
            
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
            current_date = datetime.now(timezone.utc).date()
            date_str = current_date.isoformat()
            signals_today = self.daily_signals_count.get(date_str, 0)
            
            # Минимальная сила сигнала
            if self.is_ensemble:
                min_strength = 0.003  # 0.3% для ансамблей
            else:
                min_strength = 0.6  # 60% для одиночных моделей
            
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
            
            # Фильтр стабильности: если есть противоположная позиция, требуем больше уверенности
            if self.stability_filter and prediction != 0:
                if has_position == Bias.SHORT and prediction == 1:
                    # Есть SHORT, хотим открыть LONG - нужна высокая уверенность
                    stability_threshold = max(confidence * 1.3, min_strength * 1.5)
                    if confidence < stability_threshold:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_стабильность_требует_{int(stability_threshold*100)}%_против_SHORT", 
                            current_price
                        )
                elif has_position == Bias.LONG and prediction == -1:
                    # Есть LONG, хотим открыть SHORT - нужна высокая уверенность
                    stability_threshold = max(confidence * 1.3, min_strength * 1.5)
                    if confidence < stability_threshold:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_стабильность_требует_{int(stability_threshold*100)}%_против_LONG", 
                            current_price
                        )
            
            # Дополнительные фильтры для волатильных рынков
            is_volatile_symbol = symbol in ("ETHUSDT", "SOLUSDT")
            
            # Фильтр по RSI для экстремальных зон
            rsi = row.get("rsi", np.nan)
            if prediction != 0 and np.isfinite(rsi):
                if (prediction == 1 and rsi > 85) or (prediction == -1 and rsi < 15):
                    # В экстремальных зонах требуем больше уверенности
                    extreme_threshold = confidence * 1.2
                    if confidence < extreme_threshold:
                        rsi_int = int(rsi) if np.isfinite(rsi) else 0
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_экстремальный_RSI_{rsi_int}_{strength}_{confidence_pct}%", 
                            current_price
                        )
            
            # Фильтр по объему (только для сильных сигналов > 0.7)
            if confidence > 0.7:
                volume = row.get("volume", np.nan)
                vol_sma = row.get("vol_sma", np.nan)
                if not np.isfinite(vol_sma):
                    vol_sma = row.get("volume_sma_20", np.nan)
                
                if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0:
                    volume_ratio = volume / vol_sma
                    if volume_ratio < 0.5:  # Объем меньше 50% от среднего
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_низкий_объем_{volume_ratio:.1f}x_{strength}_{confidence_pct}%", 
                            current_price
                        )
            
            # КРИТИЧНО: Проверяем, что TP/SL установлены перед генерацией LONG/SHORT сигналов
            if prediction != 0 and (tp_price is None or sl_price is None):
                # Если TP/SL не установлены, пересчитываем их принудительно
                if prediction == 1:  # LONG
                    sl_price = current_price * 0.99  # 1% ниже
                    tp_price = current_price + (abs(current_price - sl_price) * rr)
                    sl_source = sl_source or "fallback_1pct"
                elif prediction == -1:  # SHORT
                    sl_price = current_price * 1.01  # 1% выше
                    tp_price = current_price - (abs(sl_price - current_price) * rr)
                    sl_source = sl_source or "fallback_1pct"

            if prediction != 0:
                tp_pct_display = (abs(tp_price - current_price) / current_price) * 100 if tp_price else 0.0
                sl_pct_display = (abs(current_price - sl_price) / current_price) * 100 if sl_price else 0.0
            
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
                    "risk_reward": round(rr, 2),
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
                    "risk_reward": round(rr, 2),
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
    confidence_threshold: float = 0.5,
    min_signal_strength: str = "слабое",
    stability_filter: bool = True,
    leverage: int = 10,
    target_profit_pct_margin: float = 25.0,
    max_loss_pct_margin: float = 10.0,
    min_signals_per_day: int = 1,
    max_signals_per_day: int = 10,
) -> list[Signal]:
    """
    Строит сигналы на основе ML-модели для всего DataFrame.
    
    Args:
        df: DataFrame с данными (должен содержать OHLCV и индикаторы)
        model_path: Путь к обученной модели
        confidence_threshold: Минимальная уверенность для открытия позиции
        min_signal_strength: Минимальная сила сигнала
        stability_filter: Фильтр стабильности
        leverage: Плечо (default: 10)
        target_profit_pct_margin: Целевая прибыль от маржи в % (25%)
        max_loss_pct_margin: Максимальный убыток от маржи в % (10%)
    
    Returns:
        Список Signal объектов
    """
    strategy = MLStrategy(
        model_path, 
        confidence_threshold, 
        min_signal_strength, 
        stability_filter,
        min_signals_per_day=min_signals_per_day,
        max_signals_per_day=max_signals_per_day
    )
    signals: list[Signal] = []
    position_bias: Optional[Bias] = None
    
    # Убеждаемся, что DataFrame имеет правильную структуру
    df_work = df.copy()
    
    # Если timestamp в колонках, используем его как индекс
    if "timestamp" in df_work.columns:
        df_work = df_work.set_index("timestamp")
    elif not isinstance(df_work.index, pd.DatetimeIndex):
        # Пытаемся преобразовать индекс в DatetimeIndex
        try:
            df_work.index = pd.to_datetime(df_work.index)
        except:
            pass
    
    # Убеждаемся, что есть необходимые колонки OHLCV
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df_work.columns for col in required_cols):
        logger.warning(f"[ml_strategy] Warning: Missing required columns. Available: {df_work.columns.tolist()}")
        return [Signal(df_work.index[i] if len(df_work) > 0 else pd.Timestamp.now(), 
                       Action.HOLD, "ml_missing_data", 0.0) 
                for i in range(len(df_work))]
    
    # ОПТИМИЗАЦИЯ: Вычисляем фичи один раз для всего DataFrame
    try:
        # Определяем, включен ли MTF-режим
        import os
        ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "0")
        ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")

        # Базовые технические индикаторы на 15m
        df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)

        # Если включен MTF-режим, добавляем фичи 1h/4h
        if ml_mtf_enabled:
            try:
                # Строим агрегированные OHLCV для 1h и 4h из 15m данных
                ohlcv_agg = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
                df_1h = df_work.resample("60min").agg(ohlcv_agg).dropna()
                df_4h = df_work.resample("240min").agg(ohlcv_agg).dropna()

                higher_timeframes = {}
                if df_1h is not None and not df_1h.empty:
                    higher_timeframes["60"] = df_1h
                if df_4h is not None and not df_4h.empty:
                    higher_timeframes["240"] = df_4h

                if higher_timeframes:
                    df_with_features = strategy.feature_engineer.add_mtf_features(
                        df_with_features,
                        higher_timeframes,
                    )
                    logger.debug(f"[ml_strategy] MTF features enabled for ML signals (1h/4h). Columns: {len(df_with_features.columns)}")
                else:
                    logger.warning("[ml_strategy] MTF enabled but failed to build 1h/4h data – using 15m-only features")
            except Exception as mtf_err:
                logger.warning(f"[ml_strategy] Warning: failed to add MTF features in build_ml_signals: {mtf_err}")
    except Exception as e:
        logger.error(f"[ml_strategy] Error preparing features: {e}")
        return [Signal(df_work.index[i] if len(df_work) > 0 else pd.Timestamp.now(), 
                       Action.HOLD, f"ml_error_{str(e)[:20]}", 0.0) 
                for i in range(len(df_work))]
    
    for idx, row in df_with_features.iterrows():
        try:
            # Получаем данные до текущего момента
            df_until_now = df_with_features.loc[:idx]
            
            # Нужно минимум 200 баров для расчета всех индикаторов
            if len(df_until_now) < 200:
                signals.append(Signal(idx, Action.HOLD, "ml_insufficient_data", row["close"]))
                continue
            
            # Используем уже вычисленные фичи
            signal = strategy.generate_signal(
                row=row,
                df=df_until_now,
                has_position=position_bias,
                current_price=row["close"],
                leverage=leverage,
                target_profit_pct_margin=target_profit_pct_margin,
                max_loss_pct_margin=max_loss_pct_margin,
            )
            signals.append(signal)
        except Exception as e:
            logger.error(f"[ml_strategy] Error processing row {idx}: {e}")
            signals.append(Signal(idx, Action.HOLD, f"ml_error_{str(e)[:20]}", row.get("close", 0.0)))
    
    return signals