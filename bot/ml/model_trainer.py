"""
Модуль для обучения ML-моделей на исторических данных.
"""
import warnings
import os
import sys

# Подавляем предупреждения scikit-learn ДО импорта библиотек
# Устанавливаем переменную окружения ПЕРВОЙ
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['SKLEARN_WARNINGS'] = 'ignore'
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # Для joblib

# Агрессивная фильтрация всех UserWarning
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*propagate the scikit-learn configuration.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*joblib.*')

# Перехватываем предупреждения на уровне stderr для подавления sklearn warnings
class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.skip_patterns = [
            'sklearn.utils.parallel',
            'delayed',
            'joblib',
            'should be used with',
            'propagate the scikit-learn configuration'
        ]
    
    def write(self, message):
        # Пропускаем сообщения, содержащие паттерны предупреждений sklearn
        if any(pattern in message for pattern in self.skip_patterns):
            return
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # Проксируем все остальные атрибуты к оригинальному stderr
        return getattr(self.original_stderr, name)

# Сохраняем оригинальный stderr и устанавливаем фильтр
_original_stderr = sys.stderr
sys.stderr = WarningFilter(_original_stderr)

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[model_trainer] Warning: LightGBM not available. Install with: pip install lightgbm")
from collections import Counter

from bot.ml.feature_engineering import FeatureEngineer

# Импорт для LSTM (опционально)
try:
    from bot.ml.lstm_model import LSTMTrainer
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[model_trainer] Warning: LSTM not available. Install PyTorch for LSTM support.")


class PreTrainedVotingEnsemble:
    """Ансамбль с предобученными моделями для voting метода."""
    def __init__(self, rf_model, xgb_model, rf_weight, xgb_weight):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X):
        # Получаем вероятности от обеих моделей
        rf_proba = self.rf_model.predict_proba(X)
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        # XGBoost возвращает классы 0,1,2, нужно преобразовать в -1,0,1
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Взвешенное усреднение
        ensemble_proba = (self.rf_weight * rf_proba + 
                         self.xgb_weight * xgb_proba_reordered)
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class WeightedEnsemble:
    """Взвешенный ансамбль из RandomForest и XGBoost."""
    def __init__(self, rf_model, xgb_model, rf_weight=0.5, xgb_weight=0.5):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X):
        """Предсказывает вероятности для всех классов."""
        # Получаем вероятности от обеих моделей
        rf_proba = self.rf_model.predict_proba(X)
        
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        # XGBoost возвращает классы 0,1,2, нужно преобразовать в -1,0,1
        # Переупорядочиваем: [0,1,2] -> [-1,0,1]
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Взвешенное усреднение
        ensemble_proba = (self.rf_weight * rf_proba + 
                         self.xgb_weight * xgb_proba_reordered)
        return ensemble_proba
    
    def predict(self, X):
        """Предсказывает классы."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TripleEnsemble:
    """Взвешенный ансамбль из RandomForest, XGBoost и LightGBM."""
    def __init__(self, rf_model, xgb_model, lgb_model, rf_weight=0.33, xgb_weight=0.33, lgb_weight=0.34):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X):
        # Получаем вероятности от всех трех моделей
        rf_proba = self.rf_model.predict_proba(X)
        
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Для LightGBM тоже нужно преобразовать
        # Подавляем предупреждение о feature names (не критично для работы модели)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            lgb_proba = self.lgb_model.predict_proba(X)
        lgb_proba_reordered = np.zeros_like(rf_proba)
        lgb_proba_reordered[:, 0] = lgb_proba[:, 0]  # SHORT (0 -> -1)
        lgb_proba_reordered[:, 1] = lgb_proba[:, 1]  # HOLD (1 -> 0)
        lgb_proba_reordered[:, 2] = lgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Взвешенное усреднение всех трех моделей
        ensemble_proba = (self.rf_weight * rf_proba + 
                         self.xgb_weight * xgb_proba_reordered +
                         self.lgb_weight * lgb_proba_reordered)
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class QuadEnsemble:
    """
    Взвешенный ансамбль из RandomForest, XGBoost, LightGBM и LSTM.
    
    LSTM требует последовательности данных, поэтому predict_proba принимает DataFrame
    с историей для создания последовательностей.
    """
    def __init__(
        self, 
        rf_model, 
        xgb_model, 
        lgb_model, 
        lstm_trainer,  # LSTMTrainer объект
        rf_weight=0.25, 
        xgb_weight=0.25, 
        lgb_weight=0.25,
        lstm_weight=0.25,
        sequence_length: int = 60,  # Длина последовательности для LSTM
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.lstm_trainer = lstm_trainer
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.lstm_weight = lstm_weight
        self.sequence_length = sequence_length
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X, df_history: Optional[pd.DataFrame] = None):
        """
        Предсказывает вероятности для всех четырех моделей.
        
        Args:
            X: Матрица фичей (n_samples, n_features) для RF/XGB/LGB
            df_history: DataFrame с историей для LSTM (должен содержать все фичи и иметь минимум sequence_length строк)
        
        Returns:
            Массив вероятностей (n_samples, 3) в формате [SHORT, HOLD, LONG]
        """
        # Получаем вероятности от классических моделей
        rf_proba = self.rf_model.predict_proba(X)
        
        # Проверяем rf_proba на NaN
        if np.any(np.isnan(rf_proba)) or not np.all(np.isfinite(rf_proba)):
            print(f"[QuadEnsemble] Warning: RF proba contains NaN, using uniform distribution")
            rf_proba = np.ones_like(rf_proba) / 3.0
        
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # Проверяем xgb_proba на NaN
        if np.any(np.isnan(xgb_proba)) or not np.all(np.isfinite(xgb_proba)):
            print(f"[QuadEnsemble] Warning: XGB proba contains NaN, using uniform distribution")
            xgb_proba = np.ones_like(xgb_proba) / 3.0
        
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Для LightGBM тоже нужно преобразовать
        # Подавляем предупреждение о feature names (не критично для работы модели)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            lgb_proba = self.lgb_model.predict_proba(X)
        
        # Проверяем lgb_proba на NaN
        if np.any(np.isnan(lgb_proba)) or not np.all(np.isfinite(lgb_proba)):
            print(f"[QuadEnsemble] Warning: LGB proba contains NaN, using uniform distribution")
            lgb_proba = np.ones_like(lgb_proba) / 3.0
        
        lgb_proba_reordered = np.zeros_like(rf_proba)
        lgb_proba_reordered[:, 0] = lgb_proba[:, 0]  # SHORT (0 -> -1)
        lgb_proba_reordered[:, 1] = lgb_proba[:, 1]  # HOLD (1 -> 0)
        lgb_proba_reordered[:, 2] = lgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Определяем n_samples в начале (нужно для обработки исключений)
        n_samples = len(X)
        
        # Для LSTM: создаем последовательности из истории
        if df_history is not None and len(df_history) >= self.sequence_length:
            try:
                # Получаем фичи из истории (те же, что использовались при обучении LSTM)
                feature_names = self.lstm_trainer.feature_names if hasattr(self.lstm_trainer, 'feature_names') and self.lstm_trainer.feature_names is not None else None
                
                if feature_names:
                    # Проверяем, что все нужные фичи доступны
                    missing_features = [f for f in feature_names if f not in df_history.columns]
                    if missing_features:
                        # Пытаемся создать недостающие фичи динамически (только один раз, не логируем каждый раз)
                        if not hasattr(self, '_feature_creation_warned'):
                            print(f"[QuadEnsemble] Warning: Missing features in history: {missing_features[:5]}... Attempting to create them.")
                            self._feature_creation_warned = True
                        
                        # Создаем недостающие фичи, если это возможно
                        df_history_work = df_history.copy()
                        try:
                            # Используем FeatureEngineer для создания всех фичей
                            from bot.ml.feature_engineering import FeatureEngineer
                            feature_engineer = FeatureEngineer()
                            
                            # Проверяем, есть ли необходимые колонки OHLCV
                            required_cols = ['open', 'high', 'low', 'close', 'volume']
                            if all(col in df_history_work.columns for col in required_cols):
                                # Создаем все технические индикаторы
                                df_history_work = feature_engineer.create_technical_indicators(df_history_work)
                                
                                # Проверяем снова после создания
                                missing_features_after = [f for f in feature_names if f not in df_history_work.columns]
                                if missing_features_after:
                                    # Если все еще есть недостающие фичи, пытаемся создать их вручную
                                    import pandas_ta as ta
                                    
                                    # Создаем SMA фичи
                                    if 'sma_20' in missing_features_after and 'close' in df_history_work.columns:
                                        df_history_work['sma_20'] = ta.sma(df_history_work['close'], length=20)
                                    if 'sma_50' in missing_features_after and 'close' in df_history_work.columns:
                                        df_history_work['sma_50'] = ta.sma(df_history_work['close'], length=50)
                                    if 'sma_200' in missing_features_after and 'close' in df_history_work.columns:
                                        df_history_work['sma_200'] = ta.sma(df_history_work['close'], length=200)
                                    
                                    # Создаем EMA фичи
                                    if 'ema_12' in missing_features_after and 'close' in df_history_work.columns:
                                        df_history_work['ema_12'] = ta.ema(df_history_work['close'], length=12)
                                    if 'ema_26' in missing_features_after and 'close' in df_history_work.columns:
                                        df_history_work['ema_26'] = ta.ema(df_history_work['close'], length=26)
                                    
                                    # Проверяем финально
                                    missing_features_final = [f for f in feature_names if f not in df_history_work.columns]
                                    if missing_features_final:
                                        raise ValueError(f"Still missing features after creation attempt: {missing_features_final[:5]}...")
                                
                                df_history = df_history_work
                            else:
                                raise ValueError(f"Missing required OHLCV columns for feature creation: {[c for c in required_cols if c not in df_history_work.columns]}")
                        except Exception as e:
                            if not hasattr(self, '_feature_creation_error_logged'):
                                print(f"[QuadEnsemble] Failed to create missing features: {e}")
                                self._feature_creation_error_logged = True
                            raise ValueError(f"Missing features in history: {missing_features[:5]}... and could not create them automatically.")
                    
                    # Используем только нужные фичи в правильном порядке
                    df_features = df_history[feature_names].copy()
                    
                    # Проверяем на NaN перед нормализацией
                    if df_features.isna().any().any():
                        # Умное заполнение NaN: forward fill для временных рядов, затем backward fill, затем 0
                        # Это лучше чем просто 0, так как сохраняет структуру данных
                        # Логируем только один раз
                        if not hasattr(self, '_nan_warning_logged'):
                            print(f"[QuadEnsemble] Warning: Features contain NaN before scaling, filling intelligently")
                            self._nan_warning_logged = True
                        df_features = df_features.ffill().bfill().fillna(0.0)
                else:
                    # Если feature_names не установлены, пытаемся определить из scaler
                    if hasattr(self.lstm_trainer, 'scaler') and self.lstm_trainer.scaler is not None:
                        expected_features = self.lstm_trainer.scaler.n_features_in_ if hasattr(self.lstm_trainer.scaler, 'n_features_in_') else None
                        if expected_features:
                            # Используем первые expected_features фичей (как при обучении)
                            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                            feature_cols = [col for col in df_history.columns if col not in exclude_cols]
                            if len(feature_cols) >= expected_features:
                                feature_cols = feature_cols[:expected_features]
                                df_features = df_history[feature_cols].copy()
                                # Логируем только один раз
                                if not hasattr(self, '_feature_names_warning_logged'):
                                    print(f"[QuadEnsemble] Using first {expected_features} features (feature_names not set)")
                                    self._feature_names_warning_logged = True
                                
                                # Проверяем на NaN
                                if df_features.isna().any().any():
                                    # Умное заполнение NaN: forward fill, backward fill, затем 0
                                    df_features = df_features.ffill().bfill().fillna(0.0)
                            else:
                                raise ValueError(f"Not enough features: need {expected_features}, got {len(feature_cols)}")
                        else:
                            # Используем все фичи кроме OHLCV
                            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                            feature_cols = [col for col in df_history.columns if col not in exclude_cols]
                            df_features = df_history[feature_cols].copy()
                            
                            # Проверяем на NaN
                            if df_features.isna().any().any():
                                # Умное заполнение NaN: forward fill, backward fill, затем 0
                                df_features = df_features.ffill().bfill().fillna(0.0)
                    else:
                        # Используем все фичи кроме OHLCV
                        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                        feature_cols = [col for col in df_history.columns if col not in exclude_cols]
                        df_features = df_history[feature_cols].copy()
                        
                        # Проверяем на NaN
                        if df_features.isna().any().any():
                            print(f"[QuadEnsemble] Warning: Features contain NaN, filling with 0")
                            df_features = df_features.fillna(0.0)
                
                # Проверяем количество фичей перед нормализацией
                if hasattr(self.lstm_trainer, 'scaler') and self.lstm_trainer.scaler is not None:
                    # Проверяем, что количество фичей соответствует ожидаемому
                    expected_features = self.lstm_trainer.scaler.n_features_in_ if hasattr(self.lstm_trainer.scaler, 'n_features_in_') else None
                    if expected_features and len(df_features.columns) != expected_features:
                        raise ValueError(
                            f"Feature count mismatch: LSTM expects {expected_features} features, "
                            f"but got {len(df_features.columns)}. "
                            f"Expected features: {list(feature_names)[:10] if feature_names else 'unknown'}... "
                            f"Got features: {list(df_features.columns)[:10]}..."
                        )
                    
                    df_features_scaled = pd.DataFrame(
                        self.lstm_trainer.scaler.transform(df_features.values),
                        index=df_features.index,
                        columns=df_features.columns
                    )
                    
                    # Проверяем на NaN после нормализации
                    if df_features_scaled.isna().any().any():
                        # Умное заполнение NaN: forward fill, backward fill, затем 0
                        df_features_scaled = df_features_scaled.ffill().bfill().fillna(0.0)
                else:
                    df_features_scaled = df_features
                    
                    # Проверяем на NaN
                    if df_features_scaled.isna().any().any():
                        print(f"[QuadEnsemble] Warning: Features contain NaN, filling with 0")
                        df_features_scaled = df_features_scaled.fillna(0.0)
                
                # Создаем последовательности для LSTM
                lstm_proba = np.zeros((n_samples, 3))
                
                for i in range(n_samples):
                    # Берем последние sequence_length строк для последовательности
                    end_idx = len(df_features_scaled)
                    start_idx = max(0, end_idx - self.sequence_length)
                    sequence = df_features_scaled.iloc[start_idx:end_idx].values
                    
                    # Если последовательность короче нужного, дополняем первым значением
                    if len(sequence) < self.sequence_length:
                        padding = np.tile(sequence[0:1], (self.sequence_length - len(sequence), 1))
                        sequence = np.vstack([padding, sequence])
                    
                    # Проверяем последовательность на NaN перед предсказанием
                    if np.any(np.isnan(sequence)) or not np.all(np.isfinite(sequence)):
                        # Заменяем NaN на 0 или среднее значение
                        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Делаем предсказание LSTM
                    sequence_reshaped = sequence.reshape(1, self.sequence_length, -1)
                    _, proba = self.lstm_trainer.predict(sequence_reshaped)
                    
                    # Проверяем proba на NaN
                    if np.any(np.isnan(proba)) or not np.all(np.isfinite(proba)):
                        # Используем равномерное распределение для этого образца
                        proba = np.array([[0.33, 0.34, 0.33]])
                    
                    # LSTM возвращает вероятности в формате [SHORT(0), HOLD(1), LONG(2)]
                    # Преобразуем в [-1, 0, 1]
                    lstm_proba[i, 0] = proba[0, 0]  # SHORT
                    lstm_proba[i, 1] = proba[0, 1]  # HOLD
                    lstm_proba[i, 2] = proba[0, 2]  # LONG
            except Exception as e:
                # Если LSTM не может сделать предсказание, используем равномерное распределение
                print(f"[QuadEnsemble] Warning: LSTM prediction failed: {e}. Using uniform probabilities.")
                lstm_proba = np.ones((n_samples, 3)) / 3.0
        else:
            # Если нет истории, используем равномерное распределение для LSTM
            lstm_proba = np.ones((n_samples, 3)) / 3.0
        
        # Проверяем lstm_proba на NaN перед вычислением ансамбля
        if np.any(np.isnan(lstm_proba)) or not np.all(np.isfinite(lstm_proba)):
            print(f"[QuadEnsemble] Warning: LSTM proba contains NaN, using uniform distribution")
            lstm_proba = np.ones_like(lstm_proba) / 3.0
        
        # Взвешенное усреднение всех четырех моделей
        ensemble_proba = (
            self.rf_weight * rf_proba + 
            self.xgb_weight * xgb_proba_reordered +
            self.lgb_weight * lgb_proba_reordered +
            self.lstm_weight * lstm_proba
        )
        
        # Проверяем результат на NaN и нормализуем
        if np.any(np.isnan(ensemble_proba)) or not np.all(np.isfinite(ensemble_proba)):
            print(f"[QuadEnsemble] Warning: Ensemble proba contains NaN after combination, using uniform distribution")
            ensemble_proba = np.ones_like(ensemble_proba) / 3.0
        else:
            # Нормализуем вероятности (сумма должна быть 1.0)
            row_sums = ensemble_proba.sum(axis=1, keepdims=True)
            # Избегаем деления на ноль
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            ensemble_proba = ensemble_proba / row_sums
        
        return ensemble_proba
    
    def predict(self, X, df_history: Optional[pd.DataFrame] = None):
        proba = self.predict_proba(X, df_history)
        return self.classes_[np.argmax(proba, axis=1)]


class ModelTrainer:
    """
    Обучает ML-модели для предсказания движения цены.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "ml_models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
    
    def train_random_forest_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Обучает Random Forest классификатор.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (1 = LONG, -1 = SHORT, 0 = HOLD)
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            min_samples_split: Минимальное количество образцов для разделения
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов (если None, используется автоматическая балансировка)
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training Random Forest Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y + 1)}")  # +1 чтобы индексы были 0,1,2
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Определяем веса классов
        if class_weight is not None:
            # Используем переданные кастомные веса
            class_weights = class_weight
            print(f"  Using custom class weights: {class_weights}")
        else:
            # Вычисляем веса классов для более агрессивного обучения на LONG/SHORT
            # Увеличиваем вес для LONG и SHORT классов относительно HOLD
            unique_classes, class_counts = np.unique(y, return_counts=True)
            total_samples = len(y)
            class_weights = {}
            
            for cls, count in zip(unique_classes, class_counts):
                if count > 0:
                    # Используем обратную частоту, но с дополнительным весом для LONG/SHORT
                    base_weight = total_samples / (len(unique_classes) * count)
                    if cls != 0:  # LONG (1) или SHORT (-1) получают дополнительный вес
                        class_weights[int(cls)] = base_weight * 1.5  # +50% вес для торговых сигналов
                    else:  # HOLD (0) - базовый вес
                        class_weights[int(cls)] = base_weight * 0.8  # -20% вес для HOLD
            print(f"  Using auto-balanced class weights: {class_weights}")
        
        # Создаем и обучаем модель
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Используем все ядра
            class_weight=class_weights if class_weights else "balanced",  # Используем кастомные веса
        )
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        # Подавляем предупреждения при cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy", n_jobs=1)  # n_jobs=1 чтобы избежать предупреждений joblib
        
        # Обучаем на всех данных
        model.fit(X_scaled, y)
        
        # Предсказания для оценки
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "feature_importance": dict(zip(
                self.feature_engineer.get_feature_names(),
                model.feature_importances_
            )),
        }
        
        print(f"[model_trainer] Training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, metrics
    
    def train_xgboost_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """
        Обучает XGBoost классификатор.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            learning_rate: Скорость обучения
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов (если None, используется автоматическая балансировка)
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training XGBoost Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        # XGBoost может работать с ненормализованными данными, но нормализуем для консистентности
        X_scaled = self.scaler.fit_transform(X)
        
        # Преобразуем y для XGBoost (нужны индексы 0,1,2 вместо -1,0,1)
        y_xgb = y + 1  # -1,0,1 -> 0,1,2
        
        # Вычисляем веса образцов для XGBoost
        sample_weights = np.zeros(len(y_xgb))
        
        if class_weight is not None:
            # Используем переданные кастомные веса
            # Конвертируем класс-веса в веса образцов
            for orig_cls, weight in class_weight.items():
                xgb_cls = orig_cls + 1  # Преобразуем -1,0,1 -> 0,1,2
                sample_weights[y_xgb == xgb_cls] = weight
            print(f"  Using custom class weights (converted to sample_weights)")
        else:
            # Вычисляем веса классов для XGBoost (для классов 0,1,2)
            unique_classes, class_counts = np.unique(y_xgb, return_counts=True)
            total_samples = len(y_xgb)
            
            for cls, count in zip(unique_classes, class_counts):
                if count > 0:
                    base_weight = total_samples / (len(unique_classes) * count)
                    # Класс 1 (HOLD) - индекс 1 в XGBoost формате
                    if cls == 1:  # HOLD - уменьшаем вес
                        weight = base_weight * 0.8
                    else:  # LONG (2) или SHORT (0) - увеличиваем вес
                        weight = base_weight * 1.5
                    sample_weights[y_xgb == cls] = weight
            print(f"  Using auto-balanced sample weights")
        
        # Создаем и обучаем модель
        # Примечание: scale_pos_weight работает только для бинарной классификации,
        # поэтому не используем его для мультиклассовой задачи (3 класса)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            # Балансировка классов выполняется через sample_weight в fit()
        )
        
        # Обучаем с весами образцов
        model.fit(X_scaled, y_xgb, sample_weight=sample_weights)
        
        # Time-series cross-validation (без весов, так как cross_val_score не поддерживает sample_weight напрямую)
        tscv = TimeSeriesSplit(n_splits=5)
        # Подавляем предупреждения при cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = cross_val_score(model, X_scaled, y_xgb, cv=tscv, scoring="accuracy", n_jobs=1)  # n_jobs=1 чтобы избежать предупреждений joblib
        
        # Модель уже обучена выше с весами образцов
        
        # Предсказания
        y_pred_xgb = model.predict(X_scaled)
        y_pred = y_pred_xgb - 1  # Обратно в -1,0,1
        accuracy = accuracy_score(y, y_pred)
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "feature_importance": dict(zip(
                self.feature_engineer.get_feature_names(),
                model.feature_importances_
            )),
        }
        
        print(f"[model_trainer] Training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, metrics
    
    def train_lightgbm_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Обучает LightGBM классификатор.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (-1 = SHORT, 0 = HOLD, 1 = LONG)
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            learning_rate: Скорость обучения
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов (если None, используется автоматическая балансировка)
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        
        print(f"[model_trainer] Training LightGBM Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        # LightGBM может работать с ненормализованными данными, но нормализуем для консистентности
        X_scaled = self.scaler.fit_transform(X)
        
        # Преобразуем y для LightGBM (нужны индексы 0,1,2 вместо -1,0,1)
        y_lgb = y + 1  # -1,0,1 -> 0,1,2
        
        # Вычисляем веса классов для LightGBM
        if class_weight is not None:
            # Используем переданные кастомные веса
            class_weights_dict = {}
            for orig_cls, weight in class_weight.items():
                lgb_cls = orig_cls + 1  # Преобразуем -1,0,1 -> 0,1,2
                class_weights_dict[int(lgb_cls)] = weight
            print(f"  Using custom class weights: {class_weights_dict}")
        else:
            # Вычисляем веса классов автоматически
            unique_classes, class_counts = np.unique(y_lgb, return_counts=True)
            total_samples = len(y_lgb)
            class_weights_dict = {}
            
            for cls, count in zip(unique_classes, class_counts):
                if count > 0:
                    base_weight = total_samples / (len(unique_classes) * count)
                    if cls == 1:  # HOLD - уменьшаем вес
                        class_weights_dict[int(cls)] = base_weight * 0.8
                    else:  # LONG (2) или SHORT (0) - увеличиваем вес
                        class_weights_dict[int(cls)] = base_weight * 1.5
            print(f"  Using auto-balanced class weights: {class_weights_dict}")
        
        # Создаем и обучаем модель
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,  # Отключаем вывод
            class_weight=class_weights_dict if class_weights_dict else None,
            objective='multiclass',
            num_class=3,
        )
        
        # Обучаем модель
        model.fit(X_scaled, y_lgb)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = cross_val_score(model, X_scaled, y_lgb, cv=tscv, scoring="accuracy", n_jobs=1)
        
        # Предсказания
        y_pred_lgb = model.predict(X_scaled)
        y_pred = y_pred_lgb - 1  # Обратно в -1,0,1
        accuracy = accuracy_score(y, y_pred)
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "feature_importance": dict(zip(
                self.feature_engineer.get_feature_names(),
                model.feature_importances_
            )),
        }
        
        print(f"[model_trainer] Training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, metrics
    
    def save_model(
        self, 
        model: Any, 
        scaler: StandardScaler, 
        feature_names: list, 
        metrics: Dict[str, Any], 
        filename: str,
        symbol: str = "ETHUSDT",
        interval: str = "15",
        model_type: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
        class_distribution: Optional[Dict[int, int]] = None,
        training_params: Optional[Dict[str, Any]] = None,
    ):
        """Сохраняет модель, scaler, метрики и метаданные в файл."""
        filepath = self.model_dir / filename
        
        # Определяем model_type из имени файла, если не передан
        if model_type is None:
            filename_base = filename.replace('.pkl', '')
            parts = filename_base.split('_')
            if len(parts) >= 1:
                model_type = parts[0].lower()  # rf, xgb и т.д.
            else:
                model_type = "unknown"
        
        # Подготовка данных о распределении классов
        data_info = {}
        if class_distribution:
            data_info['class_distribution'] = class_distribution
            total = sum(class_distribution.values())
            data_info['total_rows'] = total
            data_info['class_percentages'] = {
                cls: (count / total * 100) for cls, count in class_distribution.items()
            } if total > 0 else {}
        
        # Для QuadEnsemble метрики находятся внутри rf_metrics, xgb_metrics и т.д.
        # Вычисляем агрегированные метрики для ансамбля
        if model_type == "quad_ensemble" and "rf_metrics" in metrics:
            # Для QuadEnsemble вычисляем средневзвешенные метрики
            rf_metrics = metrics.get("rf_metrics", {})
            xgb_metrics = metrics.get("xgb_metrics", {})
            lgb_metrics = metrics.get("lgb_metrics", {})
            lstm_metrics = metrics.get("lstm_metrics", {})
            
            # Веса моделей
            rf_weight = metrics.get("rf_weight", 0.25)
            xgb_weight = metrics.get("xgb_weight", 0.25)
            lgb_weight = metrics.get("lgb_weight", 0.25)
            lstm_weight = metrics.get("lstm_weight", 0.25)
            
            # Вычисляем средневзвешенные метрики
            ensemble_accuracy = (
                rf_metrics.get("accuracy", 0.0) * rf_weight +
                xgb_metrics.get("accuracy", 0.0) * xgb_weight +
                lgb_metrics.get("accuracy", 0.0) * lgb_weight +
                lstm_metrics.get("accuracy", 0.0) * lstm_weight
            )
            
            ensemble_cv_mean = (
                rf_metrics.get("cv_mean", 0.0) * rf_weight +
                xgb_metrics.get("cv_mean", 0.0) * xgb_weight +
                lgb_metrics.get("cv_mean", 0.0) * lgb_weight +
                (lstm_metrics.get("accuracy", 0.0) if "cv_mean" not in lstm_metrics else lstm_metrics.get("cv_mean", 0.0)) * lstm_weight
            )
            
            # Для precision, recall, f1_score берем из лучшей модели или среднее
            ensemble_precision = (
                rf_metrics.get("precision", 0.0) * rf_weight +
                xgb_metrics.get("precision", 0.0) * xgb_weight +
                lgb_metrics.get("precision", 0.0) * lgb_weight +
                lstm_metrics.get("precision", 0.0) * lstm_weight
            ) if any("precision" in m for m in [rf_metrics, xgb_metrics, lgb_metrics, lstm_metrics]) else None
            
            ensemble_recall = (
                rf_metrics.get("recall", 0.0) * rf_weight +
                xgb_metrics.get("recall", 0.0) * xgb_weight +
                lgb_metrics.get("recall", 0.0) * lgb_weight +
                lstm_metrics.get("recall", 0.0) * lstm_weight
            ) if any("recall" in m for m in [rf_metrics, xgb_metrics, lgb_metrics, lstm_metrics]) else None
            
            ensemble_f1_score = (
                rf_metrics.get("f1_score", 0.0) * rf_weight +
                xgb_metrics.get("f1_score", 0.0) * xgb_weight +
                lgb_metrics.get("f1_score", 0.0) * lgb_weight +
                lstm_metrics.get("f1_score", 0.0) * lstm_weight
            ) if any("f1_score" in m for m in [rf_metrics, xgb_metrics, lgb_metrics, lstm_metrics]) else None
            
            ensemble_cv_f1_mean = (
                rf_metrics.get("cv_f1_mean", 0.0) * rf_weight +
                xgb_metrics.get("cv_f1_mean", 0.0) * xgb_weight +
                lgb_metrics.get("cv_f1_mean", 0.0) * lgb_weight +
                lstm_metrics.get("cv_f1_mean", 0.0) * lstm_weight
            ) if any("cv_f1_mean" in m for m in [rf_metrics, xgb_metrics, lgb_metrics, lstm_metrics]) else None
            
            # Используем вычисленные метрики
            accuracy = ensemble_accuracy
            cv_mean = ensemble_cv_mean
            cv_std = (
                rf_metrics.get("cv_std", 0.0) * rf_weight +
                xgb_metrics.get("cv_std", 0.0) * xgb_weight +
                lgb_metrics.get("cv_std", 0.0) * lgb_weight +
                lstm_metrics.get("cv_std", 0.0) * lstm_weight
            )
            precision = ensemble_precision
            recall = ensemble_recall
            f1_score = ensemble_f1_score
            cv_f1_mean = ensemble_cv_f1_mean
        else:
            # Для обычных моделей используем метрики напрямую
            accuracy = metrics.get("accuracy", 0.0)
            cv_mean = metrics.get("cv_mean", 0.0)
            cv_std = metrics.get("cv_std", 0.0)
            precision = metrics.get("precision", None)
            recall = metrics.get("recall", None)
            f1_score = metrics.get("f1_score", None)
            cv_f1_mean = metrics.get("cv_f1_mean", None)
        
        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metrics": metrics,
            "model_type": model_type,  # Для обратной совместимости (старый формат)
            "timestamp": datetime.now().isoformat(),  # Для обратной совместимости
            "data_info": data_info,  # Информация о данных обучения
            "class_weights": class_weights,  # Веса классов, использованные при обучении
            "training_params": training_params,  # Параметры обучения
            "metadata": {
                "symbol": symbol,
                "interval": interval,
                "model_type": model_type,
                "trained_at": datetime.now().isoformat(),
                "accuracy": accuracy,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "precision": precision if precision is not None else 0.0,
                "recall": recall if recall is not None else 0.0,
                "f1_score": f1_score if f1_score is not None else 0.0,
                "cv_f1_mean": cv_f1_mean if cv_f1_mean is not None else 0.0,
                "rf_weight": metrics.get("rf_weight", None),  # Веса ансамбля (только для ансамблей)
                "xgb_weight": metrics.get("xgb_weight", None),
                "lgb_weight": metrics.get("lgb_weight", None),  # Для QuadEnsemble
                "lstm_weight": metrics.get("lstm_weight", None),  # Для QuadEnsemble
                "ensemble_method": metrics.get("ensemble_method", None),
            }
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"[model_trainer] Model saved to {filepath}")
        return filepath
    
    def load_model_metadata(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Загружает только метаданные модели без самой модели.
        """
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                return model_data.get("metadata", {})
        except Exception as e:
            print(f"[model_trainer] Error loading model metadata: {e}")
            return None
    
    def load_model(self, filename: str) -> Dict[str, Any]:
        """Загружает модель из файла."""
        filepath = self.model_dir / filename
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"[model_trainer] Model loaded from {filepath}")
        return model_data
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rf_n_estimators: int = 100,
        rf_max_depth: Optional[int] = 10,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        ensemble_method: str = "voting",  # "voting", "weighted_average" или "triple"
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
        include_lightgbm: bool = False,  # Включить LightGBM в ансамбль
        lgb_n_estimators: int = 100,
        lgb_max_depth: int = 6,
        lgb_learning_rate: float = 0.1,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Обучает ансамбль из RandomForest и XGBoost.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (1 = LONG, -1 = SHORT, 0 = HOLD)
            rf_n_estimators: Количество деревьев для RandomForest
            rf_max_depth: Максимальная глубина для RandomForest
            xgb_n_estimators: Количество деревьев для XGBoost
            xgb_max_depth: Максимальная глубина для XGBoost
            xgb_learning_rate: Скорость обучения для XGBoost
            ensemble_method: Метод ансамбля ("voting" или "weighted_average")
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов (передаются в обе модели)
        
        Returns:
            (ensemble_model, metrics) - обученный ансамбль и метрики
        """
        ensemble_name = f"{ensemble_method}"
        if include_lightgbm and ensemble_method == "triple":
            ensemble_name = "triple (RF+XGB+LGB)"
        print(f"[model_trainer] Training Ensemble Model ({ensemble_name})...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {Counter(y)}")
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем отдельные модели
        model_count = 3 if (include_lightgbm and ensemble_method == "triple") else 2
        
        print(f"\n  [1/{model_count}] Training RandomForest...")
        rf_model, rf_metrics = self.train_random_forest_classifier(
            X, y,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            class_weight=class_weight,  # Передаем веса классов
        )
        
        print(f"\n  [2/{model_count}] Training XGBoost...")
        xgb_model, xgb_metrics = self.train_xgboost_classifier(
            X, y,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=random_state,
            class_weight=class_weight,  # Передаем веса классов
        )
        
        lgb_model = None
        lgb_metrics = None
        
        # Если включен LightGBM и метод "triple", обучаем LightGBM
        if include_lightgbm and ensemble_method == "triple":
            if not LIGHTGBM_AVAILABLE:
                print(f"\n  ⚠️  LightGBM not available, skipping...")
                include_lightgbm = False
            else:
                print(f"\n  [3/{model_count}] Training LightGBM...")
                lgb_model, lgb_metrics = self.train_lightgbm_classifier(
                    X, y,
                    n_estimators=lgb_n_estimators,
                    max_depth=lgb_max_depth,
                    learning_rate=lgb_learning_rate,
                    random_state=random_state,
                    class_weight=class_weight,  # Передаем веса классов
                )
        
        # Вычисляем веса на основе CV метрик
        rf_cv_score = rf_metrics.get("cv_mean", 0.5)
        xgb_cv_score = xgb_metrics.get("cv_mean", 0.5)
        
        if include_lightgbm and lgb_model is not None:
            lgb_cv_score = lgb_metrics.get("cv_mean", 0.5)
            total_score = rf_cv_score + xgb_cv_score + lgb_cv_score
            if total_score > 0:
                rf_weight = rf_cv_score / total_score
                xgb_weight = xgb_cv_score / total_score
                lgb_weight = lgb_cv_score / total_score
            else:
                rf_weight = xgb_weight = lgb_weight = 1.0 / 3.0
        else:
            total_score = rf_cv_score + xgb_cv_score
            if total_score > 0:
                rf_weight = rf_cv_score / total_score
                xgb_weight = xgb_cv_score / total_score
            else:
                rf_weight = xgb_weight = 0.5
            lgb_weight = 0.0
        
        # Создаем ансамбль
        if ensemble_method == "triple" and include_lightgbm and lgb_model is not None:
            # Тройной ансамбль: RF + XGB + LGB
            ensemble = TripleEnsemble(rf_model, xgb_model, lgb_model, rf_weight, xgb_weight, lgb_weight)
            print(f"  Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}, LGB={lgb_weight:.3f}")
        elif ensemble_method == "voting":
            # Используем класс, определенный на уровне модуля
            ensemble = PreTrainedVotingEnsemble(rf_model, xgb_model, rf_weight, xgb_weight)
            print(f"  Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")
        elif ensemble_method == "weighted_average":
            # Используем класс, определенный на уровне модуля
            ensemble = WeightedEnsemble(rf_model, xgb_model, rf_weight, xgb_weight)
            print(f"  Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")
        else:
            raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
        
        # Улучшенная валидация: Walk-Forward Validation
        print(f"\n  Performing Walk-Forward Validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Обучаем модели на fold
                # Обучаем RF на fold
                rf_fold = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                rf_fold.fit(X_train_fold, y_train_fold)
                
                # Обучаем XGBoost на fold
                y_train_xgb = y_train_fold + 1
                xgb_fold = xgb.XGBClassifier(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                )
                xgb_fold.fit(X_train_fold, y_train_xgb)
                
                # Обучаем LightGBM на fold (если включен)
                lgb_fold = None
                if include_lightgbm and ensemble_method == "triple" and LIGHTGBM_AVAILABLE:
                    y_train_lgb = y_train_fold + 1
                    lgb_fold = lgb.LGBMClassifier(
                        n_estimators=lgb_n_estimators,
                        max_depth=lgb_max_depth,
                        learning_rate=lgb_learning_rate,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                        objective='multiclass',
                        num_class=3,
                    )
                    lgb_fold.fit(X_train_fold, y_train_lgb)
                
                # Создаем ансамбль для fold
                if ensemble_method == "triple" and include_lightgbm and lgb_fold is not None:
                    ensemble_fold = TripleEnsemble(rf_fold, xgb_fold, lgb_fold, rf_weight, xgb_weight, lgb_weight)
                elif ensemble_method == "voting":
                    ensemble_fold = PreTrainedVotingEnsemble(rf_fold, xgb_fold, rf_weight, xgb_weight)
                else:
                    ensemble_fold = WeightedEnsemble(rf_fold, xgb_fold, rf_weight, xgb_weight)
                
                y_pred_fold = ensemble_fold.predict(X_val_fold)
                
                # Метрики для fold
                fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val_fold, y_pred_fold, average='weighted', zero_division=0
                )
                
                cv_scores.append(fold_accuracy)
                cv_precision.append(precision)
                cv_recall.append(recall)
                cv_f1.append(f1)
                
                print(f"    Fold {fold + 1}: Accuracy={fold_accuracy:.4f}, F1={f1:.4f}")
        
        # Предсказания на всех данных
        y_pred = ensemble.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_precision_mean": np.mean(cv_precision),
            "cv_recall_mean": np.mean(cv_recall),
            "cv_f1_mean": np.mean(cv_f1),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "rf_metrics": rf_metrics,
            "xgb_metrics": xgb_metrics,
            "lgb_metrics": lgb_metrics if lgb_metrics else None,
            "ensemble_method": ensemble_method,
            "rf_weight": rf_weight,  # Веса ансамбля
            "xgb_weight": xgb_weight,
            "lgb_weight": lgb_weight if include_lightgbm else None,
        }
        
        print(f"\n[model_trainer] Ensemble training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        print(f"  CV F1-Score: {metrics['cv_f1_mean']:.4f}")
        if include_lightgbm and lgb_metrics:
            print(f"  LightGBM CV Accuracy: {lgb_metrics.get('cv_mean', 0):.4f}")
        
        return ensemble, metrics
    
    def train_quad_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,  # Полный DataFrame для LSTM (нужен для последовательностей)
        rf_n_estimators: int = 100,
        rf_max_depth: Optional[int] = 10,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        lgb_n_estimators: int = 100,
        lgb_max_depth: int = 6,
        lgb_learning_rate: float = 0.1,
        lstm_sequence_length: int = 60,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_epochs: int = 50,
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Обучает ансамбль из RandomForest, XGBoost, LightGBM и LSTM.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (-1 = SHORT, 0 = HOLD, 1 = LONG)
            df: Полный DataFrame с историей для LSTM (должен содержать все фичи)
            rf_n_estimators: Количество деревьев для RandomForest
            rf_max_depth: Максимальная глубина для RandomForest
            xgb_n_estimators: Количество деревьев для XGBoost
            xgb_max_depth: Максимальная глубина для XGBoost
            xgb_learning_rate: Скорость обучения для XGBoost
            lgb_n_estimators: Количество деревьев для LightGBM
            lgb_max_depth: Максимальная глубина для LightGBM
            lgb_learning_rate: Скорость обучения для LightGBM
            lstm_sequence_length: Длина последовательности для LSTM
            lstm_hidden_size: Размер скрытого слоя LSTM
            lstm_num_layers: Количество слоев LSTM
            lstm_epochs: Количество эпох обучения LSTM
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов
        
        Returns:
            (ensemble_model, metrics) - обученный ансамбль и метрики
        """
        if not LSTM_AVAILABLE:
            raise ImportError("LSTM is not available. Install PyTorch for LSTM support.")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        print(f"[model_trainer] Training Quad Ensemble (RF+XGB+LGB+LSTM)...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {Counter(y)}")
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем классические модели
        print(f"\n  [1/4] Training RandomForest...")
        rf_model, rf_metrics = self.train_random_forest_classifier(
            X, y,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            class_weight=class_weight,
        )
        
        print(f"\n  [2/4] Training XGBoost...")
        xgb_model, xgb_metrics = self.train_xgboost_classifier(
            X, y,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=random_state,
            class_weight=class_weight,
        )
        
        print(f"\n  [3/4] Training LightGBM...")
        lgb_model, lgb_metrics = self.train_lightgbm_classifier(
            X, y,
            n_estimators=lgb_n_estimators,
            max_depth=lgb_max_depth,
            learning_rate=lgb_learning_rate,
            random_state=random_state,
            class_weight=class_weight,
        )
        
        print(f"\n  [4/4] Training LSTM...")
        # Обучаем LSTM на полном DataFrame
        lstm_trainer = LSTMTrainer(
            sequence_length=lstm_sequence_length,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            num_epochs=lstm_epochs,
        )
        
        # Подготавливаем данные для LSTM (добавляем target колонку)
        df_lstm = df.copy()
        if 'target' not in df_lstm.columns:
            # Создаем target колонку из y (нужно синхронизировать индексы)
            # Предполагаем, что y соответствует последним len(y) строкам df
            df_lstm['target'] = 0  # Инициализируем нулями
            df_lstm.iloc[-len(y):, df_lstm.columns.get_loc('target')] = y
        
        lstm_model, lstm_metrics = lstm_trainer.train(df_lstm, validation_split=0.2)
        
        # Устанавливаем feature_names в lstm_trainer для использования при предсказании
        # Используем те же фичи, что и для классических моделей
        if hasattr(self, 'feature_engineer'):
            lstm_trainer.feature_names = self.feature_engineer.get_feature_names()
        else:
            # Если feature_engineer недоступен, извлекаем фичи из DataFrame
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']
            lstm_trainer.feature_names = [col for col in df_lstm.columns if col not in exclude_cols]
        
        # Вычисляем веса на основе CV метрик
        rf_cv_score = rf_metrics.get("cv_mean", 0.5)
        xgb_cv_score = xgb_metrics.get("cv_mean", 0.5)
        lgb_cv_score = lgb_metrics.get("cv_mean", 0.5)
        lstm_cv_score = lstm_metrics.get("accuracy", 0.5)  # LSTM использует accuracy вместо cv_mean
        
        total_score = rf_cv_score + xgb_cv_score + lgb_cv_score + lstm_cv_score
        if total_score > 0:
            rf_weight = rf_cv_score / total_score
            xgb_weight = xgb_cv_score / total_score
            lgb_weight = lgb_cv_score / total_score
            lstm_weight = lstm_cv_score / total_score
        else:
            rf_weight = xgb_weight = lgb_weight = lstm_weight = 0.25
        
        # Создаем QuadEnsemble
        ensemble = QuadEnsemble(
            rf_model, 
            xgb_model, 
            lgb_model, 
            lstm_trainer,
            rf_weight=rf_weight,
            xgb_weight=xgb_weight,
            lgb_weight=lgb_weight,
            lstm_weight=lstm_weight,
            sequence_length=lstm_sequence_length,
        )
        
        print(f"  Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}, LGB={lgb_weight:.3f}, LSTM={lstm_weight:.3f}")
        
        # Метрики ансамбля
        metrics = {
            "rf_metrics": rf_metrics,
            "xgb_metrics": xgb_metrics,
            "lgb_metrics": lgb_metrics,
            "lstm_metrics": lstm_metrics,
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
            "lgb_weight": lgb_weight,
            "lstm_weight": lstm_weight,
            "ensemble_method": "quad",
        }
        
        return ensemble, metrics

