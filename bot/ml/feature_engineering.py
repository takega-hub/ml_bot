"""
Модуль для создания фичей (признаков) из исторических данных для ML-моделей.
ИСПРАВЛЕННАЯ ВЕРСИЯ с защитой от ошибок в pandas_ta.
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import warnings

# Подавляем предупреждения
warnings.filterwarnings('ignore')

import pandas_ta as ta


class FeatureEngineer:
    """
    Создает технические индикаторы и другие фичи из OHLCV данных.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def safe_ta_indicator(self, df: pd.DataFrame, indicator_func, **kwargs):
        """Безопасное вычисление индикатора с обработкой ошибок."""
        try:
            result = indicator_func(**kwargs)
            if result is None:
                return None
            return result
        except Exception as e:
            print(f"[WARNING] Индикатор {indicator_func.__name__} не сработал: {e}")
            return None
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает технические индикаторы из OHLCV данных.
        Упрощенная и оптимизированная версия с защитой от ошибок.
        """
        if df.empty or df is None:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Проверяем необходимые колонки
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                print(f"[ERROR] Отсутствует колонка {col} в данных")
                return pd.DataFrame()
        
        # Устанавливаем timestamp как индекс если он есть
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        # === 1. ПРОСТЫЕ ИНДИКАТОРЫ (гарантированно работают) ===
        
        # Moving Averages
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["ema_12"] = ta.ema(df["close"], length=12)
        df["ema_26"] = ta.ema(df["close"], length=26)
        
        # RSI
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # ATR
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100
        
        # Volume features
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = np.where(
            df["volume_sma_20"] > 0,
            df["volume"] / df["volume_sma_20"],
            1.0
        )
        
        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()
        
        # === 2. ИНДИКАТОРЫ С ЗАЩИТОЙ ОТ ОШИБОК ===
        
        # Bollinger Bands (с защитой от разных имен колонок)
        try:
            bb_result = ta.bbands(df["close"], length=20, std=2)
            if bb_result is not None and not bb_result.empty:
                # Проверяем возможные имена колонок
                possible_names = {
                    'upper': ['BBU_20_2.0', 'BBU_20_2', 'BB_upper', 'upper'],
                    'middle': ['BBM_20_2.0', 'BBM_20_2', 'BB_middle', 'middle'],
                    'lower': ['BBL_20_2.0', 'BBL_20_2', 'BB_lower', 'lower']
                }
                
                for band, names in possible_names.items():
                    for name in names:
                        if name in bb_result.columns:
                            if band == 'upper':
                                df["bb_upper"] = bb_result[name]
                            elif band == 'middle':
                                df["bb_middle"] = bb_result[name]
                            elif band == 'lower':
                                df["bb_lower"] = bb_result[name]
                            break
                
                # Если все еще нет, создаем простые
                if "bb_upper" not in df.columns and len(bb_result.columns) >= 3:
                    df["bb_upper"] = bb_result.iloc[:, 0] if len(bb_result.columns) > 0 else df["close"]
                    df["bb_middle"] = bb_result.iloc[:, 1] if len(bb_result.columns) > 1 else df["close"]
                    df["bb_lower"] = bb_result.iloc[:, 2] if len(bb_result.columns) > 2 else df["close"]
        except Exception as e:
            print(f"[WARNING] Bollinger Bands не сработали: {e}")
            df["bb_upper"] = df["close"]
            df["bb_middle"] = df["close"]
            df["bb_lower"] = df["close"]
        
        # MACD
        try:
            macd_result = ta.macd(df["close"])
            if macd_result is not None:
                df["macd"] = macd_result.iloc[:, 0] if len(macd_result.columns) > 0 else 0
                df["macd_signal"] = macd_result.iloc[:, 1] if len(macd_result.columns) > 1 else 0
        except:
            df["macd"] = 0
            df["macd_signal"] = 0
        
        # ADX
        try:
            adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx_result is not None:
                df["adx"] = adx_result.iloc[:, 0] if len(adx_result.columns) > 0 else 0
        except:
            df["adx"] = 0
        
        # === 3. БАЗОВЫЕ ФИЧИ ===
        
        # Лаговые фичи
        for lag in [1, 2, 3]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            df[f"price_change_lag_{lag}"] = df["price_change"].shift(lag)
        
        # Волатильность
        df["volatility_10"] = df["close"].rolling(window=10, min_periods=3).std()
        
        # Дистанция до MA
        df["dist_to_sma20_pct"] = ((df["close"] - df["sma_20"]) / df["sma_20"]) * 100
        df["dist_to_ema12_pct"] = ((df["close"] - df["ema_12"]) / df["ema_12"]) * 100
        
        # === 4. ВРЕМЕННЫЕ ФИЧИ ===
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            # Циклические фичи
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        
        # === 5. ВЗАИМОДЕЙСТВИЯ ИНДИКАТОРОВ ===
        
        # RSI уровни
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        
        # Тренд по MA
        df["ema12_above_ema26"] = (df["ema_12"] > df["ema_26"]).astype(int)
        
        # ББ положение
        if all(col in df.columns for col in ["bb_upper", "bb_lower", "close"]):
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1)
            df["near_bb_upper"] = (df["close"] > df["bb_upper"] * 0.95).astype(int)
            df["near_bb_lower"] = (df["close"] < df["bb_lower"] * 1.05).astype(int)
        
        # === 6. ОБРАБОТКА NaN ===
        
        # Сначала forward fill, потом backward fill
        df = df.ffill().bfill()
        
        # Заполняем оставшиеся NaN нулями
        df = df.fillna(0)
        
        # Удаляем строки где основные цены NaN
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols, how='any')
        
        # Сохраняем имена фичей
        original_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        self.feature_names = [col for col in df.columns if col not in original_cols]
        
        print(f"[INFO] Создано {len(self.feature_names)} фичей")
        
        return df
    
    def add_mtf_features(
        self,
        df_features: pd.DataFrame,
        higher_timeframes: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Упрощенная версия добавления MTF фичей.
        """
        if not higher_timeframes or df_features is None or df_features.empty:
            return df_features
        
        df = df_features.copy()
        
        # Базовые фичи для MTF (гарантированно работающие)
        mtf_features = ["rsi", "atr_pct", "adx", "volume_ratio"]
        
        for tf_name, htf_df in higher_timeframes.items():
            if htf_df is None or htf_df.empty:
                continue
            
            try:
                # Вычисляем фичи для HTF
                fe_htf = FeatureEngineer()
                htf_with_features = fe_htf.create_technical_indicators(htf_df)
                
                if htf_with_features is None or htf_with_features.empty:
                    continue
                
                # Выбираем нужные фичи
                for feature in mtf_features:
                    if feature in htf_with_features.columns:
                        col_name = f"{feature}_{tf_name}"
                        htf_series = htf_with_features[feature]
                        
                        # Ресемплируем на базовый ТФ
                        if isinstance(df.index, pd.DatetimeIndex) and isinstance(htf_series.index, pd.DatetimeIndex):
                            # Переиндексируем с forward fill
                            htf_aligned = htf_series.reindex(df.index, method='ffill')
                            df[col_name] = htf_aligned
                        else:
                            # Просто берем значения
                            df[col_name] = htf_series.values[:len(df)] if len(htf_series) >= len(df) else 0
                        
                        if col_name not in self.feature_names:
                            self.feature_names.append(col_name)
                            
            except Exception as e:
                print(f"[WARNING] Ошибка при добавлении MTF фичей для {tf_name}: {e}")
                continue
        
        # Заполняем NaN
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_periods: int = 4,  # 4 * 15m = 1 час
        threshold_pct: float = 0.5,
        use_atr_threshold: bool = True,
        use_risk_adjusted: bool = False,  # ОТКЛЮЧЕНО для больше сигналов
        min_risk_reward_ratio: float = 1.5,
        max_hold_periods: int = 96,  # 24 часа
        min_profit_pct: float = 0.3,
    ) -> pd.DataFrame:
        """
        ИСПРАВЛЕННАЯ версия создания целевой переменной.
        """
        if df is None or df.empty or "close" not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 1. Базовое вычисление будущей цены
        current_price = df["close"].values
        future_idx = min(forward_periods, len(df) - 1)
        
        # Создаем массив будущих цен
        future_price = np.zeros_like(current_price)
        for i in range(len(current_price)):
            if i + forward_periods < len(current_price):
                future_price[i] = current_price[i + forward_periods]
            else:
                future_price[i] = current_price[-1]  # Последняя известная цена
        
        # 2. Процентное изменение
        with np.errstate(divide='ignore', invalid='ignore'):
            price_change_pct = np.where(
                current_price > 0,
                (future_price - current_price) / current_price * 100,
                0
            )
        
        # 3. Динамический порог на основе ATR
        if use_atr_threshold and "atr_pct" in df.columns:
            atr_pct = df["atr_pct"].values
            dynamic_threshold = np.minimum(threshold_pct, atr_pct * 0.8)
        else:
            dynamic_threshold = np.full(len(df), threshold_pct)
        
        # 4. Классификация (ПРОСТАЯ)
        target = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - forward_periods):
            change = price_change_pct[i]
            threshold = dynamic_threshold[i]
            
            # LONG: прибыль больше порога И больше минимальной прибыли
            if change > threshold and change >= min_profit_pct:
                target[i] = 1
            # SHORT: убыток больше порога И больше минимальной прибыли
            elif change < -threshold and abs(change) >= min_profit_pct:
                target[i] = -1
        
        df["target"] = target
        
        # 5. Удаляем последние forward_periods строк (где нет будущей цены)
        if len(df) > forward_periods:
            df = df.iloc[:-forward_periods]
        
        # 6. Анализ распределения
        unique, counts = np.unique(target, return_counts=True)
        print(f"[TARGET] Распределение классов:")
        for val, cnt in zip(unique, counts):
            pct = cnt / len(target) * 100
            name = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(val, f"UNK({val})")
            print(f"  {name}: {cnt} ({pct:.1f}%)")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список названий всех созданных фичей."""
        return self.feature_names.copy()
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения ML-модели.
        """
        if df is None or df.empty:
            return np.array([]), np.array([])
        
        # Выбираем только фичи (исключаем исходные колонки и target)
        exclude_cols = ["open", "high", "low", "close", "volume", "timestamp", "target"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Если нет фичей, создаем простые
        if not feature_cols:
            feature_cols = ["sma_20", "rsi", "atr_pct", "price_change"]
        
        X = df[feature_cols].values if feature_cols else np.zeros((len(df), 1))
        y = df["target"].values if "target" in df.columns else np.zeros(len(df))
        
        print(f"[ML PREP] X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y