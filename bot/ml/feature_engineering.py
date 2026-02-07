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
        
        # RSI уровни (очищаем от None перед сравнением)
        rsi_clean = pd.to_numeric(df["rsi"], errors='coerce').fillna(50.0)
        df["rsi_oversold"] = (rsi_clean < 30).astype(int)
        df["rsi_overbought"] = (rsi_clean > 70).astype(int)
        
        # Тренд по MA (очищаем от None перед сравнением)
        ema12_clean = pd.to_numeric(df["ema_12"], errors='coerce').fillna(0.0)
        ema26_clean = pd.to_numeric(df["ema_26"], errors='coerce').fillna(0.0)
        df["ema12_above_ema26"] = (ema12_clean > ema26_clean).astype(int)
        
        # ББ положение (очищаем от None перед сравнением)
        if all(col in df.columns for col in ["bb_upper", "bb_lower", "close"]):
            bb_upper_clean = pd.to_numeric(df["bb_upper"], errors='coerce').fillna(df["close"])
            bb_lower_clean = pd.to_numeric(df["bb_lower"], errors='coerce').fillna(df["close"])
            close_clean = pd.to_numeric(df["close"], errors='coerce').fillna(0.0)
            
            bb_range = (bb_upper_clean - bb_lower_clean).replace(0, 1)
            df["bb_position"] = (close_clean - bb_lower_clean) / bb_range
            df["near_bb_upper"] = (close_clean > bb_upper_clean * 0.95).astype(int)
            df["near_bb_lower"] = (close_clean < bb_lower_clean * 1.05).astype(int)
        
        # === 6. НОВЫЕ ФИЧИ: ВОЛАТИЛЬНОСТЬ ===
        
        # Realized volatility (rolling std returns)
        df["realized_volatility_10"] = df["price_change"].rolling(window=10, min_periods=3).std()
        df["realized_volatility_20"] = df["price_change"].rolling(window=20, min_periods=5).std()
        
        # Parkinson volatility (high-low based)
        df["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"])) ** 2)
        df["parkinson_vol_pct"] = (df["parkinson_vol"] / df["close"]) * 100
        
        # Volatility ratio (short-term / long-term)
        vol10_clean = pd.to_numeric(df["realized_volatility_10"], errors='coerce').fillna(0.0)
        vol20_clean = pd.to_numeric(df["realized_volatility_20"], errors='coerce').fillna(0.0)
        df["volatility_ratio"] = np.where(
            vol20_clean > 0,
            vol10_clean / vol20_clean,
            1.0
        )
        
        # === 7. НОВЫЕ ФИЧИ: МИКРОСТРУКТУРА РЫНКА ===
        
        # Bid-ask spread proxy (high - low) / close
        df["spread_proxy"] = ((df["high"] - df["low"]) / df["close"]) * 100
        
        # Volume imbalance за последние N свечей
        for window in [5, 10]:
            df[f"volume_imbalance_{window}"] = (
                df["volume"].rolling(window=window, min_periods=2).apply(
                    lambda x: (x.iloc[-1] - x.mean()) / x.mean() if x.mean() > 0 else 0
                )
            )
        
        # Price momentum на разных таймфреймах
        for period in [3, 5, 10]:
            df[f"momentum_{period}"] = df["close"].pct_change(periods=period)
        
        # === 8. НОВЫЕ ФИЧИ: ТРЕНДОВЫЕ ===
        
        # ADX + DI направление (если доступно)
        try:
            # Сначала убеждаемся, что входные данные не содержат None
            high_clean = df["high"].fillna(0.0).replace([None], 0.0)
            low_clean = df["low"].fillna(0.0).replace([None], 0.0)
            close_clean = df["close"].fillna(0.0).replace([None], 0.0)
            
            adx_full = ta.adx(high_clean, low_clean, close_clean, length=14)
            if adx_full is not None and len(adx_full.columns) >= 3:
                di_plus_raw = adx_full.iloc[:, 1] if len(adx_full.columns) > 1 else pd.Series([0.0] * len(df), index=df.index)
                di_minus_raw = adx_full.iloc[:, 2] if len(adx_full.columns) > 2 else pd.Series([0.0] * len(df), index=df.index)
                
                # Заполняем и NaN, и None перед сравнением
                df["di_plus"] = di_plus_raw.fillna(0.0).replace([None], 0.0)
                df["di_minus"] = di_minus_raw.fillna(0.0).replace([None], 0.0)
                
                # Убеждаемся, что это числовые значения (не None)
                df["di_plus"] = pd.to_numeric(df["di_plus"], errors='coerce').fillna(0.0)
                df["di_minus"] = pd.to_numeric(df["di_minus"], errors='coerce').fillna(0.0)
                
                # Теперь безопасное сравнение
                df["adx_trend_up"] = (df["di_plus"] > df["di_minus"]).astype(int)
            else:
                df["di_plus"] = 0.0
                df["di_minus"] = 0.0
                df["adx_trend_up"] = 0
        except Exception as e:
            print(f"[WARNING] Ошибка при создании ADX фичей: {e}")
            df["di_plus"] = 0.0
            df["di_minus"] = 0.0
            df["adx_trend_up"] = 0
        
        # === 9. НОВЫЕ ФИЧИ: ПАТТЕРНЫ СВЕЧЕЙ ===
        
        # ВАЖНО: Очищаем OHLCV от None перед вычислениями
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).replace([None], 0.0)
        
        # Body и wick размеры
        df["body_size"] = abs(df["close"] - df["open"]).fillna(0.0)
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)).fillna(0.0)
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]).fillna(0.0)
        df["total_range"] = (df["high"] - df["low"]).fillna(0.0)
        
        # Защита от деления на ноль
        df["total_range"] = df["total_range"].replace(0.0, 1.0)  # Минимум 1.0 для избежания деления на 0
        
        # Body/wick ratio
        wick_sum = (df["upper_wick"] + df["lower_wick"]).fillna(0.0)
        df["body_wick_ratio"] = np.where(
            wick_sum > 0,
            df["body_size"] / wick_sum,
            0.0
        )
        
        # Doji (body < 20% от range)
        range_ratio = (df["body_size"] / df["total_range"]).fillna(0.0)
        df["is_doji"] = (range_ratio < 0.2).astype(int)
        
        # Hammer (маленький body, длинная нижняя тень, короткая верхняя)
        body_size_clean = df["body_size"].fillna(0.0)
        lower_wick_clean = df["lower_wick"].fillna(0.0)
        upper_wick_clean = df["upper_wick"].fillna(0.0)
        df["is_hammer"] = (
            (range_ratio < 0.3) &
            (lower_wick_clean > body_size_clean * 2) &
            (upper_wick_clean < body_size_clean)
        ).astype(int)
        
        # Shooting Star (маленький body, длинная верхняя тень, короткая нижняя)
        df["is_shooting_star"] = (
            (range_ratio < 0.3) &
            (upper_wick_clean > body_size_clean * 2) &
            (lower_wick_clean < body_size_clean)
        ).astype(int)
        
        # Bullish/Bearish Engulfing
        close_clean = df["close"].fillna(0.0)
        open_clean = df["open"].fillna(0.0)
        close_prev = df["close"].shift(1).fillna(0.0)
        open_prev = df["open"].shift(1).fillna(0.0)
        
        df["is_bullish_engulfing"] = (
            (close_clean > open_clean) &  # Текущая свеча бычья
            (close_prev < open_prev) &  # Предыдущая медвежья
            (open_clean < close_prev) &  # Текущий open ниже предыдущего close
            (close_clean > open_prev)  # Текущий close выше предыдущего open
        ).astype(int)
        
        df["is_bearish_engulfing"] = (
            (close_clean < open_clean) &  # Текущая свеча медвежья
            (close_prev > open_prev) &  # Предыдущая бычья
            (open_clean > close_prev) &  # Текущий open выше предыдущего close
            (close_clean < open_prev)  # Текущий close ниже предыдущего open
        ).astype(int)
        
        # === 10. НОВЫЕ ФИЧИ: УРОВНИ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ ===
        
        # Расстояние до nearest S/R level (упрощенная версия)
        # Используем локальные минимумы/максимумы как S/R
        lookback = 20
        df["local_low"] = df["low"].rolling(window=lookback, min_periods=5).min()
        df["local_high"] = df["high"].rolling(window=lookback, min_periods=5).max()
        
        # Расстояние до поддержки (в %)
        local_low_clean = pd.to_numeric(df["local_low"], errors='coerce').fillna(0.0)
        close_clean = pd.to_numeric(df["close"], errors='coerce').fillna(0.0)
        df["dist_to_support_pct"] = np.where(
            (local_low_clean > 0) & (close_clean > 0),
            ((close_clean - local_low_clean) / close_clean) * 100,
            0.0
        )
        
        # Расстояние до сопротивления (в %)
        local_high_clean = pd.to_numeric(df["local_high"], errors='coerce').fillna(0.0)
        df["dist_to_resistance_pct"] = np.where(
            (local_high_clean > 0) & (close_clean > 0),
            ((local_high_clean - close_clean) / close_clean) * 100,
            0.0
        )
        
        # Расстояние до S/R в ATR
        atr_clean = pd.to_numeric(df["atr"], errors='coerce').fillna(0.0)
        df["dist_to_support_atr"] = np.where(
            atr_clean > 0,
            (close_clean - local_low_clean) / atr_clean,
            0.0
        )
        df["dist_to_resistance_atr"] = np.where(
            atr_clean > 0,
            (local_high_clean - close_clean) / atr_clean,
            0.0
        )
        
        # === УЛУЧШЕННЫЕ S/R ФИЧИ: Количество касаний и сила уровня ===
        # Оптимизированная версия с использованием векторных операций
        lookback_sr = 50  # Окно для поиска S/R уровней
        tolerance_pct = 0.5  # Допуск для касания уровня (0.5%)
        
        # Инициализируем фичи
        df["support_touches"] = 0.0
        df["resistance_touches"] = 0.0
        df["support_strength"] = 0.0
        df["resistance_strength"] = 0.0
        
        # Векторные вычисления для касаний
        low_clean = pd.to_numeric(df["low"], errors='coerce').fillna(0.0)
        high_clean = pd.to_numeric(df["high"], errors='coerce').fillna(0.0)
        close_clean = pd.to_numeric(df["close"], errors='coerce').fillna(0.0)
        
        # Вычисляем касания для каждого бара (векторно)
        for i in range(lookback_sr, len(df)):
            window_start = max(0, i - lookback_sr)
            window_end = i + 1
            
            current_support = local_low_clean.iloc[i]
            current_resistance = local_high_clean.iloc[i]
            
            if current_support > 0 and current_resistance > 0:
                # Толеранс для касаний
                support_tolerance = current_support * (tolerance_pct / 100)
                resistance_tolerance = current_resistance * (tolerance_pct / 100)
                
                # Окно данных
                window_low = low_clean.iloc[window_start:window_end]
                window_high = high_clean.iloc[window_start:window_end]
                window_close = close_clean.iloc[window_start:window_end]
                
                # Касания поддержки (low в пределах tolerance)
                support_touches_mask = (
                    (window_low >= (current_support - support_tolerance)) & 
                    (window_low <= (current_support + support_tolerance))
                )
                support_touches = support_touches_mask.sum()
                
                # Касания сопротивления (high в пределах tolerance)
                resistance_touches_mask = (
                    (window_high >= (current_resistance - resistance_tolerance)) & 
                    (window_high <= (current_resistance + resistance_tolerance))
                )
                resistance_touches = resistance_touches_mask.sum()
                
                # Сила поддержки = отскоки (касание + цена пошла вверх)
                if support_touches > 0:
                    # Находим индексы касаний в окне
                    touch_indices = window_low[support_touches_mask].index
                    support_bounces = 0
                    for touch_idx in touch_indices:
                        # Проверяем, пошла ли цена вверх после касания
                        touch_pos = df.index.get_loc(touch_idx)
                        if touch_pos < len(df) - 1:
                            if close_clean.iloc[touch_pos + 1] > close_clean.iloc[touch_pos]:
                                support_bounces += 1
                else:
                    support_bounces = 0
                
                # Сила сопротивления = отскоки (касание + цена пошла вниз)
                if resistance_touches > 0:
                    # Находим индексы касаний в окне
                    touch_indices = window_high[resistance_touches_mask].index
                    resistance_bounces = 0
                    for touch_idx in touch_indices:
                        # Проверяем, пошла ли цена вниз после касания
                        touch_pos = df.index.get_loc(touch_idx)
                        if touch_pos < len(df) - 1:
                            if close_clean.iloc[touch_pos + 1] < close_clean.iloc[touch_pos]:
                                resistance_bounces += 1
                else:
                    resistance_bounces = 0
                
                df.loc[df.index[i], "support_touches"] = float(support_touches)
                df.loc[df.index[i], "resistance_touches"] = float(resistance_touches)
                df.loc[df.index[i], "support_strength"] = float(support_bounces)
                df.loc[df.index[i], "resistance_strength"] = float(resistance_bounces)
        
        # Нормализуем силу (относительно количества касаний)
        support_touches_clean = pd.to_numeric(df["support_touches"], errors='coerce').fillna(0.0)
        resistance_touches_clean = pd.to_numeric(df["resistance_touches"], errors='coerce').fillna(0.0)
        support_strength_clean = pd.to_numeric(df["support_strength"], errors='coerce').fillna(0.0)
        resistance_strength_clean = pd.to_numeric(df["resistance_strength"], errors='coerce').fillna(0.0)
        
        # Сила = процент успешных отскоков от общего количества касаний
        df["support_strength_ratio"] = np.where(
            support_touches_clean > 0,
            support_strength_clean / support_touches_clean,
            0.0
        )
        df["resistance_strength_ratio"] = np.where(
            resistance_touches_clean > 0,
            resistance_strength_clean / resistance_touches_clean,
            0.0
        )
        
        # Добавляем в список фичей
        self.feature_names.extend([
            "support_touches", "resistance_touches",
            "support_strength", "resistance_strength",
            "support_strength_ratio", "resistance_strength_ratio"
        ])
        
        # === 11. ОБРАБОТКА NaN ===
        
        # Сначала forward fill, потом backward fill
        df = df.ffill().bfill()
        
        # Заполняем оставшиеся NaN нулями (с явным указанием downcast=None для избежания предупреждений)
        df = df.fillna(0, downcast=None)
        # Восстанавливаем типы после fillna
        df = df.infer_objects(copy=False)
        
        # Удаляем строки где основные цены NaN
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols, how='any')
        
        # Сохраняем имена фичей
        original_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        self.feature_names = [col for col in df.columns if col not in original_cols]
        
        # Логирование отключено для производительности (вызывается тысячи раз в бэктесте)
        # Раскомментируйте для отладки:
        # if not hasattr(self, '_features_logged'):
        #     print(f"[INFO] Создано {len(self.feature_names)} фичей (включая новые: волатильность, микроструктура, паттерны, S/R)")
        #     self._features_logged = True
        
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
                # ВАЖНО: Сначала проверяем и очищаем входные данные от None
                htf_df_clean = htf_df.copy()
                # Заполняем все None/NaN в OHLCV перед созданием фичей
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in htf_df_clean.columns:
                        htf_df_clean[col] = htf_df_clean[col].fillna(0.0)
                        # Заменяем None на 0.0
                        htf_df_clean[col] = htf_df_clean[col].replace([None], 0.0)
                
                fe_htf = FeatureEngineer()
                htf_with_features = fe_htf.create_technical_indicators(htf_df_clean)
                
                if htf_with_features is None or htf_with_features.empty:
                    continue
                
                # Дополнительная очистка: заменяем все None на 0.0 во всех колонках
                htf_with_features = htf_with_features.fillna(0.0)
                for col in htf_with_features.columns:
                    htf_with_features[col] = htf_with_features[col].replace([None], 0.0)
                
                # Выбираем нужные фичи
                for feature in mtf_features:
                    if feature in htf_with_features.columns:
                        col_name = f"{feature}_{tf_name}"
                        htf_series = htf_with_features[feature].copy()
                        
                        # Заполняем NaN перед использованием
                        htf_series = htf_series.fillna(0.0)
                        
                        # Ресемплируем на базовый ТФ
                        if isinstance(df.index, pd.DatetimeIndex) and isinstance(htf_series.index, pd.DatetimeIndex):
                            # Переиндексируем с forward fill
                            try:
                                htf_aligned = htf_series.reindex(df.index, method='ffill')
                                # Заполняем оставшиеся NaN
                                htf_aligned = htf_aligned.fillna(0.0)
                                df[col_name] = htf_aligned
                            except Exception as e:
                                print(f"[WARNING] Ошибка при реиндексации {col_name}: {e}")
                                # Fallback: просто заполняем нулями
                                df[col_name] = 0.0
                        else:
                            # Просто берем значения
                            if len(htf_series) >= len(df):
                                df[col_name] = htf_series.values[:len(df)]
                            else:
                                # Дополняем нулями если не хватает данных
                                values = list(htf_series.values) + [0.0] * (len(df) - len(htf_series))
                                df[col_name] = values[:len(df)]
                        
                        # Заполняем финальные NaN
                        df[col_name] = df[col_name].fillna(0.0)
                        
                        if col_name not in self.feature_names:
                            self.feature_names.append(col_name)
                            
            except Exception as e:
                print(f"[WARNING] Ошибка при добавлении MTF фичей для {tf_name}: {e}")
                continue
        
        # Заполняем NaN (с явным указанием downcast=None для избежания предупреждений)
        df = df.ffill().bfill().fillna(0, downcast=None)
        df = df.infer_objects(copy=False)
        
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