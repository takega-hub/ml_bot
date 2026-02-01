import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Dict, Any


# Исправление проблемы с Numba кэшированием в pandas_ta
# Если NUMBA_CACHE_DIR не установлен, используем /tmp/numba_cache или домашнюю директорию
if "NUMBA_CACHE_DIR" not in os.environ:
    # Пробуем сначала /tmp/numba_cache
    cache_dir = "/tmp/numba_cache"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
    except (PermissionError, OSError):
        # Если нет прав на /tmp, используем домашнюю директорию
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".numba_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует временную метку в DatetimeIndex и сортирует данные.
    
    Args:
        df: DataFrame с колонкой 'timestamp'
    
    Returns:
        DataFrame с DatetimeIndex
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    
    # Пробуем сначала как миллисекунды, если не получилось — пусть pandas сам парсит строки
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    except (ValueError, TypeError):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def compute_4h_context(
    df_15m: pd.DataFrame, 
    adx_length: int = 14,
    df_4h: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Вычисляет ADX на 4H таймфрейме (только фильтр тренда).
    Использует готовые данные 4H если доступны, иначе ресемплит из 15m.
    Форвард-филлит обратно на 15m индекс.
    
    Args:
        df_15m: DataFrame с 15m данными
        adx_length: Период ADX
        df_4h: Опционально готовый DataFrame с 4H данными (если есть реальные свечи)
    
    Returns:
        DataFrame с добавленным ADX
    """
    # Используем готовые данные если доступны, иначе ресемплим
    if df_4h is not None and not df_4h.empty:
        # Используем готовые данные 4H (более точные)
        df_4h = df_4h.copy()
        # Убеждаемся что индекс - DatetimeIndex
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            if 'timestamp' in df_4h.columns:
                df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', errors='coerce')
                df_4h = df_4h.set_index('timestamp')
            elif 'datetime' in df_4h.columns:
                df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], errors='coerce')
                df_4h = df_4h.set_index('datetime')
    else:
        # Ресемплим из 15m (fallback)
        ohlcv = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_4h = df_15m.resample("4h").agg(ohlcv).dropna()
    
    # Вычисляем ADX с проверкой на достаточное количество данных
    # Для расчета ADX нужно минимум adx_length + несколько дополнительных свечей
    min_required_4h = adx_length + 5  # Минимум для стабильного расчета
    if len(df_4h) >= min_required_4h:
        adx = ta.adx(high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], length=adx_length)
    elif len(df_4h) >= adx_length:
        # Минимально достаточно, но может быть менее точным
        adx = ta.adx(high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], length=adx_length)
    else:
        # Если данных недостаточно, создаем пустые колонки
        adx = pd.DataFrame(index=df_4h.index)
        adx[f"ADX_{adx_length}"] = np.nan
    
    # Проверяем, что adx не None
    if adx is None:
        adx = pd.DataFrame(index=df_4h.index)
        adx[f"ADX_{adx_length}"] = np.nan
    
    # Присоединяем ADX к df_4h
    if isinstance(adx, pd.DataFrame):
        # Ищем колонку ADX
        adx_col = None
        for col in adx.columns:
            if f"ADX_{adx_length}" in str(col):
                adx_col = col
                break
        
        if adx_col:
            df_4h[f"ADX_{adx_length}"] = adx[adx_col]
        else:
            # Если не нашли конкретную колонку, берем первую
            df_4h[f"ADX_{adx_length}"] = adx.iloc[:, 0] if len(adx.columns) > 0 else np.nan
    elif isinstance(adx, pd.Series):
        df_4h[f"ADX_{adx_length}"] = adx
    else:
        # Fallback для непредвиденных случаев
        df_4h[f"ADX_{adx_length}"] = pd.Series(index=df_4h.index, dtype=float)
    
    # Маппим обратно на 15m индекс с форвард-филлом
    # Важно: используем ffill для заполнения пропусков, но только если есть хотя бы одно валидное значение
    mapped = df_4h.reindex(df_15m.index, method="ffill")
    df_15m = df_15m.copy()
    df_15m["adx"] = mapped[f"ADX_{adx_length}"]
    
    # Дополнительная проверка: если ADX все еще NaN в начале, заполняем первым валидным значением
    # Это улучшает процент валидных данных
    if df_15m["adx"].notna().any():
        first_valid_adx = df_15m["adx"].dropna().iloc[0] if len(df_15m["adx"].dropna()) > 0 else None
        if first_valid_adx is not None:
            # Заполняем начальные NaN первым валидным значением (только если их немного)
            # Это нормально для индикаторов, которые требуют периода для расчета
            nan_count = df_15m["adx"].isna().sum()
            total_count = len(df_15m)
            # Заполняем только если NaN меньше 30% от общего количества
            if nan_count > 0 and (nan_count / total_count) < 0.3:
                df_15m["adx"] = df_15m["adx"].bfill().fillna(first_valid_adx)
    
    return df_15m


def compute_atr_higher_timeframes(
    df_15m: pd.DataFrame, 
    atr_length: int = 14,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Вычисляет ATR на 1H и 4H таймфреймах для анализа среднесрочной волатильности.
    Использует готовые данные если доступны, иначе ресемплит из 15m.
    Используется вместо 15-минутного ATR для фильтрации точек входа.
    
    Args:
        df_15m: DataFrame с 15m данными
        atr_length: Период ATR
        df_1h: Опционально готовый DataFrame с 1H данными
        df_4h: Опционально готовый DataFrame с 4H данными
    
    Returns:
        DataFrame с добавленными ATR на разных таймфреймах
    """
    df_15m = df_15m.copy()
    
    # Используем готовые данные 1H если доступны, иначе ресемплим
    if df_1h is not None and not df_1h.empty:
        df_1h = df_1h.copy()
        # Убеждаемся что индекс - DatetimeIndex
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            if 'timestamp' in df_1h.columns:
                df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms', errors='coerce')
                df_1h = df_1h.set_index('timestamp')
            elif 'datetime' in df_1h.columns:
                df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], errors='coerce')
                df_1h = df_1h.set_index('datetime')
    else:
        # Resample на 1H и вычисляем ATR
        ohlcv_1h = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_1h = df_15m.resample("1h").agg(ohlcv_1h).dropna()
    
    if len(df_1h) >= atr_length:
        atr_1h = ta.atr(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], length=atr_length)
        if atr_1h is None:
            df_1h["atr_1h"] = pd.Series(index=df_1h.index, dtype=float)
        elif isinstance(atr_1h, pd.Series):
            df_1h["atr_1h"] = atr_1h
        elif isinstance(atr_1h, pd.DataFrame) and len(atr_1h.columns) > 0:
            # Ищем колонку ATR
            atr_col = None
            for col in atr_1h.columns:
                if "ATR" in str(col).upper():
                    atr_col = col
                    break
            df_1h["atr_1h"] = atr_1h[atr_col] if atr_col else atr_1h.iloc[:, 0]
        else:
            df_1h["atr_1h"] = pd.Series(index=df_1h.index, dtype=float)
    else:
        df_1h["atr_1h"] = pd.Series(index=df_1h.index, dtype=float)
    
    # Используем готовые данные 4H если доступны, иначе ресемплим
    if df_4h is not None and not df_4h.empty:
        df_4h = df_4h.copy()
        # Убеждаемся что индекс - DatetimeIndex
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            if 'timestamp' in df_4h.columns:
                df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', errors='coerce')
                df_4h = df_4h.set_index('timestamp')
            elif 'datetime' in df_4h.columns:
                df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], errors='coerce')
                df_4h = df_4h.set_index('datetime')
    else:
        # Resample на 4H и вычисляем ATR
        ohlcv_4h = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_4h = df_15m.resample("4h").agg(ohlcv_4h).dropna()
    
    if len(df_4h) >= atr_length:
        atr_4h = ta.atr(high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], length=atr_length)
        if atr_4h is None:
            df_4h["atr_4h"] = pd.Series(index=df_4h.index, dtype=float)
        elif isinstance(atr_4h, pd.Series):
            df_4h["atr_4h"] = atr_4h
        elif isinstance(atr_4h, pd.DataFrame) and len(atr_4h.columns) > 0:
            # Ищем колонку ATR
            atr_col = None
            for col in atr_4h.columns:
                if "ATR" in str(col).upper():
                    atr_col = col
                    break
            df_4h["atr_4h"] = atr_4h[atr_col] if atr_col else atr_4h.iloc[:, 0]
        else:
            df_4h["atr_4h"] = pd.Series(index=df_4h.index, dtype=float)
    else:
        df_4h["atr_4h"] = pd.Series(index=df_4h.index, dtype=float)
    
    # Map обратно на 15m индекс, используя forward fill
    mapped_1h = df_1h.reindex(df_15m.index, method="ffill")
    mapped_4h = df_4h.reindex(df_15m.index, method="ffill")
    
    # Используем среднее значение ATR с 1H и 4H
    df_15m["atr_1h"] = mapped_1h["atr_1h"]
    df_15m["atr_4h"] = mapped_4h["atr_4h"]
    
    # Среднее значение ATR для анализа среднесрочной волатильности
    df_15m["atr_avg"] = (df_15m["atr_1h"] + df_15m["atr_4h"]) / 2
    
    # Также оставляем максимальное значение для более консервативного подхода (опционально)
    df_15m["atr_max"] = df_15m[["atr_1h", "atr_4h"]].max(axis=1)
    
    return df_15m


def compute_1h_context(
    df_15m: pd.DataFrame, 
    di_length: int = 14,
    df_1h: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Вычисляет PlusDI/MinusDI на 1H таймфрейме (направление).
    Использует готовые данные 1H если доступны, иначе ресемплит из 15m.
    Форвард-филлит обратно на 15m индекс.
    
    Args:
        df_15m: DataFrame с 15m данными
        di_length: Период DI
        df_1h: Опционально готовый DataFrame с 1H данными (если есть реальные свечи)
    
    Returns:
        DataFrame с добавленными PlusDI и MinusDI
    """
    # Используем готовые данные если доступны, иначе ресемплим
    if df_1h is not None and not df_1h.empty:
        # Используем готовые данные 1H (более точные)
        df_1h = df_1h.copy()
        # Убеждаемся что индекс - DatetimeIndex
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            if 'timestamp' in df_1h.columns:
                df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms', errors='coerce')
                df_1h = df_1h.set_index('timestamp')
            elif 'datetime' in df_1h.columns:
                df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], errors='coerce')
                df_1h = df_1h.set_index('datetime')
    else:
        # Ресемплим из 15m (fallback)
        ohlcv = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_1h = df_15m.resample("1h").agg(ohlcv).dropna()
    
    # Вычисляем ADX (включая DI) с проверкой на достаточное количество данных
    # Для расчета DI нужно минимум di_length + несколько дополнительных свечей
    min_required_1h = di_length + 5  # Минимум для стабильного расчета
    if len(df_1h) >= min_required_1h:
        adx_result = ta.adx(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], length=di_length)
    elif len(df_1h) >= di_length:
        # Минимально достаточно, но может быть менее точным
        adx_result = ta.adx(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], length=di_length)
    else:
        # Если данных недостаточно, создаем пустые колонки
        adx_result = pd.DataFrame(index=df_1h.index)
        adx_result[f"DMP_{di_length}"] = np.nan
        adx_result[f"DMN_{di_length}"] = np.nan
    
    # Проверяем, что adx_result не None
    if adx_result is None:
        adx_result = pd.DataFrame(index=df_1h.index)
        adx_result[f"DMP_{di_length}"] = np.nan
        adx_result[f"DMN_{di_length}"] = np.nan
    
    # Присоединяем результат к df_1h
    if isinstance(adx_result, pd.DataFrame):
        # Ищем колонки DMP и DMN
        dmp_col = None
        dmn_col = None
        for col in adx_result.columns:
            if f"DMP_{di_length}" in str(col):
                dmp_col = col
            if f"DMN_{di_length}" in str(col):
                dmn_col = col
        
        if dmp_col:
            df_1h[f"DMP_{di_length}"] = adx_result[dmp_col]
        else:
            # Если не нашли конкретную колонку, пытаемся определить
            for col in adx_result.columns:
                if "DMP" in str(col) or "DMI_P" in str(col):
                    df_1h[f"DMP_{di_length}"] = adx_result[col]
                    break
            else:
                df_1h[f"DMP_{di_length}"] = pd.Series(index=df_1h.index, dtype=float)
        
        if dmn_col:
            df_1h[f"DMN_{di_length}"] = adx_result[dmn_col]
        else:
            # Если не нашли конкретную колонку, пытаемся определить
            for col in adx_result.columns:
                if "DMN" in str(col) or "DMI_N" in str(col):
                    df_1h[f"DMN_{di_length}"] = adx_result[col]
                    break
            else:
                df_1h[f"DMN_{di_length}"] = pd.Series(index=df_1h.index, dtype=float)
    else:
        # Fallback для непредвиденных случаев
        df_1h[f"DMP_{di_length}"] = pd.Series(index=df_1h.index, dtype=float)
        df_1h[f"DMN_{di_length}"] = pd.Series(index=df_1h.index, dtype=float)
    
    # Map обратно на 15m индекс с форвард-филлом
    mapped = df_1h.reindex(df_15m.index, method="ffill")
    df_15m = df_15m.copy()
    df_15m["plus_di"] = mapped[f"DMP_{di_length}"]
    df_15m["minus_di"] = mapped[f"DMN_{di_length}"]
    
    # Дополнительная проверка: заполняем начальные NaN первым валидным значением для DI
    # Это улучшает процент валидных данных
    for di_col in ["plus_di", "minus_di"]:
        if df_15m[di_col].notna().any():
            first_valid_di = df_15m[di_col].dropna().iloc[0] if len(df_15m[di_col].dropna()) > 0 else None
            if first_valid_di is not None:
                nan_count = df_15m[di_col].isna().sum()
                total_count = len(df_15m)
                # Заполняем только если NaN меньше 30% от общего количества
                if nan_count > 0 and (nan_count / total_count) < 0.3:
                    df_15m[di_col] = df_15m[di_col].bfill().fillna(first_valid_di)
    
    return df_15m


def compute_15m_features(
    df_15m: pd.DataFrame,
    sma_length: int = 20,
    rsi_length: int = 14,
    breakout_lookback: int = 20,
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
    ema_short: int = 9,
    ema_long: int = 21,
) -> pd.DataFrame:
    """
    Добавляет 15-минутные индикаторы, необходимые для точек входа и управления позицией.
    
    Args:
        df_15m: DataFrame с 15m данными
        sma_length: Период SMA
        rsi_length: Период RSI
        breakout_lookback: Период для поиска пробоев
        bb_length: Период Bollinger Bands
        bb_std: Стандартное отклонение для Bollinger Bands
        atr_length: Период ATR
        ema_short: Период короткой EMA
        ema_long: Период длинной EMA
    
    Returns:
        DataFrame с добавленными индикаторами
    """
    df = df_15m.copy()
    
    # SMA: используем rolling с min_periods=1 чтобы избежать NaN на коротких сериях
    df["sma"] = df["close"].rolling(window=sma_length, min_periods=1).mean()
    # Предыдущее значение SMA (sma_prev) требуется для генератора трендовых сигналов
    df["sma_prev"] = df["sma"].shift(1)
    
    # Также вычисляем трендовую EMA (быстрее) для трендовой стратегии
    try:
        df["ema_trend"] = ta.ema(df["close"], length=sma_length)
    except Exception:
        # Fallback к простой реализации EMA
        df["ema_trend"] = df["close"].ewm(span=sma_length, adjust=False).mean()
    df["ema_prev"] = df["ema_trend"].shift(1)
    
    # Вычисляем пару короткой/длинной EMA для кроссовера
    try:
        df["ema_short"] = ta.ema(df["close"], length=ema_short)
        df["ema_long"] = ta.ema(df["close"], length=ema_long)
    except Exception:
        df["ema_short"] = df["close"].ewm(span=ema_short, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=ema_long, adjust=False).mean()
    
    # RSI: пробуем pandas_ta, но гарантируем отсутствие NaN (fallback к консервативному 50)
    try:
        df["rsi"] = ta.rsi(df["close"], length=rsi_length)
    except Exception:
        df["rsi"] = pd.Series(index=df.index, dtype=float)
    
    # Проверяем результат RSI
    if isinstance(df["rsi"], pd.Series):
        df["rsi"] = df["rsi"].bfill().fillna(50.0)
    else:
        df["rsi"] = pd.Series(index=df.index, dtype=float).fillna(50.0)
    
    # Скользящие средние объема: гарантируем, что vol_sma и vol_avg5 всегда присутствуют
    df["vol_sma"] = df["volume"].rolling(window=breakout_lookback, min_periods=1).mean()
    df["vol_avg5"] = df["volume"].rolling(window=5, min_periods=1).mean()
    df["recent_high"] = df["high"].rolling(window=breakout_lookback).max().shift(1)
    df["recent_low"] = df["low"].rolling(window=breakout_lookback).min().shift(1)
    
    # ATR для определения волатильности и точек выхода
    if len(df) >= atr_length:
        atr = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_length)
    else:
        atr = pd.Series(index=df.index, dtype=float)
    
    # Обработка результата ATR
    if atr is None:
        df["atr"] = pd.Series(index=df.index, dtype=float)
    elif isinstance(atr, pd.Series):
        df["atr"] = atr
    elif isinstance(atr, pd.DataFrame) and len(atr.columns) > 0:
        # Если вернулся DataFrame, берем первую колонку или колонку с именем ATR
        atr_col = None
        for col in atr.columns:
            if "ATR" in str(col).upper():
                atr_col = col
                break
        df["atr"] = atr[atr_col] if atr_col else atr.iloc[:, 0]
    else:
        # Fallback: пытаемся преобразовать в Series
        df["atr"] = pd.Series(atr, index=df.index) if hasattr(atr, '__iter__') else pd.Series(index=df.index, dtype=float)
    
    # Bollinger Bands для флэтовой стратегии
    bb = None
    try:
        bb = ta.bbands(df["close"], length=bb_length, lower_std=bb_std, upper_std=bb_std)
    except Exception:
        bb = None
    
    if isinstance(bb, pd.DataFrame) and len(bb.columns) >= 3:
        # Пробуем стандартные имена колонок pandas_ta
        try:
            bb_col_suffix = f"_{bb_length}_{bb_std}_{bb_std}"
            upper_col = f"BBU{bb_col_suffix}"
            middle_col = f"BBM{bb_col_suffix}"
            lower_col = f"BBL{bb_col_suffix}"
            
            if upper_col in bb.columns and middle_col in bb.columns and lower_col in bb.columns:
                df["bb_upper"] = bb[upper_col]
                df["bb_middle"] = bb[middle_col]
                df["bb_lower"] = bb[lower_col]
            else:
                # Fallback: берем первые три колонки
                df["bb_upper"] = bb.iloc[:, 0]
                df["bb_middle"] = bb.iloc[:, 1]
                df["bb_lower"] = bb.iloc[:, 2]
        except Exception:
            # Резервный вариант: вычисляем вручную
            bb = None
    
    if bb is None:
        # Ручное вычисление Bollinger Bands
        mid = df["close"].rolling(window=bb_length, min_periods=1).mean()
        std = df["close"].rolling(window=bb_length, min_periods=1).std()
        df["bb_middle"] = mid
        df["bb_upper"] = mid + bb_std * std
        df["bb_lower"] = mid - bb_std * std
    
    # Гарантируем, что колонки BB не содержат NaN
    df["bb_middle"] = df["bb_middle"].bfill().fillna(df["close"].rolling(window=bb_length, min_periods=1).mean())
    df["bb_upper"] = df["bb_upper"].bfill().fillna(df["bb_middle"])
    df["bb_lower"] = df["bb_lower"].bfill().fillna(df["bb_middle"])
    
    # Ширина Bollinger Bands (относительная) — используется флэтовыми стратегиями
    mid_abs = df["bb_middle"].replace(0, np.nan).abs()
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / mid_abs).fillna(0.0)
    df["bbw"] = df["bb_width"]  # алиас, используемый в некоторых местах
    
    # MACD для анализа тренда и момента
    macd_result = None
    try:
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    except Exception:
        macd_result = None
    
    if isinstance(macd_result, pd.DataFrame):
        # Ищем правильные колонки
        macd_col = None
        signal_col = None
        hist_col = None
        
        for col in macd_result.columns:
            col_str = str(col)
            if "MACD_" in col_str and "12_26_9" in col_str and "H" not in col_str and "S" not in col_str:
                macd_col = col
            elif "MACDS" in col_str or "MACD_S" in col_str or ("MACD" in col_str and "SIGNAL" in col_str):
                signal_col = col
            elif "MACDH" in col_str or "MACD_H" in col_str or ("MACD" in col_str and "HIST" in col_str):
                hist_col = col
        
        if macd_col:
            df["macd"] = macd_result[macd_col]
        else:
            df["macd"] = macd_result.iloc[:, 0] if len(macd_result.columns) > 0 else pd.Series(index=df.index)
        
        if signal_col:
            df["macd_signal"] = macd_result[signal_col]
        else:
            df["macd_signal"] = macd_result.iloc[:, 1] if len(macd_result.columns) > 1 else pd.Series(index=df.index)
        
        if hist_col:
            df["macd_hist"] = macd_result[hist_col]
        else:
            df["macd_hist"] = macd_result.iloc[:, 2] if len(macd_result.columns) > 2 else pd.Series(index=df.index)
            
    elif isinstance(macd_result, pd.Series):
        df["macd"] = macd_result
        df["macd_signal"] = pd.Series(index=df.index, dtype=float)
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)
    else:
        # fallback: создаем пустые числовые колонки
        df["macd"] = pd.Series(index=df.index, dtype=float)
        df["macd_signal"] = pd.Series(index=df.index, dtype=float)
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)
    
    return df


def compute_vwap(df: pd.DataFrame, anchor: str = "D") -> pd.DataFrame:
    """
    Вычисляет VWAP (Volume Weighted Average Price) с ежедневным сбросом.
    
    Args:
        df: DataFrame с OHLCV данными и DatetimeIndex
        anchor: Период сброса VWAP ("D" = ежедневно, "W" = еженедельно, "M" = ежемесячно)
    
    Returns:
        DataFrame с добавленной колонкой vwap
    """
    df = df.copy()
    
    # Проверяем, что индекс - DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for VWAP calculation")
    
    # Вычисляем VWAP с помощью pandas_ta
    try:
        vwap_result = ta.vwap(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            anchor=anchor
        )
    except Exception:
        vwap_result = None
    
    if isinstance(vwap_result, pd.Series):
        df["vwap"] = vwap_result
    elif isinstance(vwap_result, pd.DataFrame):
        # Если вернулся DataFrame, берем первую колонку (обычно это VWAP_D)
        df["vwap"] = vwap_result.iloc[:, 0]
    else:
        df["vwap"] = pd.Series(index=df.index, dtype=float)
    
    return df


def compute_donchian_channels(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """
    Вычисляет Donchian Channels (каналы Дончиана).
    Показывают максимумы и минимумы за период.
    
    Args:
        df: DataFrame с OHLCV данными
        length: Период для вычисления каналов (по умолчанию 20)
    
    Returns:
        DataFrame с добавленными колонками donchian_upper, donchian_lower, donchian_middle
    """
    df = df.copy()
    
    # Donchian Channels: верхняя граница = максимум high за период, нижняя = минимум low за период
    df["donchian_upper"] = df["high"].rolling(window=length).max()
    df["donchian_lower"] = df["low"].rolling(window=length).min()
    df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2
    
    return df


def compute_ema_indicators(df: pd.DataFrame, ema_fast_length: int = 20, ema_slow_length: int = 50) -> pd.DataFrame:
    """
    Вычисляет EMA индикаторы для стратегии импульсного пробоя.
    
    Args:
        df: DataFrame с данными свечей
        ema_fast_length: Период быстрой EMA (по умолчанию 20)
        ema_slow_length: Период медленной EMA (по умолчанию 50)
    
    Returns:
        DataFrame с добавленными колонками ema_fast и ema_slow
    """
    df = df.copy()
    try:
        df["ema_fast"] = ta.ema(df["close"], length=ema_fast_length)
    except Exception:
        df["ema_fast"] = df["close"].ewm(span=ema_fast_length, adjust=False).mean()
    
    try:
        df["ema_slow"] = ta.ema(df["close"], length=ema_slow_length)
    except Exception:
        df["ema_slow"] = df["close"].ewm(span=ema_slow_length, adjust=False).mean()
    
    return df


def compute_higher_timeframe_ema(
    df_15m: pd.DataFrame,
    timeframe: str = "1h",
    ema_fast_length: int = 20,
    ema_slow_length: int = 50,
    df_htf: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Вычисляет EMA на более высоком таймфрейме (1h или 4h) для стратегии импульсного пробоя.
    Использует готовые данные если доступны, иначе ресемплит из 15m.
    Затем маппит значения обратно на 15m индекс.
    
    Args:
        df_15m: DataFrame с 15m свечами
        timeframe: Таймфрейм для вычисления EMA ("1h" или "4h")
        ema_fast_length: Период быстрой EMA
        ema_slow_length: Период медленной EMA
        df_htf: Опционально готовый DataFrame с данными высшего таймфрейма
    
    Returns:
        DataFrame с добавленными колонками ema_fast_htf и ema_slow_htf (higher timeframe)
    """
    df_15m = df_15m.copy()
    
    # Используем готовые данные если доступны, иначе ресемплим
    if df_htf is not None and not df_htf.empty:
        # Используем готовые данные (более точные)
        df_htf = df_htf.copy()
        # Убеждаемся что индекс - DatetimeIndex
        if not isinstance(df_htf.index, pd.DatetimeIndex):
            if 'timestamp' in df_htf.columns:
                df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms', errors='coerce')
                df_htf = df_htf.set_index('timestamp')
            elif 'datetime' in df_htf.columns:
                df_htf['datetime'] = pd.to_datetime(df_htf['datetime'], errors='coerce')
                df_htf = df_htf.set_index('datetime')
    else:
        # Resample на более высокий таймфрейм (fallback)
        ohlcv = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_htf = df_15m.resample(timeframe).agg(ohlcv).dropna()
    
    # Вычисляем EMA на высоком таймфрейме
    try:
        df_htf["ema_fast"] = ta.ema(df_htf["close"], length=ema_fast_length)
    except Exception:
        df_htf["ema_fast"] = df_htf["close"].ewm(span=ema_fast_length, adjust=False).mean()
    
    try:
        df_htf["ema_slow"] = ta.ema(df_htf["close"], length=ema_slow_length)
    except Exception:
        df_htf["ema_slow"] = df_htf["close"].ewm(span=ema_slow_length, adjust=False).mean()
    
    # Map обратно на 15m индекс, используя forward fill
    mapped = df_htf.reindex(df_15m.index, method="ffill")
    df_15m[f"ema_fast_{timeframe}"] = mapped["ema_fast"]
    df_15m[f"ema_slow_{timeframe}"] = mapped["ema_slow"]
    
    return df_15m


def compute_support_resistance_levels(df: pd.DataFrame, lookback: int = 20, min_touches: int = 2) -> pd.DataFrame:
    """
    Вычисляет уровни поддержки и сопротивления на основе локальных максимумов и минимумов.
    
    Args:
        df: DataFrame с OHLCV данными
        lookback: Период для поиска локальных экстремумов (по умолчанию 20)
        min_touches: Минимальное количество касаний для подтверждения уровня (по умолчанию 2)
    
    Returns:
        DataFrame с добавленными колонками поддержки/сопротивления
    """
    df = df.copy()
    
    # Упрощенный подход: используем recent_high и recent_low как базовые уровни
    # Инициализируем колонки
    df["nearest_resistance"] = np.nan
    df["nearest_support"] = np.nan
    
    # Используем recent_high и recent_low как базовые уровни сопротивления/поддержки
    if "recent_high" in df.columns:
        df["nearest_resistance"] = df["recent_high"]
    if "recent_low" in df.columns:
        df["nearest_support"] = df["recent_low"]
    
    # Дополняем уровнями из Donchian Channels (более надежные)
    if "donchian_upper" in df.columns:
        # Используем Donchian верх как сопротивление, если он выше recent_high
        df["nearest_resistance"] = df[["nearest_resistance", "donchian_upper"]].max(axis=1)
        df["donchian_resistance"] = df["donchian_upper"]
    
    if "donchian_lower" in df.columns:
        # Используем Donchian низ как поддержку, если он ниже recent_low
        df["nearest_support"] = df[["nearest_support", "donchian_lower"]].min(axis=1)
        df["donchian_support"] = df["donchian_lower"]
    
    # Дополняем уровнями из Bollinger Bands (для флэта)
    if "bb_upper" in df.columns:
        df["bb_resistance"] = df["bb_upper"]
    if "bb_lower" in df.columns:
        df["bb_support"] = df["bb_lower"]
    
    return df


def prepare_with_indicators(
    df_raw: pd.DataFrame,
    adx_length: int = 14,
    di_length: int = 14,
    sma_length: int = 20,
    rsi_length: int = 14,
    breakout_lookback: int = 20,
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
    ema_fast_length: int = 20,
    ema_slow_length: int = 50,
    ema_timeframe: str = "1h",
    ema_short: int = 9,
    ema_long: int = 21,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Основная функция для подготовки данных со всеми индикаторами.
    Поддерживает мультитаймфреймовый анализ с использованием готовых данных высших таймфреймов.
    
    Преимущества использования готовых данных (df_1h, df_4h):
    - Более точные индикаторы (используются реальные свечи с биржи вместо ресемплинга)
    - Лучшая синхронизация с данными биржи
    - Более надежный анализ тренда на высших таймфреймах
    
    Если готовые данные не переданы, используется ресемплинг из 15m (обратная совместимость).
    
    Args:
        df_raw: Исходный DataFrame с сырыми данными (15m)
        adx_length: Период ADX на 4H
        di_length: Период DI на 1H
        sma_length: Период SMA на 15m
        rsi_length: Период RSI на 15m
        breakout_lookback: Период для поиска пробоев
        bb_length: Период Bollinger Bands
        bb_std: Стандартное отклонение для Bollinger Bands
        atr_length: Период ATR
        ema_fast_length: Период быстрой EMA на высоком таймфрейме
        ema_slow_length: Период медленной EMA на высоком таймфрейме
        ema_timeframe: Таймфрейм для EMA высокого уровня
        ema_short: Период короткой EMA на 15m
        ema_long: Период длинной EMA на 15m
        df_1h: Опционально готовый DataFrame с 1H данными (для более точного анализа).
               Должен иметь DatetimeIndex или колонки 'timestamp'/'datetime'.
        df_4h: Опционально готовый DataFrame с 4H данными (для более точного анализа).
               Должен иметь DatetimeIndex или колонки 'timestamp'/'datetime'.
    
    Returns:
        DataFrame со всеми вычисленными индикаторами
    """
    # Базовые преобразования
    df = add_time_index(df_raw)
    
    # Контекстные индикаторы на высоких таймфреймах
    # Используем готовые данные если доступны (более точные), иначе ресемплим
    df = compute_4h_context(df, adx_length=adx_length, df_4h=df_4h)  # ADX на 4H для фильтра тренда
    df = compute_1h_context(df, di_length=di_length, df_1h=df_1h)    # DI на 1H для направления
    
    # 15-минутные индикаторы
    df = compute_15m_features(
        df,
        sma_length=sma_length,
        rsi_length=rsi_length,
        breakout_lookback=breakout_lookback,
        bb_length=bb_length,
        bb_std=bb_std,
        atr_length=atr_length,
        ema_short=ema_short,
        ema_long=ema_long,
    )
    
    # ATR на высоких таймфреймах
    # Используем готовые данные если доступны
    df = compute_atr_higher_timeframes(df, atr_length=atr_length, df_1h=df_1h, df_4h=df_4h)
    
    # EMA на высоком таймфрейме
    # Выбираем соответствующий DataFrame для высшего таймфрейма
    df_htf_for_ema = df_1h if ema_timeframe == "1h" else (df_4h if ema_timeframe == "4h" else None)
    df = compute_higher_timeframe_ema(
        df, 
        timeframe=ema_timeframe, 
        ema_fast_length=ema_fast_length, 
        ema_slow_length=ema_slow_length,
        df_htf=df_htf_for_ema
    )
    
    # Дополнительные индикаторы
    df = compute_vwap(df, anchor="D")  # VWAP с ежедневным сбросом
    df = compute_donchian_channels(df, length=20)  # Каналы Дончиана
    
    # Уровни поддержки и сопротивления
    df = compute_support_resistance_levels(df, lookback=breakout_lookback, min_touches=2)
    
    # Очистка данных
    key_columns = ["open", "high", "low", "close", "volume"]
    if all(col in df.columns for col in key_columns):
        # Удаляем строки, где все основные колонки NaN
        df = df[df[key_columns].notna().any(axis=1)]
    
    # Удаляем только строки, где все значения NaN
    df = df.dropna(how='all')
    
    # Если после обработки DataFrame пуст, пытаемся вернуть fallback данные
    if len(df) == 0:
        print(f"[indicators] ⚠️ Warning: All rows removed after processing")
        # Попробуем вернуть исходные данные с минимальной обработкой
        df_fallback = add_time_index(df_raw)
        if all(col in df_fallback.columns for col in key_columns):
            df_fallback = df_fallback[df_fallback[key_columns].notna().any(axis=1)]
        if len(df_fallback) > 0:
            print(f"[indicators] ⚠️ Returning fallback data with {len(df_fallback)} rows")
            return df_fallback
    
    return df