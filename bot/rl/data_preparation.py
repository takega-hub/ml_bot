"""
Подготовка данных для PPO обучения: MTF фичи + уровни S/R.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging
import glob

from bot.ml.feature_engineering import FeatureEngineer
from bot.indicators import (
    compute_support_resistance_levels,
    compute_donchian_channels,
    prepare_with_indicators,
)

logger = logging.getLogger(__name__)


def prepare_ppo_data(
    csv_path: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "15",
    enable_mtf: bool = True,
) -> pd.DataFrame:
    """
    Подготавливает данные для PPO обучения.
    
    Args:
        csv_path: Путь к CSV файлу с 15m данными
        symbol: Символ (для логирования)
        timeframe: Таймфрейм (15, 60, 240)
        enable_mtf: Включить MTF фичи (1h/4h)
    
    Returns:
        DataFrame с OHLCV, техническими индикаторами, MTF фичами и уровнями S/R
    """
    logger.info(f"Loading data from {csv_path}...")
    
    # Загружаем данные (поддержка glob масок)
    if any(ch in csv_path for ch in ["*", "?", "[", "]"]):
        matched_files = sorted(glob.glob(csv_path))
        if not matched_files:
            raise FileNotFoundError(f"No CSV files matched pattern: {csv_path}")
        df_list = []
        for file_path in matched_files:
            df_file = pd.read_csv(file_path)
            # Если CSV с разделителем ';' (часто в выгрузках), пытаемся перечитать
            if df_file.shape[1] == 1:
                df_file_alt = pd.read_csv(file_path, sep=";")
                if df_file_alt.shape[1] > df_file.shape[1]:
                    df_file = df_file_alt
            df_list.append(df_file)
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 1:
            df_alt = pd.read_csv(csv_path, sep=";")
            if df_alt.shape[1] > df.shape[1]:
                df = df_alt

    # Нормализуем названия колонок
    if df is not None and len(df.columns) > 0:
        df.columns = [str(c).strip().lower() for c in df.columns]

    # Если данные пустые — прекращаем с понятной ошибкой
    if df is None or df.empty:
        raise ValueError(
            f"No rows loaded from {csv_path}. "
            "Check file pattern, delimiter, and file contents."
        )
    
    # Преобразуем timestamp
    if "timestamp" in df.columns:
        # Если timestamp уже в строковом формате, парсим без unit
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    
    # Удаляем строки с NaT в индексе перед ресемплом
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.isna()]

    # Сортируем по времени
    df = df.sort_index()

    # Дедупликация по timestamp (оставляем последнюю запись)
    if isinstance(df.index, pd.DatetimeIndex):
        dup_count = df.index.duplicated(keep="last").sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicated timestamps; keeping last occurrence")
            df = df[~df.index.duplicated(keep="last")]
    
    # Проверяем необходимые колонки
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    
    # 1. Создаем базовые технические индикаторы на 15m
    logger.info("Creating 15m technical indicators...")
    fe = FeatureEngineer()
    df = fe.create_technical_indicators(df)
    
    # 2. Добавляем MTF фичи (1h/4h)
    if enable_mtf:
        logger.info("Adding MTF features (1h/4h)...")
        
        # Resample на 1h и 4h
        ohlcv_agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        
        df_1h = df[required_cols].resample("60min").agg(ohlcv_agg).dropna()
        df_4h = df[required_cols].resample("240min").agg(ohlcv_agg).dropna()
        
        higher_timeframes = {}
        if not df_1h.empty:
            higher_timeframes["60"] = df_1h
        if not df_4h.empty:
            higher_timeframes["240"] = df_4h
        
        if higher_timeframes:
            df = fe.add_mtf_features(df, higher_timeframes)
            logger.info(f"Added MTF features. Total columns: {len(df.columns)}")

        # Добавляем EMA тренд на 1h как фичу для бонуса
        try:
            df = _add_htf_ema_trend(df, df_1h, timeframe_suffix="1h", ema_fast=20, ema_slow=50)
        except Exception as ema_err:
            logger.warning(f"Failed to add 1h EMA trend features: {ema_err}")
    
    # 3. Добавляем уровни поддержки/сопротивления
    logger.info("Computing support/resistance levels...")
    
    # Сначала добавляем Donchian Channels (нужны для уровней)
    if "donchian_upper" not in df.columns:
        df = compute_donchian_channels(df, length=20)
    
    # Добавляем уровни S/R
    df = compute_support_resistance_levels(df, lookback=60, min_touches=2)
    
    # 4. Добавляем derived фичи для уровней (расстояния в долях ATR)
    logger.info("Adding level distance features...")
    df = _add_level_distance_features(df)
    
    # 5. Очистка данных
    # Удаляем строки где основные колонки NaN
    key_columns = ["open", "high", "low", "close", "volume"]
    df = df[df[key_columns].notna().any(axis=1)]
    
    # Заполняем NaN в фичах
    feature_cols = [c for c in df.columns if c not in key_columns]
    if feature_cols:
        df[feature_cols] = df[feature_cols].ffill().fillna(0)
    
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def _add_htf_ema_trend(
    df_base: pd.DataFrame,
    df_htf: pd.DataFrame,
    timeframe_suffix: str = "1h",
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.DataFrame:
    """Добавляет EMA тренд с HTF, замапленный на базовый индекс."""
    df = df_base.copy()
    if df_htf is None or df_htf.empty:
        return df

    df_htf = df_htf.copy()
    if not isinstance(df_htf.index, pd.DatetimeIndex):
        return df

    df_htf["ema_fast"] = df_htf["close"].ewm(span=ema_fast, adjust=False).mean()
    df_htf["ema_slow"] = df_htf["close"].ewm(span=ema_slow, adjust=False).mean()

    mapped = df_htf.reindex(df.index, method="ffill")
    df[f"ema_fast_{timeframe_suffix}"] = mapped["ema_fast"]
    df[f"ema_slow_{timeframe_suffix}"] = mapped["ema_slow"]
    return df


def _add_level_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи расстояний до уровней S/R в долях ATR.
    """
    df = df.copy()
    
    # Получаем ATR
    atr = df.get("atr")
    if atr is None or atr.isna().all():
        # Fallback: используем процент от цены
        atr = df["close"] * 0.01
        logger.warning("ATR not available, using fallback (1% of price)")
    
    current_price = df["close"]
    
    # Расстояние до поддержки (в долях ATR)
    support = df.get("nearest_support")
    if support is not None:
        d_support = (current_price - support) / atr
        d_support = d_support.replace([np.inf, -np.inf], 0).fillna(0)
        df["d_support_atr"] = d_support
        df["has_support"] = (support.notna() & (support < current_price)).astype(float)
    else:
        df["d_support_atr"] = 0.0
        df["has_support"] = 0.0
    
    # Расстояние до сопротивления (в долях ATR)
    resistance = df.get("nearest_resistance")
    if resistance is not None:
        d_resistance = (resistance - current_price) / atr
        d_resistance = d_resistance.replace([np.inf, -np.inf], 0).fillna(0)
        df["d_resistance_atr"] = d_resistance
        df["has_resistance"] = (resistance.notna() & (resistance > current_price)).astype(float)
    else:
        df["d_resistance_atr"] = 0.0
        df["has_resistance"] = 0.0
    
    # Ширина диапазона (resistance - support) в долях ATR
    if support is not None and resistance is not None:
        range_width = (resistance - support) / atr
        range_width = range_width.replace([np.inf, -np.inf], 0).fillna(0)
        df["range_width_atr"] = range_width
    else:
        df["range_width_atr"] = 0.0
    
    # Положение цены в диапазоне (0 = на поддержке, 1 = на сопротивлении)
    if support is not None and resistance is not None:
        price_position = (current_price - support) / (resistance - support)
        price_position = price_position.replace([np.inf, -np.inf], 0.5).fillna(0.5)
        price_position = price_position.clip(0, 1)
        df["price_position_in_range"] = price_position
    else:
        df["price_position_in_range"] = 0.5
    
    return df


def split_data(
    df: pd.DataFrame,
    val_days: int = 7,
    oos_days: int = 14,
    train_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделяет данные на train/val/oos сплиты.
    
    Args:
        df: DataFrame с данными
        train_pct: Доля данных для обучения (остальное идет в val+oos)
        val_days: Количество дней для валидации (перед OOS)
        oos_days: Количество дней для out-of-sample теста (последние дни)
    
    Returns:
        (df_train, df_val, df_oos)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Сортируем по времени
    df = df.sort_index()
    
    # Определяем границы
    total_days = (df.index[-1] - df.index[0]).days
    logger.info(f"Total data span: {total_days} days")
    
    # OOS: последние oos_days дней
    oos_start = df.index[-1] - pd.Timedelta(days=oos_days)
    df_oos = df[df.index >= oos_start].copy()
    
    # Val: val_days дней перед OOS
    val_start = oos_start - pd.Timedelta(days=val_days)
    df_val = df[(df.index >= val_start) & (df.index < oos_start)].copy()
    
    # Train: заданный период перед Val (или все остальное)
    if train_days is not None:
        train_start = val_start - pd.Timedelta(days=train_days)
        df_train = df[(df.index >= train_start) & (df.index < val_start)].copy()
    else:
        df_train = df[df.index < val_start].copy()
    
    logger.info(f"Split: Train={len(df_train)} rows, Val={len(df_val)} rows, OOS={len(df_oos)} rows")
    logger.info(f"Train: {df_train.index[0]} to {df_train.index[-1]}")
    logger.info(f"Val: {df_val.index[0]} to {df_val.index[-1]}")
    logger.info(f"OOS: {df_oos.index[0]} to {df_oos.index[-1]}")
    
    return df_train, df_val, df_oos
