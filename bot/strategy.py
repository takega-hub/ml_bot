import typing as t
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Dict


class Action(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class Bias(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class MarketPhase(Enum):
    TREND = "trend"
    FLAT = "flat"
    MOMENTUM = "momentum"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    action: Action
    reason: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing: Optional[dict] = None
    indicators_info: Optional[dict] = None


def enrich_for_strategy(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Passthrough/enrichment placeholder. Existing code expects this function.
    Currently returns df unchanged. In future can add indicator prepping here.
    """
    # For backward compatibility, try to ensure standard columns exist
    df_out = df.copy()
    # Ensure timestamp column exists if index is DatetimeIndex
    if isinstance(df_out.index, pd.DatetimeIndex) and 'timestamp' not in df_out.columns:
        df_out = df_out.reset_index().rename(columns={'index': 'timestamp'}).set_index(pd.Index(df_out.index))
    return df_out


def build_signals(
    df: pd.DataFrame,
    strategy_obj: t.Any,
    use_momentum: bool = False,
    use_liquidity: bool = False,
    state: Optional[dict] = None,
    params: Optional[dict] = None,
    **kwargs,
) -> List[Signal]:
    """Backward-compatible adapter used across the codebase.
    
    NOTE: This function now only supports ML strategy. Other strategies are removed.
    
    Parameters from existing callers preserved: use_momentum/use_liquidity.
    strategy_obj can be a string ("ML") or a settings object.
    """
    state = state or {}
    
    # Вспомогательная функция для безопасного получения значения из params
    # Поддерживает как словари, так и объекты dataclass (StrategyParams)
    def get_param(key: str, default: t.Any, alt_key: t.Optional[str] = None):
        if params is None:
            return default
        if isinstance(params, dict):
            # Сначала пробуем основной ключ
            if key in params:
                return params[key]
            # Если есть альтернативный ключ, пробуем его
            if alt_key and alt_key in params:
                return params[alt_key]
            return default
        # Если это объект (dataclass), используем getattr
        if hasattr(params, key):
            return getattr(params, key, default)
        # Если есть альтернативный ключ, пробуем его
        if alt_key and hasattr(params, alt_key):
            return getattr(params, alt_key, default)
        return default
    
    out: List[Signal] = []

    # derive name
    name = None
    if isinstance(strategy_obj, str):
        name = strategy_obj.upper()
    else:
        # try common attributes
        for attr in ('strategy', 'name', 'strategy_name'):
            if hasattr(strategy_obj, attr):
                try:
                    val = getattr(strategy_obj, attr)
                    if isinstance(val, str):
                        name = val.upper()
                        break
                except Exception:
                    pass
    
    # Если не удалось определить имя стратегии, используем ML по умолчанию
    if name is None:
        name = 'ML'
    
    # Теперь поддерживаем только ML стратегию
    if name != 'ML':
        # Временный заглушка для обратной совместимости
        print(f"⚠️  Strategy '{name}' is deprecated. Only ML strategy is supported.")
        name = 'ML'
    
    try:
        # Генерируем ML сигнал (здесь должна быть интеграция с ML моделью)
        # В реальной реализации это будет вызываться из MLStrategy
        res = generate_ml_signal(
            df,
            state=state,
            confidence_threshold=get_param('confidence_threshold', 0.5),
            min_signal_strength=get_param('min_signal_strength', 'слабое'),
            stability_filter=get_param('stability_filter', True),
        )
    except Exception as e:
        # Логируем ошибку для диагностики
        import traceback
        print(f"[build_signals] ERROR generating ML signal: {e}")
        print(f"[build_signals] Traceback: {traceback.format_exc()}")
        return out

    if res and res.get('signal') is not None:
        action = Action.LONG if res.get('signal') == 'LONG' else (Action.SHORT if res.get('signal') == 'SHORT' else Action.HOLD)
        reason = res.get('reason', '')
        price = float(df['close'].iloc[-1]) if 'close' in df.columns and len(df) > 0 else 0.0
        # Prefer explicit indicators_info, but attach SL/TP/trailing into indicators so downstream
        # systems that only accept a Signal object still have access to exit params.
        indicators = dict(res.get('indicators_info', {}) or {})
        # attach stop/take/trailing into indicators for downstream consumers
        if res.get('stop_loss') is not None:
            try:
                indicators['stop_loss'] = float(res.get('stop_loss'))
            except Exception:
                indicators['stop_loss'] = res.get('stop_loss')
        if res.get('take_profit') is not None:
            try:
                indicators['take_profit'] = float(res.get('take_profit'))
            except Exception:
                indicators['take_profit'] = res.get('take_profit')
        if res.get('trailing') is not None:
            indicators['trailing'] = res.get('trailing')
        # prefer timestamp from df index if available
        try:
            ts = pd.Timestamp(df.index[-1])
        except Exception:
            ts = pd.Timestamp.now()
        # Prefer explicit stop/take/trailing fields on Signal for downstream consumers
        sig = Signal(
            timestamp=ts,
            action=action,
            reason=reason,
            price=price,
            stop_loss=res.get('stop_loss') or res.get('indicators_info', {}).get('sl'),
            take_profit=res.get('take_profit') or res.get('indicators_info', {}).get('tp'),
            trailing=res.get('trailing'),
            indicators_info=indicators,
        )
        out.append(sig)

    return out


def detect_market_phase(row_or_df: t.Union[pd.Series, pd.DataFrame], strategy_name: Optional[str] = None) -> Optional[MarketPhase]:
    """
    Простая детекция рыночной фазы для совместимости с остальным кодом.
    Принимает либо одну строку (Series) с индикаторами, либо DataFrame.
    Если доступны индикаторы ('adx', 'atr' и т.д.), пытается определить фазу.

    Возвращает MarketPhase или None.
    """
    try:
        # If DataFrame passed, use last row
        if isinstance(row_or_df, pd.DataFrame):
            row = row_or_df.iloc[-1]
        else:
            row = row_or_df

        # Prefer explicit strategy_name hints
        name_hint = None
        if isinstance(strategy_name, str):
            name_hint = strategy_name.upper()
        elif strategy_name is not None:
            # try common attributes if it's a settings object
            for attr in ('strategy', 'name', 'strategy_name'):
                if hasattr(strategy_name, attr):
                    try:
                        val = getattr(strategy_name, attr)
                        if isinstance(val, str):
                            name_hint = val.upper()
                            break
                    except Exception:
                        pass

        if name_hint == 'ML':
            # Для ML стратегии определяем фазу по индикаторам
            # ADX-based heuristic if available
            adx = row.get('adx') if hasattr(row, 'get') else None
            if adx is not None:
                try:
                    adx_v = float(adx)
                    if adx_v > 25:
                        return MarketPhase.TREND
                    if adx_v < 20:
                        return MarketPhase.FLAT
                except Exception:
                    pass

            # Volatility-based fallback using atr
            atr = row.get('atr') if hasattr(row, 'get') else None
            if atr is not None:
                try:
                    atr_v = float(atr)
                    # crude thresholds - kept conservative
                    if atr_v > 0.5:
                        return MarketPhase.MOMENTUM
                    return MarketPhase.FLAT
                except Exception:
                    pass

        return None
    except Exception:
        return None


def detect_market_bias(row: pd.Series) -> Optional[Bias]:
    # Ищем DI под любыми именами
    p_di = row.get('plus_di') or row.get('DMP_14') or row.get('ADX_14_pos')
    m_di = row.get('minus_di') or row.get('DMN_14') or row.get('ADX_14_neg')

    if pd.notnull(p_di) and pd.notnull(m_di):
        return Bias.LONG if float(p_di) > float(m_di) else Bias.SHORT
    
    # Если DI нет, смотрим на SMA (обязательно!)
    close = row.get('close')
    sma = row.get('sma') or row.get('sma_200')
    if pd.notnull(close) and pd.notnull(sma):
        return Bias.LONG if float(close) > float(sma) else Bias.SHORT
    
    return None


def generate_range_signal(row: pd.Series, position_bias: Optional[Bias], settings: t.Any) -> Signal:
    """Compatibility wrapper for legacy code that expects a row-based range signal.
    
    NOTE: This function is deprecated. ML strategy should be used instead.
    """
    print("⚠️  generate_range_signal is deprecated. Use ML strategy instead.")
    return Signal(
        timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(), 
        action=Action.HOLD, 
        reason='deprecated_use_ml', 
        price=float(row.get('close', 0.0)), 
        indicators_info={}
    )


def generate_momentum_breakout_signal(row: pd.Series, position_bias: Optional[Bias], settings: t.Any) -> Signal:
    """Compatibility wrapper for legacy momentum breakout signature.
    
    NOTE: This function is deprecated. ML strategy should be used instead.
    """
    print("⚠️  generate_momentum_breakout_signal is deprecated. Use ML strategy instead.")
    return Signal(
        timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(), 
        action=Action.HOLD, 
        reason='deprecated_use_ml', 
        price=float(row.get('close', 0.0)), 
        indicators_info={}
    )


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    # Use min_periods=1 to produce values on short series (prevents NaN for small datasets)
    return series.rolling(period, min_periods=1).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Backfill then default to neutral 50 to avoid None values in downstream checks/logs
    rsi = rsi.bfill().fillna(50.0)
    return rsi


def _bb_width(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    mid = _sma(df['close'], period)
    # ensure std is computed with min_periods to avoid NaNs on short series
    std = df['close'].rolling(period, min_periods=1).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / (mid.replace(0, np.nan).abs())
    return width


def _ensure_history(df: pd.DataFrame, required: int) -> bool:
    return len(df) >= required


def _get_higher_timeframe_bias(
    df: pd.DataFrame,
    timeframe: str = '1h',
    ema_period: int = 50,
    end_idx: Optional[int] = None,
    df_htf: Optional[pd.DataFrame] = None,
) -> t.Optional[str]:
    """
    Определяет глобальный тренд через положение цены относительно EMA на высшем таймфрейме.
    
    Args:
        df: DataFrame с данными текущего таймфрейма (обычно 15m)
        timeframe: Целевой таймфрейм для анализа ('1h', '4h')
        ema_period: Период EMA для определения тренда
        end_idx: Индекс до которого анализировать (для backtesting)
        df_htf: Опционально готовый DataFrame с высшим таймфреймом (если есть готовые данные)
    
    Returns:
        'bullish' если цена выше EMA, 'bearish' если ниже, None если недостаточно данных
    """
    try:
        # Если передан готовый DataFrame с высшим таймфреймом - используем его
        if df_htf is not None and not df_htf.empty:
            df_htf_use = df_htf.copy()
        else:
            # Иначе ресемплим из текущего таймфрейма
            df_curr = df.iloc[:end_idx+1] if end_idx is not None else df
            if len(df_curr) < ema_period * 2:  # Нужно достаточно данных для ресемплинга и EMA
                return None
            
            # Ресемплинг на высший таймфрейм
            if timeframe == '1h':
                df_htf_use = df_curr.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            elif timeframe == '4h':
                df_htf_use = df_curr.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                return None
        
        if len(df_htf_use) < ema_period:
            return None
        
        # Рассчитываем EMA на высшем таймфрейме
        ema = df_htf_use['close'].ewm(span=ema_period, adjust=False).mean()
        last_close = df_htf_use['close'].iloc[-1]
        last_ema = ema.iloc[-1]
        
        if last_close > last_ema:
            return 'bullish'
        elif last_close < last_ema:
            return 'bearish'
        return None
    except Exception:
        return None


def _get_multi_timeframe_consensus(
    df_15m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    ema_period: int = 50,
    end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Определяет общую ситуацию на рынке используя все доступные таймфреймы.
    
    Args:
        df_15m: DataFrame с данными 15m таймфрейма
        df_1h: Опционально готовый DataFrame с 1h данными
        df_4h: Опционально готовый DataFrame с 4h данными
        ema_period: Период EMA для определения тренда
        end_idx: Индекс до которого анализировать (для backtesting)
    
    Returns:
        Словарь с информацией о тренде на всех таймфреймах:
        {
            '1h_bias': 'bullish'/'bearish'/None,
            '4h_bias': 'bullish'/'bearish'/None,
            'consensus': 'bullish'/'bearish'/'neutral'/None,
            'trend_strength': float (0-1),  # Сила консенсуса
        }
    """
    result = {
        '1h_bias': None,
        '4h_bias': None,
        'consensus': None,
        'trend_strength': 0.0,
    }
    
    # Анализ 1H таймфрейма
    if df_1h is not None:
        result['1h_bias'] = _get_higher_timeframe_bias(
            df_15m, timeframe='1h', ema_period=ema_period, end_idx=end_idx, df_htf=df_1h
        )
    else:
        result['1h_bias'] = _get_higher_timeframe_bias(
            df_15m, timeframe='1h', ema_period=ema_period, end_idx=end_idx
        )
    
    # Анализ 4H таймфрейма
    if df_4h is not None:
        result['4h_bias'] = _get_higher_timeframe_bias(
            df_15m, timeframe='4h', ema_period=ema_period, end_idx=end_idx, df_htf=df_4h
        )
    else:
        result['4h_bias'] = _get_higher_timeframe_bias(
            df_15m, timeframe='4h', ema_period=ema_period, end_idx=end_idx
        )
    
    # Определение консенсуса
    biases = [b for b in [result['1h_bias'], result['4h_bias']] if b is not None]
    
    if not biases:
        result['consensus'] = None
        result['trend_strength'] = 0.0
    elif len(biases) == 1:
        result['consensus'] = biases[0]
        result['trend_strength'] = 0.5  # Средняя сила (только один таймфрейм)
    else:
        # Оба таймфрейма дали результат
        if biases[0] == biases[1]:
            result['consensus'] = biases[0]
            result['trend_strength'] = 1.0  # Сильный консенсус
        else:
            result['consensus'] = 'neutral'  # Противоречивые сигналы
            result['trend_strength'] = 0.3  # Слабая сила
    
    return result


def generate_trend_signal(*args, **kwargs):
    """DEPRECATED: Trend strategy is no longer supported. Use ML strategy instead."""
    print("⚠️  generate_trend_signal is deprecated. Use ML strategy instead.")
    return {"signal": None, "stop_loss": None, "indicators_info": {}, "reason": "deprecated_use_ml"}


def generate_flat_signal(*args, **kwargs):
    """DEPRECATED: Flat strategy is no longer supported. Use ML strategy instead."""
    print("⚠️  generate_flat_signal is deprecated. Use ML strategy instead.")
    return {"signal": None, "indicators_info": {}, "reason": "deprecated_use_ml"}


def generate_momentum_signal(*args, **kwargs):
    """DEPRECATED: Momentum strategy is no longer supported. Use ML strategy instead."""
    print("⚠️  generate_momentum_signal is deprecated. Use ML strategy instead.")
    return {"signal": None, "indicators_info": {}, "reason": "deprecated_use_ml"}


def generate_ml_signal(
    df: pd.DataFrame,
    state: t.Optional[dict] = None,
    confidence_threshold: float = 0.5,
    min_signal_strength: str = "слабое",
    stability_filter: bool = True,
) -> t.Dict:
    """
    Генерация сигнала ML стратегии.
    
    NOTE: This is a placeholder function. In real implementation,
    this should call ML model for prediction.
    
    Args:
        df: DataFrame с историческими данными
        state: Словарь состояния
        confidence_threshold: Порог уверенности для открытия позиции
        min_signal_strength: Минимальная сила сигнала
        stability_filter: Фильтр стабильности сигналов
    
    Returns:
        Dict с сигналом и метаданными
    """
    state = state or {}
    indicators_info = {}
    
    # Это заглушка - в реальной реализации здесь будет вызов ML модели
    # из модуля bot.ml.strategy_ml
    
    # Для обратной совместимости возвращаем HOLD
    # В реальном боте эта функция будет заменена вызовом MLStrategy
    
    indicators_info['strategy'] = 'ML'
    indicators_info['confidence_threshold'] = confidence_threshold
    indicators_info['min_signal_strength'] = min_signal_strength
    indicators_info['stability_filter'] = stability_filter
    
    return {"signal": None, "stop_loss": None, "take_profit": None, "indicators_info": indicators_info, "reason": "ml_placeholder"}


if __name__ == '__main__':
    # quick smoke test (runs only when module executed directly)
    import json

    # create dummy data
    idx = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='T')
    df = pd.DataFrame(index=idx)
    df['open'] = np.linspace(100, 120, len(df)) + np.random.randn(len(df))
    df['high'] = df['open'] + np.random.rand(len(df)) * 1.5
    df['low'] = df['open'] - np.random.rand(len(df)) * 1.5
    df['close'] = df['open'] + np.random.randn(len(df)) * 0.5
    df['volume'] = np.random.randint(1, 100, len(df))

    print("Testing ML signal generation:")
    print(json.dumps(generate_ml_signal(df), indent=2))
    
    print("\nTesting deprecated strategies (should show warnings):")
    print(json.dumps(generate_trend_signal(df, state={'long_pyramid': 0}), indent=2))
    print(json.dumps(generate_flat_signal(df), indent=2))
    print(json.dumps(generate_momentum_signal(df), indent=2))