from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except Exception:
    ta = None
    PANDAS_TA_AVAILABLE = False


def _sma(series: pd.Series, length: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window=length, min_periods=1).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=length, adjust=False, min_periods=1).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean().fillna(0.0)


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    c = pd.to_numeric(close, errors="coerce")
    mid = c.rolling(window=length, min_periods=1).mean()
    dev = c.rolling(window=length, min_periods=1).std(ddof=0).fillna(0.0)
    upper = mid + std * dev
    lower = mid - std * dev
    return pd.DataFrame(
        {
            "BBU_20_2.0": upper,
            "BBM_20_2.0": mid,
            "BBL_20_2.0": lower,
        },
        index=c.index,
    )


def _macd(close: pd.Series) -> pd.DataFrame:
    c = pd.to_numeric(close, errors="coerce")
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=1).mean()
    hist = macd_line - signal
    return pd.DataFrame(
        {
            "MACD_12_26_9": macd_line,
            "MACDs_12_26_9": signal,
            "MACDh_12_26_9": hist,
        },
        index=c.index,
    )


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h.index)
    atr = _atr(h, l, c, length).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean()
    return pd.DataFrame(
        {
            "ADX_14": adx.fillna(0.0),
            "DMP_14": plus_di.fillna(0.0),
            "DMN_14": minus_di.fillna(0.0),
        },
        index=h.index,
    )


class _TAFallback:
    @staticmethod
    def sma(series: pd.Series, length: int = 20):
        return _sma(series, length)

    @staticmethod
    def ema(series: pd.Series, length: int = 20):
        return _ema(series, length)

    @staticmethod
    def rsi(series: pd.Series, length: int = 14):
        return _rsi(series, length)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
        return _atr(high, low, close, length)

    @staticmethod
    def bbands(series: pd.Series, length: int = 20, std: float = 2.0):
        return _bbands(series, length, std)

    @staticmethod
    def macd(series: pd.Series):
        return _macd(series)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
        return _adx(high, low, close, length)


if not PANDAS_TA_AVAILABLE:
    ta = _TAFallback()


def compute_orderbook_imbalance(ob_response: Dict[str, Any], depth: int = 10) -> float:
    if not ob_response or not isinstance(ob_response, dict):
        return 0.0
    result = ob_response.get("result") if ob_response.get("retCode") == 0 else ob_response.get("result")
    if not result or not isinstance(result, dict):
        return 0.0

    bids = result.get("b", []) or []
    asks = result.get("a", []) or []

    def _sum(levels):
        s = 0.0
        for lvl in levels[:depth]:
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                try:
                    s += float(lvl[1])
                except Exception:
                    pass
        return s

    bid_vol = _sum(bids)
    ask_vol = _sum(asks)
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return (bid_vol - ask_vol) / total


class FeatureEngineer:
    def __init__(self):
        self.feature_names: List[str] = []

    def safe_ta_indicator(self, df: pd.DataFrame, indicator_func, **kwargs):
        try:
            return indicator_func(**kwargs)
        except Exception:
            return None

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # ОПТИМИЗАЦИЯ: Если DataFrame слишком большой, берем только необходимые для индикаторов последние N строк
        # Максимальное окно у нас - 100 (для touches), возьмем 200 для запаса.
        # ВАЖНО: Для обучения/бэктеста используем всё, для живой торговли - хвост.
        is_live = len(df) < 1000

        out = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in out.columns:
                found = False
                for c in out.columns:
                    if c.lower() == col:
                        out[col] = out[c]
                        found = True
                        break
                if not found:
                    return pd.DataFrame()
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Переводим индекс в datetime один раз
        if not isinstance(out.index, pd.DatetimeIndex) and "timestamp" in out.columns:
            try:
                out.index = pd.to_datetime(out["timestamp"], unit="ms" if out["timestamp"].iloc[0] > 1e12 else "s")
            except:
                pass

        # Основные фичи (Векторные операции Pandas/NumPy очень быстры)
        close_vals = out["close"].values
        high_vals = out["high"].values
        low_vals = out["low"].values
        vol_vals = out["volume"].values

        out["turnover"] = close_vals * vol_vals
        out["sma_20"] = ta.sma(out["close"], length=20)
        out["sma_50"] = ta.sma(out["close"], length=50)
        out["ema_12"] = ta.ema(out["close"], length=12)
        out["ema_26"] = ta.ema(out["close"], length=26)
        out["rsi"] = ta.rsi(out["close"], length=14)
        out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
        out["atr_pct"] = (out["atr"] / out["close"]) * 100

        v_sma20 = ta.sma(out["volume"], length=20)
        out["volume_sma_20"] = v_sma20
        out["volume_ratio"] = np.where(v_sma20 > 0, vol_vals / v_sma20, 1.0)

        out["price_change"] = out["close"].pct_change()
        out["price_change_abs"] = out["price_change"].abs()

        # Bollinger Bands
        bb = ta.bbands(out["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            out["bb_upper"] = bb.iloc[:, 0]
            out["bb_middle"] = bb.iloc[:, 1]
            out["bb_lower"] = bb.iloc[:, 2]

        # MACD
        macd = ta.macd(out["close"])
        if macd is not None and not macd.empty:
            out["macd"] = macd.iloc[:, 0]
            out["macd_signal"] = macd.iloc[:, 1]

        # ADX
        adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            out["adx"] = adx_df.iloc[:, 0]
            out["di_plus"] = adx_df.iloc[:, 1]
            out["di_minus"] = adx_df.iloc[:, 2]

        # Lags
        for lag in [1, 2, 3]:
            out[f"close_lag_{lag}"] = out["close"].shift(lag)
            out[f"volume_lag_{lag}"] = out["volume"].shift(lag)

        # Volatility
        out["volatility_10"] = out["close"].rolling(window=10).std()
        out["realized_volatility_10"] = out["price_change"].rolling(window=10).std()
        out["realized_volatility_20"] = out["price_change"].rolling(window=20).std()

        # Parkinson Volatility (Оптимизировано через NumPy)
        log_hl = np.log(high_vals / low_vals)**2
        out["parkinson_vol"] = np.sqrt(1 / (4 * np.log(2)) * pd.Series(log_hl).rolling(window=20).mean()).values

        out["dist_to_sma20_pct"] = ((out["close"] - out["sma_20"]) / out["sma_20"]) * 100

        # Time features
        if isinstance(out.index, pd.DatetimeIndex):
            hours = out.index.hour
            dows = out.index.dayofweek
            out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # Bollinger position
        out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)

        # Volume Imbalance (Approximation)
        range_total = (out["high"] - out["low"]).replace(0, np.nan)
        buy_p = (out["close"] - out["low"]) / range_total
        sell_p = (out["high"] - out["close"]) / range_total
        vol_imbalance = (buy_p - sell_p).fillna(0.0)
        out["volume_imbalance_10"] = vol_imbalance.rolling(window=10).mean()

        # Support/Resistance (ОПТИМИЗИРОВАНО: убрано center=True для live)
        # Для живой торговли нам нужны ТЕКУЩИЕ локальные экстремумы, а не заглядывание в будущее
        window_sr = 20
        out["local_low"] = out["low"].rolling(window=window_sr).min()
        out["local_high"] = out["high"].rolling(window=window_sr).max()

        out["dist_to_support_pct"] = (out["close"] - out["local_low"]) / out["close"] * 100
        out["dist_to_resistance_pct"] = (out["local_high"] - out["close"]) / out["close"] * 100

        # Relative Volume (RVOL)
        out["rvol"] = out["volume"] / out["volume"].rolling(window=48).mean()

        # MTF Context
        out["trend_1h"] = out["close"].rolling(window=4).mean()
        out["trend_4h"] = out["close"].rolling(window=16).mean()

        # Финальная очистка
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        base_cols = {"open", "high", "low", "close", "volume", "timestamp", "target", "meta_target"}
        self.feature_names = [c for c in out.columns if c not in base_cols]
        return out

    def create_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        pt_sl_ratio: float = 2.0,
        volatility_lookback: int = 20,
        vertical_barrier_candles: int = 24,
        min_vol_pct: float = 0.1
    ) -> pd.DataFrame:
        """
        Implementation of the Triple Barrier Method (de Prado).
        Labels: 1 (Long Opportunity), -1 (Short Opportunity), 0 (No clear trend/Horizontal hit).
        """
        if df is None or df.empty or "close" not in df.columns:
            return pd.DataFrame()

        out = df.copy()
        close = pd.to_numeric(out["close"], errors="coerce").values

        # 1. Dynamic volatility (Daily-scaled volatility or rolling window)
        returns = out["close"].pct_change()
        volatility = returns.rolling(window=volatility_lookback).std()
        volatility = volatility.fillna(method='bfill').fillna(0.005).values

        # Ensure minimum volatility
        min_vol = min_vol_pct / 100.0
        volatility = np.clip(volatility, min_vol, None)

        # 2. Add Volatility regime feature
        out["volatility_regime"] = volatility
        out["volatility_ma"] = out["volatility_regime"].rolling(window=100).mean().fillna(method='bfill')
        out["is_high_volatility"] = (out["volatility_regime"] > out["volatility_ma"] * 1.5).astype(int)

        labels = np.zeros(len(out))

        for i in range(len(out) - vertical_barrier_candles):
            price_start = close[i]
            vol = volatility[i]

            # Barriers for Long
            l_tp = price_start * (1 + vol * pt_sl_ratio)
            l_sl = price_start * (1 - vol)

            # Barriers for Short
            s_tp = price_start * (1 - vol * pt_sl_ratio)
            s_sl = price_start * (1 + vol)

            hit = 0
            for j in range(1, vertical_barrier_candles + 1):
                p_curr = close[i + j]

                if p_curr >= l_tp:
                    hit = 1
                    break
                elif p_curr <= s_tp:
                    hit = -1
                    break

                if p_curr <= l_sl or p_curr >= s_sl:
                    hit = 0
                    break

            labels[i] = hit

        out["target"] = labels.astype(int)
        return out.iloc[:-vertical_barrier_candles]

    def compute_meta_labels(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
        """
        Computes meta-labels (0/1) for a secondary 'Signal Filter' model.
        Meta-label is 1 if the primary model's signal was correct (hit TP), 0 otherwise.

        predictions: Array of primary model predictions (-1, 0, 1)
        """
        if "target" not in df.columns or len(predictions) != len(df):
            return pd.Series(0, index=df.index)

        target = df["target"].values
        # Meta label is 1 ONLY if prediction matches a non-zero target
        meta_labels = np.where((predictions != 0) & (predictions == target), 1, 0)
        return pd.Series(meta_labels, index=df.index, name="meta_target")

    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_periods: int = 4,
        threshold_pct: float = 0.5,
        use_atr_threshold: bool = True,
        use_risk_adjusted: bool = False,
        min_risk_reward_ratio: float = 1.5,
        max_hold_periods: int = 96,
        min_profit_pct: float = 0.3,
        threshold: Optional[float] = None,
        min_profit_for_signal: Optional[float] = None,
        risk_reward_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        _ = use_risk_adjusted, min_risk_reward_ratio, max_hold_periods, risk_reward_ratio
        if df is None or df.empty or "close" not in df.columns:
            return pd.DataFrame()

        out = df.copy()
        if threshold is not None:
            threshold_pct = float(threshold)
        if min_profit_for_signal is not None:
            min_profit_pct = float(min_profit_for_signal)

        current_price = pd.to_numeric(out["close"], errors="coerce").fillna(0.0).values
        future_price = np.zeros_like(current_price)
        for i in range(len(current_price)):
            if i + forward_periods < len(current_price):
                future_price[i] = current_price[i + forward_periods]
            else:
                future_price[i] = current_price[-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            pct_change = np.where(current_price > 0, (future_price - current_price) / current_price * 100, 0.0)

        if use_atr_threshold and "atr_pct" in out.columns:
            dyn = np.minimum(float(threshold_pct), pd.to_numeric(out["atr_pct"], errors="coerce").fillna(0.0).values * 0.8)
        else:
            dyn = np.full(len(out), float(threshold_pct))

        target = np.zeros(len(out), dtype=int)
        for i in range(max(0, len(out) - forward_periods)):
            ch = pct_change[i]
            thr = dyn[i]
            if ch > thr and ch >= float(min_profit_pct):
                target[i] = 1
            elif ch < -thr and abs(ch) >= float(min_profit_pct):
                target[i] = -1

        out["target"] = target
        if len(out) > forward_periods:
            out = out.iloc[:-forward_periods]
        return out

    def set_orderbook_imbalance_last_row(self, df: pd.DataFrame, ob_response: Dict[str, Any], depth: int = 10) -> None:
        if df is None or df.empty:
            return
        main = compute_orderbook_imbalance(ob_response, depth=depth)
        d5 = compute_orderbook_imbalance(ob_response, depth=5)
        d20 = compute_orderbook_imbalance(ob_response, depth=20)
        if "ob_imbalance" in df.columns:
            df.iloc[-1, df.columns.get_loc("ob_imbalance")] = main
        if "ob_imbalance_5" in df.columns:
            df.iloc[-1, df.columns.get_loc("ob_imbalance_5")] = d5
        if "ob_imbalance_20" in df.columns:
            df.iloc[-1, df.columns.get_loc("ob_imbalance_20")] = d20

    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()

    def prepare_features_for_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df is None or df.empty:
            return np.array([]), np.array([])
        exclude = {"open", "high", "low", "close", "volume", "timestamp", "target"}
        feature_cols = [c for c in df.columns if c not in exclude]
        if not feature_cols:
            feature_cols = ["sma_20", "rsi", "atr_pct", "price_change"]
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        X = df[feature_cols].values
        y = df["target"].values if "target" in df.columns else np.zeros(len(df))
        self.feature_names = feature_cols
        return X, y
