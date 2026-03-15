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

        out = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in out.columns:
                return pd.DataFrame()
            out[col] = pd.to_numeric(out[col], errors="coerce")

        if "timestamp" in out.columns:
            out = out.set_index("timestamp")

        out["sma_20"] = ta.sma(out["close"], length=20)
        out["sma_50"] = ta.sma(out["close"], length=50)
        out["ema_12"] = ta.ema(out["close"], length=12)
        out["ema_26"] = ta.ema(out["close"], length=26)
        out["rsi"] = ta.rsi(out["close"], length=14)
        out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
        out["atr_pct"] = (out["atr"] / out["close"].replace(0, np.nan)) * 100
        out["volume_sma_20"] = ta.sma(out["volume"], length=20)
        out["volume_ratio"] = np.where(out["volume_sma_20"] > 0, out["volume"] / out["volume_sma_20"], 1.0)

        bb = ta.bbands(out["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            out["bb_upper"] = bb.iloc[:, 0]
            out["bb_middle"] = bb.iloc[:, 1]
            out["bb_lower"] = bb.iloc[:, 2]
        else:
            out["bb_upper"] = out["close"]
            out["bb_middle"] = out["close"]
            out["bb_lower"] = out["close"]

        macd = ta.macd(out["close"])
        if macd is not None and not macd.empty:
            out["macd"] = macd.iloc[:, 0]
            out["macd_signal"] = macd.iloc[:, 1] if len(macd.columns) > 1 else 0.0
        else:
            out["macd"] = 0.0
            out["macd_signal"] = 0.0

        adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            out["adx"] = adx_df.iloc[:, 0]
            out["di_plus"] = adx_df.iloc[:, 1] if len(adx_df.columns) > 1 else 0.0
            out["di_minus"] = adx_df.iloc[:, 2] if len(adx_df.columns) > 2 else 0.0
        else:
            out["adx"] = 0.0
            out["di_plus"] = 0.0
            out["di_minus"] = 0.0

        out["adx_trend_up"] = (pd.to_numeric(out["di_plus"], errors="coerce").fillna(0.0) > pd.to_numeric(out["di_minus"], errors="coerce").fillna(0.0)).astype(int)
        out["price_change"] = out["close"].pct_change()
        out["price_change_abs"] = out["price_change"].abs()
        out["volatility_10"] = out["close"].rolling(window=10, min_periods=3).std()
        out["realized_volatility_20"] = out["price_change"].rolling(window=20, min_periods=5).std()
        out["momentum_3"] = out["close"].pct_change(periods=3)
        out["momentum_5"] = out["close"].pct_change(periods=5)
        out["momentum_10"] = out["close"].pct_change(periods=10)
        out["dist_to_sma20_pct"] = ((out["close"] - out["sma_20"]) / out["sma_20"].replace(0, np.nan)) * 100
        out["dist_to_ema12_pct"] = ((out["close"] - out["ema_12"]) / out["ema_12"].replace(0, np.nan)) * 100
        out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
        out["near_bb_upper"] = (out["close"] > out["bb_upper"] * 0.95).astype(int)
        out["near_bb_lower"] = (out["close"] < out["bb_lower"] * 1.05).astype(int)
        out["rsi_oversold"] = (pd.to_numeric(out["rsi"], errors="coerce").fillna(50.0) < 30).astype(int)
        out["rsi_overbought"] = (pd.to_numeric(out["rsi"], errors="coerce").fillna(50.0) > 70).astype(int)
        out["ema12_above_ema26"] = (pd.to_numeric(out["ema_12"], errors="coerce").fillna(0.0) > pd.to_numeric(out["ema_26"], errors="coerce").fillna(0.0)).astype(int)
        out["ob_imbalance"] = 0.0
        out["ob_imbalance_5"] = 0.0
        out["ob_imbalance_20"] = 0.0

        if isinstance(out.index, pd.DatetimeIndex):
            out["hour"] = out.index.hour
            out["day_of_week"] = out.index.dayofweek
            out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
            out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
            out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

        for lag in [1, 2, 3]:
            out[f"close_lag_{lag}"] = out["close"].shift(lag)
            out[f"volume_lag_{lag}"] = out["volume"].shift(lag)
            out[f"price_change_lag_{lag}"] = out["price_change"].shift(lag)

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        base_cols = {"open", "high", "low", "close", "volume", "timestamp"}
        self.feature_names = [c for c in out.columns if c not in base_cols]
        return out

    def add_mtf_features(self, df_features: pd.DataFrame, higher_timeframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if df_features is None or df_features.empty or not higher_timeframes:
            return df_features
        out = df_features.copy()
        for tf_name, htf_df in higher_timeframes.items():
            if htf_df is None or htf_df.empty:
                continue
            htf_feat = FeatureEngineer().create_technical_indicators(htf_df)
            for col in ["rsi", "atr_pct", "adx", "volume_ratio"]:
                if col in htf_feat.columns:
                    src = htf_feat[col]
                    if isinstance(out.index, pd.DatetimeIndex) and isinstance(src.index, pd.DatetimeIndex):
                        out[f"{col}_{tf_name}"] = src.reindex(out.index, method="ffill").fillna(0.0)
                    else:
                        vals = list(src.values)
                        if len(vals) < len(out):
                            vals += [0.0] * (len(out) - len(vals))
                        out[f"{col}_{tf_name}"] = vals[: len(out)]
        out = out.ffill().bfill().fillna(0.0)
        return out

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
