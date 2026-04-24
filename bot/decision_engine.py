from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class DecisionEngineWeights:
    w_ml_confidence: float = 1.2
    w_mtf_alignment: float = 0.6
    w_atr_regime: float = 0.6
    w_sr_proximity: float = 0.9
    w_trend_slope: float = 0.3
    w_history_edge: float = 1.0


@dataclass
class DecisionEngineThresholds:
    allow_score: float = 0.35
    reduce_score: float = 0.10


@dataclass
class DecisionEngineConfig:
    enabled: bool = False
    mode: str = "shadow"
    weights: DecisionEngineWeights = field(default_factory=DecisionEngineWeights)
    thresholds: DecisionEngineThresholds = field(default_factory=DecisionEngineThresholds)
    sr_pivot_span: int = 2
    sr_lookback: int = 40
    atr_period: int = 14
    trend_lookback: int = 10
    atr_prefer_min_pct: float = 0.35
    atr_prefer_max_pct: float = 1.60


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _atr(bars: List[Dict[str, Any]], period: int) -> Optional[float]:
    if not isinstance(bars, list) or len(bars) < period + 1:
        return None
    trs: List[float] = []
    prev_close = _safe_float(bars[-period - 1].get("close"))
    if prev_close is None:
        return None
    for b in bars[-period:]:
        h = _safe_float(b.get("high"))
        l = _safe_float(b.get("low"))
        c = _safe_float(b.get("close"))
        if h is None or l is None or c is None:
            return None
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if not trs:
        return None
    return sum(trs) / len(trs)


def _pivot_levels(bars: List[Dict[str, Any]], span: int, lookback: int) -> Tuple[List[float], List[float]]:
    if not isinstance(bars, list) or len(bars) < (2 * span + 3):
        return [], []
    xs = bars[-lookback:] if lookback > 0 else bars
    highs: List[float] = []
    lows: List[float] = []
    hs = [_safe_float(b.get("high")) for b in xs]
    ls = [_safe_float(b.get("low")) for b in xs]
    if any(v is None for v in hs) or any(v is None for v in ls):
        return [], []
    for i in range(span, len(xs) - span):
        h = float(hs[i])
        l = float(ls[i])
        if all(h > float(hs[j]) for j in range(i - span, i + span + 1) if j != i):
            highs.append(h)
        if all(l < float(ls[j]) for j in range(i - span, i + span + 1) if j != i):
            lows.append(l)
    highs = sorted(set(round(x, 6) for x in highs))
    lows = sorted(set(round(x, 6) for x in lows))
    return highs, lows


def _nearest_above(levels: List[float], price: float) -> Optional[float]:
    for x in levels:
        if x > price:
            return x
    return None


def _nearest_below(levels: List[float], price: float) -> Optional[float]:
    for x in reversed(levels):
        if x < price:
            return x
    return None


def _trend_slope(bars: List[Dict[str, Any]], lookback: int, atr_val: Optional[float]) -> Optional[float]:
    if not isinstance(bars, list) or len(bars) < lookback + 1:
        return None
    c0 = _safe_float(bars[-lookback - 1].get("close"))
    c1 = _safe_float(bars[-1].get("close"))
    if c0 is None or c1 is None:
        return None
    denom = float(atr_val) if atr_val and atr_val > 0 else None
    if denom:
        return (float(c1) - float(c0)) / (lookback * denom)
    return (float(c1) - float(c0)) / lookback


def _atr_pct(price: float, atr_val: Optional[float]) -> Optional[float]:
    if not atr_val or atr_val <= 0 or price <= 0:
        return None
    return (atr_val / price) * 100.0


class AIAuditStats:
    def __init__(self, path: Path):
        self.path = path
        self._mtime: Optional[float] = None
        self._stats: Dict[str, Any] = {}

    def load_or_rebuild(self) -> Dict[str, Any]:
        try:
            mtime = self.path.stat().st_mtime
        except Exception:
            return {}
        if self._mtime == mtime and self._stats:
            return self._stats
        self._mtime = mtime
        self._stats = self._build()
        return self._stats

    def _build(self) -> Dict[str, Any]:
        total = 0
        outcomes = 0
        wins = 0
        by_symbol: Dict[str, Dict[str, float]] = {}
        by_code: Dict[str, Dict[str, float]] = {}
        confirm: Dict[str, Dict[str, Any]] = {}

        def _inc(d: Dict[str, Dict[str, float]], k: str, field: str, v: float = 1.0):
            if k not in d:
                d[k] = {"n": 0.0, "wins": 0.0, "pnl_sum": 0.0}
            d[k][field] = d[k].get(field, 0.0) + v

        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    et = rec.get("event_type")
                    if et == "confirm_entry":
                        total += 1
                        did = rec.get("decision_id") or rec.get("response", {}).get("decision_id")
                        if isinstance(did, str) and did:
                            confirm[did] = rec
                    elif et == "trade_outcome":
                        did = rec.get("decision_id")
                        if not isinstance(did, str) or not did:
                            continue
                        c = confirm.get(did)
                        if not isinstance(c, dict):
                            continue
                        outcomes += 1
                        symbol = c.get("symbol") or rec.get("symbol") or ""
                        pnl_usd = _safe_float(rec.get("pnl_usd"), 0.0) or 0.0
                        is_win = 1.0 if pnl_usd > 0 else 0.0
                        wins += int(is_win)
                        _inc(by_symbol, str(symbol), "n", 1.0)
                        _inc(by_symbol, str(symbol), "wins", is_win)
                        _inc(by_symbol, str(symbol), "pnl_sum", pnl_usd)
                        resp = c.get("response") if isinstance(c.get("response"), dict) else {}
                        codes = resp.get("reason_codes") if isinstance(resp.get("reason_codes"), list) else []
                        for code in codes:
                            _inc(by_code, str(code), "n", 1.0)
                            _inc(by_code, str(code), "wins", is_win)
                            _inc(by_code, str(code), "pnl_sum", pnl_usd)
        except Exception:
            return {}

        win_rate = (wins / outcomes) if outcomes else 0.0
        return {
            "total_confirm": total,
            "total_outcomes": outcomes,
            "win_rate": win_rate,
            "by_symbol": by_symbol,
            "by_reason_code": by_code,
        }


def _get_market_regime(ohlcv: List[Dict[str, Any]], adx_period: int = 14) -> Dict[str, Any]:
    """
    Determines if market is in Trending or Flat state.
    Returns: {'regime': 'trend'|'flat'|'volatile', 'adx': float, 'atr_ratio': float}
    """
    if len(ohlcv) < 30:
        return {"regime": "unknown", "adx": 0, "atr_ratio": 1.0}

    # Simple ADX approximation
    atr_val = _atr(ohlcv, adx_period)
    atr_short = _atr(ohlcv, 5)
    atr_ratio = atr_short / atr_val if (atr_val and atr_val > 0) else 1.0

    # Trend detection via price displacement
    c_start = _safe_float(ohlcv[-20].get("close"), 0)
    c_end = _safe_float(ohlcv[-1].get("close"), 0)
    price_move = abs(c_end - c_start) / c_start * 100 if c_start > 0 else 0

    regime = "flat"
    if price_move > 1.5 or atr_ratio > 1.5:
        regime = "volatile"
    elif price_move > 0.7:
        regime = "trend"

    return {"regime": regime, "price_move": price_move, "atr_ratio": atr_ratio}

class SignalDecisionEngine:
    def __init__(self, config: DecisionEngineConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.audit = AIAuditStats(project_root / "logs" / "ai_entry_audit.jsonl")

    def _history_edge(self, symbol: str) -> float:
        stats = self.audit.load_or_rebuild()
        baseline = float(stats.get("win_rate", 0.0) or 0.0)
        by_symbol = stats.get("by_symbol") if isinstance(stats.get("by_symbol"), dict) else {}
        sym = by_symbol.get(symbol) if isinstance(by_symbol, dict) else None
        if not isinstance(sym, dict):
            return 0.0
        n = float(sym.get("n", 0.0) or 0.0)
        if n <= 0:
            return 0.0
        wr = float(sym.get("wins", 0.0) or 0.0) / n
        return _clamp(wr - baseline, -0.25, 0.25)

    def evaluate(
        self,
        symbol: str,
        side: str,
        signal_payload: Dict[str, Any],
        ohlcv: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cfg = self.config
        now = datetime.now(timezone.utc).isoformat()

        # 0. Detect Market Regime
        regime_info = _get_market_regime(ohlcv)
        regime = regime_info["regime"]

        action = str(signal_payload.get("action") or "").upper()
        # Identify strategy type from payload
        indicators = signal_payload.get("indicators_info", {})
        strat_name = indicators.get("strategy", "UNKNOWN").upper()
        is_scalper = "SCALP" in strat_name or "5M" in strat_name or "5" == str(indicators.get("interval"))

        price = _safe_float(signal_payload.get("price"))
        ml_conf = _safe_float(signal_payload.get("confidence"), 0.0) or 0.0
        strength = str(signal_payload.get("strength") or "")
        sl_price = _safe_float(signal_payload.get("stop_loss"))
        tp_price = _safe_float(signal_payload.get("take_profit"))
        pred_1h = signal_payload.get("1h_pred")
        conf_1h = _safe_float(signal_payload.get("1h_conf"))
        pred_15m = signal_payload.get("15m_pred")
        conf_15m = _safe_float(signal_payload.get("15m_conf"))
        pred_4h = signal_payload.get("4h_pred")
        conf_4h = _safe_float(signal_payload.get("4h_conf"))

        if not price or price <= 0:
            return {
                "decision": "veto",
                "size_multiplier": 1.0,
                "score": -1.0,
                "reason_codes": ["ENGINE_NO_PRICE"],
                "notes": "Decision engine: missing price",
                "timestamp_utc": now,
                "engine": {"enabled": cfg.enabled, "mode": cfg.mode},
            }

        atr_val = _atr(ohlcv, cfg.atr_period)
        atr_pct = _atr_pct(float(price), atr_val)
        slope = _trend_slope(ohlcv, cfg.trend_lookback, atr_val)
        highs, lows = _pivot_levels(ohlcv, cfg.sr_pivot_span, cfg.sr_lookback)
        r1 = _nearest_above(highs, float(price))
        s1 = _nearest_below(lows, float(price))

        if atr_val and atr_val > 0:
            dist_to_r_atr = ((r1 - float(price)) / atr_val) if r1 else None
            dist_to_s_atr = ((float(price) - s1) / atr_val) if s1 else None
        else:
            dist_to_r_atr = None
            dist_to_s_atr = None

        sr_score = 0.0
        if action == "LONG":
            if dist_to_r_atr is not None:
                sr_score -= _clamp(1.0 - (dist_to_r_atr / 1.5), 0.0, 1.0)
            if dist_to_s_atr is not None:
                sr_score += _clamp(1.0 - (dist_to_s_atr / 1.5), 0.0, 1.0) * 0.5
        elif action == "SHORT":
            if dist_to_s_atr is not None:
                sr_score -= _clamp(1.0 - (dist_to_s_atr / 1.5), 0.0, 1.0)
            if dist_to_r_atr is not None:
                sr_score += _clamp(1.0 - (dist_to_r_atr / 1.5), 0.0, 1.0) * 0.5

        atr_score = 0.0
        if atr_pct is not None:
            if atr_pct < cfg.atr_prefer_min_pct:
                atr_score -= _clamp((cfg.atr_prefer_min_pct - atr_pct) / cfg.atr_prefer_min_pct, 0.0, 1.0)
            elif atr_pct > cfg.atr_prefer_max_pct:
                atr_score -= _clamp((atr_pct - cfg.atr_prefer_max_pct) / cfg.atr_prefer_max_pct, 0.0, 1.0)
            else:
                atr_score += 0.25

        slope_score = 0.0
        if slope is not None:
            if action == "LONG":
                slope_score = _clamp(slope, -0.25, 0.25)
            elif action == "SHORT":
                slope_score = _clamp(-slope, -0.25, 0.25)

        mtf_conf = None
        if isinstance(pred_1h, (int, float)) and isinstance(pred_15m, (int, float)) and conf_1h is not None and conf_15m is not None:
            mtf_conf = (float(conf_1h) + float(conf_15m)) / 2.0

        conf_for_score = mtf_conf if mtf_conf is not None else ml_conf
        ml_score = _clamp((conf_for_score - 0.5) * 2.0, -1.0, 1.0)

        desired_pred = 1 if action == "LONG" else (-1 if action == "SHORT" else 0)
        align_score = 0.0
        align_w = 0.0
        for p, c in ((pred_1h, conf_1h), (pred_15m, conf_15m), (pred_4h, conf_4h)):
            if desired_pred == 0:
                continue
            if not isinstance(p, (int, float)) or c is None:
                continue
            p_int = int(p)
            if p_int == 0:
                continue
            wv = float(c)
            align_w += wv
            align_score += wv * (1.0 if p_int == desired_pred else -1.0)
        if align_w > 0:
            align_score = _clamp(align_score / align_w, -1.0, 1.0)
        hist_edge = self._history_edge(symbol)

        rr_ratio = None
        rr_score = 0.0
        if tp_price and sl_price and float(price) > 0:
            denom = abs(float(price) - float(sl_price))
            if denom > 0:
                rr_ratio = abs(float(tp_price) - float(price)) / denom
                rr_score = _clamp((rr_ratio - 1.2) / 1.0, -1.0, 1.0)

        barrier_score = 0.0
        if tp_price and float(tp_price) > 0:
            tp_dist = abs(float(tp_price) - float(price))
            if tp_dist > 0:
                if action == "LONG" and r1 is not None and float(tp_price) > float(price):
                    if float(r1) > float(price) and float(r1) < float(tp_price):
                        frac = (float(r1) - float(price)) / tp_dist
                        barrier_score -= _clamp((0.60 - frac) / 0.60, 0.0, 1.0)
                if action == "SHORT" and s1 is not None and float(tp_price) < float(price):
                    if float(s1) < float(price) and float(s1) > float(tp_price):
                        frac = (float(price) - float(s1)) / tp_dist
                        barrier_score -= _clamp((0.60 - frac) / 0.60, 0.0, 1.0)

        w = cfg.weights

        # --- Regime-based weight adjustments ---
        effective_w_ml = w.w_ml_confidence
        effective_w_align = w.w_mtf_alignment
        effective_w_sr = w.w_sr_proximity

        if regime == "flat":
            # In flat market, we trust S/R more and MTF-trend less
            effective_w_align *= 0.5
            effective_w_sr *= 1.4
            if is_scalper:
                effective_w_ml *= 1.2 # Scalper might be better in chop
        elif regime == "volatile":
            # In volatile market, reduce size or be more picky
            effective_w_ml *= 0.8
            effective_w_align *= 1.2 # MTF alignment is crucial when it's wild

        score = (
            effective_w_ml * ml_score
            + w.w_atr_regime * atr_score
            + effective_w_align * align_score
            + effective_w_sr * sr_score
            + w.w_trend_slope * slope_score
            + w.w_history_edge * hist_edge
        )

        if align_w > 0 and align_score < -0.15:
            score -= 0.8

        # Strategy-specific adjustments
        if is_scalper:
            if regime == "trend":
                score -= 0.2 # Scalpers can struggle in strong one-way trends if they hunt reversals
            elif regime == "flat":
                score += 0.15 # Bonus for scalping in chop

        score += 0.35 * rr_score
        score += 0.25 * barrier_score

        decision = "veto"
        size_multiplier = 1.0
        if score >= cfg.thresholds.allow_score:
            decision = "allow"
        elif score >= cfg.thresholds.reduce_score:
            decision = "reduce"
            size_multiplier = 0.5 if score >= (cfg.thresholds.reduce_score + 0.1) else 0.25
        else:
            decision = "veto"

        reason_codes: List[str] = ["ENGINE"]
        if decision == "allow":
            reason_codes.append("ENGINE_ALLOW")
        elif decision == "reduce":
            reason_codes.append("ENGINE_REDUCE")
        else:
            reason_codes.append("ENGINE_VETO")

        if atr_pct is None:
            reason_codes.append("ENGINE_NO_ATR")
        if align_w > 0 and align_score < -0.15:
            reason_codes.append("ENGINE_MTF_CONFLICT")
        if rr_ratio is not None and rr_ratio < 0.9:
            reason_codes.append("ENGINE_BAD_RR")
        if r1 is None or s1 is None:
            reason_codes.append("ENGINE_WEAK_SR")
        if strength:
            reason_codes.append(f"ENGINE_STRENGTH_{strength}")

        return {
            "decision": decision,
            "size_multiplier": float(size_multiplier),
            "score": float(score),
            "reason_codes": reason_codes,
            "notes": "Decision engine evaluation",
            "timestamp_utc": now,
            "engine": {
                "ml_conf": ml_conf,
                "ml_score": ml_score,
                "mtf_alignment": align_score,
                "rr_ratio": rr_ratio,
                "rr_score": rr_score,
                "tp_barrier_score": barrier_score,
                "atr_pct": atr_pct,
                "atr_score": atr_score,
                "sr_score": sr_score,
                "trend_slope": slope,
                "trend_score": slope_score,
                "history_edge": hist_edge,
                "dist_to_res_atr": dist_to_r_atr,
                "dist_to_sup_atr": dist_to_s_atr,
                "levels": {"res_1": r1, "sup_1": s1},
                "weights": {
                    "w_ml_confidence": w.w_ml_confidence,
                    "w_mtf_alignment": w.w_mtf_alignment,
                    "w_atr_regime": w.w_atr_regime,
                    "w_sr_proximity": w.w_sr_proximity,
                    "w_trend_slope": w.w_trend_slope,
                    "w_history_edge": w.w_history_edge,
                },
                "thresholds": {
                    "allow": cfg.thresholds.allow_score,
                    "reduce": cfg.thresholds.reduce_score,
                },
                "enabled": cfg.enabled,
                "mode": cfg.mode,
            },
        }
