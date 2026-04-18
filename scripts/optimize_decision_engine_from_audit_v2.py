import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_ml_settings(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class Row:
    ts: datetime
    symbol: str
    pnl_usd: float
    x: np.ndarray  # 8 features: ml_score, mtf, atr_score, sr_score, trend_score, hist, rr_score, barrier
    score_logged: float


def load_rows(audit_path: Path) -> List[Row]:
    entries: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    for line in audit_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        did = rec.get("decision_id")
        if not isinstance(did, str) or not did:
            continue
        et = rec.get("event_type")
        if et == "engine_entry":
            entries[did] = rec
        elif et == "engine_trade_outcome":
            outcomes[did] = rec

    rows: List[Row] = []
    for did, e in entries.items():
        o = outcomes.get(did)
        if not isinstance(o, dict):
            continue
        ts = _parse_dt(str(e.get("timestamp_utc") or ""))
        if ts is None:
            continue
        symbol = str(e.get("symbol") or "")
        pnl = _safe_float(o.get("pnl_usd"), 0.0)
        engine_eval = e.get("engine_eval") if isinstance(e.get("engine_eval"), dict) else {}
        engine_ctx = engine_eval.get("engine") if isinstance(engine_eval.get("engine"), dict) else {}
        x = np.array(
            [
                _safe_float(engine_ctx.get("ml_score"), 0.0),
                _safe_float(engine_ctx.get("mtf_alignment"), 0.0),
                _safe_float(engine_ctx.get("atr_score"), 0.0),
                _safe_float(engine_ctx.get("sr_score"), 0.0),
                _safe_float(engine_ctx.get("trend_score"), 0.0),
                _safe_float(engine_ctx.get("history_edge"), 0.0),
                _safe_float(engine_ctx.get("rr_score"), 0.0),
                _safe_float(engine_ctx.get("tp_barrier_score"), 0.0),
            ],
            dtype=float,
        )
        rows.append(Row(ts=ts, symbol=symbol, pnl_usd=pnl, x=x, score_logged=_safe_float(engine_eval.get("score"), 0.0)))
    rows.sort(key=lambda r: r.ts)
    return rows


def compute_scores(X: np.ndarray, w: np.ndarray, rr_weight: float = 0.35, barrier_weight: float = 0.25) -> np.ndarray:
    base = X[:, :6] @ w
    rr = X[:, 6] * rr_weight
    barrier = X[:, 7] * barrier_weight
    mtf = X[:, 1]
    conflict = np.where(mtf < -0.15, -0.8, 0.0)
    return base + rr + barrier + conflict


def simulate(pnl: np.ndarray, t: np.ndarray, scores: np.ndarray, allow_th: float, reduce_th: float) -> Dict[str, float]:
    idx = np.where(scores >= reduce_th)[0]
    if idx.size == 0:
        return {"n": 0.0, "pnl_sum": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

    pnl_t = pnl[idx].copy()
    mult = np.ones_like(pnl_t)
    allow_mask = scores[idx] >= allow_th
    reduce_mask = ~allow_mask
    if reduce_mask.any():
        s = scores[idx][reduce_mask]
        mult[reduce_mask] = np.where(s >= (reduce_th + 0.1), 0.5, 0.25)
    pnl_t *= mult

    n = float(pnl_t.size)
    win_rate = float((pnl_t > 0).sum()) / n if n else 0.0
    gross_win = float(pnl_t[pnl_t > 0].sum())
    gross_loss = float(np.abs(pnl_t[pnl_t < 0]).sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    pnl_sum = float(pnl_t.sum())

    order = np.argsort(t[idx])
    equity = np.cumsum(pnl_t[order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_dd = float(dd.max()) if dd.size else 0.0
    return {"n": n, "pnl_sum": pnl_sum, "win_rate": win_rate, "profit_factor": pf, "max_dd": max_dd}


def objective(train: Dict[str, float], valid: Dict[str, float]) -> float:
    if valid["n"] <= 0:
        return -1e18
    obj = 0.75 * valid["pnl_sum"] + 0.25 * train["pnl_sum"]
    obj -= 0.08 * valid["max_dd"] + 0.02 * train["max_dd"]
    obj -= 0.002 * valid["n"]
    if math.isfinite(valid["profit_factor"]):
        obj += 0.05 * min(5.0, valid["profit_factor"])
    if math.isfinite(train["profit_factor"]):
        obj += 0.01 * min(5.0, train["profit_factor"])
    obj -= 0.10 * abs(valid["pnl_sum"] - train["pnl_sum"])
    return obj


def optimize(rows: List[Row], base_w: np.ndarray, base_allow: float, base_reduce: float, trials: int = 12000, seed: int = 17) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = len(rows)
    split = int(0.70 * n)
    train_rows = rows[:split]
    valid_rows = rows[split:]

    X_train = np.stack([r.x for r in train_rows]) if train_rows else np.zeros((0, 8))
    X_valid = np.stack([r.x for r in valid_rows]) if valid_rows else np.zeros((0, 8))
    pnl_train = np.array([r.pnl_usd for r in train_rows], dtype=float)
    pnl_valid = np.array([r.pnl_usd for r in valid_rows], dtype=float)
    t_train = np.array([r.ts.timestamp() for r in train_rows], dtype=float)
    t_valid = np.array([r.ts.timestamp() for r in valid_rows], dtype=float)

    min_valid_n = max(3, int(0.10 * len(valid_rows))) if valid_rows else 0
    max_valid_n = max(min_valid_n, int(0.45 * len(valid_rows))) if valid_rows else 0

    def sample_w() -> np.ndarray:
        w = base_w.copy()
        for i in range(w.size):
            w[i] *= math.exp(rng.uniform(-0.5, 0.5))
        w[0] = float(min(1.8, max(0.4, w[0])))
        w[1] = float(min(1.6, max(0.1, w[1])))
        w[2] = float(min(1.4, max(0.0, w[2])))
        w[3] = float(min(1.8, max(0.0, w[3])))
        w[4] = float(min(0.40, max(0.0, w[4])))
        w[5] = float(min(1.0, max(0.0, w[5])))
        return w

    def sample_thresholds() -> Tuple[float, float]:
        allow = base_allow + rng.uniform(-0.4, 0.6)
        reduce = base_reduce + rng.uniform(-0.6, 0.4)
        reduce = min(reduce, allow - 0.05)
        return float(max(0.0, allow)), float(max(0.0, reduce))

    candidates: List[Dict[str, Any]] = []
    for _ in range(trials):
        w = sample_w()
        allow_th, reduce_th = sample_thresholds()
        sc_train = compute_scores(X_train, w) if train_rows else np.array([], dtype=float)
        sc_valid = compute_scores(X_valid, w) if valid_rows else np.array([], dtype=float)
        m_train = simulate(pnl_train, t_train, sc_train, allow_th, reduce_th) if train_rows else {"n": 0.0, "pnl_sum": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}
        m_valid = simulate(pnl_valid, t_valid, sc_valid, allow_th, reduce_th) if valid_rows else {"n": 0.0, "pnl_sum": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

        if valid_rows and (m_valid["n"] < min_valid_n or m_valid["n"] > max_valid_n):
            continue
        obj = objective(m_train, m_valid)
        candidates.append(
            {
                "objective": obj,
                "allow": allow_th,
                "reduce": reduce_th,
                "weights": {
                    "w_ml_confidence": float(w[0]),
                    "w_mtf_alignment": float(w[1]),
                    "w_atr_regime": float(w[2]),
                    "w_sr_proximity": float(w[3]),
                    "w_trend_slope": float(w[4]),
                    "w_history_edge": float(w[5]),
                },
                "train": m_train,
                "valid": m_valid,
            }
        )

    candidates.sort(key=lambda x: x["objective"], reverse=True)
    return {
        "n_total": n,
        "split_index": split,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "valid_constraints": {"min_n": min_valid_n, "max_n": max_valid_n},
        "best": candidates[0] if candidates else None,
        "top": candidates[:25],
    }


def main():
    root = Path(__file__).resolve().parent.parent
    audit = root / "logs" / "decision_engine_audit.jsonl"
    ml_path = root / "ml_settings.json"
    rows = load_rows(audit)
    ml = load_ml_settings(ml_path)

    base_w = np.array(
        [
            float(ml.get("decision_engine_w_ml_confidence", 1.0) or 0.0),
            float(ml.get("decision_engine_w_mtf_alignment", 0.7) or 0.0),
            float(ml.get("decision_engine_w_atr_regime", 0.4) or 0.0),
            float(ml.get("decision_engine_w_sr_proximity", 0.7) or 0.0),
            float(ml.get("decision_engine_w_trend_slope", 0.05) or 0.0),
            float(ml.get("decision_engine_w_history_edge", 0.0) or 0.0),
        ],
        dtype=float,
    )
    base_allow = float(ml.get("decision_engine_allow_score", 1.5) or 1.5)
    base_reduce = float(ml.get("decision_engine_reduce_score", 1.25) or 1.25)

    res = optimize(rows, base_w, base_allow, base_reduce, trials=15000, seed=19)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {"audit": str(audit), "ml_settings": str(ml_path)},
        "base": {"allow": base_allow, "reduce": base_reduce, "weights": base_w.tolist()},
        "result": res,
    }
    out_path = root / "docs" / "decision_engine_audit_optimization_v2.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)
    best = res.get("best") if isinstance(res, dict) else None
    if isinstance(best, dict):
        print("best_valid", best.get("valid"), "best_train", best.get("train"))
        print("allow", best.get("allow"), "reduce", best.get("reduce"), "weights", best.get("weights"))


if __name__ == "__main__":
    main()
