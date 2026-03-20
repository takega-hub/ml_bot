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


@dataclass
class Row:
    decision_id: str
    ts: datetime
    symbol: str
    side: str
    pnl_usd: float
    commission: float
    f_ml: float
    f_mtf: float
    f_atr: float
    f_sr: float
    f_trend: float
    f_hist: float
    f_rr: float
    f_barrier: float
    score_logged: float


def load_rows(path: Path) -> List[Row]:
    if not path.exists():
        return []
    entries: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        et = rec.get("event_type")
        did = rec.get("decision_id")
        if not isinstance(did, str) or not did:
            continue
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
        side = str(e.get("side") or "")
        pnl = _safe_float(o.get("pnl_usd"), 0.0)
        commission = _safe_float(o.get("commission"), 0.0)
        engine_eval = e.get("engine_eval") if isinstance(e.get("engine_eval"), dict) else {}
        engine_ctx = engine_eval.get("engine") if isinstance(engine_eval.get("engine"), dict) else {}

        f_ml = _safe_float(engine_ctx.get("ml_score"), 0.0)
        f_mtf = _safe_float(engine_ctx.get("mtf_alignment"), 0.0)
        f_atr = _safe_float(engine_ctx.get("atr_score"), 0.0)
        f_sr = _safe_float(engine_ctx.get("sr_score"), 0.0)
        f_trend = _safe_float(engine_ctx.get("trend_score"), 0.0)
        f_hist = _safe_float(engine_ctx.get("history_edge"), 0.0)
        f_rr = _safe_float(engine_ctx.get("rr_score"), 0.0)
        f_barrier = _safe_float(engine_ctx.get("tp_barrier_score"), 0.0)
        score_logged = _safe_float(engine_eval.get("score"), 0.0)

        rows.append(
            Row(
                decision_id=did,
                ts=ts,
                symbol=symbol,
                side=side,
                pnl_usd=pnl,
                commission=commission,
                f_ml=f_ml,
                f_mtf=f_mtf,
                f_atr=f_atr,
                f_sr=f_sr,
                f_trend=f_trend,
                f_hist=f_hist,
                f_rr=f_rr,
                f_barrier=f_barrier,
                score_logged=score_logged,
            )
        )
    rows.sort(key=lambda r: r.ts)
    return rows


def compute_scores(
    X: np.ndarray,
    w: np.ndarray,
    rr_weight: float = 0.35,
    barrier_weight: float = 0.25,
    mtf_conflict_penalty: float = -0.8,
) -> np.ndarray:
    base = X[:, :6] @ w
    rr = X[:, 6] * rr_weight
    barrier = X[:, 7] * barrier_weight
    mtf = X[:, 1]
    conflict = np.where(mtf < -0.15, mtf_conflict_penalty, 0.0)
    return base + rr + barrier + conflict


def simulate(
    pnl: np.ndarray,
    ts_ord: np.ndarray,
    scores: np.ndarray,
    allow_th: float,
    reduce_th: float,
) -> Dict[str, float]:
    taken = np.where(scores >= reduce_th)[0]
    if taken.size == 0:
        return {"n": 0.0, "pnl_sum": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

    pnl_taken = pnl[taken].copy()
    mult = np.ones_like(pnl_taken)
    allow_mask = scores[taken] >= allow_th
    reduce_mask = ~allow_mask
    if reduce_mask.any():
        reduce_scores = scores[taken][reduce_mask]
        mult[reduce_mask] = np.where(reduce_scores >= (reduce_th + 0.1), 0.5, 0.25)
    pnl_taken *= mult

    n = float(pnl_taken.size)
    wins = float((pnl_taken > 0).sum())
    win_rate = wins / n if n else 0.0
    gross_win = float(pnl_taken[pnl_taken > 0].sum())
    gross_loss = float(np.abs(pnl_taken[pnl_taken < 0]).sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    pnl_sum = float(pnl_taken.sum())

    order = np.argsort(ts_ord[taken])
    equity = np.cumsum(pnl_taken[order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_dd = float(dd.max()) if dd.size else 0.0

    return {
        "n": n,
        "pnl_sum": pnl_sum,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
    }


def random_search(rows: List[Row], trials: int = 5000, seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    X = np.array(
        [[r.f_ml, r.f_mtf, r.f_atr, r.f_sr, r.f_trend, r.f_hist, r.f_rr, r.f_barrier] for r in rows],
        dtype=float,
    )
    pnl = np.array([r.pnl_usd for r in rows], dtype=float)
    ts_ord = np.array([r.ts.timestamp() for r in rows], dtype=float)

    n_total = len(rows)
    min_n = max(6, int(0.35 * n_total))
    max_n = max(min_n, int(0.90 * n_total))

    def eval_one(w: np.ndarray, allow_th: float, reduce_th: float) -> Tuple[float, Dict[str, float]]:
        scores = compute_scores(X, w)
        m = simulate(pnl, ts_ord, scores, allow_th=allow_th, reduce_th=reduce_th)
        if m["n"] < min_n or m["n"] > max_n:
            return -1e18, m
        obj = m["pnl_sum"]
        obj -= 0.05 * m["max_dd"]
        obj -= 0.002 * m["n"]
        if math.isfinite(m["profit_factor"]):
            obj += 0.05 * min(5.0, m["profit_factor"])
        return obj, m

    candidates: List[Tuple[float, Dict[str, Any]]] = []

    def sample_w() -> np.ndarray:
        w_ml = rng.uniform(0.6, 1.6)
        w_mtf = rng.uniform(0.2, 1.4)
        w_atr = rng.uniform(0.0, 1.2)
        w_sr = rng.uniform(0.0, 1.6)
        w_trend = rng.uniform(0.0, 0.35)
        w_hist = rng.uniform(0.0, 0.6)
        return np.array([w_ml, w_mtf, w_atr, w_sr, w_trend, w_hist], dtype=float)

    for _ in range(trials):
        w = sample_w()
        allow_th = rng.uniform(0.6, 2.0)
        reduce_th = rng.uniform(0.2, min(allow_th, 1.8))
        obj, metrics = eval_one(w, allow_th, reduce_th)
        candidates.append(
            (
                obj,
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
                    "metrics": metrics,
                },
            )
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [c[1] for c in candidates[:25]]
    best = top[0] if top else None

    return {"best": best, "top": top, "constraints": {"min_n": min_n, "max_n": max_n}, "n_total": n_total}


def load_ml_settings(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_recompute_error(rows: List[Row], ml_settings: Dict[str, Any]) -> Dict[str, Any]:
    if not rows:
        return {}
    w = np.array(
        [
            float(ml_settings.get("decision_engine_w_ml_confidence", 1.0) or 0.0),
            float(ml_settings.get("decision_engine_w_mtf_alignment", 0.7) or 0.0),
            float(ml_settings.get("decision_engine_w_atr_regime", 0.4) or 0.0),
            float(ml_settings.get("decision_engine_w_sr_proximity", 0.7) or 0.0),
            float(ml_settings.get("decision_engine_w_trend_slope", 0.05) or 0.0),
            float(ml_settings.get("decision_engine_w_history_edge", 0.0) or 0.0),
        ],
        dtype=float,
    )
    X = np.array(
        [[r.f_ml, r.f_mtf, r.f_atr, r.f_sr, r.f_trend, r.f_hist, r.f_rr, r.f_barrier] for r in rows],
        dtype=float,
    )
    scores = compute_scores(X, w)
    logged = np.array([r.score_logged for r in rows], dtype=float)
    err = np.abs(scores - logged)
    return {
        "n": int(err.size),
        "mae": float(err.mean()),
        "p95": float(np.quantile(err, 0.95)),
        "max": float(err.max()),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    audit_path = root / "logs" / "decision_engine_audit.jsonl"
    ml_path = root / "ml_settings.json"

    rows = load_rows(audit_path)
    ml_settings = load_ml_settings(ml_path)
    recon = baseline_recompute_error(rows, ml_settings)

    result = random_search(rows, trials=6000, seed=11) if rows else {"best": None, "top": [], "n_total": 0}

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {"audit": str(audit_path), "ml_settings": str(ml_path)},
        "rows": len(rows),
        "recompute_error_vs_logged": recon,
        "result": result,
    }
    out_path = root / "docs" / "decision_engine_audit_optimization.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)
    best = result.get("best") if isinstance(result, dict) else None
    if isinstance(best, dict):
        print("best", best.get("metrics"), "allow", best.get("allow"), "reduce", best.get("reduce"))


if __name__ == "__main__":
    main()

