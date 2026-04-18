import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
    rows = []
    for did, e in entries.items():
        o = outcomes.get(did)
        if not isinstance(o, dict):
            continue
        engine_eval = e.get("engine_eval") if isinstance(e.get("engine_eval"), dict) else {}
        engine_ctx = engine_eval.get("engine") if isinstance(engine_eval.get("engine"), dict) else {}
        rows.append(
            {
                "decision_id": did,
                "timestamp_utc": e.get("timestamp_utc"),
                "symbol": e.get("symbol"),
                "side": e.get("side"),
                "engine_decision": engine_eval.get("decision"),
                "engine_score": _safe_float(engine_eval.get("score"), 0.0),
                "engine_mult": _safe_float(engine_eval.get("size_multiplier"), 1.0),
                "ml_conf": _safe_float(engine_ctx.get("ml_conf"), 0.0),
                "ml_score": _safe_float(engine_ctx.get("ml_score"), 0.0),
                "mtf_alignment": _safe_float(engine_ctx.get("mtf_alignment"), 0.0),
                "atr_pct": _safe_float(engine_ctx.get("atr_pct"), 0.0),
                "atr_score": _safe_float(engine_ctx.get("atr_score"), 0.0),
                "sr_score": _safe_float(engine_ctx.get("sr_score"), 0.0),
                "trend_score": _safe_float(engine_ctx.get("trend_score"), 0.0),
                "history_edge": _safe_float(engine_ctx.get("history_edge"), 0.0),
                "rr_ratio": _safe_float(engine_ctx.get("rr_ratio"), 0.0),
                "rr_score": _safe_float(engine_ctx.get("rr_score"), 0.0),
                "tp_barrier_score": _safe_float(engine_ctx.get("tp_barrier_score"), 0.0),
                "pnl_usd": _safe_float(o.get("pnl_usd"), 0.0),
                "pnl_pct": _safe_float(o.get("pnl_pct"), 0.0),
                "commission": _safe_float(o.get("commission"), 0.0),
                "exit_reason": o.get("exit_reason"),
            }
        )
    rows.sort(key=lambda r: str(r.get("timestamp_utc") or ""))
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    pnl = sum(float(r.get("pnl_usd") or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl_usd") or 0.0) > 0)
    gross_win = sum(float(r.get("pnl_usd") or 0.0) for r in rows if float(r.get("pnl_usd") or 0.0) > 0)
    gross_loss = sum(abs(float(r.get("pnl_usd") or 0.0)) for r in rows if float(r.get("pnl_usd") or 0.0) < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    return {"n": n, "pnl_sum": pnl, "win_rate": wins / n, "profit_factor": pf}


def score_bins(rows: List[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    xs = sorted(float(r.get("engine_score") or 0.0) for r in rows)
    if not xs:
        return []
    cuts = []
    for i in range(1, bins):
        idx = int((i / bins) * (len(xs) - 1))
        cuts.append(xs[idx])
    out = []
    for bi in range(bins):
        lo = -1e9 if bi == 0 else cuts[bi - 1]
        hi = 1e9 if bi == bins - 1 else cuts[bi]
        bucket = [r for r in rows if float(r.get("engine_score") or 0.0) >= lo and float(r.get("engine_score") or 0.0) <= hi]
        s = _summary(bucket)
        out.append({"bin": bi + 1, "lo": lo, "hi": hi, **s})
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    audit = root / "logs" / "decision_engine_audit.jsonl"
    rows = load_pairs(audit)

    by_symbol = defaultdict(list)
    by_dec = defaultdict(list)
    for r in rows:
        by_symbol[str(r.get("symbol") or "")].append(r)
        by_dec[str(r.get("engine_decision") or "")].append(r)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "overall": _summary(rows),
        "by_engine_decision": {k: _summary(v) for k, v in by_dec.items()},
        "top_symbols_by_n": sorted(
            [{"symbol": k, **_summary(v)} for k, v in by_symbol.items()],
            key=lambda x: x["n"],
            reverse=True,
        )[:20],
        "score_bins": score_bins(rows, bins=10),
    }

    out_path = root / "docs" / "decision_engine_audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)
    print("rows:", report["rows"], "overall:", report["overall"])


if __name__ == "__main__":
    main()

