import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.decision_engine import DecisionEngineConfig, DecisionEngineThresholds, DecisionEngineWeights, SignalDecisionEngine


def load_config_from_ml_settings(root: Path) -> DecisionEngineConfig:
    cfg = DecisionEngineConfig()
    ml_file = root / "ml_settings.json"
    if not ml_file.exists():
        return cfg
    try:
        data = json.loads(ml_file.read_text(encoding="utf-8"))
    except Exception:
        return cfg

    cfg.enabled = bool(data.get("decision_engine_enabled", cfg.enabled))
    cfg.mode = str(data.get("decision_engine_mode", cfg.mode))
    cfg.thresholds = DecisionEngineThresholds(
        allow_score=float(data.get("decision_engine_allow_score", cfg.thresholds.allow_score)),
        reduce_score=float(data.get("decision_engine_reduce_score", cfg.thresholds.reduce_score)),
    )
    cfg.weights = DecisionEngineWeights(
        w_ml_confidence=float(data.get("decision_engine_w_ml_confidence", cfg.weights.w_ml_confidence)),
        w_mtf_alignment=float(data.get("decision_engine_w_mtf_alignment", cfg.weights.w_mtf_alignment)),
        w_atr_regime=float(data.get("decision_engine_w_atr_regime", cfg.weights.w_atr_regime)),
        w_sr_proximity=float(data.get("decision_engine_w_sr_proximity", cfg.weights.w_sr_proximity)),
        w_trend_slope=float(data.get("decision_engine_w_trend_slope", cfg.weights.w_trend_slope)),
        w_history_edge=float(data.get("decision_engine_w_history_edge", cfg.weights.w_history_edge)),
    )
    cfg.atr_prefer_min_pct = float(data.get("decision_engine_atr_prefer_min_pct", cfg.atr_prefer_min_pct))
    cfg.atr_prefer_max_pct = float(data.get("decision_engine_atr_prefer_max_pct", cfg.atr_prefer_max_pct))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=str, default="logs/ai_entry_audit.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    audit_path = (root / args.audit).resolve()
    if not audit_path.exists():
        raise SystemExit(f"missing: {audit_path}")

    cfg = load_config_from_ml_settings(root)
    engine = SignalDecisionEngine(cfg, root)

    confirm = {}
    outcomes = {}
    with open(audit_path, "r", encoding="utf-8", errors="ignore") as f:
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
                did = rec.get("decision_id") or rec.get("response", {}).get("decision_id")
                if isinstance(did, str) and did:
                    confirm[did] = rec
            if et == "trade_outcome":
                did = rec.get("decision_id")
                if isinstance(did, str) and did:
                    outcomes[did] = rec

    paired = [(did, confirm[did], outcomes[did]) for did in confirm.keys() if did in outcomes]
    if args.limit and args.limit > 0:
        paired = paired[-args.limit :]

    rows = 0
    by_decision = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_sum": 0.0})
    by_symbol_decision = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_sum": 0.0})

    for did, c, o in paired:
        req = c.get("request") if isinstance(c.get("request"), dict) else {}
        sig = req.get("signal") if isinstance(req.get("signal"), dict) else {}
        mc = req.get("market_context") if isinstance(req.get("market_context"), dict) else {}
        ohlcv = mc.get("ohlcv") if isinstance(mc.get("ohlcv"), list) else []
        symbol = req.get("symbol") or c.get("symbol") or "UNKNOWN"
        side = req.get("bot_context", {}).get("side") if isinstance(req.get("bot_context"), dict) else None
        side = side or c.get("side") or ""
        eng = engine.evaluate(symbol=str(symbol), side=str(side), signal_payload=sig, ohlcv=ohlcv)
        d = eng.get("decision", "veto")

        try:
            pnl = float(o.get("pnl_usd") or 0.0)
        except Exception:
            pnl = 0.0
        win = 1 if pnl > 0 else 0

        rows += 1
        by_decision[d]["n"] += 1
        by_decision[d]["wins"] += win
        by_decision[d]["pnl_sum"] += pnl
        key = f"{symbol}:{d}"
        by_symbol_decision[key]["n"] += 1
        by_symbol_decision[key]["wins"] += win
        by_symbol_decision[key]["pnl_sum"] += pnl

    print("paired outcomes:", rows)
    for d in ("allow", "reduce", "veto"):
        n = by_decision[d]["n"]
        wins = by_decision[d]["wins"]
        pnl_sum = by_decision[d]["pnl_sum"]
        wr = (wins / n) if n else 0.0
        print("decision", d, "n", n, "win_rate", round(wr, 3), "pnl_sum", round(pnl_sum, 2))

    top = sorted(by_symbol_decision.items(), key=lambda x: x[1]["n"], reverse=True)[:20]
    print("\nby symbol (top n):")
    for k, v in top:
        n = v["n"]
        wr = (v["wins"] / n) if n else 0.0
        print(k, "n", n, "wr", round(wr, 3), "pnl_sum", round(v["pnl_sum"], 2))


if __name__ == "__main__":
    main()
