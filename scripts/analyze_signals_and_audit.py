import json
import re
from collections import defaultdict
from pathlib import Path


def parse_signals_log(path: Path, max_lines: int = 20000):
    stats = {
        "total": 0,
        "by_symbol": defaultdict(int),
        "by_action": defaultdict(int),
        "conf_sum": defaultdict(float),
        "conf_n": defaultdict(int),
    }
    if not path.exists():
        return stats

    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    lines = lines[-max_lines:]

    pat = re.compile(r"SIGNAL GEN:\s+(?P<symbol>\w+)\s+(?P<action>\w+)\s+Conf=(?P<conf>[0-9.]+)")
    for l in lines:
        m = pat.search(l)
        if not m:
            continue
        symbol = m.group("symbol")
        action = m.group("action")
        try:
            conf = float(m.group("conf"))
        except Exception:
            conf = None
        stats["total"] += 1
        stats["by_symbol"][symbol] += 1
        stats["by_action"][action] += 1
        if conf is not None:
            k = f"{symbol}:{action}"
            stats["conf_sum"][k] += conf
            stats["conf_n"][k] += 1
    return stats


def parse_ai_audit(path: Path):
    stats = {
        "total_confirm": 0,
        "total_outcome": 0,
        "decisions": defaultdict(int),
        "outcomes_by_decision": defaultdict(int),
        "wins_by_decision": defaultdict(int),
        "pnl_sum_by_decision": defaultdict(float),
        "by_symbol_decision": defaultdict(int),
    }
    if not path.exists():
        return stats

    confirm = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
                stats["total_confirm"] += 1
                did = rec.get("decision_id") or rec.get("response", {}).get("decision_id")
                if isinstance(did, str) and did:
                    confirm[did] = rec
                    resp = rec.get("response") if isinstance(rec.get("response"), dict) else {}
                    d = resp.get("decision")
                    if isinstance(d, str):
                        stats["decisions"][d] += 1
                        sym = rec.get("symbol")
                        if sym:
                            stats["by_symbol_decision"][f"{sym}:{d}"] += 1
            if et == "trade_outcome":
                did = rec.get("decision_id")
                if not isinstance(did, str) or did not in confirm:
                    continue
                c = confirm[did]
                resp = c.get("response") if isinstance(c.get("response"), dict) else {}
                d = resp.get("decision")
                if not isinstance(d, str):
                    d = "unknown"
                stats["total_outcome"] += 1
                stats["outcomes_by_decision"][d] += 1
                pnl = rec.get("pnl_usd")
                try:
                    pnl = float(pnl)
                except Exception:
                    pnl = 0.0
                stats["pnl_sum_by_decision"][d] += pnl
                if pnl > 0:
                    stats["wins_by_decision"][d] += 1
    return stats


def main():
    root = Path(__file__).resolve().parent.parent
    signals_path = root / "logs" / "signals.log"
    audit_path = root / "logs" / "ai_entry_audit.jsonl"

    s = parse_signals_log(signals_path)
    a = parse_ai_audit(audit_path)

    print("== signals.log ==")
    print("total SIGNAL GEN:", s["total"])
    top_symbols = sorted(s["by_symbol"].items(), key=lambda x: x[1], reverse=True)[:10]
    print("top symbols:", top_symbols)
    print("by action:", dict(sorted(s["by_action"].items(), key=lambda x: x[0])))
    for k, n in sorted(s["conf_n"].items(), key=lambda x: x[1], reverse=True)[:10]:
        avg = s["conf_sum"][k] / n if n else 0.0
        print("avg conf", k, round(avg, 4), "n=", n)

    print("\n== ai_entry_audit.jsonl ==")
    print("confirm:", a["total_confirm"], "outcome:", a["total_outcome"])
    print("decision counts:", dict(a["decisions"]))
    for d, n in sorted(a["outcomes_by_decision"].items(), key=lambda x: x[1], reverse=True):
        wins = a["wins_by_decision"].get(d, 0)
        pnl = a["pnl_sum_by_decision"].get(d, 0.0)
        wr = wins / n if n else 0.0
        print("decision", d, "outcomes", n, "win_rate", round(wr, 3), "pnl_sum", round(pnl, 2))


if __name__ == "__main__":
    main()

