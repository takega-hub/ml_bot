import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_ts_prefix(line: str) -> Optional[datetime]:
    try:
        # format: "2026-03-17 12:45:15 - ..."
        ts = line[:19]
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


@dataclass
class EngineEval:
    ts: datetime
    symbol: str
    action: str  # LONG/SHORT
    score: float


@dataclass
class TradeEvent:
    ts: datetime
    symbol: str
    side: str  # Buy/Sell


@dataclass
class TradeClose:
    ts: datetime
    symbol: str
    pnl_usd: float
    pnl_pct: float
    reason: str


@dataclass
class LabeledTrade:
    ts_open: datetime
    ts_close: datetime
    symbol: str
    action: str  # LONG/SHORT
    score: float
    pnl_usd: float
    pnl_pct: float
    close_reason: str


def load_engine_evals(signals_log: Path, max_lines: int = 400000) -> List[EngineEval]:
    if not signals_log.exists():
        return []
    lines = signals_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    re_eval = re.compile(
        r"ENGINE EVAL:\s+(?P<symbol>\w+)\s+action=(?P<action>\w+)\s+score=(?P<score>-?[0-9.]+)\s+decision=(?P<decision>\w+)"
    )
    out: List[EngineEval] = []
    for l in lines:
        m = re_eval.search(l)
        if not m:
            continue
        ts = _parse_ts_prefix(l)
        if ts is None:
            continue
        symbol = m.group("symbol")
        action = m.group("action").upper()
        try:
            score = float(m.group("score"))
        except Exception:
            continue
        out.append(EngineEval(ts=ts, symbol=symbol, action=action, score=score))
    out.sort(key=lambda x: x.ts)
    return out


def load_trade_events(trades_log: Path, max_lines: int = 400000) -> Tuple[List[TradeEvent], List[TradeClose]]:
    if not trades_log.exists():
        return [], []
    lines = trades_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    re_open = re.compile(r"TRADE OPEN:\s+(?P<symbol>\w+)\s+\|\s+Side:\s+(?P<side>\w+)\s+\|")
    re_close = re.compile(
        r"TRADE CLOSE:\s+(?P<symbol>\w+)\s+\|\s+Exit:\s+(?P<exit>[0-9.]+)\s+\|\s+PnL:\s+\$(?P<pnl>-?[0-9.]+)\s+\((?P<pnlpct>-?[0-9.]+)%\)\s+(?:\|\s+Fee:\s+\$(?P<fee>-?[0-9.]+)\s+)?\|\s+Reason:\s+(?P<reason>.+?)(?:\s+\|.*)?$"
    )
    opens: List[TradeEvent] = []
    closes: List[TradeClose] = []
    for l in lines:
        ts = _parse_ts_prefix(l)
        if ts is None:
            continue
        mo = re_open.search(l)
        if mo:
            opens.append(TradeEvent(ts=ts, symbol=mo.group("symbol"), side=mo.group("side")))
            continue
        mc = re_close.search(l)
        if mc:
            reason = mc.group("reason").strip()
            reason = reason.split("|", 1)[0].strip()
            try:
                pnl = float(mc.group("pnl"))
                pnlpct = float(mc.group("pnlpct"))
            except Exception:
                continue
            closes.append(
                TradeClose(ts=ts, symbol=mc.group("symbol"), pnl_usd=pnl, pnl_pct=pnlpct, reason=reason)
            )
    opens.sort(key=lambda x: x.ts)
    closes.sort(key=lambda x: x.ts)
    return opens, closes


def _side_to_action(side: str) -> Optional[str]:
    s = str(side).lower()
    if s == "buy":
        return "LONG"
    if s == "sell":
        return "SHORT"
    return None


def label_trades(
    engine_evals: List[EngineEval],
    opens: List[TradeEvent],
    closes: List[TradeClose],
    match_window_seconds: int = 21600,
) -> List[LabeledTrade]:
    evals_by_symbol: Dict[str, List[EngineEval]] = {}
    for e in engine_evals:
        evals_by_symbol.setdefault(e.symbol, []).append(e)

    open_stack: Dict[str, List[TradeEvent]] = {}
    for o in opens:
        open_stack.setdefault(o.symbol, []).append(o)

    labeled: List[LabeledTrade] = []
    for c in closes:
        st = open_stack.get(c.symbol)
        if not st:
            continue
        o = st.pop(0)
        action = _side_to_action(o.side)
        if action is None:
            continue

        evs = evals_by_symbol.get(c.symbol, [])
        best: Optional[EngineEval] = None
        for e in reversed(evs):
            if e.ts > o.ts:
                continue
            if (o.ts - e.ts).total_seconds() > match_window_seconds:
                break
            if e.action == action:
                best = e
                break
        if best is None:
            continue

        labeled.append(
            LabeledTrade(
                ts_open=o.ts,
                ts_close=c.ts,
                symbol=c.symbol,
                action=action,
                score=best.score,
                pnl_usd=c.pnl_usd,
                pnl_pct=c.pnl_pct,
                close_reason=c.reason,
            )
        )
    return labeled


def simulate(labeled: List[LabeledTrade], allow: float, reduce: float) -> Dict[str, float]:
    pnl = 0.0
    n = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    for t in labeled:
        if t.score >= allow:
            mult = 1.0
        elif t.score >= reduce:
            mult = 0.25
        else:
            continue
        pnl_t = t.pnl_usd * mult
        pnl += pnl_t
        n += 1
        if pnl_t > 0:
            wins += 1
            gross_win += pnl_t
        else:
            gross_loss += abs(pnl_t)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    wr = (wins / n) if n else 0.0
    return {"pnl_sum": pnl, "n": float(n), "win_rate": wr, "profit_factor": pf}


def grid_search(labeled: List[LabeledTrade]) -> Dict[str, object]:
    if not labeled:
        return {"best": None, "top": [], "n_labeled": 0}
    base = simulate(labeled, allow=-999.0, reduce=-999.0)
    base_n = int(base["n"])
    min_n = max(5, int(0.10 * base_n))

    results = []
    allow_grid = [x / 100 for x in range(20, 161, 5)]  # 0.20..1.60
    reduce_grid = [x / 100 for x in range(-20, 121, 5)]  # -0.20..1.20
    for a in allow_grid:
        for r in reduce_grid:
            if r > a:
                continue
            m = simulate(labeled, allow=a, reduce=r)
            if int(m["n"]) < min_n:
                continue
            score = float(m["pnl_sum"])  # primary objective
            # mild penalty for too many trades (encourage filtering)
            score -= 0.0005 * float(m["n"])
            results.append((score, a, r, m))
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:20]
    best = top[0] if top else None
    return {
        "n_labeled": len(labeled),
        "baseline": base,
        "min_trades_constraint": min_n,
        "best": None
        if best is None
        else {
            "objective": best[0],
            "allow": best[1],
            "reduce": best[2],
            "metrics": best[3],
        },
        "top": [
            {"objective": s, "allow": a, "reduce": r, "metrics": m}
            for (s, a, r, m) in top
        ],
    }


def main():
    root = Path(__file__).resolve().parent.parent
    signals = root / "logs" / "signals.log"
    trades = root / "logs" / "trades.log"

    engine_evals = load_engine_evals(signals)
    opens, closes = load_trade_events(trades)
    labeled = label_trades(engine_evals, opens, closes, match_window_seconds=21600)
    search = grid_search(labeled)

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "paths": {"signals": str(signals), "trades": str(trades)},
        "engine_evals": len(engine_evals),
        "trade_opens": len(opens),
        "trade_closes": len(closes),
        "match_window_seconds": 21600,
        "result": search,
    }

    out_path = root / "docs" / "decision_engine_optimization.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)
    print("labeled:", search.get("n_labeled", 0))
    best = search.get("best") if isinstance(search, dict) else None
    if isinstance(best, dict):
        print("best_allow:", best.get("allow"), "best_reduce:", best.get("reduce"), "metrics:", best.get("metrics"))


if __name__ == "__main__":
    main()
