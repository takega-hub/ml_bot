import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pct(x: float) -> str:
    return f"{x:.2f}%"


def _usd(x: float) -> str:
    return f"{x:+.2f}$"


@dataclass
class TradeRow:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    leverage: int
    margin_usd: float
    commission: float
    dca_count: int
    confidence: float
    sl: Optional[float]
    tp: Optional[float]
    de_score: Optional[float]
    de_decision: Optional[str]


def load_state_trades(path: Path) -> List[TradeRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    trades = data.get("trades") if isinstance(data, dict) else None
    if not isinstance(trades, list):
        return []

    out: List[TradeRow] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        if t.get("status") != "closed":
            continue
        symbol = str(t.get("symbol") or "")
        side = str(t.get("side") or "")
        entry = _safe_float(t.get("entry_price"))
        exitp = _safe_float(t.get("exit_price"))
        pnl_usd = _safe_float(t.get("pnl_usd"), 0.0) or 0.0
        pnl_pct = _safe_float(t.get("pnl_pct"), 0.0) or 0.0
        exit_reason = str(t.get("exit_reason") or "")
        leverage = int(_safe_float(t.get("leverage"), 1) or 1)
        margin = _safe_float(t.get("margin_usd"), 0.0) or 0.0
        commission = _safe_float(t.get("commission"), 0.0) or 0.0
        dca_count = int(_safe_float(t.get("dca_count"), 0) or 0)
        confidence = _safe_float(t.get("confidence"), 0.0) or 0.0
        sl = _safe_float(t.get("stop_loss"))
        tp = _safe_float(t.get("take_profit"))

        de_score = None
        de_decision = None
        params = t.get("signal_parameters") if isinstance(t.get("signal_parameters"), dict) else {}
        de = params.get("decision_engine_eval") if isinstance(params.get("decision_engine_eval"), dict) else None
        if isinstance(de, dict):
            de_score = _safe_float(de.get("score"))
            dd = de.get("decision")
            if isinstance(dd, str):
                de_decision = dd

        if entry is None or exitp is None:
            continue

        out.append(
            TradeRow(
                symbol=symbol,
                side=side,
                entry_price=float(entry),
                exit_price=float(exitp),
                pnl_usd=float(pnl_usd),
                pnl_pct=float(pnl_pct),
                exit_reason=exit_reason,
                leverage=leverage,
                margin_usd=float(margin),
                commission=float(commission),
                dca_count=dca_count,
                confidence=float(confidence),
                sl=sl,
                tp=tp,
                de_score=de_score,
                de_decision=de_decision,
            )
        )
    return out


def parse_trades_log(path: Path, max_lines: int = 200000) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = lines[-max_lines:]
    close_re_fee = re.compile(
        r"TRADE CLOSE:\s+(?P<symbol>\w+)\s+\|\s+Exit:\s+(?P<exit>[0-9.]+)\s+\|\s+PnL:\s+\$(?P<pnl>-?[0-9.]+)\s+\((?P<pnlpct>-?[0-9.]+)%\)\s+\|\s+Fee:\s+\$(?P<fee>-?[0-9.]+)\s+\|\s+Reason:\s+(?P<reason>.+)$"
    )
    close_re_simple = re.compile(
        r"TRADE CLOSE:\s+(?P<symbol>\w+)\s+\|\s+Exit:\s+(?P<exit>[0-9.]+)\s+\|\s+PnL:\s+\$(?P<pnl>-?[0-9.]+)\s+\((?P<pnlpct>-?[0-9.]+)%\)\s+\|\s+Reason:\s+(?P<reason>.+)$"
    )
    reasons = defaultdict(int)
    pnl_sum = 0.0
    pnls: List[float] = []
    wins = 0
    n = 0
    for l in lines:
        m = close_re_fee.search(l) or close_re_simple.search(l)
        if not m:
            continue
        n += 1
        pnl = float(m.group("pnl"))
        pnl_sum += pnl
        pnls.append(pnl)
        if pnl > 0:
            wins += 1
        reason = m.group("reason").strip()
        reason = reason.split("|", 1)[0].strip()
        reasons[reason] += 1
    pnls_sorted = sorted(pnls)
    median = pnls_sorted[len(pnls_sorted) // 2] if pnls_sorted else 0.0
    p95_loss = 0.0
    losses = sorted([x for x in pnls_sorted if x < 0])
    if losses:
        p95_loss = losses[int(0.05 * (len(losses) - 1))]
    return {
        "n_close_lines": n,
        "pnl_sum": pnl_sum,
        "win_rate": (wins / n) if n else 0.0,
        "pnl_avg": (pnl_sum / n) if n else 0.0,
        "pnl_median": median,
        "p95_loss": p95_loss,
        "reasons": dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True)),
    }


def summarize(trades: List[TradeRow]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(trades)
    out["n_closed"] = n
    if n == 0:
        return out

    pnl_sum = sum(t.pnl_usd for t in trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    out["pnl_sum_usd"] = pnl_sum
    out["win_rate"] = wins / n

    by_reason = defaultdict(lambda: {"n": 0, "pnl_sum": 0.0})
    by_symbol = defaultdict(lambda: {"n": 0, "pnl_sum": 0.0, "wins": 0})
    sl_ratios = []
    tp_ratios = []
    for t in trades:
        by_reason[t.exit_reason]["n"] += 1
        by_reason[t.exit_reason]["pnl_sum"] += t.pnl_usd
        by_symbol[t.symbol]["n"] += 1
        by_symbol[t.symbol]["pnl_sum"] += t.pnl_usd
        by_symbol[t.symbol]["wins"] += 1 if t.pnl_usd > 0 else 0
        if t.sl and t.entry_price > 0:
            sl_ratios.append(abs(t.entry_price - t.sl) / t.entry_price)
        if t.tp and t.entry_price > 0:
            tp_ratios.append(abs(t.tp - t.entry_price) / t.entry_price)

    out["by_reason"] = dict(by_reason)
    out["by_symbol_top"] = sorted(
        [{"symbol": k, **v, "win_rate": (v["wins"] / v["n"]) if v["n"] else 0.0} for k, v in by_symbol.items()],
        key=lambda x: (x["n"], x["pnl_sum"]),
        reverse=True,
    )[:20]
    if sl_ratios:
        out["sl_ratio_avg_pct"] = 100.0 * (sum(sl_ratios) / len(sl_ratios))
        out["sl_ratio_p95_pct"] = 100.0 * sorted(sl_ratios)[int(0.95 * (len(sl_ratios) - 1))]
    if tp_ratios:
        out["tp_ratio_avg_pct"] = 100.0 * (sum(tp_ratios) / len(tp_ratios))
        out["tp_ratio_p95_pct"] = 100.0 * sorted(tp_ratios)[int(0.95 * (len(tp_ratios) - 1))]

    de_scores = [t.de_score for t in trades if t.de_score is not None]
    if de_scores:
        de_scores_sorted = sorted(float(x) for x in de_scores)
        out["de_score_avg"] = sum(de_scores_sorted) / len(de_scores_sorted)
        out["de_score_p25"] = de_scores_sorted[int(0.25 * (len(de_scores_sorted) - 1))]
        out["de_score_p75"] = de_scores_sorted[int(0.75 * (len(de_scores_sorted) - 1))]
    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_signals_engine(path: Path, max_lines: int = 300000) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = lines[-max_lines:]
    re_eval = re.compile(r"ENGINE EVAL:\s+(?P<symbol>\w+)\s+action=(?P<action>\w+)\s+score=(?P<score>-?[0-9.]+)\s+decision=(?P<decision>\w+)")
    by_decision = defaultdict(int)
    scores_by_decision: Dict[str, List[float]] = defaultdict(list)
    n = 0
    for l in lines:
        m = re_eval.search(l)
        if not m:
            continue
        n += 1
        d = m.group("decision").lower()
        by_decision[d] += 1
        try:
            scores_by_decision[d].append(float(m.group("score")))
        except Exception:
            pass
    out: Dict[str, Any] = {"n_engine_eval": n, "by_decision": dict(by_decision)}
    for d, xs in scores_by_decision.items():
        xs = sorted(xs)
        if not xs:
            continue
        out[f"score_{d}_avg"] = sum(xs) / len(xs)
        out[f"score_{d}_p25"] = xs[int(0.25 * (len(xs) - 1))]
        out[f"score_{d}_p75"] = xs[int(0.75 * (len(xs) - 1))]
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    state_path = root / "runtime_state.json"
    trades_log_path = root / "logs" / "trades.log"
    signals_log_path = root / "logs" / "signals.log"
    risk_path = root / "risk_settings.json"
    ml_path = root / "ml_settings.json"

    report = {
        "risk_settings": load_json(risk_path),
        "ml_settings": load_json(ml_path),
        "state_closed_trades": summarize(load_state_trades(state_path)) if state_path.exists() else {},
        "trades_log_scan": parse_trades_log(trades_log_path),
        "signals_engine_scan": parse_signals_engine(signals_log_path),
    }

    out_path = root / "docs" / "trade_diagnostics_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)
    print("closed_trades:", report["state_closed_trades"].get("n_closed", 0))
    print("pnl_sum_usd:", round(report["state_closed_trades"].get("pnl_sum_usd", 0.0), 4))


if __name__ == "__main__":
    main()
