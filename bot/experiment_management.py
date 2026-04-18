import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_code_version(project_root: Path) -> Dict[str, Any]:
    sha = os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA")
    dirty: Optional[bool] = None
    if not sha:
        try:
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(project_root),
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except Exception:
            sha = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return {"git_sha": sha, "git_dirty": dirty}


def build_param_signature(
    symbol: str,
    experiment_type: str,
    params: Dict[str, Any],
    hyperparams: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "symbol": symbol.upper(),
        "type": experiment_type,
        "params": params or {},
        "hyperparams": hyperparams or {},
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


@dataclass(frozen=True)
class ExperimentCriteria:
    min_total_trades: int = 30
    min_profit_factor: float = 1.1
    min_total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 25.0


def evaluate_quality_gates(metrics: Dict[str, Any], criteria: Optional[ExperimentCriteria] = None) -> Dict[str, Any]:
    c = criteria or ExperimentCriteria()
    trades = int(metrics.get("total_trades") or 0)
    pf = float(metrics.get("profit_factor") or 0.0)
    pnl = float(metrics.get("total_pnl_pct") or 0.0)
    dd = float(metrics.get("max_drawdown_pct") or metrics.get("max_drawdown") or 0.0)
    checks = {
        "min_total_trades": trades >= c.min_total_trades,
        "min_profit_factor": pf >= c.min_profit_factor,
        "min_total_pnl_pct": pnl >= c.min_total_pnl_pct,
        "max_drawdown_pct": dd <= c.max_drawdown_pct,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "criteria": {
            "min_total_trades": c.min_total_trades,
            "min_profit_factor": c.min_profit_factor,
            "min_total_pnl_pct": c.min_total_pnl_pct,
            "max_drawdown_pct": c.max_drawdown_pct,
        },
    }


def compute_unified_score(metrics: Dict[str, Any], criteria: Optional[ExperimentCriteria] = None) -> Dict[str, Any]:
    gates = evaluate_quality_gates(metrics, criteria=criteria)
    pnl = float(metrics.get("total_pnl_pct") or 0.0)
    dd = float(metrics.get("max_drawdown_pct") or metrics.get("max_drawdown") or 0.0)
    sharpe = float(metrics.get("sharpe_ratio") or 0.0)
    win_rate = float(metrics.get("win_rate") or 0.0)
    pf = float(metrics.get("profit_factor") or 0.0)
    trades = float(metrics.get("total_trades") or 0.0)
    stability = max(0.0, min(1.0, (sharpe + 1.0) / 3.0))
    quality = max(0.0, min(1.0, ((win_rate / 100.0) * 0.6) + (min(max(pf, 0.0), 3.0) / 3.0) * 0.4))
    score = pnl - (0.5 * dd) + (8.0 * stability) + (6.0 * quality) + (0.01 * trades)
    if not gates["passed"]:
        score -= 5.0
    return {
        "score": score,
        "components": {
            "pnl": pnl,
            "drawdown": dd,
            "stability": stability,
            "trade_quality": quality,
            "trades": trades,
        },
        "gates": gates,
    }


class ExperimentStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def read_all(self) -> Dict[str, Dict[str, Any]]:
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        tmp = self.file_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(self.file_path)

    def upsert(self, experiment_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        data = self.read_all()
        current = data.get(experiment_id, {})
        merged = {**current, **patch}
        data[experiment_id] = merged
        self.write_all(data)
        return merged

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        data = self.read_all()
        exp = data.get(experiment_id)
        return exp if isinstance(exp, dict) else None

    def list(self) -> List[Dict[str, Any]]:
        data = self.read_all()
        experiments: List[Dict[str, Any]] = []
        for experiment_id, exp in data.items():
            if not isinstance(exp, dict):
                continue
            if not isinstance(exp.get("id"), str) or not str(exp.get("id") or "").strip():
                exp = dict(exp)
                exp["id"] = str(experiment_id)
            experiments.append(exp)
        experiments.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        return experiments


class ExperimentAnalyzer:
    def __init__(self, experiments: Iterable[Dict[str, Any]]):
        self.experiments = [e for e in experiments if isinstance(e, dict)]

    def filter(self, symbol: Optional[str] = None, experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        out = self.experiments
        if symbol:
            out = [e for e in out if str(e.get("symbol", "")).upper() == symbol.upper()]
        if experiment_type:
            out = [e for e in out if str(e.get("type", "")) == experiment_type]
        return out

    def _extract_score_row(self, e: Dict[str, Any]) -> Dict[str, Any]:
        results = e.get("results") if isinstance(e.get("results"), dict) else {}
        return {
            "id": e.get("id"),
            "created_at": e.get("created_at"),
            "status": e.get("status"),
            "symbol": e.get("symbol"),
            "type": e.get("type"),
            "interval": (e.get("params") or {}).get("interval") if isinstance(e.get("params"), dict) else None,
            "no_mtf": (e.get("params") or {}).get("no_mtf") if isinstance(e.get("params"), dict) else None,
            "total_pnl_pct": results.get("total_pnl_pct"),
            "profit_factor": results.get("profit_factor"),
            "win_rate": results.get("win_rate"),
            "max_drawdown_pct": results.get("max_drawdown_pct") or results.get("max_drawdown"),
            "total_trades": results.get("total_trades"),
            "models": results.get("models") if isinstance(results.get("models"), dict) else None,
            "signature": e.get("param_signature"),
            "market_regime": e.get("market_regime") if isinstance(e.get("market_regime"), dict) else None,
            "analysis_summary": results.get("analysis_summary") or e.get("analysis_summary"),
            "hypothesis": e.get("hypothesis") or results.get("hypothesis"),
            "expected_outcome": e.get("expected_outcome") or results.get("expected_outcome"),
            "param_changes": e.get("param_changes") if isinstance(e.get("param_changes"), dict) else None,
        }

    def summarize(self, symbol: Optional[str] = None, experiment_type: Optional[str] = None) -> Dict[str, Any]:
        subset = self.filter(symbol=symbol, experiment_type=experiment_type)
        rows = [self._extract_score_row(e) for e in subset]
        completed = [r for r in rows if r.get("status") == "completed"]
        completed_sorted = sorted(
            completed,
            key=lambda r: (r.get("total_pnl_pct") is None, -(float(r.get("total_pnl_pct") or 0.0))),
        )
        best = completed_sorted[0] if completed_sorted else None
        regime_breakdown: Dict[str, Dict[str, Any]] = {}
        for r in completed:
            regime = r.get("market_regime") if isinstance(r.get("market_regime"), dict) else {}
            regime_key = str(regime.get("regime") or "unknown")
            stats = regime_breakdown.setdefault(regime_key, {"count": 0, "pnls": [], "dds": [], "win_rates": []})
            stats["count"] += 1
            if r.get("total_pnl_pct") is not None:
                stats["pnls"].append(float(r.get("total_pnl_pct") or 0.0))
            if r.get("max_drawdown_pct") is not None:
                stats["dds"].append(float(r.get("max_drawdown_pct") or 0.0))
            if r.get("win_rate") is not None:
                stats["win_rates"].append(float(r.get("win_rate") or 0.0))
        regime_stats: List[Dict[str, Any]] = []
        for regime_key, stats in sorted(regime_breakdown.items(), key=lambda kv: kv[0]):
            pnls = stats.get("pnls") or []
            dds = stats.get("dds") or []
            wrs = stats.get("win_rates") or []
            regime_stats.append(
                {
                    "regime": regime_key,
                    "count": stats["count"],
                    "avg_total_pnl_pct": sum(pnls) / len(pnls) if pnls else None,
                    "avg_max_drawdown_pct": sum(dds) / len(dds) if dds else None,
                    "avg_win_rate": sum(wrs) / len(wrs) if wrs else None,
                }
            )
        return {"count": len(rows), "rows": rows, "best": best, "regime_stats": regime_stats}

    def compute_param_impact(self, symbol: str) -> Dict[str, Any]:
        subset = [r for r in (self._extract_score_row(e) for e in self.filter(symbol=symbol)) if r.get("status") == "completed"]
        by_interval: Dict[str, List[Dict[str, Any]]] = {}
        for r in subset:
            interval = str(r.get("interval") or "unknown")
            by_interval.setdefault(interval, []).append(r)
        interval_stats = []
        for interval, rows in sorted(by_interval.items(), key=lambda kv: kv[0]):
            pnls = [float(x.get("total_pnl_pct") or 0.0) for x in rows if x.get("total_pnl_pct") is not None]
            pfs = [float(x.get("profit_factor") or 0.0) for x in rows if x.get("profit_factor") is not None]
            interval_stats.append(
                {
                    "interval": interval,
                    "count": len(rows),
                    "avg_total_pnl_pct": sum(pnls) / len(pnls) if pnls else None,
                    "avg_profit_factor": sum(pfs) / len(pfs) if pfs else None,
                }
            )
        return {"symbol": symbol.upper(), "interval_stats": interval_stats, "experiments": subset}

    def summarize_regime_memory(self, symbol: str) -> Dict[str, Any]:
        subset = [r for r in (self._extract_score_row(e) for e in self.filter(symbol=symbol)) if r.get("status") == "completed"]

        def _parse_iso(ts: Any) -> Optional[datetime]:
            if not isinstance(ts, str) or not ts.strip():
                return None
            raw = ts.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return None

        def _safe_std(values: List[float]) -> Optional[float]:
            if len(values) <= 1:
                return None
            mean = sum(values) / float(len(values))
            var = sum((float(v) - mean) ** 2 for v in values) / float(len(values))
            return var ** 0.5

        def _to_float(v: Any) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v.strip():
                try:
                    return float(v.strip())
                except Exception:
                    return None
            return None

        def _top_fail_reason(fail_counts: Dict[str, int]) -> Optional[str]:
            if not fail_counts:
                return None
            return max(fail_counts.items(), key=lambda kv: (int(kv[1]), kv[0]))[0]

        grouped: Dict[str, Dict[str, Any]] = {}
        for r in subset:
            regime = r.get("market_regime") if isinstance(r.get("market_regime"), dict) else {}
            regime_key = str(regime.get("regime") or "unknown")
            row = grouped.setdefault(
                regime_key,
                {
                    "regime": regime_key,
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "best_experiment_id": None,
                    "best_score": None,
                    "best_total_pnl_pct": None,
                    "common_param_changes": {},
                    "avoid_signatures": [],
                    "avoid_signatures_meta": {},
                    "fail_reason_counts": {},
                    "sample_size_effective": 0,
                    "pnl_values": [],
                    "dd_values": [],
                    "wr_values": [],
                    "score_values": [],
                    "first_seen_at": None,
                    "last_seen_at": None,
                },
            )
            created_dt = _parse_iso(r.get("created_at"))
            if created_dt is not None:
                prev_first = row.get("first_seen_at")
                prev_last = row.get("last_seen_at")
                if prev_first is None or created_dt < prev_first:
                    row["first_seen_at"] = created_dt
                if prev_last is None or created_dt > prev_last:
                    row["last_seen_at"] = created_dt
            metrics = {
                "total_pnl_pct": r.get("total_pnl_pct"),
                "profit_factor": r.get("profit_factor"),
                "win_rate": r.get("win_rate"),
                "max_drawdown_pct": r.get("max_drawdown_pct"),
                "total_trades": r.get("total_trades"),
            }
            evaluation = compute_unified_score(metrics)
            score = float(evaluation.get("score") or 0.0)
            passed = bool((evaluation.get("gates") or {}).get("passed"))
            row["count"] += 1
            row["score_values"].append(score)
            pnl_value = _to_float(r.get("total_pnl_pct"))
            dd_value = _to_float(r.get("max_drawdown_pct"))
            wr_value = _to_float(r.get("win_rate"))
            trades_value = _to_float(r.get("total_trades"))
            if pnl_value is not None and trades_value is not None:
                row["sample_size_effective"] += 1
            if pnl_value is not None:
                row["pnl_values"].append(pnl_value)
            if dd_value is not None:
                row["dd_values"].append(dd_value)
            if wr_value is not None:
                row["wr_values"].append(wr_value)
            if passed:
                row["successful"] += 1
            else:
                row["failed"] += 1
                checks = (evaluation.get("gates") or {}).get("checks") if isinstance((evaluation.get("gates") or {}).get("checks"), dict) else {}
                current_fail_reasons = []
                for key, ok in checks.items():
                    if not bool(ok):
                        reason = str(key)
                        current_fail_reasons.append(reason)
                        row["fail_reason_counts"][reason] = int(row["fail_reason_counts"].get(reason) or 0) + 1
                signature = r.get("signature")
                if isinstance(signature, str) and signature and signature not in row["avoid_signatures"]:
                    row["avoid_signatures"].append(signature)
                if isinstance(signature, str) and signature:
                    sign_meta = row["avoid_signatures_meta"].setdefault(
                        signature,
                        {
                            "signature": signature,
                            "failed_count": 0,
                            "last_failed_at": None,
                            "reason": None,
                        },
                    )
                    sign_meta["failed_count"] = int(sign_meta.get("failed_count") or 0) + 1
                    if created_dt is not None:
                        prev_fail_dt = sign_meta.get("last_failed_at")
                        if not isinstance(prev_fail_dt, datetime) or created_dt > prev_fail_dt:
                            sign_meta["last_failed_at"] = created_dt
                    if sign_meta.get("reason") is None:
                        sign_meta["reason"] = _top_fail_reason({x: 1 for x in current_fail_reasons}) or "quality_gates_failed"
            if row["best_score"] is None or score > float(row["best_score"]):
                row["best_score"] = score
                row["best_experiment_id"] = r.get("id")
                row["best_total_pnl_pct"] = r.get("total_pnl_pct")
            param_changes = r.get("param_changes") if isinstance(r.get("param_changes"), dict) else {}
            for key, value in param_changes.items():
                bucket = row["common_param_changes"].setdefault(key, {})
                label = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                bucket[label] = int(bucket.get(label) or 0) + 1
        memories = []
        for regime_key, row in sorted(grouped.items(), key=lambda kv: kv[0]):
            common_changes = []
            for key, counts in row["common_param_changes"].items():
                best_label = max(counts.items(), key=lambda kv: kv[1])[0]
                common_changes.append({"param": key, "value": best_label, "count": counts[best_label]})
            count = int(row.get("count") or 0)
            successful = int(row.get("successful") or 0)
            failed = int(row.get("failed") or 0)
            success_rate = (float(successful) / float(count)) if count > 0 else 0.0
            pnl_std = _safe_std([float(x) for x in row.get("pnl_values") or []])
            dd_std = _safe_std([float(x) for x in row.get("dd_values") or []])
            wr_std = _safe_std([float(x) for x in row.get("wr_values") or []])
            score_std = _safe_std([float(x) for x in row.get("score_values") or []])
            sample_size_effective = int(row.get("sample_size_effective") or 0)
            sample_factor = min(1.0, float(sample_size_effective) / 10.0)
            stability_penalty = 0.0
            if pnl_std is not None:
                stability_penalty += min(1.0, float(pnl_std) / 20.0)
            if dd_std is not None:
                stability_penalty += min(1.0, float(dd_std) / 15.0)
            if wr_std is not None:
                stability_penalty += min(1.0, float(wr_std) / 30.0)
            stability_penalty /= 3.0
            fail_pressure = min(1.0, float(failed) / float(max(count, 1)))
            confidence = 0.15 + (0.55 * sample_factor) + (0.35 * success_rate) - (0.25 * stability_penalty) - (0.25 * fail_pressure)
            confidence = max(0.0, min(1.0, confidence))
            fail_reason_counts = row.get("fail_reason_counts") if isinstance(row.get("fail_reason_counts"), dict) else {}
            gate_fail_reasons_top = [
                {"reason": k, "count": int(v)}
                for k, v in sorted(fail_reason_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:5]
            ]
            avoid_signatures_meta = []
            for _, item in sorted(
                (row.get("avoid_signatures_meta") or {}).items(),
                key=lambda kv: (-int((kv[1] or {}).get("failed_count") or 0), kv[0]),
            )[:10]:
                fail_dt = item.get("last_failed_at")
                avoid_signatures_meta.append(
                    {
                        "signature": item.get("signature"),
                        "failed_count": int(item.get("failed_count") or 0),
                        "last_failed_at": fail_dt.isoformat() if isinstance(fail_dt, datetime) else None,
                        "reason": item.get("reason"),
                    }
                )
            first_seen_at = row.get("first_seen_at")
            last_seen_at = row.get("last_seen_at")
            memories.append(
                {
                    "regime": regime_key,
                    "count": count,
                    "sample_size_effective": sample_size_effective,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": success_rate,
                    "best_experiment_id": row["best_experiment_id"],
                    "best_score": row["best_score"],
                    "best_total_pnl_pct": row["best_total_pnl_pct"],
                    "first_seen_at": first_seen_at.isoformat() if isinstance(first_seen_at, datetime) else None,
                    "last_seen_at": last_seen_at.isoformat() if isinstance(last_seen_at, datetime) else None,
                    "stability": {
                        "pnl_std": pnl_std,
                        "dd_std": dd_std,
                        "wr_std": wr_std,
                        "score_std": score_std,
                    },
                    "gate_fail_reasons_top": gate_fail_reasons_top,
                    "common_param_changes": sorted(common_changes, key=lambda x: (-int(x["count"]), x["param"]))[:5],
                    "avoid_signatures": row["avoid_signatures"][:10],
                    "avoid_signatures_meta": avoid_signatures_meta,
                    "confidence": confidence,
                }
            )
        return {"symbol": symbol.upper(), "regimes": memories}

    def build_campaign_notebook(self, root_experiment_id: str) -> Dict[str, Any]:
        related = []
        for e in self.experiments:
            campaign = e.get("ai_campaign") if isinstance(e.get("ai_campaign"), dict) else {}
            root_id = str(campaign.get("root_experiment_id") or e.get("id") or "")
            if root_id == str(root_experiment_id):
                related.append(e)
        related.sort(key=lambda x: int(((x.get("ai_campaign") or {}).get("iteration") or 0)))
        entries = []
        prev_metrics: Optional[Dict[str, float]] = None
        prev_entry_id: Optional[str] = None
        stop_reason: Optional[str] = None
        stop_reason_detail: Optional[Dict[str, Any]] = None

        def _to_float(v: Any) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v.strip():
                try:
                    return float(v.strip())
                except Exception:
                    return None
            return None

        def _expected_intent(expected_text: Optional[str]) -> str:
            if not isinstance(expected_text, str):
                return "generic"
            text = expected_text.lower()
            if ("сниз" in text or "reduce" in text) and ("dd" in text or "drawdown" in text):
                return "reduce_drawdown"
            if "стабил" in text or "stability" in text:
                return "improve_stability"
            if "pnl" in text or "профит" in text or "доход" in text:
                return "increase_pnl"
            return "generic"

        def _build_expected_vs_actual(
            expected_text: Optional[str],
            score: Optional[float],
            pnl: Optional[float],
            dd: Optional[float],
            wr: Optional[float],
            trades: Optional[float],
            prev_vals: Optional[Dict[str, float]],
        ) -> Dict[str, Any]:
            intent = _expected_intent(expected_text)
            checks: Dict[str, Optional[bool]] = {
                "profit_positive": (pnl is not None and pnl > 0.0) if pnl is not None else None,
                "drawdown_within_guardrail": (dd is not None and dd <= 25.0) if dd is not None else None,
                "trades_sufficient": (trades is not None and trades >= 30.0) if trades is not None else None,
                "score_positive": (score is not None and score > 0.0) if score is not None else None,
            }
            if intent == "reduce_drawdown":
                prev_dd = prev_vals.get("dd") if isinstance(prev_vals, dict) else None
                checks["intent_match"] = (dd is not None and prev_dd is not None and dd <= prev_dd) if (dd is not None and prev_dd is not None) else None
            elif intent == "increase_pnl":
                prev_pnl = prev_vals.get("pnl") if isinstance(prev_vals, dict) else None
                checks["intent_match"] = (pnl is not None and prev_pnl is not None and pnl >= prev_pnl) if (pnl is not None and prev_pnl is not None) else None
            elif intent == "improve_stability":
                checks["intent_match"] = checks.get("drawdown_within_guardrail")
            else:
                checks["intent_match"] = checks.get("score_positive")
            check_values = [v for v in checks.values() if isinstance(v, bool)]
            overall = all(check_values) if check_values else None
            return {
                "intent": intent,
                "expected_text": expected_text,
                "actual": {
                    "score": score,
                    "pnl_pct": pnl,
                    "max_drawdown_pct": dd,
                    "win_rate": wr,
                    "total_trades": trades,
                },
                "checks": checks,
                "overall_met": overall,
            }

        def _build_decision_trace(
            tactic: Optional[str],
            score: Optional[float],
            selection_payload: Dict[str, Any],
            rationale: Optional[str],
        ) -> Dict[str, Any]:
            candidates = selection_payload.get("candidates") if isinstance(selection_payload.get("candidates"), list) else []
            normalized = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                cand_tactic = str(c.get("tactic") or "")
                cand_score = _to_float(c.get("score"))
                normalized.append(
                    {
                        "tactic": cand_tactic,
                        "score": cand_score,
                        "oos_pass": bool(c.get("oos_pass")),
                        "walk_forward_stability_pass": bool(c.get("walk_forward_stability_pass")),
                        "quality_gates_passed": bool(((c.get("quality_gates") or {}).get("passed")) if isinstance(c.get("quality_gates"), dict) else False),
                    }
                )
            normalized.sort(key=lambda x: (x.get("score") is None, -(float(x.get("score") or 0.0))))
            chosen = next((x for x in normalized if x.get("tactic") == str(tactic or "")), None)
            alternatives = [x for x in normalized if x.get("tactic") != str(tactic or "")][:2]
            why_not_others = []
            chosen_score = _to_float(score if score is not None else (chosen or {}).get("score"))
            for alt in alternatives:
                alt_score = _to_float(alt.get("score"))
                why_not_others.append(
                    {
                        "tactic": alt.get("tactic"),
                        "score": alt_score,
                        "delta_vs_chosen": (chosen_score - alt_score) if (chosen_score is not None and alt_score is not None) else None,
                    }
                )
            top_tradeoffs = []
            if isinstance(chosen, dict):
                top_tradeoffs = [
                    {"metric": "quality_gates_passed", "value": chosen.get("quality_gates_passed")},
                    {"metric": "oos_pass", "value": chosen.get("oos_pass")},
                    {"metric": "walk_forward_stability_pass", "value": chosen.get("walk_forward_stability_pass")},
                ]
            return {
                "chosen_tactic": tactic,
                "chosen_score": chosen_score,
                "why_this_tactic": rationale or f"Выбран {str(tactic or 'unknown')} по максимальному composite score.",
                "why_not_others": why_not_others,
                "top_tradeoffs": top_tradeoffs,
                "candidate_rankings": normalized[:4],
            }

        for e in related:
            results = e.get("results") if isinstance(e.get("results"), dict) else {}
            tactic = results.get("recommended_tactic")
            wf = results.get("walk_forward") if isinstance(results.get("walk_forward"), dict) else {}
            oos = results.get("oos_validation") if isinstance(results.get("oos_validation"), dict) else {}
            wf_for_tactic = wf.get(tactic) if isinstance(wf.get(tactic), dict) else {}
            oos_for_tactic = oos.get(tactic) if isinstance(oos.get(tactic), dict) else {}
            oos_eval = oos_for_tactic.get("evaluation") if isinstance(oos_for_tactic, dict) else {}
            expected = e.get("expected_outcome") or results.get("expected_outcome")
            actual_pnl = results.get("total_pnl_pct")
            deviation = None
            if isinstance(expected, str):
                exp_lower = expected.lower()
                if "сниз" in exp_lower and isinstance(results.get("max_drawdown_pct"), (int, float)):
                    deviation = f"DD={float(results.get('max_drawdown_pct') or 0.0):.2f}%"
                elif isinstance(actual_pnl, (int, float)):
                    deviation = f"PnL={float(actual_pnl):.2f}%"
            score_value = _to_float((results.get("selection") or {}).get("recommended_score") if isinstance(results.get("selection"), dict) else None)
            pnl_value = _to_float(results.get("total_pnl_pct"))
            dd_value = _to_float(results.get("max_drawdown_pct") or results.get("max_drawdown"))
            wr_value = _to_float(results.get("win_rate"))
            trades_value = _to_float(results.get("total_trades"))
            delta_vs_prev = {
                "previous_experiment_id": prev_entry_id,
                "score_delta": (score_value - prev_metrics["score"]) if (score_value is not None and isinstance(prev_metrics, dict) and prev_metrics.get("score") is not None) else None,
                "pnl_delta": (pnl_value - prev_metrics["pnl"]) if (pnl_value is not None and isinstance(prev_metrics, dict) and prev_metrics.get("pnl") is not None) else None,
                "drawdown_delta": (dd_value - prev_metrics["dd"]) if (dd_value is not None and isinstance(prev_metrics, dict) and prev_metrics.get("dd") is not None) else None,
                "win_rate_delta": (wr_value - prev_metrics["wr"]) if (wr_value is not None and isinstance(prev_metrics, dict) and prev_metrics.get("wr") is not None) else None,
                "trades_delta": (trades_value - prev_metrics["trades"]) if (trades_value is not None and isinstance(prev_metrics, dict) and prev_metrics.get("trades") is not None) else None,
            }
            expected_vs_actual = _build_expected_vs_actual(
                expected_text=expected if isinstance(expected, str) else None,
                score=score_value,
                pnl=pnl_value,
                dd=dd_value,
                wr=wr_value,
                trades=trades_value,
                prev_vals=prev_metrics,
            )
            decision_trace = _build_decision_trace(
                tactic=tactic if isinstance(tactic, str) else None,
                score=score_value,
                selection_payload=results.get("selection") if isinstance(results.get("selection"), dict) else {},
                rationale=e.get("rationale") if isinstance(e.get("rationale"), str) else (results.get("rationale") if isinstance(results.get("rationale"), str) else None),
            )
            campaign_status = e.get("campaign_status")
            campaign_stop_reason = results.get("campaign_stop_reason") or e.get("campaign_stop_reason")
            campaign_stop_detail = (
                results.get("campaign_stop_reason_detail")
                if isinstance(results.get("campaign_stop_reason_detail"), dict)
                else (e.get("campaign_stop_reason_detail") if isinstance(e.get("campaign_stop_reason_detail"), dict) else None)
            )
            if isinstance(campaign_stop_reason, str):
                stop_reason = campaign_stop_reason
                stop_reason_detail = campaign_stop_detail
            elif isinstance(campaign_status, str) and campaign_status in {"failed_to_schedule_next", "stopped", "paused"}:
                stop_reason = campaign_status
                stop_reason_detail = {
                    "next_experiment_error": e.get("next_experiment_error"),
                    "next_experiment_blocked_reasons": e.get("next_experiment_blocked_reasons"),
                }
            entries.append(
                {
                    "experiment_id": e.get("id"),
                    "iteration": (e.get("ai_campaign") or {}).get("iteration"),
                    "status": e.get("status"),
                    "created_at": e.get("created_at"),
                    "hypothesis": e.get("hypothesis") or results.get("hypothesis"),
                    "expected_outcome": expected,
                    "analysis_summary": results.get("analysis_summary") or e.get("analysis_summary"),
                    "recommended_tactic": tactic,
                    "score": score_value,
                    "oos_score": oos_eval.get("score") if isinstance(oos_eval, dict) else None,
                    "walk_forward_stability_pass": wf_for_tactic.get("stability_pass") if isinstance(wf_for_tactic, dict) else None,
                    "deviation": deviation,
                    "next_experiment_id": results.get("next_experiment_id") or e.get("next_experiment_id"),
                    "market_regime": e.get("market_regime") if isinstance(e.get("market_regime"), dict) else None,
                    "param_changes": e.get("param_changes") if isinstance(e.get("param_changes"), dict) else None,
                    "campaign_status": campaign_status,
                    "decision_trace": decision_trace,
                    "delta_vs_prev": delta_vs_prev,
                    "expected_vs_actual": expected_vs_actual,
                    "stop_reason": campaign_stop_reason,
                    "stop_reason_detail": campaign_stop_detail,
                }
            )
            prev_entry_id = e.get("id") if isinstance(e.get("id"), str) else prev_entry_id
            prev_metrics = {
                "score": score_value,
                "pnl": pnl_value,
                "dd": dd_value,
                "wr": wr_value,
                "trades": trades_value,
            }
        if stop_reason is None:
            last = related[-1] if related else {}
            last_status = last.get("status")
            last_campaign = last.get("ai_campaign") if isinstance(last.get("ai_campaign"), dict) else {}
            if last.get("next_experiment_id"):
                stop_reason = "scheduled_next"
            elif isinstance(last.get("campaign_status"), str):
                stop_reason = str(last.get("campaign_status"))
            elif bool(last_campaign.get("auto_chain", False)) and int(last_campaign.get("remaining_steps") or 0) <= 0 and str(last_status) == "completed":
                stop_reason = "max_steps_reached"
            elif not bool(last_campaign.get("auto_chain", True)):
                stop_reason = "manual_stop"
            elif str(last_status) != "completed":
                stop_reason = "not_completed"
            else:
                stop_reason = "no_next_iteration"
        summary = {
            "root_experiment_id": root_experiment_id,
            "iterations": len(entries),
            "completed": len([x for x in entries if x.get("status") == "completed"]),
            "scheduled_next": len([x for x in entries if x.get("next_experiment_id")]),
            "stop_reason": stop_reason,
            "stop_reason_detail": stop_reason_detail,
        }
        return {"summary": summary, "entries": entries}


def derive_market_regime_from_metrics(symbol: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = metrics if isinstance(metrics, dict) else {}
    trend_hint = str(payload.get("trend") or payload.get("market_trend") or "").lower()
    volatility_raw = payload.get("volatility")
    try:
        volatility = float(volatility_raw) if volatility_raw is not None else None
    except Exception:
        volatility = None
    regime = "sideways_normal_vol"
    trend = "sideways"
    vol_bucket = "normal"
    if trend_hint in {"bullish", "up", "uptrend", "trend_up"}:
        trend = "trend_up"
    elif trend_hint in {"bearish", "down", "downtrend", "trend_down"}:
        trend = "trend_down"
    if isinstance(volatility_raw, str):
        raw = volatility_raw.lower()
        if raw in {"high", "low", "normal"}:
            vol_bucket = raw
    elif volatility is not None:
        if volatility >= 3.0:
            vol_bucket = "high"
        elif volatility <= 1.0:
            vol_bucket = "low"
    if trend == "trend_up":
        regime = f"trend_up_{vol_bucket}_vol"
    elif trend == "trend_down":
        regime = f"trend_down_{vol_bucket}_vol"
    else:
        regime = f"sideways_{vol_bucket}_vol"
    return {
        "symbol": symbol.upper(),
        "regime": regime,
        "trend": trend,
        "volatility": vol_bucket,
        "source": "heuristic_metrics",
    }


def build_regime_plan_context(
    symbol: str,
    experiment_type: str,
    summary: Optional[Dict[str, Any]] = None,
    regime_memory: Optional[Dict[str, Any]] = None,
    market_regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    regime = market_regime if isinstance(market_regime, dict) else derive_market_regime_from_metrics(symbol)
    memory_payload = regime_memory if isinstance(regime_memory, dict) else {"symbol": symbol.upper(), "regimes": []}
    current_regime = str(regime.get("regime") or "unknown")
    memory_rows = memory_payload.get("regimes") if isinstance(memory_payload.get("regimes"), list) else []
    current_memory = next((r for r in memory_rows if str(r.get("regime")) == current_regime), None)

    interval = "15m"
    use_mtf = experiment_type != "aggressive"
    safe_mode = experiment_type != "aggressive"
    backtest_days = 10 if experiment_type == "balanced" else (14 if experiment_type == "conservative" else 7)
    param_changes: Dict[str, Any] = {}
    next_experiments: List[str] = []
    tags = ["regime_aware", current_regime, experiment_type]

    if current_regime.startswith("trend_up"):
        interval = "15m"
        use_mtf = True
        safe_mode = experiment_type != "aggressive"
        backtest_days = max(backtest_days, 10)
        param_changes.update({"trend_bias": "long", "confirmation_mode": "trend_follow"})
        next_experiments.extend([
            "Проверить более агрессивный trend-follow вход на high-vol окне",
            "Сравнить MTF с single 15m в том же восходящем режиме",
        ])
    elif current_regime.startswith("trend_down"):
        interval = "15m"
        use_mtf = True
        safe_mode = True
        backtest_days = max(backtest_days, 10)
        param_changes.update({"trend_bias": "short", "confirmation_mode": "trend_follow"})
        next_experiments.extend([
            "Проверить более строгий фильтр для short сигналов",
            "Сравнить поведение safe_mode в нисходящем режиме",
        ])
    elif current_regime.startswith("sideways"):
        interval = "1h" if experiment_type == "conservative" else "15m"
        use_mtf = experiment_type != "aggressive"
        safe_mode = True
        backtest_days = max(backtest_days, 12)
        param_changes.update({"trend_bias": "neutral", "confirmation_mode": "mean_reversion_filter"})
        next_experiments.extend([
            "Ужесточить фильтр ложных пробоев во флэте",
            "Увеличить окно бэктеста для проверки устойчивости во флэте",
        ])

    if current_memory and int(current_memory.get("failed") or 0) > int(current_memory.get("successful") or 0):
        safe_mode = True
        backtest_days = max(backtest_days, 14)
        param_changes["risk_tuning"] = "tighten"
        next_experiments.insert(0, "Избегать ранее слабых конфигураций в этом режиме рынка")
    if current_memory and isinstance(current_memory.get("common_param_changes"), list) and current_memory["common_param_changes"]:
        for item in current_memory["common_param_changes"][:2]:
            key = item.get("param")
            value = item.get("value")
            if isinstance(key, str) and key and key not in param_changes:
                param_changes[key] = value

    best = summary.get("best") if isinstance(summary, dict) else None
    if isinstance(best, dict) and float(best.get("total_pnl_pct") or 0.0) < 0.0:
        safe_mode = True
        param_changes["regularization"] = "increase"
        param_changes["confidence_thresholds"] = "tighten"

    return {
        "market_regime": regime,
        "regime_memory": current_memory,
        "plan_defaults": {
            "interval": interval,
            "use_mtf": use_mtf,
            "safe_mode": safe_mode,
            "backtest_days": backtest_days,
            "param_changes": param_changes,
            "next_experiments": next_experiments[:4],
            "tags": tags,
            "hyperparams": {
                "training_days_15m": 30,
                "training_days_1h": 180,
                "forward_periods_15m": 5,
                "forward_periods_1h": 8,
                "threshold_pct_15m": 0.3,
                "threshold_pct_1h": 0.8,
                "min_profit_pct_15m": 0.3,
                "min_profit_pct_1h": 0.8,
                "rf_n_estimators": 120,
                "rf_max_depth": 10,
                "xgb_n_estimators": 140,
                "xgb_max_depth": 6,
                "xgb_learning_rate": 0.08,
                "use_meta_labeling": True,
            },
        },
    }


def build_hyperparameter_search_strategy(
    *,
    symbol: str,
    experiment_type: str,
    market_regime: Optional[Dict[str, Any]] = None,
    regime_memory: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
    previous_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    regime = market_regime if isinstance(market_regime, dict) else derive_market_regime_from_metrics(symbol)
    regime_name = str(regime.get("regime") or "unknown")
    volatility = str(regime.get("volatility") or "normal")
    trend = str(regime.get("trend") or "sideways")
    prev = previous_hyperparams if isinstance(previous_hyperparams, dict) else {}
    objective_weights = {
        "pnl": 1.0,
        "drawdown": -0.5,
        "stability": 0.8,
        "trade_quality": 0.6,
    }
    if experiment_type == "conservative":
        objective_weights = {"pnl": 0.8, "drawdown": -0.8, "stability": 1.0, "trade_quality": 0.7}
    elif experiment_type == "aggressive":
        objective_weights = {"pnl": 1.2, "drawdown": -0.35, "stability": 0.6, "trade_quality": 0.5}
    if volatility == "high":
        objective_weights["drawdown"] = min(float(objective_weights["drawdown"]), -0.9)
        objective_weights["stability"] = max(float(objective_weights["stability"]), 1.0)

    base = {
        "training_days_15m": int(prev.get("training_days_15m") or 30),
        "training_days_1h": int(prev.get("training_days_1h") or 180),
        "forward_periods_15m": int(prev.get("forward_periods_15m") or 5),
        "forward_periods_1h": int(prev.get("forward_periods_1h") or 8),
        "threshold_pct_15m": float(prev.get("threshold_pct_15m") or 0.3),
        "threshold_pct_1h": float(prev.get("threshold_pct_1h") or 0.8),
        "min_profit_pct_15m": float(prev.get("min_profit_pct_15m") or 0.3),
        "min_profit_pct_1h": float(prev.get("min_profit_pct_1h") or 0.8),
        "rf_n_estimators": int(prev.get("rf_n_estimators") or 120),
        "rf_max_depth": int(prev.get("rf_max_depth") or 10),
        "xgb_n_estimators": int(prev.get("xgb_n_estimators") or 140),
        "xgb_max_depth": int(prev.get("xgb_max_depth") or 6),
        "xgb_learning_rate": float(prev.get("xgb_learning_rate") or 0.08),
        # Triple Barrier Method (TBM) defaults
        "use_triple_barrier": bool(prev.get("use_triple_barrier", False)),
        "tbm_pt_sl_ratio": float(prev.get("tbm_pt_sl_ratio") or 2.0),
        "tbm_vertical_barrier": int(prev.get("tbm_vertical_barrier") or 24),
        "tbm_volatility_lookback": int(prev.get("tbm_volatility_lookback") or 20),
        "use_meta_labeling": bool(prev.get("use_meta_labeling", True)),
    }

    # Smarter evolutionary mutation logic
    import random
    def mutate(params: Dict[str, Any], strength: float = 0.1) -> Dict[str, Any]:
        new_params = params.copy()
        for k, v in new_params.items():
            if isinstance(v, bool):
                if random.random() < 0.2: # 20% chance to flip bool
                    new_params[k] = not v
            elif isinstance(v, int):
                delta = int(v * strength * random.uniform(-1, 1))
                new_params[k] = v + (delta if delta != 0 else random.choice([-1, 1]))
            elif isinstance(v, float):
                new_params[k] = v * (1 + strength * random.uniform(-1, 1))
        return new_params

    if trend.startswith("trend_up") or trend.startswith("trend_down"):
        base["forward_periods_15m"] = max(base["forward_periods_15m"], 6)
        base["threshold_pct_15m"] = max(base["threshold_pct_15m"], 0.35)
    if regime_name.startswith("sideways"):
        base["threshold_pct_15m"] = min(base["threshold_pct_15m"], 0.28)
        base["min_profit_pct_15m"] = min(base["min_profit_pct_15m"], 0.28)
        base["rf_max_depth"] = max(base["rf_max_depth"], 12)
    if volatility == "high":
        base["training_days_15m"] = max(base["training_days_15m"], 40)
        base["threshold_pct_15m"] = max(base["threshold_pct_15m"], 0.4)
        base["min_profit_pct_15m"] = max(base["min_profit_pct_15m"], 0.35)
        base["xgb_learning_rate"] = min(base["xgb_learning_rate"], 0.07)

    mem = regime_memory if isinstance(regime_memory, dict) else {}
    memory_success_rate = float(mem.get("success_rate") or 0.0)
    memory_confidence = float(mem.get("confidence") or 0.0)
    memory_stability = mem.get("stability") if isinstance(mem.get("stability"), dict) else {}
    memory_pnl_std = float(memory_stability.get("pnl_std") or 0.0)
    memory_dd_std = float(memory_stability.get("dd_std") or 0.0)
    memory_wr_std = float(memory_stability.get("wr_std") or 0.0)
    memory_fail_reasons = mem.get("gate_fail_reasons_top") if isinstance(mem.get("gate_fail_reasons_top"), list) else []

    if int(mem.get("failed") or 0) > int(mem.get("successful") or 0):
        base["training_days_15m"] = max(base["training_days_15m"], 45)
        base["rf_n_estimators"] = max(base["rf_n_estimators"], 160)
        base["xgb_n_estimators"] = max(base["xgb_n_estimators"], 180)
        base["xgb_learning_rate"] = min(base["xgb_learning_rate"], 0.06)
    if memory_confidence < 0.45 or memory_success_rate < 0.45:
        base["training_days_15m"] = max(base["training_days_15m"], 50)
        base["rf_n_estimators"] = max(base["rf_n_estimators"], 180)
        base["xgb_n_estimators"] = max(base["xgb_n_estimators"], 220)
        base["xgb_learning_rate"] = min(base["xgb_learning_rate"], 0.055)
        base["threshold_pct_15m"] = max(base["threshold_pct_15m"], 0.35)
        base["min_profit_pct_15m"] = max(base["min_profit_pct_15m"], 0.33)
    if memory_pnl_std > 8.0 or memory_dd_std > 6.0 or memory_wr_std > 10.0:
        base["training_days_15m"] = max(base["training_days_15m"], 55)
        base["xgb_learning_rate"] = min(base["xgb_learning_rate"], 0.05)
    if memory_confidence > 0.7 and memory_success_rate > 0.6 and memory_pnl_std < 5.0:
        base["threshold_pct_15m"] = min(base["threshold_pct_15m"], 0.26)
        base["min_profit_pct_15m"] = min(base["min_profit_pct_15m"], 0.25)

    candidates: List[Dict[str, Any]] = []

    def _clamp_int(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(v)))

    def _clamp_float(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def _normalized(hp: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "training_days_15m": _clamp_int(hp.get("training_days_15m", base["training_days_15m"]), 20, 120),
            "training_days_1h": _clamp_int(hp.get("training_days_1h", base["training_days_1h"]), 60, 365),
            "forward_periods_15m": _clamp_int(hp.get("forward_periods_15m", base["forward_periods_15m"]), 3, 16),
            "forward_periods_1h": _clamp_int(hp.get("forward_periods_1h", base["forward_periods_1h"]), 4, 20),
            "threshold_pct_15m": _clamp_float(hp.get("threshold_pct_15m", base["threshold_pct_15m"]), 0.15, 1.5),
            "threshold_pct_1h": _clamp_float(hp.get("threshold_pct_1h", base["threshold_pct_1h"]), 0.2, 2.5),
            "min_profit_pct_15m": _clamp_float(hp.get("min_profit_pct_15m", base["min_profit_pct_15m"]), 0.1, 1.2),
            "min_profit_pct_1h": _clamp_float(hp.get("min_profit_pct_1h", base["min_profit_pct_1h"]), 0.2, 2.0),
            "rf_n_estimators": _clamp_int(hp.get("rf_n_estimators", base["rf_n_estimators"]), 80, 500),
            "rf_max_depth": _clamp_int(hp.get("rf_max_depth", base["rf_max_depth"]), 4, 20),
            "xgb_n_estimators": _clamp_int(hp.get("xgb_n_estimators", base["xgb_n_estimators"]), 80, 600),
            "xgb_max_depth": _clamp_int(hp.get("xgb_max_depth", base["xgb_max_depth"]), 3, 12),
            "xgb_learning_rate": _clamp_float(hp.get("xgb_learning_rate", base["xgb_learning_rate"]), 0.02, 0.3),
            "use_triple_barrier": bool(hp.get("use_triple_barrier", base["use_triple_barrier"])),
            "tbm_pt_sl_ratio": _clamp_float(hp.get("tbm_pt_sl_ratio", base["tbm_pt_sl_ratio"]), 1.0, 5.0),
            "tbm_vertical_barrier": _clamp_int(hp.get("tbm_vertical_barrier", base["tbm_vertical_barrier"]), 12, 96),
            "tbm_volatility_lookback": _clamp_int(hp.get("tbm_volatility_lookback", base["tbm_volatility_lookback"]), 10, 100),
            "use_meta_labeling": bool(hp.get("use_meta_labeling", base["use_meta_labeling"])),
        }

    candidates.append(
        {
            "candidate_id": "base_local",
            "strategy": "local_base",
            "hyperparams": _normalized(base),
            "rationale": "Локальный базовый набор под текущий режим.",
        }
    )

    # Evolutionary mutations based on base (or previously successful)
    candidates.append(
        {
            "candidate_id": "mutated_evolutionary_1",
            "strategy": "evolutionary",
            "hyperparams": _normalized(mutate(base, strength=0.15)),
            "rationale": "Эволюционная мутация (15%) базовых параметров для поиска локальных оптимумов.",
        }
    )

    candidates.append(
        {
            "candidate_id": "stability_tuned",
            "strategy": "local_stability",
            "hyperparams": _normalized(
                {
                    **base,
                    "training_days_15m": int(base["training_days_15m"]) + 10,
                    "rf_n_estimators": int(base["rf_n_estimators"]) + 40,
                    "xgb_n_estimators": int(base["xgb_n_estimators"]) + 40,
                    "xgb_learning_rate": float(base["xgb_learning_rate"]) - 0.01,
                }
            ),
            "rationale": "Усиление устойчивости: больше данных и более плавный бустинг.",
        }
    )

    candidates.append(
        {
            "candidate_id": "meta_labeling_test",
            "strategy": "meta_labeling",
            "hyperparams": _normalized({**base, "use_meta_labeling": True}),
            "rationale": "Использование Meta-Labeling для фильтрации потенциально убыточных сигналов.",
        }
    )

    candidates.append(
        {
            "candidate_id": "tbm_optimized",
            "strategy": "triple_barrier",
            "hyperparams": _normalized(
                {
                    **base,
                    "use_triple_barrier": True,
                    "tbm_pt_sl_ratio": 2.0,
                    "tbm_vertical_barrier": 24,
                    "threshold_pct_15m": 0.25,
                }
            ),
            "rationale": "Использование Triple Barrier Method для качественной разметки волатильности.",
        }
    )

    selection_payload = selection if isinstance(selection, dict) else {}
    selected = candidates[0]
    memory_prefers_stability = (
        memory_confidence < 0.45
        or memory_success_rate < 0.45
        or memory_pnl_std > 8.0
        or memory_dd_std > 6.0
        or any(str((x or {}).get("reason") or "").startswith("max_drawdown_pct") for x in memory_fail_reasons if isinstance(x, dict))
    )
    memory_prefers_pnl = (
        memory_confidence > 0.7
        and memory_success_rate > 0.62
        and memory_pnl_std < 5.0
        and memory_dd_std < 4.0
    )
    if memory_prefers_stability:
        selected = candidates[1]
    elif memory_prefers_pnl:
        selected = candidates[3]
    if isinstance(selection_payload.get("candidates"), list):
        gating_fail = 0
        for c in selection_payload.get("candidates") or []:
            if isinstance(c, dict):
                gates = c.get("quality_gates") if isinstance(c.get("quality_gates"), dict) else {}
                if not bool((gates or {}).get("passed")):
                    gating_fail += 1
        if gating_fail >= 2:
            selected = candidates[1]
        else:
            top_tactic = str(selection_payload.get("recommended_tactic") or "")
            if "single" in top_tactic:
                selected = candidates[1] if memory_prefers_stability else candidates[2]
            elif "mtf" in top_tactic:
                selected = candidates[0]
            else:
                selected = candidates[1] if memory_prefers_stability else candidates[3]

    return {
        "version": "p2.4_local_search_v2",
        "symbol": symbol.upper(),
        "experiment_type": experiment_type,
        "market_regime": regime,
        "regime_memory_snapshot": {
            "count": int(mem.get("count") or 0),
            "sample_size_effective": int(mem.get("sample_size_effective") or 0),
            "success_rate": memory_success_rate,
            "confidence": memory_confidence,
            "stability": {
                "pnl_std": memory_pnl_std,
                "dd_std": memory_dd_std,
                "wr_std": memory_wr_std,
            },
            "top_fail_reasons": memory_fail_reasons[:3] if isinstance(memory_fail_reasons, list) else [],
            "last_seen_at": mem.get("last_seen_at"),
        },
        "objective_weights": objective_weights,
        "base_hyperparams": _normalized(base),
        "candidates": candidates,
        "chosen_candidate": selected,
        "search_space": {
            "training_days_15m": [20, 120],
            "forward_periods_15m": [3, 16],
            "threshold_pct_15m": [0.15, 1.5],
            "min_profit_pct_15m": [0.1, 1.2],
            "rf_n_estimators": [80, 500],
            "xgb_n_estimators": [80, 600],
            "xgb_learning_rate": [0.02, 0.3],
            "tbm_pt_sl_ratio": [1.0, 5.0],
            "tbm_vertical_barrier": [12, 96],
        },
    }


class HypothesisGenerator:
    def __init__(self, criteria: ExperimentCriteria):
        self.criteria = criteria

    def _success(self, r: Dict[str, Any]) -> bool:
        trades = int(r.get("total_trades") or 0)
        pf = float(r.get("profit_factor") or 0.0)
        pnl = float(r.get("total_pnl_pct") or 0.0)
        dd = float(r.get("max_drawdown_pct") or 0.0)
        return (
            trades >= self.criteria.min_total_trades
            and pf >= self.criteria.min_profit_factor
            and pnl >= self.criteria.min_total_pnl_pct
            and dd <= self.criteria.max_drawdown_pct
        )

    def propose(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = analysis.get("rows", [])
        completed = [r for r in rows if r.get("status") == "completed"]
        completed_sorted = sorted(
            completed,
            key=lambda r: (r.get("total_pnl_pct") is None, -(float(r.get("total_pnl_pct") or 0.0))),
        )
        best = completed_sorted[0] if completed_sorted else None
        hypotheses: List[Dict[str, Any]] = []

        # New TBM Hypothesis
        hypotheses.append(
            {
                "title": "Переход на Triple Barrier Method",
                "changes": {
                    "labeling": {
                        "method": "triple_barrier",
                        "pt_sl_ratio": 2.0,
                        "vertical_barrier": 24
                    }
                },
                "rationale": "Triple Barrier Method лучше учитывает волатильность и временные ограничения сделок, чем фиксированный forward window.",
                "expected": "Повышение стабильности сигналов и снижение просадок за счет динамических стопов.",
            }
        )

        if best and self._success(best):
            hypotheses.append(
                {
                    "title": "Уточнение вокруг лучшего эксперимента",
                    "changes": {
                        "training": {
                            "search": "local",
                            "around_experiment_id": best.get("id"),
                        }
                    },
                    "rationale": "Лучший эксперимент уже проходит минимальные критерии качества; есть смысл уточнить параметры локально.",
                    "expected": "Рост PnL при сохранении acceptable drawdown и числа сделок.",
                }
            )
        else:
            hypotheses.append(
                {
                    "title": "Стабилизация и фильтрация ложных сигналов",
                    "changes": {
                        "risk": {"safe_mode": True},
                        "model": {"regularization": "increase"},
                    },
                    "rationale": "История не показывает устойчиво успешного результата; сначала нужно повысить качество сигналов и снизить шум.",
                    "expected": "Снижение drawdown и рост profit factor даже при меньшем числе сделок.",
                }
            )

        if any(not self._success(r) and float(r.get("max_drawdown_pct") or 0.0) > self.criteria.max_drawdown_pct for r in completed_sorted[:5]):
            hypotheses.append(
                {
                    "title": "Снижение просадки",
                    "changes": {"risk": {"max_positions": 1, "atr_stop_multiplier": 1.2}},
                    "rationale": "Последние эксперименты демонстрируют повышенную просадку.",
                    "expected": "Сокращение max_drawdown_pct с умеренной потерей доходности.",
                }
            )

        if any(int(r.get("total_trades") or 0) < self.criteria.min_total_trades for r in completed_sorted[:5]):
            hypotheses.append(
                {
                    "title": "Повышение частоты сделок без деградации качества",
                    "changes": {"signal_generation": {"entry_threshold": "slightly_looser"}},
                    "rationale": "Мало сделок затрудняет статистически устойчивую оценку результатов.",
                    "expected": "Увеличение total_trades до минимально достаточного уровня при сохранении profit factor.",
                }
            )

        return hypotheses[:3]


def build_experiment_report(experiment: Dict[str, Any]) -> Dict[str, Any]:
    results = experiment.get("results") if isinstance(experiment.get("results"), dict) else {}
    report = {
        "id": experiment.get("id"),
        "symbol": experiment.get("symbol"),
        "type": experiment.get("type"),
        "status": experiment.get("status"),
        "hypothesis": experiment.get("hypothesis"),
        "expected_outcome": experiment.get("expected_outcome"),
        "rationale": experiment.get("rationale"),
        "param_changes": experiment.get("param_changes"),
        "created_at": experiment.get("created_at"),
        "updated_at": experiment.get("updated_at"),
        "results": {
            "total_pnl_pct": results.get("total_pnl_pct"),
            "win_rate": results.get("win_rate"),
            "profit_factor": results.get("profit_factor"),
            "max_drawdown_pct": results.get("max_drawdown_pct") or results.get("max_drawdown"),
            "total_trades": results.get("total_trades"),
            "recommended_tactic": results.get("recommended_tactic"),
            "analysis_summary": results.get("analysis_summary"),
        },
        "next_experiments": experiment.get("next_experiments") or results.get("next_experiments"),
    }
    return report


class ExperimentReportBuilder:
    def build_markdown(self, experiment: Dict[str, Any], impact: Optional[Dict[str, Any]] = None) -> str:
        exp = experiment if isinstance(experiment, dict) else {}
        results = exp.get("results") if isinstance(exp.get("results"), dict) else {}
        exp_id = str(exp.get("id") or "unknown")
        symbol = str(exp.get("symbol") or "UNKNOWN")
        exp_type = str(exp.get("type") or "custom")
        status = str(exp.get("status") or "unknown")
        total_pnl = results.get("total_pnl_pct")
        win_rate = results.get("win_rate")
        max_dd = results.get("max_drawdown_pct") or results.get("max_drawdown")
        trades = results.get("total_trades")
        tactic = results.get("recommended_tactic")
        created_at = exp.get("created_at")
        completed_at = exp.get("completed_at")
        lines: List[str] = []
        lines.append(f"# Experiment Report: {exp_id}")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- Symbol: {symbol}")
        lines.append(f"- Type: {exp_type}")
        lines.append(f"- Status: {status}")
        if created_at:
            lines.append(f"- Created at: {created_at}")
        if completed_at:
            lines.append(f"- Completed at: {completed_at}")
        lines.append("")
        lines.append("## Performance")
        lines.append(f"- Total PnL (%): {total_pnl}")
        lines.append(f"- Win rate (%): {win_rate}")
        lines.append(f"- Max drawdown (%): {max_dd}")
        lines.append(f"- Total trades: {trades}")
        lines.append(f"- Recommended tactic: {tactic}")
        analysis_summary = results.get("analysis_summary") or exp.get("analysis_summary")
        if analysis_summary:
            lines.append("")
            lines.append("## Analysis Summary")
            lines.append(str(analysis_summary))
        if isinstance(impact, dict):
            lines.append("")
            lines.append("## Parameter Impact")
            for key, value in impact.items():
                lines.append(f"- {key}: {value}")
        next_experiments = exp.get("next_experiments") or results.get("next_experiments")
        if isinstance(next_experiments, list) and next_experiments:
            lines.append("")
            lines.append("## Next Experiments")
            for item in next_experiments:
                lines.append(f"- {item}")
        return "\n".join(lines)
