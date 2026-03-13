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
        experiments = list(self.read_all().values())
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
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
        return {"count": len(rows), "rows": rows, "best": best}

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
                    "expected": "Стабилизировать результат и снизить разброс метрик без потери прибыльности.",
                    "rationale": "Локальный поиск вокруг лучшего найденного решения снижает риск деградации и даёт прирост за счёт тонкой настройки.",
                }
            )
        else:
            hypotheses.append(
                {
                    "title": "Снижение переобучения и стабилизация",
                    "changes": {
                        "lstm": {
                            "dropout": "+0.05",
                            "weight_decay": "keep",
                            "early_stopping_patience": "decrease",
                        }
                    },
                    "expected": "Снизить разрыв между train/validation и улучшить обобщение.",
                    "rationale": "Повышение регуляризации и более ранняя остановка уменьшают variance и переобучение.",
                }
            )
            hypotheses.append(
                {
                    "title": "Повышение выразительности при недообучении",
                    "changes": {
                        "lstm": {
                            "hidden_size": "+",
                            "num_layers": "+1",
                        }
                    },
                    "expected": "Увеличить способность модели описывать нелинейные паттерны рынка.",
                    "rationale": "Увеличение емкости модели может помочь, если качество низкое и нет признаков переобучения.",
                }
            )

        return hypotheses


class ExperimentReportBuilder:
    def build_markdown(self, experiment: Dict[str, Any], impact: Optional[Dict[str, Any]] = None) -> str:
        exp_id = experiment.get("id")
        symbol = experiment.get("symbol")
        status = experiment.get("status")
        exp_type = experiment.get("type")
        created_at = experiment.get("created_at")
        updated_at = experiment.get("updated_at")
        signature = experiment.get("param_signature")
        hypothesis = experiment.get("hypothesis")
        expected = experiment.get("expected_outcome")
        rationale = experiment.get("rationale")
        baseline = experiment.get("baseline")
        params = experiment.get("params")
        changes = experiment.get("param_changes")
        results = experiment.get("results")
        lines: List[str] = []
        lines.append(f"# Experiment Report — {exp_id}")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- Symbol: {symbol}")
        lines.append(f"- Type: {exp_type}")
        lines.append(f"- Status: {status}")
        lines.append(f"- Created: {created_at}")
        lines.append(f"- Updated: {updated_at}")
        if signature:
            lines.append(f"- Param signature: {signature}")
        lines.append("")
        if hypothesis or expected or rationale:
            lines.append("## Hypothesis")
            if hypothesis:
                lines.append(f"- Hypothesis: {hypothesis}")
            if expected:
                lines.append(f"- Expected: {expected}")
            if rationale:
                lines.append(f"- Rationale: {rationale}")
            lines.append("")
        if baseline:
            lines.append("## Baseline")
            lines.append("```json")
            lines.append(json.dumps(baseline, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
        if params:
            lines.append("## Params")
            lines.append("```json")
            lines.append(json.dumps(params, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
        if changes:
            lines.append("## Param Changes")
            lines.append("```json")
            lines.append(json.dumps(changes, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
        if results:
            lines.append("## Results")
            lines.append("```json")
            lines.append(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
        if impact:
            lines.append("## Param Impact (Symbol-level)")
            lines.append("```json")
            lines.append(json.dumps(impact, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
            lines.append("")
        return "\n".join(lines)
