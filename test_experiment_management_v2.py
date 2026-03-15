from bot.experiment_management import (
    ExperimentAnalyzer,
    build_hyperparameter_search_strategy,
    derive_market_regime_from_metrics,
)


def _exp(
    exp_id: str,
    *,
    symbol: str = "BTCUSDT",
    regime: str = "sideways_normal_vol",
    created_at: str = "2026-03-15T10:00:00+00:00",
    status: str = "completed",
    pnl: float | None = 5.0,
    pf: float | None = 1.2,
    wr: float | None = 55.0,
    dd: float | None = 10.0,
    trades: int | None = 35,
    campaign: dict | None = None,
    param_changes: dict | None = None,
    results_extra: dict | None = None,
    top_level_extra: dict | None = None,
):
    payload = {
        "id": exp_id,
        "created_at": created_at,
        "status": status,
        "symbol": symbol,
        "type": "balanced",
        "market_regime": {
            "symbol": symbol,
            "regime": regime,
            "trend": "sideways",
            "volatility": "normal",
        },
        "param_signature": f"sig_{exp_id}",
        "param_changes": param_changes or {},
        "ai_campaign": campaign or {},
        "results": {
            "total_pnl_pct": pnl,
            "profit_factor": pf,
            "win_rate": wr,
            "max_drawdown_pct": dd,
            "total_trades": trades,
            "recommended_tactic": "single_15m",
            "selection": {
                "recommended_tactic": "single_15m",
                "recommended_score": 3.2,
                "candidates": [
                    {
                        "tactic": "single_15m",
                        "score": 3.2,
                        "oos_pass": True,
                        "walk_forward_stability_pass": True,
                        "quality_gates": {"passed": True},
                    },
                    {
                        "tactic": "mtf",
                        "score": 2.7,
                        "oos_pass": True,
                        "walk_forward_stability_pass": False,
                        "quality_gates": {"passed": True},
                    },
                ],
            },
            "oos_validation": {
                "single_15m": {
                    "evaluation": {"score": 1.1, "gates": {"passed": True}},
                    "passed": True,
                }
            },
            "walk_forward": {
                "single_15m": {
                    "stability_pass": True,
                    "pnl_spread_pct": 2.1,
                }
            },
            "analysis_summary": "ok",
        },
    }
    if isinstance(results_extra, dict):
        payload["results"].update(results_extra)
    if isinstance(top_level_extra, dict):
        payload.update(top_level_extra)
    return payload


def test_memory_v2_fields_present_and_consistent():
    experiments = [
        _exp(
            "exp_pass_1",
            created_at="2026-03-15T10:00:00+00:00",
            pnl=8.0,
            pf=1.4,
            wr=60.0,
            dd=8.0,
            trades=45,
        ),
        _exp(
            "exp_fail_1",
            created_at="2026-03-15T11:00:00+00:00",
            pnl=-3.0,
            pf=0.8,
            wr=42.0,
            dd=31.0,
            trades=12,
            param_changes={"risk_tuning": "tighten"},
        ),
        _exp(
            "exp_pass_2",
            created_at="2026-03-15T12:00:00+00:00",
            pnl=5.0,
            pf=1.2,
            wr=53.0,
            dd=11.0,
            trades=33,
            param_changes={"trend_bias": "neutral"},
        ),
    ]
    analyzer = ExperimentAnalyzer(experiments)
    memory = analyzer.summarize_regime_memory("BTCUSDT")
    assert memory["symbol"] == "BTCUSDT"
    assert isinstance(memory.get("regimes"), list) and memory["regimes"]
    row = memory["regimes"][0]
    for key in [
        "success_rate",
        "sample_size_effective",
        "first_seen_at",
        "last_seen_at",
        "stability",
        "gate_fail_reasons_top",
        "avoid_signatures_meta",
        "confidence",
    ]:
        assert key in row
    assert row["count"] == 3
    assert row["sample_size_effective"] == 3
    assert abs(float(row["success_rate"]) - (2.0 / 3.0)) < 1e-9
    assert isinstance(row["stability"], dict)
    assert 0.0 <= float(row["confidence"]) <= 1.0
    assert row["first_seen_at"] <= row["last_seen_at"]
    assert isinstance(row["gate_fail_reasons_top"], list)


def test_notebook_v2_contains_decision_delta_expected_and_stop_reason():
    root_id = "exp_root"
    experiments = [
        _exp(
            root_id,
            symbol="ETHUSDT",
            created_at="2026-03-15T10:00:00+00:00",
            campaign={
                "root_experiment_id": root_id,
                "iteration": 1,
                "remaining_steps": 1,
                "auto_chain": True,
            },
            results_extra={
                "campaign_stop_reason": "scheduled_next",
                "next_experiment_id": "exp_child",
                "total_pnl_pct": 6.0,
                "max_drawdown_pct": 9.0,
                "total_trades": 41,
            },
            top_level_extra={"expected_outcome": "Увеличить PnL"},
        ),
        _exp(
            "exp_child",
            symbol="ETHUSDT",
            created_at="2026-03-15T11:00:00+00:00",
            campaign={
                "root_experiment_id": root_id,
                "iteration": 2,
                "remaining_steps": 0,
                "auto_chain": False,
                "parent_experiment_id": root_id,
            },
            results_extra={
                "campaign_stop_reason": "manual_stop",
                "campaign_stop_reason_detail": {"operator": "user"},
                "total_pnl_pct": 4.0,
                "max_drawdown_pct": 10.0,
                "total_trades": 35,
            },
            top_level_extra={"expected_outcome": "Снизить DD"},
        ),
    ]
    analyzer = ExperimentAnalyzer(experiments)
    notebook = analyzer.build_campaign_notebook(root_id)
    assert isinstance(notebook.get("summary"), dict)
    assert notebook["summary"].get("stop_reason") in {
        "manual_stop",
        "scheduled_next",
        "max_steps_reached",
        "no_next_iteration",
        "failed_to_schedule_next",
    }
    entries = notebook.get("entries")
    assert isinstance(entries, list) and len(entries) == 2
    last = entries[-1]
    assert isinstance(last.get("decision_trace"), dict)
    assert isinstance(last.get("delta_vs_prev"), dict)
    assert isinstance(last.get("expected_vs_actual"), dict)
    assert "overall_met" in (last.get("expected_vs_actual") or {})
    assert "stop_reason" in last


def test_hyperparameter_strategy_prefers_stability_when_memory_is_weak():
    regime = derive_market_regime_from_metrics("BTCUSDT")
    weak_memory = {
        "regime": regime["regime"],
        "count": 5,
        "sample_size_effective": 3,
        "success_rate": 0.2,
        "confidence": 0.25,
        "stability": {"pnl_std": 12.0, "dd_std": 7.0, "wr_std": 16.0},
        "gate_fail_reasons_top": [{"reason": "max_drawdown_pct", "count": 3}],
        "last_seen_at": "2026-03-15T12:00:00+00:00",
        "failed": 4,
        "successful": 1,
    }
    strategy = build_hyperparameter_search_strategy(
        symbol="BTCUSDT",
        experiment_type="balanced",
        market_regime=regime,
        regime_memory=weak_memory,
        selection=None,
        previous_hyperparams=None,
    )
    assert strategy["version"] == "p2.4_local_search_v2"
    assert isinstance(strategy.get("regime_memory_snapshot"), dict)
    assert (strategy.get("chosen_candidate") or {}).get("candidate_id") == "stability_tuned"


def test_hyperparameter_strategy_prefers_pnl_when_memory_is_strong():
    regime = derive_market_regime_from_metrics("ETHUSDT")
    strong_memory = {
        "regime": regime["regime"],
        "count": 12,
        "sample_size_effective": 12,
        "success_rate": 0.82,
        "confidence": 0.88,
        "stability": {"pnl_std": 2.0, "dd_std": 1.8, "wr_std": 3.0},
        "gate_fail_reasons_top": [],
        "last_seen_at": "2026-03-15T12:00:00+00:00",
        "failed": 2,
        "successful": 10,
    }
    strategy = build_hyperparameter_search_strategy(
        symbol="ETHUSDT",
        experiment_type="balanced",
        market_regime=regime,
        regime_memory=strong_memory,
        selection=None,
        previous_hyperparams=None,
    )
    assert strategy["version"] == "p2.4_local_search_v2"
    assert (strategy.get("chosen_candidate") or {}).get("candidate_id") == "pnl_tuned"
