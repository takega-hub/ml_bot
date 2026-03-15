from fastapi.testclient import TestClient

from bot.api_server import create_app


class _DummyState:
    def __init__(self):
        self.trades = []

    def get_strategy_config(self, symbol):
        return None


class _DummySettings:
    leverage = 2


def test_research_status_exposes_iteration_d_aliases(monkeypatch):
    experiment_payload = {
        "id": "exp_test_1",
        "symbol": "XRPUSDT",
        "type": "balanced",
        "status": "completed",
        "updated_at": "2026-03-15T13:00:00+00:00",
        "results": {
            "recommended_tactic": "single_15m",
            "total_pnl_pct": 4.2,
            "win_rate": 57.1,
            "total_trades": 33,
            "profit_factor": 1.2,
            "max_drawdown_pct": 12.0,
            "analysis_summary": "ok",
            "oos_validation": {
                "single_15m": {
                    "evaluation": {"score": 1.3, "gates": {"passed": True}},
                    "passed": True,
                }
            },
            "walk_forward": {
                "single_15m": {
                    "stability_pass": True,
                    "pnl_spread_pct": 2.1,
                }
            },
            "oos_metrics": {
                "recommended_tactic": "single_15m",
                "passed": True,
                "source": "oos_validation",
            },
            "drift_signals": {
                "recommended_tactic": "single_15m",
                "level": "low",
                "drift_score": 0.1,
            },
            "stress_results": {
                "recommended_tactic": "single_15m",
                "stress_passed": True,
                "robustness_score": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        "bot.ai_agent_service.AIAgentService.get_research_experiments",
        lambda self: [experiment_payload],
    )

    app = create_app(
        state=_DummyState(),
        bybit_client=None,
        settings=_DummySettings(),
        trading_loop=None,
        model_manager=None,
        tg_bot=None,
    )

    with TestClient(app) as client:
        response = client.get("/api/ai/research/status")
    assert response.status_code == 200
    body = response.json()
    assert "experiments" in body and isinstance(body["experiments"], list)
    exp = body["experiments"][0]
    assert exp["oos_metrics"]["recommended_tactic"] == "single_15m"
    assert exp["drift_signals"]["level"] == "low"
    assert exp["stress_results"]["stress_passed"] is True
    assert exp["analysis_summary"] == "ok"


def test_research_status_has_alias_keys_even_when_results_missing(monkeypatch):
    experiment_payload = {
        "id": "exp_test_2",
        "symbol": "BTCUSDT",
        "type": "balanced",
        "status": "starting",
        "results": {},
    }

    monkeypatch.setattr(
        "bot.ai_agent_service.AIAgentService.get_research_experiments",
        lambda self: [experiment_payload],
    )

    app = create_app(
        state=_DummyState(),
        bybit_client=None,
        settings=_DummySettings(),
        trading_loop=None,
        model_manager=None,
        tg_bot=None,
    )

    with TestClient(app) as client:
        response = client.get("/api/ai/research/status")
    assert response.status_code == 200
    exp = response.json()["experiments"][0]
    assert "oos_metrics" in exp
    assert "drift_signals" in exp
    assert "stress_results" in exp
    assert exp["oos_metrics"] is None
    assert exp["drift_signals"] is None
    assert exp["stress_results"] is None
