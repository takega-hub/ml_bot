import sys
from pathlib import Path
import asyncio
from datetime import datetime, timezone
import uuid

# Корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.ai_agent_service import AIAgentService


async def main():
    agent = AIAgentService()
    payload = {
        "request_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "signal": {
            "action": "LONG",
            "reason": "test",
            "price": 65000.0,
            "stop_loss": 64000.0,
            "take_profit": 66500.0,
            "signal_timestamp": "2026-03-13 12:30:00",
            "confidence": 0.62,
            "strength": "слабое",
        },
        "bot_context": {
            "side": "Buy",
            "leverage": 10,
            "position_horizon": "short_term",
            "risk_settings": {
                "base_order_usd": 50.0,
                "margin_pct_balance": 0.2,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.015,
                "max_position_usd": 200.0,
            },
            "ai_fallback_policy": {
                "force_enabled": False,
                "spread_reduce_pct": 0.10,
                "spread_veto_pct": 0.25,
                "min_depth_usd_5": 0.0,
                "imbalance_abs_reduce": 0.60,
                "orderflow_ratio_low": 0.40,
                "orderflow_ratio_high": 2.50,
            },
        },
        "market_context": {"ohlcv": [], "orderbook": {}, "recent_trades": {}},
    }
    agent.validate_confirm_entry_request(payload)
    res = await agent.confirm_entry(payload)
    assert isinstance(res, dict)
    assert res.get("decision") in ("allow", "reduce", "veto")
    assert res.get("decision_id") == payload["request_id"]
    assert "timestamp_utc" in res
    assert isinstance(res.get("size_multiplier"), (int, float))
    if res.get("decision") == "reduce":
        assert float(res["size_multiplier"]) in (0.1, 0.25, 0.5)
    print("OK", res.get("decision"), res.get("reason_codes"))


if __name__ == "__main__":
    asyncio.run(main())
