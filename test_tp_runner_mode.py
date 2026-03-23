import asyncio
from pathlib import Path

import pandas as pd

from bot.config import AppSettings
from bot.state import BotState, TradeRecord
from bot.strategy import Signal, Action
from bot.trading_loop import TradingLoop


class DummyBybit:
    def __init__(self):
        self.last_place_order_kwargs = None

    def get_wallet_balance(self, account_type: str = "UNIFIED"):
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "coin": [
                            {
                                "coin": "USDT",
                                "walletBalance": "1000",
                            }
                        ]
                    }
                ]
            },
        }

    def get_qty_step(self, symbol: str) -> float:
        return 0.001

    def place_order(self, **kwargs):
        self.last_place_order_kwargs = dict(kwargs)
        return {"retCode": 0, "retMsg": "OK", "result": {}}

    def set_trading_stop(self, **kwargs):
        return {"retCode": 0, "retMsg": "OK", "result": {}}


def _tmp_state(tmp_path: Path) -> BotState:
    return BotState(state_file=str(tmp_path / "state.json"))


def test_execute_trade_does_not_send_tp_when_disabled(tmp_path: Path):
    settings = AppSettings()
    settings.active_symbols = ["BTCUSDT"]
    settings.primary_symbol = "BTCUSDT"
    settings.risk.base_order_usd = 10.0
    settings.risk.use_take_profit = False

    state = _tmp_state(tmp_path)
    bybit = DummyBybit()
    loop = TradingLoop(settings=settings, state=state, bybit=bybit)

    sig = Signal(
        timestamp=pd.Timestamp.utcnow(),
        action=Action.LONG,
        reason="test",
        price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        indicators_info={},
    )

    asyncio.run(loop.execute_trade(symbol="BTCUSDT", side="Buy", signal=sig, is_add=False))
    assert bybit.last_place_order_kwargs is not None
    assert bybit.last_place_order_kwargs.get("take_profit") is None


def test_partial_close_falls_back_to_local_tp_when_exchange_tp_missing(tmp_path: Path):
    settings = AppSettings()
    settings.active_symbols = ["BTCUSDT"]
    settings.primary_symbol = "BTCUSDT"
    settings.risk.enable_partial_close = True
    settings.risk.partial_close_levels = [(0.5, 0.5)]

    state = _tmp_state(tmp_path)
    bybit = DummyBybit()
    loop = TradingLoop(settings=settings, state=state, bybit=bybit)

    state.add_trade(
        TradeRecord(
            symbol="BTCUSDT",
            side="Buy",
            entry_price=100.0,
            qty=2.0,
            status="open",
            take_profit=102.0,
            stop_loss=99.0,
        )
    )

    position_info = {
        "size": "2",
        "side": "Buy",
        "avgPrice": "100",
        "markPrice": "101.2",
        "takeProfit": "",
    }

    asyncio.run(loop.check_partial_close("BTCUSDT", position_info))
    assert bybit.last_place_order_kwargs is not None
    assert bybit.last_place_order_kwargs.get("reduce_only") is True
