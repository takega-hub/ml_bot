import sys
from pathlib import Path

import asyncio
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot.config import AppSettings
from bot.strategy import Action, Signal
from bot.trading_loop import TradingLoop


class DummyState:
    def __init__(self):
        self.symbol_models = {}
        self.trades = []

    def get_open_position(self, symbol):
        return None

    def add_trade(self, trade):
        self.trades.append(trade)


class DummyBybit:
    def __init__(self):
        self.open_orders = {}
        self.placed = []
        self.canceled = []

    def round_price(self, price: float, symbol: str) -> float:
        return float(price)

    def get_qty_step(self, symbol: str) -> float:
        return 0.001

    def get_wallet_balance(self):
        return {"retCode": 0, "result": {"list": [{"coin": [{"coin": "USDT", "walletBalance": "100000"}]}]}}

    def place_order(self, **payload):
        self.placed.append(payload)
        link = payload.get("orderLinkId") or payload.get("order_link_id")
        if link:
            self.open_orders[link] = {"orderId": "OID1", "orderLinkId": link}
        return {"retCode": 0, "retMsg": "OK", "result": {}}

    def get_open_orders(self, symbol=None, order_id=None, order_link_id=None):
        if order_link_id and order_link_id in self.open_orders:
            return {"retCode": 0, "result": {"list": [self.open_orders[order_link_id]]}}
        return {"retCode": 0, "result": {"list": []}}

    def cancel_order(self, symbol: str, order_id=None, order_link_id=None):
        self.canceled.append((symbol, order_id, order_link_id))
        if order_link_id and order_link_id in self.open_orders:
            del self.open_orders[order_link_id]
        return {"retCode": 0, "retMsg": "OK", "result": {}}


async def main():
    settings = AppSettings()
    settings.timeframe = "15m"
    settings.ml_strategy.pullback_enabled = True
    settings.ml_strategy.pullback_entry_mode = "limit_roll"
    settings.ml_strategy.pullback_pct = 0.003
    settings.ml_strategy.pullback_max_bars = 3
    settings.ml_strategy.pullback_limit_roll_min_requote_pct = 0.01
    settings.ml_strategy.pullback_limit_roll_conf_drop_pct = 0.05

    loop = TradingLoop.__new__(TradingLoop)
    loop.settings = settings
    loop.state = DummyState()
    loop.bybit = DummyBybit()
    loop.pending_pullback_entry_orders = {}
    loop.pending_pullback_signals = {}

    t0 = pd.Timestamp.now()
    signal = Signal(
        timestamp=t0,
        action=Action.LONG,
        reason="x",
        price=100.0,
        take_profit=101.0,
        stop_loss=99.0,
        indicators_info={"confidence": 0.7},
    )
    await loop._place_pullback_limit_entry_order("BTCUSDT", signal, t0, 100.0, 99.0)
    rec = loop.pending_pullback_entry_orders["BTCUSDT"]
    assert abs(rec["limit_price"] - 99.7) < 1e-9
    assert len(loop.bybit.placed) == 1

    t1 = t0 + pd.Timedelta(minutes=15)
    signal2 = Signal(
        timestamp=t1,
        action=Action.LONG,
        reason="x2",
        price=100.0,
        take_profit=101.0,
        stop_loss=99.0,
        indicators_info={"confidence": 0.7},
    )
    await loop._handle_pullback_limit_roll_signal("BTCUSDT", signal2, t1, 100.2, 99.0)
    assert "BTCUSDT" in loop.pending_pullback_entry_orders
    assert len(loop.bybit.placed) == 1
    assert len(loop.bybit.canceled) == 0

    t2 = t1 + pd.Timedelta(minutes=15)
    signal3 = Signal(
        timestamp=t2,
        action=Action.LONG,
        reason="x3",
        price=100.0,
        take_profit=102.0,
        stop_loss=99.0,
        indicators_info={"confidence": 0.7},
    )
    await loop._handle_pullback_limit_roll_signal("BTCUSDT", signal3, t2, 103.0, 99.0)
    assert "BTCUSDT" in loop.pending_pullback_entry_orders
    assert len(loop.bybit.placed) == 2
    assert len(loop.bybit.canceled) == 1

    t3 = t2 + pd.Timedelta(minutes=15)
    signal4 = Signal(
        timestamp=t3,
        action=Action.LONG,
        reason="x4",
        price=100.0,
        take_profit=102.0,
        stop_loss=99.0,
        indicators_info={"confidence": 0.1},
    )
    await loop._handle_pullback_limit_roll_signal("BTCUSDT", signal4, t3, 103.0, 99.0)
    assert "BTCUSDT" not in loop.pending_pullback_entry_orders
    assert len(loop.bybit.placed) == 2
    assert len(loop.bybit.canceled) == 2


if __name__ == "__main__":
    asyncio.run(main())
