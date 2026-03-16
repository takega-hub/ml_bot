import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.trading_loop import TradingLoop


def main():
    decision_id = "test-id"

    res = TradingLoop._normalize_ai_confirm_entry_result({"decision": "REJECT"}, decision_id)
    assert res["decision"] == "veto"
    assert res["decision_id"] == decision_id

    res = TradingLoop._normalize_ai_confirm_entry_result({"decision": "WAIT"}, decision_id)
    assert res["decision"] == "veto"

    res = TradingLoop._normalize_ai_confirm_entry_result({"decision": "APPROVE"}, decision_id)
    assert res["decision"] == "allow"

    res = TradingLoop._normalize_ai_confirm_entry_result({"decision": "reduce", "size_multiplier": 2}, decision_id)
    assert res["decision"] == "reduce"
    assert res["size_multiplier"] in (0.1, 0.25, 0.5)

    print("OK")


if __name__ == "__main__":
    main()

