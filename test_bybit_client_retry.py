import pytest

from bot.exchange.bybit_client import BybitClient


def test_retry_request_retries_when_pybit_rate_limit_header_missing(monkeypatch):
    client = BybitClient.__new__(BybitClient)
    sleep_calls = []

    monkeypatch.setattr("bot.exchange.bybit_client.time.sleep", lambda value: sleep_calls.append(value))

    state = {"calls": 0}

    def flaky_request():
        state["calls"] += 1
        if state["calls"] == 1:
            raise KeyError("x-bapi-limit-reset-timestamp")
        return {"retCode": 0, "result": {"list": []}}

    response = client._retry_request(flaky_request, max_retries=3, delay=1.0)

    assert response["retCode"] == 0
    assert state["calls"] == 2
    assert sleep_calls == [1.0]


def test_retry_request_raises_after_exhausted_missing_header_retries(monkeypatch):
    client = BybitClient.__new__(BybitClient)
    sleep_calls = []

    monkeypatch.setattr("bot.exchange.bybit_client.time.sleep", lambda value: sleep_calls.append(value))

    def always_fails():
        raise KeyError("x-bapi-limit-reset-timestamp")

    with pytest.raises(KeyError):
        client._retry_request(always_fails, max_retries=2, delay=1.0)

    assert sleep_calls == [1.0]


def test_retry_request_retries_on_generic_10006_rate_limit_exception(monkeypatch):
    client = BybitClient.__new__(BybitClient)
    sleep_calls = []

    monkeypatch.setattr("bot.exchange.bybit_client.time.sleep", lambda value: sleep_calls.append(value))

    state = {"calls": 0}

    def flaky_request():
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("Too many visits. Exceeded the API Rate Limit. (ErrCode: 10006)")
        return {"retCode": 0}

    response = client._retry_request(flaky_request, max_retries=3, delay=0.5)

    assert response["retCode"] == 0
    assert state["calls"] == 2
    assert sleep_calls == [0.5]
