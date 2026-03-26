"""
REST API для мобильного приложения (iPhone).
Полный функционал как в Telegram-админке: статус, пары, риск, ML, модели, история, экстренные действия.
Аутентификация: заголовок X-API-Key (MOBILE_API_KEY в .env).
"""
import asyncio
from collections import defaultdict, deque
import hashlib
import json
import logging
import os
import re
import subprocess
import statistics
import sys
import time
import urllib.request
import urllib.parse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, List, Dict
import shutil
import signal
from .audit_logger import append_jsonl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SIGNAL_LOG_PATTERN_SIGNAL_GEN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?SIGNAL GEN:\s+"
    r"(?P<pair>[A-Z0-9]+)\s+(?P<direction>LONG|SHORT)\s+Conf=(?P<conf>[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_SIGNAL_LOG_PATTERN_SIGNAL_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?\[(?P<pair>[A-Z0-9]+)\]\s+"
    r"Signal:\s+(?P<direction>LONG|SHORT|HOLD)\b.*?Confidence:\s+(?P<conf>[0-9]*\.?[0-9]+)%",
    re.IGNORECASE,
)
_TIMEFRAME_TO_MINUTES = {"15m": 15, "1h": 60}


def _parse_datetime_filter(value: Optional[str], *, is_end: bool = False) -> Optional[datetime]:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    parsed: Optional[datetime] = None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
            if is_end:
                parsed = parsed.replace(hour=23, minute=59, second=59)
        except Exception as exc:
            raise ValueError(f"Invalid datetime format: {value}") from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _slot_floor(timestamp: datetime, timeframe: str) -> datetime:
    minutes = _TIMEFRAME_TO_MINUTES[timeframe]
    floored_minute = (timestamp.minute // minutes) * minutes
    return timestamp.replace(minute=floored_minute, second=0, microsecond=0)


def _normalize_signal(direction: str, conf: float) -> float:
    if direction == "HOLD":
        return 0.0
    return conf if direction == "LONG" else -conf


def _parse_signal_line(line: str) -> Optional[Dict[str, Any]]:
    m = _SIGNAL_LOG_PATTERN_SIGNAL_GEN.search(line)
    source = "signal_gen"
    if not m:
        m = _SIGNAL_LOG_PATTERN_SIGNAL_LINE.search(line)
        source = "signal_line"
    if not m:
        return None

    try:
        ts = datetime.strptime(m.group("timestamp"), "%Y-%m-%d %H:%M:%S")
        pair = m.group("pair").upper()
        direction = m.group("direction").upper()
        conf = float(m.group("conf"))
    except Exception:
        return None

    if source == "signal_line":
        conf = conf / 100.0

    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0

    if direction not in {"LONG", "SHORT", "HOLD"}:
        return None

    return {"timestamp": ts, "pair": pair, "direction": direction, "conf": conf}


def _parse_signals_for_dashboard(
    log_paths: List[Path],
    timeframe: str,
    agg_method: str,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    pairs_filter: Optional[set[str]],
    strong_threshold: float,
    max_lines: int,
) -> Dict[str, Any]:
    existing_paths = [p for p in log_paths if p.exists()]
    if not existing_paths:
        return {
            "points": [],
            "pairs": [],
            "time_slots": [],
            "total_raw_signals": 0,
            "total_points": 0,
            "strong_threshold": strong_threshold,
        }

    grouped: Dict[tuple[datetime, str], List[Dict[str, Any]]] = defaultdict(list)
    pair_activity: Dict[str, int] = defaultdict(int)
    total_raw_signals = 0
    seen: set[tuple[str, str, str, float]] = set()

    for log_path in existing_paths:
        tail_lines: deque[str] = deque(maxlen=max_lines)
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tail_lines.append(line.rstrip("\n"))

        for line in tail_lines:
            rec = _parse_signal_line(line)
            if not rec:
                continue

            ts = rec["timestamp"]
            pair = rec["pair"]
            direction = rec["direction"]
            conf = float(rec["conf"])

            key = (ts.isoformat(), pair, direction, round(conf, 6))
            if key in seen:
                continue
            seen.add(key)

            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            if pairs_filter and pair not in pairs_filter:
                continue

            total_raw_signals += 1
            pair_activity[pair] += 1
            slot = _slot_floor(ts, timeframe)
            y_value = _normalize_signal(direction, conf)
            grouped[(slot, pair)].append(
                {
                    "timestamp": ts,
                    "pair": pair,
                    "direction": direction,
                    "conf": conf,
                    "y": y_value,
                }
            )

    points: List[Dict[str, Any]] = []
    for (slot, pair), entries in grouped.items():
        if agg_method == "mean":
            y_value = sum(e["y"] for e in entries) / len(entries)
            conf_value = sum(e["conf"] for e in entries) / len(entries)
            direction = "HOLD" if abs(y_value) < 1e-12 else ("LONG" if y_value >= 0 else "SHORT")
            source_ts = max(e["timestamp"] for e in entries)
        else:
            latest = max(entries, key=lambda item: item["timestamp"])
            y_value = latest["y"]
            conf_value = latest["conf"]
            direction = latest["direction"]
            source_ts = latest["timestamp"]

        points.append(
            {
                "time_slot": slot.isoformat(),
                "timestamp": source_ts.isoformat(),
                "pair": pair,
                "direction": direction,
                "conf": round(conf_value, 6),
                "y_value": round(y_value, 6),
                "is_strong": (direction != "HOLD") and (abs(conf_value) >= strong_threshold),
            }
        )

    points.sort(key=lambda p: (p["time_slot"], p["pair"]))
    pairs_sorted = sorted(pair_activity.items(), key=lambda kv: kv[1], reverse=True)
    time_slots_sorted = sorted({p["time_slot"] for p in points})
    return {
        "points": points,
        "pairs": [p for p, _ in pairs_sorted],
        "pair_activity": [{"pair": p, "count": c} for p, c in pairs_sorted],
        "time_slots": time_slots_sorted,
        "total_raw_signals": total_raw_signals,
        "total_points": len(points),
        "strong_threshold": strong_threshold,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _get_status_data(state, bybit_client, settings, trading_loop=None) -> Dict[str, Any]:
    """Собирает данные для /api/status (без блокировки event loop надолго)."""
    wallet_balance = 0.0
    available = 0.0
    total_margin = 0.0
    open_positions: List[Dict[str, Any]] = []

    if bybit_client:
        try:
            balance_info = bybit_client.get_wallet_balance()
            if balance_info.get("retCode") == 0:
                result = balance_info.get("result", {})
                list_data = result.get("list", [])
                if list_data:
                    wallet = list_data[0].get("coin", [])
                    usdt = next((c for c in wallet if c.get("coin") == "USDT"), None)
                    if usdt:
                        wallet_balance = _safe_float(usdt.get("walletBalance"), 0)
        except Exception as e:
            logger.error(f"API: error getting balance: {e}")

        try:
            for symbol in state.active_symbols:
                pos_info = bybit_client.get_position_info(symbol=symbol)
                if pos_info.get("retCode") != 0:
                    continue
                for p in pos_info.get("result", {}).get("list", []):
                    size = _safe_float(p.get("size"), 0)
                    if size <= 0:
                        continue
                    side = p.get("side", "Buy")
                    entry_price = _safe_float(p.get("avgPrice"), 0)
                    mark_price = _safe_float(p.get("markPrice"), 0) or _safe_float(p.get("lastPrice"), entry_price) or entry_price
                    unrealised_pnl = _safe_float(p.get("unrealisedPnl"), 0)
                    if unrealised_pnl == 0:
                        # Иногда Bybit не возвращает unrealisedPnl сразу, считаем сами
                        if side == "Buy":
                            unrealised_pnl = (mark_price - entry_price) * size
                        else:
                            unrealised_pnl = (entry_price - mark_price) * size
                    sym_for_lev = p.get("symbol", "")
                    default_lev = settings.get_leverage_for_symbol(sym_for_lev)
                    leverage = _safe_float(p.get("leverage", default_lev), default_lev)
                    margin = _safe_float(p.get("positionMargin"), 0) or _safe_float(p.get("positionIM"), 0)
                    if margin == 0:
                        pv = _safe_float(p.get("positionValue"), 0)
                        if pv > 0 and leverage > 0:
                            margin = pv / leverage
                    pnl_pct = ((mark_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - mark_price) / entry_price * 100)
                    open_positions.append({
                        "symbol": symbol,
                        "side": side,
                        "size": size,
                        "entry": entry_price,
                        "current": mark_price,
                        "pnl": unrealised_pnl,
                        "pnl_pct": round(pnl_pct, 2),
                        "leverage": leverage,
                        "margin": margin,
                        "tp": float(p["takeProfit"]) if p.get("takeProfit") else None,
                        "sl": float(p["stopLoss"]) if p.get("stopLoss") else None,
                    })
                    total_margin += margin
        except Exception as e:
            logger.error(f"API: error getting positions: {e}")

    available = max(0.0, wallet_balance - total_margin)

    strategies: List[Dict[str, Any]] = []
    for symbol in state.active_symbols:
        strategy_info: Dict[str, Any] = {"symbol": symbol, "model": None, "mtf": False}
        
        # Try to get from running strategy first
        strategy = None
        if trading_loop and getattr(trading_loop, "strategies", None):
            strategy = trading_loop.strategies.get(symbol)
            
        if strategy:
            if getattr(strategy, "predict_combined", None):
                strategy_info["mtf"] = True
                strategy_info["model_1h"] = getattr(strategy, "model_1h_path", None)
                strategy_info["model_15m"] = getattr(strategy, "model_15m_path", None)
            else:
                strategy_info["model"] = getattr(strategy, "model_path", None)
        else:
            # Fallback to state config
            config = state.get_strategy_config(symbol) if hasattr(state, "get_strategy_config") else None
            if config:
                mode = config.get("mode", "single")
                if mode == "mtf":
                    strategy_info["mtf"] = True
                    strategy_info["model_1h"] = config.get("model_1h_path")
                    strategy_info["model_15m"] = config.get("model_15m_path")
                else:
                    strategy_info["model"] = config.get("model_path") or state.symbol_models.get(symbol)
            else:
                strategy_info["model"] = state.symbol_models.get(symbol)
                
        cooldown = state.get_cooldown_info(symbol) if hasattr(state, "get_cooldown_info") else None
        if cooldown and cooldown.get("active"):
            strategy_info["cooldown"] = {"hours_left": cooldown.get("hours_left"), "reason": cooldown.get("reason", "")}
        strategies.append(strategy_info)

    stats = state.get_stats()
    unrealized_pnl = sum(p["pnl"] for p in open_positions)
    
    # Calculate Today PnL
    from datetime import datetime
    today = datetime.now().date()
    closed = [t for t in state.trades if t.status == "closed"]
    today_trades = [t for t in closed if t.exit_time and datetime.fromisoformat(t.exit_time).date() == today]
    today_pnl = sum(t.pnl_usd for t in today_trades)
    
    return {
        "is_running": state.is_running,
        "wallet_balance": round(wallet_balance, 2),
        "available_balance": round(available, 2),
        "total_margin": round(total_margin, 2),
        "positions": open_positions,
        "strategies": strategies,
        "active_symbols": list(state.active_symbols),
        "total_pnl": round(stats["total_pnl"], 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "today_pnl": round(today_pnl, 2),
        "win_rate": round(stats["win_rate"], 1),
        "total_trades": stats["total_trades"],
    }


def _get_dashboard_data(state, bybit_client, settings) -> Dict[str, Any]:
    """Собирает данные для /api/dashboard."""
    from datetime import datetime, timedelta

    wallet_balance = 0.0
    total_margin = 0.0
    open_count = 0
    total_pnl_open = 0.0

    if bybit_client:
        try:
            balance_info = bybit_client.get_wallet_balance()
            if balance_info.get("retCode") == 0:
                result = balance_info.get("result", {})
                list_data = result.get("list", [])
                if list_data:
                    wallet = list_data[0].get("coin", [])
                    usdt = next((c for c in wallet if c.get("coin") == "USDT"), None)
                    if usdt:
                        wallet_balance = _safe_float(usdt.get("walletBalance"), 0)
        except Exception as e:
            logger.error(f"API: error getting balance: {e}")

        try:
            for symbol in state.active_symbols:
                pos_info = bybit_client.get_position_info(symbol=symbol)
                if pos_info.get("retCode") != 0:
                    continue
                for p in pos_info.get("result", {}).get("list", []):
                    size = _safe_float(p.get("size"), 0)
                    if size <= 0:
                        continue
                    open_count += 1
                    total_pnl_open += _safe_float(p.get("unrealisedPnl"), 0)
                    margin = _safe_float(p.get("positionMargin"), 0) or _safe_float(p.get("positionIM"), 0)
                    if margin == 0:
                        pv = _safe_float(p.get("positionValue"), 0)
                        lev = _safe_float(p.get("leverage", settings.leverage), settings.leverage)
                        if pv > 0 and lev > 0:
                            margin = pv / lev
                    total_margin += margin
        except Exception as e:
            logger.error(f"API: error getting positions: {e}")

    available = max(0.0, wallet_balance - total_margin)
    stats = state.get_stats()
    total_pnl_pct = (stats["total_pnl"] / wallet_balance * 100) if wallet_balance > 0 else 0

    today = datetime.now().date()
    week_ago = datetime.now() - timedelta(days=7)
    closed = [t for t in state.trades if t.status == "closed"]
    today_trades = [t for t in closed if t.exit_time and datetime.fromisoformat(t.exit_time).date() == today]
    week_trades = [t for t in closed if t.exit_time and datetime.fromisoformat(t.exit_time) >= week_ago]

    today_pnl = sum(t.pnl_usd for t in today_trades)
    week_pnl = sum(t.pnl_usd for t in week_trades)
    week_winrate = (len([t for t in week_trades if t.pnl_usd > 0]) / len(week_trades) * 100) if week_trades else 0

    return {
        "wallet_balance": round(wallet_balance, 2),
        "available_balance": round(available, 2),
        "total_margin": round(total_margin, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "open_positions_count": open_count,
        "open_positions_pnl": round(total_pnl_open, 2),
        "today_trades": len(today_trades),
        "today_pnl": round(today_pnl, 2),
        "week_pnl": round(week_pnl, 2),
        "week_winrate": round(week_winrate, 1),
        "week_trades": len(week_trades),
        "is_running": state.is_running,
        "active_symbols_count": len(state.active_symbols),
    }


def create_app(state, bybit_client, settings, trading_loop=None, model_manager=None, tg_bot=None):
    """Создаёт FastAPI приложение с инжектированными зависимостями."""
    logger.info("[Mobile API] create_app: импорт FastAPI...")
    try:
        from fastapi import FastAPI, Depends, HTTPException, Header, Body, BackgroundTasks, Response
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError as e:
        logger.error(f"[Mobile API] create_app: ошибка импорта FastAPI: {e}")
        raise ImportError("Установите fastapi и uvicorn: pip install fastapi uvicorn") from e

    # Инициализация AI Agent
    from bot.ai_agent_service import AIAgentService
    ai_agent = AIAgentService()
    
    # Get paper trading manager from trading loop
    paper_trading_manager = None
    if trading_loop and hasattr(trading_loop, 'paper_trading_manager'):
        paper_trading_manager = trading_loop.paper_trading_manager

    logger.info("[Mobile API] create_app: создание app и middleware...")
    app = FastAPI(title="ML Trading Bot API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key = (os.getenv("MOBILE_API_KEY") or "").strip()
    if not api_key:
        api_key = os.getenv("ALLOWED_USER_ID", "").strip()  # fallback: Telegram user id как ключ

    def _send_telegram_sync(text: str):
        if not tg_bot or not tg_bot.settings.telegram_token or not tg_bot.settings.allowed_user_id:
            return
        try:
            url = f"https://api.telegram.org/bot{tg_bot.settings.telegram_token}/sendMessage"
            data = {
                "chat_id": tg_bot.settings.allowed_user_id,
                "text": text
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                pass
        except Exception as e:
            logger.error(f"Failed to send telegram notification sync: {e}")

    def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if not api_key:
            return True
        if not x_api_key or x_api_key.strip() != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    runbook_state_file = PROJECT_ROOT / "runbook_incident_state.json"
    planner_policy_config_file = PROJECT_ROOT / "planner_policy_config.json"
    planner_policy_alert_state_file = PROJECT_ROOT / "planner_policy_alert_state.json"

    def _load_runbook_state() -> Dict[str, Any]:
        try:
            if runbook_state_file.exists():
                with open(runbook_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    incidents = data.get("incidents")
                    if not isinstance(incidents, list):
                        data["incidents"] = []
                    return data
        except Exception:
            pass
        return {"incidents": []}

    def _save_runbook_state(payload: Dict[str, Any]) -> None:
        try:
            runbook_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(runbook_state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save runbook state: {e}")

    def _incident_stats(window_hours: int = 24) -> Dict[str, Any]:
        state_payload = _load_runbook_state()
        incidents = state_payload.get("incidents")
        if not isinstance(incidents, list):
            incidents = []
        cutoff = datetime.now(timezone.utc).timestamp() - (max(1, int(window_hours)) * 3600)
        in_window: List[Dict[str, Any]] = []
        for row in incidents:
            if not isinstance(row, dict):
                continue
            ts_raw = row.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            s = ts_raw
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(s)
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.timestamp() >= cutoff:
                in_window.append(row)
        high_count = 0
        critical_count = 0
        for row in in_window:
            level = str(row.get("level") or "").lower()
            if level == "high":
                high_count += 1
            elif level == "critical":
                critical_count += 1
        return {
            "window_hours": window_hours,
            "count_all": len(in_window),
            "count_high": high_count,
            "count_critical": critical_count,
            "count_high_or_critical": high_count + critical_count,
        }

    def _record_incident(level: str, symbol: Optional[str], details: Dict[str, Any]) -> Dict[str, Any]:
        state_payload = _load_runbook_state()
        incidents = state_payload.get("incidents")
        if not isinstance(incidents, list):
            incidents = []
        incidents.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": str(level or "").lower(),
                "symbol": symbol,
                "details": details if isinstance(details, dict) else {},
            }
        )
        state_payload["incidents"] = incidents[-500:]
        _save_runbook_state(state_payload)
        return _incident_stats(24)

    def _default_planner_policy_config() -> Dict[str, Any]:
        return {
            "active_profile": "balanced",
            "min_actionable_for_alert": 5,
            "min_conversion_actionable": 35.0,
            "alert_cooldown_minutes": 60,
        }

    def _normalize_planner_policy_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = _default_planner_policy_config()
        data = raw if isinstance(raw, dict) else {}
        profile = str(data.get("active_profile") or cfg["active_profile"]).strip().lower()
        if profile not in {"conservative", "balanced", "aggressive"}:
            profile = str(cfg["active_profile"])
        cfg["active_profile"] = profile
        cfg["min_actionable_for_alert"] = max(
            1,
            min(1000, int(_safe_float(data.get("min_actionable_for_alert"), cfg["min_actionable_for_alert"]))),
        )
        cfg["min_conversion_actionable"] = max(
            0.0,
            min(100.0, float(_safe_float(data.get("min_conversion_actionable"), cfg["min_conversion_actionable"]))),
        )
        cfg["alert_cooldown_minutes"] = max(
            1,
            min(24 * 60, int(_safe_float(data.get("alert_cooldown_minutes"), cfg["alert_cooldown_minutes"]))),
        )
        return cfg

    def _load_planner_policy_config() -> Dict[str, Any]:
        try:
            if planner_policy_config_file.exists():
                with open(planner_policy_config_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return _normalize_planner_policy_config(payload if isinstance(payload, dict) else {})
        except Exception:
            pass
        return _normalize_planner_policy_config({})

    def _save_planner_policy_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _normalize_planner_policy_config(payload)
        planner_policy_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(planner_policy_config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
        return cfg

    def _load_planner_policy_alert_state() -> Dict[str, Any]:
        try:
            if planner_policy_alert_state_file.exists():
                with open(planner_policy_alert_state_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    sent = payload.get("last_sent")
                    if not isinstance(sent, dict):
                        payload["last_sent"] = {}
                    return payload
        except Exception:
            pass
        return {"last_sent": {}}

    def _save_planner_policy_alert_state(payload: Dict[str, Any]) -> None:
        try:
            planner_policy_alert_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(planner_policy_alert_state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    def _emit_policy_alerts(alerts: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
        cooldown_sec = max(60, int(_safe_float(cfg.get("alert_cooldown_minutes"), 60) * 60))
        state_payload = _load_planner_policy_alert_state()
        sent = state_payload.get("last_sent")
        if not isinstance(sent, dict):
            sent = {}
        now_ts = time.time()
        emitted = 0
        suppressed = 0
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            key = (
                str(alert.get("type") or "unknown")
                + "|"
                + str(alert.get("profile") or cfg.get("active_profile") or "unknown")
            )
            last_ts = _safe_float(sent.get(key), 0.0)
            if last_ts > 0 and now_ts - last_ts < cooldown_sec:
                suppressed += 1
                continue
            sent[key] = now_ts
            emitted += 1
            summary = json.dumps(alert, ensure_ascii=False, default=str)
            record = {
                "event_type": "planner_policy_alert",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
                "tool_name": "planner_policy_stats",
                "goal": "policy_alerting",
                "risk_tier": "medium",
                "ok": True,
                "summary": summary[:1200],
                "arguments": {
                    "active_profile": cfg.get("active_profile"),
                    "alert_cooldown_minutes": cfg.get("alert_cooldown_minutes"),
                },
            }
            append_jsonl(str(PROJECT_ROOT / "logs" / "tool_execution_log.jsonl"), record)
            try:
                text_path = PROJECT_ROOT / "logs" / "tool_execution_log.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                with open(text_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{record['timestamp_utc']} | PLANNER_POLICY_ALERT | risk=medium | ok=True | goal=policy_alerting | {record['summary']}\n"
                    )
            except Exception:
                pass
            _send_telegram_sync(
                f"Planner policy alert: {alert.get('type')} | profile={cfg.get('active_profile')} | details={summary[:500]}"
            )
        state_payload["last_sent"] = sent
        _save_planner_policy_alert_state(state_payload)
        return {"emitted": emitted, "suppressed": suppressed, "cooldown_seconds": cooldown_sec}

    @app.get("/")
    def root():
        """Корень: подсказка для проверки API."""
        return {
            "service": "ML Trading Bot API",
            "status": "ok",
            "health": "/api/health",
            "docs": "/docs",
        }

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.on_event("startup")
    async def _log_startup():
        logger.info("[Mobile API] FastAPI готов, запросы принимаются. Проверка: GET /api/health")

    @app.get("/api/status", dependencies=[Depends(verify_api_key)])
    def get_status():
        return _get_status_data(state, bybit_client, settings, trading_loop)

    @app.get("/api/dashboard", dependencies=[Depends(verify_api_key)])
    def get_dashboard():
        return _get_dashboard_data(state, bybit_client, settings)

    @app.post("/api/start", dependencies=[Depends(verify_api_key)])
    async def post_start():
        state.set_running(True)
        if tg_bot:
            await tg_bot.send_notification("🟢 Bot started via Mobile App")
        state.add_notification("Bot started via Mobile App", "success")
        return {"ok": True, "is_running": True}

    @app.post("/api/stop", dependencies=[Depends(verify_api_key)])
    async def post_stop():
        state.set_running(False)
        if tg_bot:
            await tg_bot.send_notification("🔴 Bot stopped via Mobile App")
        state.add_notification("Bot stopped via Mobile App", "warning")
        return {"ok": True, "is_running": False}

    @app.get("/api/settings", dependencies=[Depends(verify_api_key)])
    def get_settings():
        """Только чтение: активные пары, плечо, пороги ML (без секретов)."""
        return {
            "active_symbols": list(state.active_symbols),
            "known_symbols": list(state.known_symbols),
            "leverage": settings.leverage,
            "confidence_threshold": getattr(settings.ml_strategy, "confidence_threshold", 0.35),
            "mtf_confidence_1h": getattr(settings.ml_strategy, "mtf_confidence_threshold_1h", 0.5),
            "mtf_confidence_15m": getattr(settings.ml_strategy, "mtf_confidence_threshold_15m", 0.35),
        }

    # --- Pairs ---
    @app.get("/api/pairs", dependencies=[Depends(verify_api_key)])
    def get_pairs():
        """Список известных и активных пар с cooldown и статистикой."""
        known = {s for s in state.known_symbols if isinstance(s, str) and s.endswith("USDT")}
        active = set(state.active_symbols)
        all_possible = sorted(list(known | active))
        
        cooldowns = {}
        stats = {}
        
        from datetime import datetime, timedelta
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        for s in all_possible:
            # Cooldowns
            info = state.get_cooldown_info(s) if hasattr(state, "get_cooldown_info") else None
            if info and info.get("active"):
                cooldowns[s] = {"hours_left": info.get("hours_left"), "reason": info.get("reason", "")}
            
            # Stats
            s_trades = [t for t in state.trades if t.symbol == s and t.status == "closed"]
            
            def calc_period_stats(trades_subset):
                if not trades_subset:
                    return {"count": 0, "win_rate": 0.0, "pnl": 0.0}
                wins = len([t for t in trades_subset if t.pnl_usd > 0])
                pnl = sum(t.pnl_usd for t in trades_subset)
                return {
                    "count": len(trades_subset),
                    "win_rate": round(wins / len(trades_subset) * 100, 1),
                    "pnl": round(pnl, 2)
                }

            week_trades = [t for t in s_trades if t.exit_time and datetime.fromisoformat(t.exit_time) >= week_ago]
            month_trades = [t for t in s_trades if t.exit_time and datetime.fromisoformat(t.exit_time) >= month_ago]
            
            stats[s] = {
                "leverage": settings.get_leverage_for_symbol(s),
                "week": calc_period_stats(week_trades),
                "month": calc_period_stats(month_trades),
                "all": calc_period_stats(s_trades)
            }

        return {
            "known_symbols": list(state.known_symbols),
            "active_symbols": list(state.active_symbols),
            "max_active": getattr(state, "max_active_symbols", 5),
            "cooldowns": cooldowns,
            "stats": stats
        }

    class TogglePairBody(BaseModel):
        symbol: str

    @app.post("/api/pairs/toggle", dependencies=[Depends(verify_api_key)])
    def post_pairs_toggle(body: TogglePairBody):
        sym = body.symbol.upper()
        if not sym.endswith("USDT"):
            raise HTTPException(status_code=400, detail="Symbol must end with USDT")
        res = state.toggle_symbol(sym)
        if res is None:
            raise HTTPException(status_code=400, detail="Max active pairs reached")
        return {"ok": True, "symbol": sym, "enabled": res}

    class RemoveCooldownBody(BaseModel):
        symbol: str

    @app.post("/api/pairs/remove_cooldown", dependencies=[Depends(verify_api_key)])
    def post_pairs_remove_cooldown(body: RemoveCooldownBody):
        sym = body.symbol.upper()
        state.remove_cooldown(sym)
        return {"ok": True, "symbol": sym}

    class AddPairBody(BaseModel):
        symbol: str

    @app.post("/api/pairs/add", dependencies=[Depends(verify_api_key)])
    def post_pairs_add(body: AddPairBody):
        sym = body.symbol.upper()
        if not sym.endswith("USDT"):
            raise HTTPException(status_code=400, detail="Symbol must end with USDT")
        if sym in state.active_symbols:
            return {"ok": True, "symbol": sym, "message": "Already active"}
        try:
            if bybit_client:
                info = bybit_client.get_instrument_info(sym)
                if not info or not info.get("symbol"):
                    raise HTTPException(status_code=400, detail=f"Symbol {sym} not found on exchange")
            state.add_known_symbol(sym)
            res = state.enable_symbol(sym)
            if res is None:
                return {"ok": False, "symbol": sym, "message": "Max active pairs reached"}
            return {"ok": True, "symbol": sym, "enabled": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error adding pair")
            raise HTTPException(status_code=500, detail=str(e))

    class PairLeverageBody(BaseModel):
        leverage: int

    @app.post("/api/pairs/{symbol}/leverage", dependencies=[Depends(verify_api_key)])
    def post_pair_leverage(symbol: str, body: PairLeverageBody):
        sym = symbol.upper()
        if not (1 <= body.leverage <= 100):
            raise HTTPException(status_code=400, detail="Leverage must be between 1 and 100")
            
        ml_settings = settings.get_ml_settings_for_symbol(sym)
        ml_settings.leverage = body.leverage
        settings.set_ml_settings_for_symbol(sym, ml_settings)
        
        from bot.config import save_symbol_ml_settings
        save_symbol_ml_settings(settings)
        
        if trading_loop:
            asyncio.create_task(trading_loop.update_leverage_for_symbol(sym, body.leverage))
            
        return {"ok": True, "symbol": sym, "leverage": body.leverage}

    # --- Risk (read + write to file and in-memory) ---
    def _risk_to_dict(r):
        if is_dataclass(r):
            try:
                payload = asdict(r)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        try:
            return {
                k: v
                for k, v in vars(r).items()
                if not str(k).startswith("_")
            }
        except Exception:
            return {}

    def _normalize_risk_key(name: Any) -> str:
        k = str(name or "").strip().lower()
        k = k.replace(" ", "_").replace("-", "_")
        aliases = {
            "stop_loss": "stop_loss_pct",
            "stop_loss_percent": "stop_loss_pct",
            "take_profit": "take_profit_pct",
            "take_profit_percent": "take_profit_pct",
            "trailing_stop_activation": "trailing_stop_activation_pct",
            "trailing_stop_distance": "trailing_stop_distance_pct",
            "breakeven_level1_activation": "breakeven_level1_activation_pct",
            "breakeven_level2_activation": "breakeven_level2_activation_pct",
        }
        return aliases.get(k, k)

    @app.get("/api/risk", dependencies=[Depends(verify_api_key)])
    def get_risk():
        return _risk_to_dict(settings.risk)

    @app.put("/api/risk", dependencies=[Depends(verify_api_key)])
    def put_risk(body: Dict[str, Any] = Body(...)):
        # Capture old settings for history tracking
        old_settings = _risk_to_dict(settings.risk)
        
        risk = settings.risk
        applied_keys: List[str] = []
        ignored_keys: List[str] = []
        for key, val in body.items():
            key = _normalize_risk_key(key)
            if hasattr(risk, key):
                if key in ("margin_pct_balance", "stop_loss_pct", "take_profit_pct", "trailing_stop_activation_pct",
                           "trailing_stop_distance_pct", "breakeven_level1_activation_pct", "breakeven_level1_sl_pct",
                           "breakeven_level2_activation_pct", "breakeven_level2_sl_pct", "fee_rate",
                           "mid_term_tp_pct", "long_term_tp_pct", "long_term_sl_pct", "dca_drawdown_pct",
                           "dca_min_confidence", "reverse_min_confidence") and isinstance(val, (int, float)):
                    if val >= 1:
                        val = val / 100.0
                setattr(risk, key, val)
                applied_keys.append(key)
            else:
                ignored_keys.append(str(key))

        if not applied_keys:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No valid risk fields to update",
                    "ignored_keys": ignored_keys,
                },
            )
        
        # Capture new settings
        new_settings = _risk_to_dict(risk)
        
        risk_file = PROJECT_ROOT / "risk_settings.json"
        existing_payload: Dict[str, Any] = {}
        try:
            if risk_file.exists():
                with open(risk_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_payload = loaded
        except Exception:
            existing_payload = {}
        existing_payload.update(new_settings)
        tmp_file = risk_file.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(existing_payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, risk_file)
            
        # Notify AI Agent about changes
        try:
             # Get total trade count to track observation period
             total_trades = len(state.trades)
             # Pass old and new settings to record history
             ai_agent.on_risk_settings_updated(total_trades, old_settings, new_settings)
        except Exception as e:
             logger.error(f"Failed to notify AI agent about risk update: {e}")
             
        return {
            "ok": True,
            "risk": new_settings,
            "applied_keys": applied_keys,
            "ignored_keys": ignored_keys,
        }

    @app.get("/api/ai/risk_history", dependencies=[Depends(verify_api_key)])
    def get_ai_risk_history():
        """Returns the history of risk setting changes."""
        history = ai_agent.get_risk_history()
        normalized: List[Dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
            if not changes:
                continue
            normalized.append(
                {
                    "timestamp": item.get("timestamp"),
                    "trade_count": item.get("total_trades"),
                    "total_trades": item.get("total_trades"),
                    "changes": changes,
                }
            )
        return {"history": normalized[-200:]}

    @app.post("/api/ai/risk_history/clear", dependencies=[Depends(verify_api_key)])
    def post_ai_risk_history_clear():
        return ai_agent.clear_risk_history(clear_last_analysis=True)

    # --- ML settings (read + write) ---
    def _ml_to_dict(m):
        return {
            "use_mtf_strategy": getattr(m, "use_mtf_strategy", False),
            "mtf_confidence_threshold_1h": getattr(m, "mtf_confidence_threshold_1h", 0.5),
            "mtf_confidence_threshold_15m": getattr(m, "mtf_confidence_threshold_15m", 0.35),
            "mtf_alignment_mode": getattr(m, "mtf_alignment_mode", "strict"),
            "atr_filter_enabled": getattr(m, "atr_filter_enabled", False),
            "pullback_enabled": getattr(m, "pullback_enabled", True),
            "pullback_enter_on_continuation": getattr(m, "pullback_enter_on_continuation", True),
            "pullback_entry_mode": getattr(m, "pullback_entry_mode", "pending"),
            "pullback_limit_roll_min_requote_pct": getattr(m, "pullback_limit_roll_min_requote_pct", 0.001),
            "pullback_limit_roll_conf_drop_pct": getattr(m, "pullback_limit_roll_conf_drop_pct", 0.05),
            "follow_btc_filter_enabled": getattr(m, "follow_btc_filter_enabled", True),
            "follow_btc_override_confidence": getattr(m, "follow_btc_override_confidence", 0.80),
            "auto_optimize_strategies": getattr(m, "auto_optimize_strategies", False),
            "auto_optimize_day": getattr(m, "auto_optimize_day", "sunday"),
            "auto_optimize_hour": getattr(m, "auto_optimize_hour", 3),
            "use_fixed_sl_from_risk": getattr(m, "use_fixed_sl_from_risk", False),
            "ai_entry_confirmation_enabled": getattr(m, "ai_entry_confirmation_enabled", False),
            "ai_entry_confirmation_mode": getattr(m, "ai_entry_confirmation_mode", "enforce"),
            "ai_fallback_force_enabled": getattr(m, "ai_fallback_force_enabled", False),
            "ai_fallback_spread_reduce_pct": getattr(m, "ai_fallback_spread_reduce_pct", 0.10),
            "ai_fallback_spread_veto_pct": getattr(m, "ai_fallback_spread_veto_pct", 0.25),
            "ai_fallback_min_depth_usd_5": getattr(m, "ai_fallback_min_depth_usd_5", 0.0),
            "ai_fallback_imbalance_abs_reduce": getattr(m, "ai_fallback_imbalance_abs_reduce", 0.60),
            "ai_fallback_orderflow_ratio_low": getattr(m, "ai_fallback_orderflow_ratio_low", 0.40),
            "ai_fallback_orderflow_ratio_high": getattr(m, "ai_fallback_orderflow_ratio_high", 2.50),
            "decision_engine_enabled": getattr(m, "decision_engine_enabled", False),
            "decision_engine_mode": getattr(m, "decision_engine_mode", "shadow"),
            "decision_engine_allow_score": getattr(m, "decision_engine_allow_score", 0.35),
            "decision_engine_reduce_score": getattr(m, "decision_engine_reduce_score", 0.10),
            "decision_engine_w_ml_confidence": getattr(m, "decision_engine_w_ml_confidence", 1.2),
            "decision_engine_w_mtf_alignment": getattr(m, "decision_engine_w_mtf_alignment", 0.6),
            "decision_engine_w_atr_regime": getattr(m, "decision_engine_w_atr_regime", 0.6),
            "decision_engine_w_sr_proximity": getattr(m, "decision_engine_w_sr_proximity", 0.9),
            "decision_engine_w_trend_slope": getattr(m, "decision_engine_w_trend_slope", 0.3),
            "decision_engine_w_history_edge": getattr(m, "decision_engine_w_history_edge", 1.0),
            "decision_engine_atr_prefer_min_pct": getattr(m, "decision_engine_atr_prefer_min_pct", 0.35),
            "decision_engine_atr_prefer_max_pct": getattr(m, "decision_engine_atr_prefer_max_pct", 1.60),
            "confidence_threshold": getattr(m, "confidence_threshold", 0.35),
            "min_confidence_for_trade": getattr(m, "min_confidence_for_trade", 0.5),
        }

    @app.get("/api/ml", dependencies=[Depends(verify_api_key)])
    def get_ml():
        return _ml_to_dict(settings.ml_strategy)

    @app.put("/api/ml", dependencies=[Depends(verify_api_key)])
    def put_ml(body: Dict[str, Any] = Body(...)):
        m = settings.ml_strategy
        ml_file = PROJECT_ROOT / "ml_settings.json"
        data = {}
        if ml_file.exists():
            try:
                with open(ml_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        for key in (
            "use_mtf_strategy",
            "atr_filter_enabled",
            "pullback_enabled",
            "pullback_enter_on_continuation",
            "follow_btc_filter_enabled",
            "auto_optimize_strategies",
            "use_fixed_sl_from_risk",
            "ai_entry_confirmation_enabled",
            "ai_fallback_force_enabled",
            "decision_engine_enabled",
        ):
            if key in body:
                data[key] = bool(body[key])
        for key in (
            "mtf_confidence_threshold_1h",
            "mtf_confidence_threshold_15m",
            "confidence_threshold",
            "min_confidence_for_trade",
            "follow_btc_override_confidence",
            "ai_fallback_spread_reduce_pct",
            "ai_fallback_spread_veto_pct",
            "ai_fallback_min_depth_usd_5",
            "ai_fallback_imbalance_abs_reduce",
            "ai_fallback_orderflow_ratio_low",
            "ai_fallback_orderflow_ratio_high",
            "decision_engine_allow_score",
            "decision_engine_reduce_score",
            "decision_engine_w_ml_confidence",
            "decision_engine_w_mtf_alignment",
            "decision_engine_w_atr_regime",
            "decision_engine_w_sr_proximity",
            "decision_engine_w_trend_slope",
            "decision_engine_w_history_edge",
            "decision_engine_atr_prefer_min_pct",
            "decision_engine_atr_prefer_max_pct",
            "pullback_limit_roll_min_requote_pct",
            "pullback_limit_roll_conf_drop_pct",
        ):
            if key in body:
                v = float(body[key])
                if key in (
                    "mtf_confidence_threshold_1h",
                    "mtf_confidence_threshold_15m",
                    "confidence_threshold",
                    "min_confidence_for_trade",
                    "follow_btc_override_confidence",
                    "ai_fallback_spread_reduce_pct",
                    "ai_fallback_spread_veto_pct",
                    "pullback_limit_roll_min_requote_pct",
                    "pullback_limit_roll_conf_drop_pct",
                ) and v >= 1:
                    v = v / 100.0
                data[key] = v
                setattr(m, key, v)
        for key in ("mtf_alignment_mode", "auto_optimize_day", "decision_engine_mode", "ai_entry_confirmation_mode", "pullback_entry_mode"):
            if key in body:
                data[key] = body[key]
                setattr(m, key, body[key])
        if "auto_optimize_hour" in body:
            data["auto_optimize_hour"] = int(body["auto_optimize_hour"])
            m.auto_optimize_hour = data["auto_optimize_hour"]
        for k, v in data.items():
            if hasattr(m, k):
                if k in (
                    "confidence_threshold",
                    "min_confidence_for_trade",
                    "mtf_confidence_threshold_1h",
                    "mtf_confidence_threshold_15m",
                    "follow_btc_override_confidence",
                    "ai_fallback_spread_reduce_pct",
                    "ai_fallback_spread_veto_pct",
                ) and isinstance(v, (int, float)) and v >= 1:
                    v = v / 100.0
                setattr(m, k, v)
        with open(ml_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"ok": True, "ml": _ml_to_dict(m)}

    # --- Models ---
    @app.get("/api/models", dependencies=[Depends(verify_api_key)])
    def get_models_list():
        """Текущие модели по активным парам с деталями стратегий."""
        out = []
        from datetime import datetime
        
        for symbol in state.active_symbols:
            # Get config
            config = state.get_strategy_config(symbol) if hasattr(state, "get_strategy_config") else None
            active_mode = config.get("mode", "single") if config else "single"
            
            # Если стратегия запущена прямо сейчас в trading_loop, возьмем режим оттуда
            if trading_loop and getattr(trading_loop, "strategies", None):
                strategy = trading_loop.strategies.get(symbol)
                if strategy:
                     if getattr(strategy, "predict_combined", None):
                         active_mode = "mtf"
                     else:
                         active_mode = "single"

            # Single Info
            single_path = state.symbol_models.get(symbol)
            if config and config.get("mode") == "single":
                single_path = config.get("model_path") or single_path
            
            single_info = {"name": None, "updated": None}
            if single_path:
                p = Path(single_path)
                if p.exists():
                    single_info["name"] = p.stem
                    single_info["updated"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            
            # MTF Info
            mtf_info = {"model_1h": None, "model_15m": None, "updated": None}
            m1h_path = config.get("model_1h_path") if config else None
            m15m_path = config.get("model_15m_path") if config else None
            
            # Если MTF активен в trading_loop, берем пути оттуда, так как они самые точные
            if active_mode == "mtf" and trading_loop and getattr(trading_loop, "strategies", None):
                strategy = trading_loop.strategies.get(symbol)
                if strategy:
                    m1h_path = getattr(strategy, "model_1h_path", m1h_path)
                    m15m_path = getattr(strategy, "model_15m_path", m15m_path)
            
            if m1h_path and m15m_path:
                p1h = Path(m1h_path)
                p15m = Path(m15m_path)
                if p1h.exists() and p15m.exists():
                    mtf_info["model_1h"] = p1h.stem
                    mtf_info["model_15m"] = p15m.stem
                    # Use latest update time
                    ts = max(p1h.stat().st_mtime, p15m.stat().st_mtime)
                    mtf_info["updated"] = datetime.fromtimestamp(ts).isoformat()

            out.append({
                "symbol": symbol,
                "active_mode": active_mode,
                "single": single_info,
                "mtf": mtf_info
            })
        return {"symbols": out}

    def _get_mtf_candidates(symbol: str) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            # Find latest comparison files
            comp_files = sorted(list(PROJECT_ROOT.glob("ml_models_comparison_*.csv")), reverse=True)
            if not comp_files:
                return []
            
            # Iterate through files to find one that contains the symbol
            df = None
            for file_path in comp_files:
                try:
                    temp_df = pd.read_csv(file_path)
                    if symbol in temp_df["symbol"].values:
                        df = temp_df
                        # Found the latest file containing this symbol
                        break
                except Exception:
                    continue
            
            if df is None or df.empty:
                return []
            
            # Filter by symbol
            df = df[df["symbol"] == symbol]
            if df.empty:
                return []
                
            # Split by timeframe (mode_suffix)
            df_1h = df[df["mode_suffix"] == "1h"].sort_values("total_pnl_pct", ascending=False)
            df_15m = df[df["mode_suffix"] == "15m"].sort_values("total_pnl_pct", ascending=False)
            
            candidates = []
            
            top_1h = df_1h.head(3).to_dict('records')
            top_15m = df_15m.head(3).to_dict('records')
            
            if not top_1h or not top_15m:
                return []
                
            def make_cand(h_row, m_row, label):
                 h_path = str(h_row["model_path"])
                 m_path = str(m_row["model_path"])
                 
                 # Check existence
                 if not Path(h_path).exists() or not Path(m_path).exists():
                     return None
                     
                 return {
                    "name": label,
                    "model_1h": h_path,
                    "model_15m": m_path,
                    "pnl_1h": h_row["total_pnl_pct"],
                    "pnl_15m": m_row["total_pnl_pct"],
                    "total_pnl": h_row["total_pnl_pct"] + m_row["total_pnl_pct"]
                }

            # Combo 1: Best of both
            c1 = make_cand(top_1h[0], top_15m[0], f"Best Combo")
            if c1: candidates.append(c1)
            
            # Combo 2: Best 1h + 2nd 15m
            if len(top_15m) > 1:
                 c2 = make_cand(top_1h[0], top_15m[1], f"Alt Combo 1")
                 if c2: candidates.append(c2)
            
            # Combo 3: 2nd 1h + Best 15m
            if len(top_1h) > 1:
                 c3 = make_cand(top_1h[1], top_15m[0], f"Alt Combo 2")
                 if c3: candidates.append(c3)
                
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting MTF candidates: {e}")
            return []

    @app.get("/api/models/{symbol}", dependencies=[Depends(verify_api_key)])
    def get_models_for_symbol(symbol: str):
        symbol = symbol.upper()
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")
        models = model_manager.find_models_for_symbol(symbol)
        results = model_manager.get_model_test_results(symbol)
        
        # Get MTF candidates
        mtf_candidates = _get_mtf_candidates(symbol)
        
        # Determine current models
        current_1h = None
        current_15m = None
        current_single = None
        mode = "single"

        if trading_loop and getattr(trading_loop, "strategies", None):
            strategy = trading_loop.strategies.get(symbol)
            if strategy:
                if getattr(strategy, "predict_combined", None):
                    mode = "mtf"
                    current_1h = getattr(strategy, "model_1h_path", None)
                    current_15m = getattr(strategy, "model_15m_path", None)
                else:
                    mode = "single"
                    current_single = getattr(strategy, "model_path", None)
        
        if not current_single and not current_1h and not current_15m:
             # Try to get from config
             config = state.get_strategy_config(symbol) if hasattr(state, "get_strategy_config") else None
             if config:
                 mode = config.get("mode", "single")
                 if mode == "mtf":
                     current_1h = config.get("model_1h_path")
                     current_15m = config.get("model_15m_path")
                 else:
                     current_single = config.get("model_path")
             
             if not current_single and not current_1h:
                  current_single = state.symbol_models.get(symbol)

        # Calculate real stats per model
        real_stats = {}
        for t in state.trades:
            if t.symbol == symbol and t.model_name:
                model_key = Path(str(t.model_name)).stem
                if not model_key:
                    model_key = str(t.model_name)
                
                if model_key not in real_stats:
                    real_stats[model_key] = {"pnl": 0.0, "wins": 0, "count": 0}
                
                if t.status == "closed":
                    real_stats[model_key]["pnl"] += t.pnl_usd
                    real_stats[model_key]["count"] += 1
                    if t.pnl_usd > 0:
                        real_stats[model_key]["wins"] += 1

        list_models = []
        models_1h = []
        models_15m = []

        for i, mp in enumerate(models):
            mp_str = str(mp)
            m_name = Path(mp).stem
            res_test = results.get(mp_str, {})
            
            # Get real stats
            r_stats = real_stats.get(m_name, {"pnl": 0.0, "wins": 0, "count": 0})
            win_rate = (r_stats["wins"] / r_stats["count"] * 100) if r_stats["count"] > 0 else 0.0
            
            res_real = {
                "total_pnl": r_stats["pnl"],
                "win_rate": win_rate,
                "total_trades": r_stats["count"]
            }

            entry = {
                "index": i,
                "path": mp_str,
                "name": m_name,
                "is_active_single": mp_str == current_single,
                "is_active_1h": mp_str == current_1h,
                "is_active_15m": mp_str == current_15m,
                "test": res_test,
                "real": res_real,
            }
            list_models.append(entry)
            
            if "_1h" in m_name or "1h" in m_name:
                models_1h.append(entry)
            elif "_15m" in m_name or "15m" in m_name:
                models_15m.append(entry)
            
        return {
            "symbol": symbol, 
            "models": list_models,
            "models_1h": models_1h,
            "models_15m": models_15m,
            "mode": mode,
            "current_single": current_single,
            "current_1h": current_1h,
            "current_15m": current_15m,
            "mtf_candidates": mtf_candidates
        }

    class ModelsCleanupBody(BaseModel):
        min_age_days: int = 7
        dry_run: bool = False

    @app.post("/api/models/cleanup_old_inactive", dependencies=[Depends(verify_api_key)])
    def post_cleanup_old_inactive_models(body: ModelsCleanupBody):
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")

        active_paths: set[str] = set()
        with state.lock:
            for p in state.symbol_models.values():
                if isinstance(p, str) and p.strip():
                    active_paths.add(p)
            strategy_map = getattr(state, "symbol_strategies", {})
            if isinstance(strategy_map, dict):
                for cfg in strategy_map.values():
                    if not isinstance(cfg, dict):
                        continue
                    for key in ("model_path", "model_1h_path", "model_15m_path"):
                        p = cfg.get(key)
                        if isinstance(p, str) and p.strip():
                            active_paths.add(p)

        if trading_loop and getattr(trading_loop, "strategies", None):
            for strategy in trading_loop.strategies.values():
                for attr in ("model_path", "model_1h_path", "model_15m_path"):
                    p = getattr(strategy, attr, None)
                    if isinstance(p, str) and p.strip():
                        active_paths.add(p)

        result = model_manager.cleanup_old_inactive_models(
            active_model_paths=active_paths,
            min_age_days=body.min_age_days,
            dry_run=body.dry_run,
        )
        return result

    @app.post("/api/models/{symbol}/apply_best", dependencies=[Depends(verify_api_key)])
    def post_apply_best_mtf(symbol: str):
        symbol = symbol.upper()
        candidates = _get_mtf_candidates(symbol)
        if not candidates:
            raise HTTPException(status_code=404, detail="No MTF candidates found")
            
        best = candidates[0] # First one is "Best Combo"
        
        config = {
            "mode": "mtf",
            "model_1h_path": best["model_1h"],
            "model_15m_path": best["model_15m"]
        }
        
        state.set_strategy_config(symbol, config)
        logger.info(f"Applied Best MTF for {symbol}: {best['name']}")
        
        return {"ok": True, "applied": best}

    class ApplyModelBody(BaseModel):
        model_path: Optional[str] = None
        model_1h_path: Optional[str] = None
        model_15m_path: Optional[str] = None
        mode: str = "single"  # single or mtf

    @app.post("/api/models/{symbol}/apply", dependencies=[Depends(verify_api_key)])
    def post_apply_model(symbol: str, body: ApplyModelBody):
        symbol = symbol.upper()
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")
            
        config = {"mode": body.mode}
        
        if body.mode == "mtf":
            if not body.model_1h_path or not body.model_15m_path:
                 raise HTTPException(status_code=400, detail="Both 1h and 15m models required for MTF")
            
            config["model_1h_path"] = body.model_1h_path
            config["model_15m_path"] = body.model_15m_path
            
            logger.info(f"Applying MTF for {symbol}: 1h={body.model_1h_path}, 15m={body.model_15m_path}")
            state.set_strategy_config(symbol, config)
            
            return {"ok": True, "mode": "mtf"}
            
        else:
            if not body.model_path:
                raise HTTPException(status_code=400, detail="Model path required for Single mode")
            
            config["model_path"] = body.model_path
            
            model_manager.apply_model(symbol, body.model_path)
            state.symbol_models[symbol] = body.model_path
            state.set_strategy_config(symbol, config)
            
            return {"ok": True, "mode": "single", "model_path": body.model_path}

    class TestModelBody(BaseModel):
        symbol: str
        model_path: Optional[str] = None
        days: int = 14

    @app.post("/api/models/test", dependencies=[Depends(verify_api_key)])
    def post_test_model(body: TestModelBody, background_tasks: BackgroundTasks):
        symbol = body.symbol.upper()
        
        # Determine model path
        model_path = body.model_path
        if not model_path:
             model_path = state.symbol_models.get(symbol)
        
        if not model_path:
            raise HTTPException(status_code=400, detail=f"No active model for {symbol}")

        if not model_manager:
             raise HTTPException(status_code=501, detail="Model manager not available")

        # Run in background
        def _run_test():
            try:
                if tg_bot:
                    _send_telegram_sync(f"🧪 Started testing model {model_path} for {symbol}...")
                state.add_notification(f"Started testing {symbol}", "info")
                logger.info(f"Starting background test for {symbol} model {model_path}")
                results = model_manager.test_model(model_path, symbol, days=body.days)
                if results:
                    model_manager.save_model_test_result(symbol, model_path, results)
                    logger.info(f"Test finished for {symbol}")
                    pnl = results.get('total_pnl_pct', 0)
                    if tg_bot:
                        _send_telegram_sync(f"✅ Testing finished for {symbol}\nPnL: {pnl:.2f}%")
                    state.add_notification(f"Testing finished for {symbol}. PnL: {pnl:.2f}%", "success")
            except Exception as e:
                logger.error(f"Error in background test for {symbol}: {e}")
                if tg_bot:
                    _send_telegram_sync(f"❌ Testing failed for {symbol}: {e}")
                state.add_notification(f"Testing failed for {symbol}", "error")

        background_tasks.add_task(_run_test)
        
        return {"ok": True, "symbol": symbol, "message": "Test started in background"}

    class TestAllSingleModelsBody(BaseModel):
        days: int = 14
        symbols: Optional[List[str]] = None

    @app.post("/api/models/test_all_single_models", dependencies=[Depends(verify_api_key)])
    def post_test_all_single_models(body: TestAllSingleModelsBody, background_tasks: BackgroundTasks):
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")

        symbols = body.symbols if body.symbols else list(state.active_symbols)
        symbols = [s.upper() for s in symbols]

        def _run_test_all_single():
            try:
                if tg_bot:
                    _send_telegram_sync(
                        f"🧪 Started testing ALL single models for {len(symbols)} active symbols..."
                    )
                state.add_notification("Started testing all single models", "info")

                for sym in symbols:
                    models = []
                    if hasattr(model_manager, "find_single_models_for_symbol"):
                        models = model_manager.find_single_models_for_symbol(sym)
                    else:
                        models = model_manager.find_models_for_symbol(sym)
                        models = [
                            p for p in models
                            if not (p.stem.lower().endswith("_mtf") or "_mtf_" in p.stem.lower())
                        ]

                    if not models:
                        logger.info(f"No models found for {sym} in ml_models")
                        continue

                    logger.info(f"Testing {len(models)} models for {sym}")
                    if tg_bot:
                        _send_telegram_sync(f"🧪 {sym}: testing {len(models)} models...")

                    ok_count = 0
                    for p in models:
                        try:
                            results = model_manager.test_model(str(p), sym, days=body.days)
                            if results:
                                model_manager.save_model_test_result(sym, str(p), results)
                                ok_count += 1
                        except Exception as e:
                            logger.error(f"Error testing model {p} for {sym}: {e}")

                    logger.info(f"Testing finished for {sym}: {ok_count}/{len(models)} models saved")
                    state.add_notification(
                        f"Testing finished for {sym}: {ok_count}/{len(models)} models",
                        "success",
                    )
                    if tg_bot:
                        _send_telegram_sync(
                            f"✅ {sym}: testing finished ({ok_count}/{len(models)} models)."
                        )

                state.add_notification("Testing all single models finished", "success")
                if tg_bot:
                    _send_telegram_sync("✅ Testing ALL single models finished.")
            except Exception as e:
                logger.error(f"Error in test_all_single_models: {e}")
                state.add_notification("Testing all single models failed", "error")
                if tg_bot:
                    _send_telegram_sync(f"❌ Testing ALL single models failed: {e}")

        background_tasks.add_task(_run_test_all_single)
        return {"ok": True, "message": "Test started in background", "symbols": symbols}

    class TestAllBody(BaseModel):
        symbol: str
        days: int = 14
        mode: str = "mtf"  # single or mtf

    @app.post("/api/models/test_all_combinations", dependencies=[Depends(verify_api_key)])
    def post_test_all_combinations(body: TestAllBody, background_tasks: BackgroundTasks):
        """Запускает тестирование всех моделей и выбор лучшей."""
        symbol = body.symbol.upper()
        if not model_manager:
             raise HTTPException(status_code=501, detail="Model manager not available")

        # Run in background
        def _run_test_all():
            try:
                # Loop removed, using sync telegram
                
                test_type = "Single Strategy" if body.mode == "single" else "MTF Strategy"
                if tg_bot:
                    _send_telegram_sync(f"🧪 Started {test_type} test for {symbol}...")
                state.add_notification(f"Started {test_type} test for {symbol}", "info")
                logger.info(f"Starting {test_type} test for {symbol}...")
                
                # 1. Запускаем compare_ml_models.py для бэктеста всех моделей
                script_path = PROJECT_ROOT / "compare_ml_models.py"
                cmd = [
                    sys.executable, 
                    str(script_path), 
                    "--symbol", symbol, 
                    "--days", str(body.days),
                    "--output", "csv"
                ]
                
                # Если Single mode, можно (опционально) фильтровать, но лучше протестировать всё
                # и выбрать лучшее из результатов.
                
                process = subprocess.run(
                    cmd, 
                    cwd=str(PROJECT_ROOT), 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if process.returncode != 0:
                    logger.error(f"Compare models failed: {process.stderr}")
                    raise Exception(f"Backtest script failed: {process.stderr[:200]}")
                
                # Обновляем результаты в model_manager из CSV
                try:
                    import pandas as pd
                    comp_files = sorted(list(PROJECT_ROOT.glob("ml_models_comparison_*.csv")), reverse=True)
                    if comp_files:
                        df = pd.read_csv(comp_files[0])
                        for _, row in df.iterrows():
                            if row['symbol'] == symbol:
                                m_path = row['model_path']
                                # Convert row to dict (cleaning nan values)
                                res = row.to_dict()
                                clean_res = {k: v for k, v in res.items() if pd.notna(v)}
                                model_manager.save_model_test_result(symbol, m_path, clean_res)
                        logger.info(f"Updated test results for {symbol} from {comp_files[0].name}")
                except Exception as e:
                    logger.error(f"Error updating model results from CSV: {e}")

                # 2. Выбираем лучшие модели на основе результатов
                from bot.ml.model_selector import select_best_models, select_best_single_model
                
                selected_info = ""
                
                if body.mode == "single":
                    # Выбираем лучшую Single модель
                    best_model_path, info = select_best_single_model(symbol, use_best_from_comparison=True)
                    
                    if best_model_path:
                        # Применяем модель
                        model_manager.apply_model(symbol, best_model_path)
                        # Обновляем конфиг
                        config = {"mode": "single", "model_path": best_model_path}
                        state.symbol_models[symbol] = best_model_path
                        state.set_strategy_config(symbol, config)
                        
                        pnl = info.get('pnl_pct', 0)
                        name = info.get('model_name', 'Unknown')
                        selected_info = f"Selected: {name} (PnL: {pnl:.2f}%)"
                    else:
                        selected_info = "No suitable single model found"
                        
                else:
                    # Выбираем лучшие MTF модели (как раньше)
                    # select_best_models сохраняет результат в internal state? 
                    # Нет, она возвращает пути. Нам нужно их применить.
                    m1h, m15m, info = select_best_models(symbol, use_best_from_comparison=True)
                    
                    if m1h and m15m:
                        config = {
                            "mode": "mtf", 
                            "model_1h_path": m1h, 
                            "model_15m_path": m15m
                        }
                        state.set_strategy_config(symbol, config)
                        # Нужно ли применять? MTF стратегия читает пути из конфига/state при запуске.
                        # Но лучше обновить активную стратегию если она запущена.
                        # В текущей архитектуре restart стратегии может потребоваться.
                        
                        selected_info = f"Selected MTF: 1h={info.get('model_1h')}, 15m={info.get('model_15m')}"
                    else:
                        selected_info = "No suitable MTF combination found"

                logger.info(f"{test_type} finished for {symbol}. {selected_info}")
                
                if tg_bot:
                    _send_telegram_sync(f"✅ {test_type} finished for {symbol}.\n{selected_info}")
                state.add_notification(f"{test_type} finished for {symbol}", "success")
                
            except Exception as e:
                logger.error(f"Error in {test_type} for {symbol}: {e}")
                if tg_bot:
                    _send_telegram_sync(f"❌ Testing failed for {symbol}: {e}")
                state.add_notification(f"Testing failed for {symbol}", "error")

        background_tasks.add_task(_run_test_all)
        
        return {"ok": True, "symbol": symbol, "message": "Test started in background"}

    class RetrainBody(BaseModel):
        symbol: str

    @app.post("/api/models/retrain", dependencies=[Depends(verify_api_key)])
    def post_retrain(body: RetrainBody, background_tasks: BackgroundTasks):
        """Запуск переобучения в фоне (subprocess)."""
        sym = body.symbol.upper()
        script = PROJECT_ROOT / "retrain_ml_optimized.py"
        if not script.exists():
            raise HTTPException(status_code=501, detail="retrain_ml_optimized.py not found")
            
        def _run_retrain_task():
             try:
                 if tg_bot:
                     _send_telegram_sync(f"🔄 Retraining started for {sym} via Mobile App...")
                 state.add_notification(f"Started retraining {sym}", "info")
                 
                 # Run synchronously (blocking thread)
                 result = subprocess.run(
                     [sys.executable, str(script), "--symbol", sym],
                     cwd=str(PROJECT_ROOT),
                     capture_output=True,
                     text=True,
                     encoding='utf-8',
                     errors='replace'
                 )
                 
                 if result.returncode == 0:
                     logger.info(f"Retrain finished for {sym}")
                     if tg_bot:
                         _send_telegram_sync(f"✅ Retraining finished for {sym} successfully.")
                     state.add_notification(f"Retraining finished for {sym}", "success")
                 else:
                     logger.error(f"Retrain failed for {sym}: {result.stderr}")
                     if tg_bot:
                         err_msg = result.stderr[:200] if result.stderr else "Unknown error"
                         _send_telegram_sync(f"❌ Retraining failed for {sym}.\nError: {err_msg}")
                     state.add_notification(f"Retraining failed for {sym}", "error")
             except Exception as e:
                 logger.error(f"Retrain exception: {e}")
                 if tg_bot:
                     _send_telegram_sync(f"❌ Retraining error for {sym}: {e}")
                 state.add_notification(f"Retraining error for {sym}", "error")

        background_tasks.add_task(_run_retrain_task)
        return {"ok": True, "symbol": sym, "message": "Retrain started in background"}

    # --- Optimization ---
    class OptimizeBody(BaseModel):
        symbols: Optional[List[str]] = None
        days: int = 30
        skip_training: bool = False
        skip_comparison: bool = False
        skip_mtf: bool = False

    @app.post("/api/optimization/run", dependencies=[Depends(verify_api_key)])
    def post_optimize(body: OptimizeBody, background_tasks: BackgroundTasks):
        """Запуск полной оптимизации стратегий."""
        script = PROJECT_ROOT / "auto_strategy_optimizer.py"
        if not script.exists():
            raise HTTPException(status_code=501, detail="auto_strategy_optimizer.py not found")
            
        def _run_optimizer_task():
             try:
                 syms = ",".join(body.symbols) if body.symbols else ""
                 if tg_bot:
                     _send_telegram_sync(f"🚀 Optimization started via Mobile App...")
                 state.add_notification(f"Started strategy optimization", "info")
                 
                 cmd = [sys.executable, str(script), "--days", str(body.days)]
                 if syms:
                     cmd.extend(["--symbols", syms])
                 if body.skip_training:
                     cmd.append("--skip-training")
                 if body.skip_comparison:
                     cmd.append("--skip-comparison")
                 if body.skip_mtf:
                     cmd.append("--skip-mtf-testing")
                 
                 # Run synchronously (blocking thread)
                 result = subprocess.run(
                     cmd,
                     cwd=str(PROJECT_ROOT),
                     capture_output=True,
                     text=True,
                     encoding='utf-8',
                     errors='replace'
                 )
                 
                 if result.returncode == 0:
                     logger.info(f"Optimization finished successfully")
                     if tg_bot:
                         _send_telegram_sync(f"✅ Strategy optimization completed successfully.")
                     state.add_notification(f"Optimization completed", "success")
                 else:
                     logger.error(f"Optimization failed: {result.stderr}")
                     if tg_bot:
                         err_msg = result.stderr[:200] if result.stderr else "Unknown error"
                         _send_telegram_sync(f"❌ Optimization failed.\nError: {err_msg}")
                     state.add_notification(f"Optimization failed", "error")
             except Exception as e:
                 logger.error(f"Optimization exception: {e}")
                 state.add_notification(f"Optimization error", "error")

        background_tasks.add_task(_run_optimizer_task)
        return {"ok": True, "message": "Optimization started in background"}

    @app.get("/api/optimization/latest", dependencies=[Depends(verify_api_key)])
    def get_latest_optimization():
        """Возвращает последний отчет об оптимизации."""
        opt_dir = PROJECT_ROOT / "optimization_results"
        if not opt_dir.exists():
             return {"found": False}
        
        files = sorted(opt_dir.glob("best_strategies_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
             return {"found": False}
             
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"found": True, "data": data, "filename": files[0].name, "timestamp": files[0].stat().st_mtime}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/optimization/chart", dependencies=[Depends(verify_api_key)])
    def get_optimization_chart():
        """Возвращает последний график сравнения."""
        from fastapi.responses import FileResponse
        opt_dir = PROJECT_ROOT / "optimization_results"
        if not opt_dir.exists():
             raise HTTPException(status_code=404, detail="No optimization results found")
        
        files = sorted(opt_dir.glob("comparison_chart_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
             raise HTTPException(status_code=404, detail="No chart found")
             
        return FileResponse(files[0])

    # --- History & stats ---
    @app.get("/api/stats", dependencies=[Depends(verify_api_key)])
    def get_stats():
        """Детальная статистика как в Telegram."""
        stats = state.get_stats()
        closed = [t for t in state.trades if t.status == "closed"]
        open_trades = [t for t in state.trades if t.status == "open"]
        wins = [t for t in closed if t.pnl_usd > 0]
        losses = [t for t in closed if t.pnl_usd < 0]
        out = {
            "total_pnl": stats["total_pnl"],
            "win_rate": stats["win_rate"],
            "total_trades": len(state.trades),
            "closed_count": len(closed),
            "open_count": len(open_trades),
            "wins_count": len(wins),
            "losses_count": len(losses),
            "avg_win": sum(t.pnl_usd for t in wins) / len(wins) if wins else None,
            "avg_loss": sum(t.pnl_usd for t in losses) / len(losses) if losses else None,
        }
        return out

    @app.get("/api/history/trades", dependencies=[Depends(verify_api_key)])
    def get_history_trades(limit: int = 50):
        closed = [t for t in state.trades if t.status == "closed"][-limit:]
        from dataclasses import asdict
        return {"trades": [asdict(t) for t in reversed(closed)]}

    @app.get("/api/history/signals", dependencies=[Depends(verify_api_key)])
    def get_history_signals(limit: int = 30):
        sigs = state.signals[-limit:]
        from dataclasses import asdict
        return {"signals": [asdict(s) for s in reversed(sigs)]}

    @app.get("/api/logs", dependencies=[Depends(verify_api_key)])
    def get_logs(log_type: str = "bot", lines: int = 100):
        """log_type: bot, trades, signals, errors, ai."""
        path_map = {"bot": "logs/bot.log", "trades": "logs/trades.log", "signals": "logs/signals.log", "errors": "logs/errors.log", "ai": "logs/ai_entry_audit.jsonl"}
        path = PROJECT_ROOT / path_map.get(log_type, path_map["bot"])
        if not path.exists():
            return {"lines": [], "path": str(path)}
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.readlines()
            last = raw[-lines:] if len(raw) > lines else raw
            if log_type == "ai":
                out = []
                for l in last:
                    s = l.strip()
                    if not s:
                        continue
                    try:
                        import json as _json
                        rec = _json.loads(s)
                        if isinstance(rec, dict):
                            event_type = rec.get("event_type")
                            ts = rec.get("timestamp_utc") or rec.get("response", {}).get("timestamp_utc")
                            symbol = rec.get("symbol")
                            side = rec.get("side")
                            decision_id = rec.get("decision_id") or rec.get("response", {}).get("decision_id")
                            if event_type == "confirm_entry":
                                resp = rec.get("response", {}) if isinstance(rec.get("response"), dict) else {}
                                decision = resp.get("decision")
                                mult = resp.get("size_multiplier", resp.get("mult"))
                                codes = resp.get("reason_codes")
                                if not isinstance(codes, list):
                                    codes = resp.get("risk_flags")
                                latency = resp.get("latency_ms", rec.get("latency_ms"))
                                out.append(
                                    f"{ts} AI confirm_entry {symbol} {side} decision={decision} mult={mult} codes={codes} latency_ms={latency} id={decision_id}"
                                )
                                continue
                            if event_type == "trade_outcome":
                                pnl_usd = rec.get("pnl_usd")
                                pnl_pct = rec.get("pnl_pct")
                                exit_reason = rec.get("exit_reason")
                                out.append(
                                    f"{ts} AI trade_outcome {symbol} {side} pnl_usd={pnl_usd} pnl_pct={pnl_pct} exit_reason={exit_reason} id={decision_id}"
                                )
                                continue
                            if event_type == "entry_blocked":
                                ai = rec.get("ai", {}) if isinstance(rec.get("ai"), dict) else {}
                                decision = ai.get("decision", "veto")
                                codes = rec.get("reason_codes", ai.get("reason_codes"))
                                notes = rec.get("notes", ai.get("notes"))
                                out.append(
                                    f"{ts} AI entry_blocked {symbol} {side} decision={decision} codes={codes} notes={notes} id={decision_id}"
                                )
                                continue
                    except Exception:
                        pass
                    out.append(s[:2000])
                return {"lines": out, "path": str(path)}
            return {"lines": [l.rstrip() for l in last], "path": str(path)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class LogsClearBody(BaseModel):
        log_type: str = "bot"

    def _truncate_rotating_handler_file(target_path: Path) -> bool:
        try:
            from logging.handlers import RotatingFileHandler
        except Exception:
            return False

        try:
            target_resolved = str(target_path.resolve())
        except Exception:
            target_resolved = str(target_path)

        candidates = [logging.getLogger(), logging.getLogger("main")]
        for lg in candidates:
            for h in getattr(lg, "handlers", []) or []:
                try:
                    if not isinstance(h, RotatingFileHandler):
                        continue
                    base = getattr(h, "baseFilename", None)
                    if not base:
                        continue
                    try:
                        base_resolved = str(Path(base).resolve())
                    except Exception:
                        base_resolved = str(base)
                    if base_resolved != target_resolved:
                        continue
                    h.acquire()
                    try:
                        if h.stream is None:
                            h.stream = h._open()
                        h.stream.seek(0)
                        h.stream.truncate(0)
                        h.stream.flush()
                    finally:
                        h.release()
                    return True
                except Exception:
                    continue
        return False

    def _clear_log_group(base_file: Path) -> Dict[str, Any]:
        cleared: list[str] = []
        failed: list[Dict[str, str]] = []

        base_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if base_file.exists():
                if not _truncate_rotating_handler_file(base_file):
                    try:
                        base_file.unlink(missing_ok=True)
                        cleared.append(base_file.name)
                    except Exception:
                        with open(base_file, "w", encoding="utf-8"):
                            pass
                        cleared.append(base_file.name)
                else:
                    cleared.append(base_file.name)
            else:
                with open(base_file, "w", encoding="utf-8"):
                    pass
                cleared.append(base_file.name)
        except Exception as e:
            failed.append({"file": base_file.name, "error": str(e)})

        for p in sorted(base_file.parent.glob(f"{base_file.name}.*")):
            if not p.is_file():
                continue
            try:
                p.unlink(missing_ok=True)
                cleared.append(p.name)
            except Exception as e:
                failed.append({"file": p.name, "error": str(e)})

        return {"cleared_files": cleared, "failed_files": failed}

    @app.post("/api/logs/clear", dependencies=[Depends(verify_api_key)])
    def post_clear_logs(body: LogsClearBody):
        log_type = (body.log_type or "").strip().lower()
        if log_type not in ("bot", "errors"):
            raise HTTPException(status_code=400, detail="Only bot and errors logs can be cleared")

        path_map = {"bot": "logs/bot.log", "errors": "logs/errors.log"}
        base = PROJECT_ROOT / path_map[log_type]
        result = _clear_log_group(base)
        return {
            "ok": len(result["failed_files"]) == 0,
            "log_type": log_type,
            **result,
        }

    # --- Analytics: equity curve for PnL visualization ---
    @app.get("/api/analytics/equity_curve", dependencies=[Depends(verify_api_key)])
    def get_equity_curve():
        """Кривая капитала по закрытым сделкам (для графика PnL)."""
        closed = [t for t in state.trades if t.status == "closed"]
        if not closed:
            return {"points": [], "total_pnl": 0}
        sorted_trades = sorted(closed, key=lambda t: t.exit_time or t.entry_time)
        points = []
        cum = 0.0
        for t in sorted_trades:
            cum += t.pnl_usd
            points.append({"time": t.exit_time or t.entry_time, "pnl_cum": round(cum, 2), "trade_pnl": round(t.pnl_usd, 2)})
        return {"points": points, "total_pnl": round(cum, 2)}

    @app.get("/api/analytics/daily_pnl", dependencies=[Depends(verify_api_key)])
    def get_daily_pnl(days: int = 7):
        """PnL по дням (бары) за последние N дней."""
        from datetime import datetime, timedelta
        
        now = datetime.now().date()
        start_date = now - timedelta(days=days-1)
        
        # Initialize map with 0.0 for all days
        daily_map = {}
        for i in range(days):
            d = start_date + timedelta(days=i)
            daily_map[d.isoformat()] = 0.0
            
        closed = [t for t in state.trades if t.status == "closed"]
        
        for t in closed:
            if not t.exit_time:
                continue
            try:
                # Handle ISO format with potential timezone or fractional seconds
                # Using simple split 'T' or assuming fromisoformat works
                dt = datetime.fromisoformat(t.exit_time).date()
                if dt >= start_date and dt <= now:
                    iso = dt.isoformat()
                    if iso in daily_map:
                        daily_map[iso] += t.pnl_usd
            except Exception:
                pass
                    
        # Convert to list sorted by date
        result = [{"date": k, "pnl": round(v, 2)} for k, v in sorted(daily_map.items())]
        return {"data": result}

    @app.get("/api/analytics/signals_dashboard", dependencies=[Depends(verify_api_key)])
    def get_signals_dashboard(
        timeframe: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pairs: Optional[str] = None,
        agg_method: str = "last",
        strong_threshold: float = 0.65,
        limit_pairs: int = 10,
        max_lines: int = 50000,
    ):
        tf = (timeframe or "15m").strip().lower()
        if tf not in _TIMEFRAME_TO_MINUTES:
            raise HTTPException(status_code=400, detail="timeframe must be '15m' or '1h'")

        agg = (agg_method or "last").strip().lower()
        if agg not in {"last", "mean"}:
            raise HTTPException(status_code=400, detail="agg_method must be 'last' or 'mean'")

        if strong_threshold < 0:
            raise HTTPException(status_code=400, detail="strong_threshold must be >= 0")
        if limit_pairs <= 0:
            raise HTTPException(status_code=400, detail="limit_pairs must be > 0")
        if max_lines <= 0:
            raise HTTPException(status_code=400, detail="max_lines must be > 0")

        try:
            start_dt = _parse_datetime_filter(start_date, is_end=False)
            end_dt = _parse_datetime_filter(end_date, is_end=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if start_dt and end_dt and start_dt > end_dt:
            raise HTTPException(status_code=400, detail="start_date must be <= end_date")

        pairs_filter: Optional[set[str]] = None
        if pairs:
            parsed_pairs = {
                token.strip().upper()
                for token in pairs.split(",")
                if token and token.strip()
            }
            if parsed_pairs:
                pairs_filter = parsed_pairs

        base_data = _parse_signals_for_dashboard(
            log_paths=[
                PROJECT_ROOT / "logs" / "signals.log",
                PROJECT_ROOT / "logs" / "bot.log",
            ],
            timeframe=tf,
            agg_method=agg,
            start_dt=start_dt,
            end_dt=end_dt,
            pairs_filter=None,
            strong_threshold=strong_threshold,
            max_lines=max_lines,
        )
        top_pairs = base_data["pairs"][:limit_pairs]
        if pairs_filter:
            selected_pairs = [p for p in top_pairs if p in pairs_filter]
            if not selected_pairs:
                selected_pairs = top_pairs
        else:
            selected_pairs = top_pairs
        selected_set = set(selected_pairs)
        filtered_points = [p for p in base_data["points"] if p["pair"] in selected_set]

        return {
            "timeframe": tf,
            "agg_method": agg,
            "strong_threshold": strong_threshold,
            "start_date": start_date,
            "end_date": end_date,
            "pairs_requested": sorted(list(pairs_filter)) if pairs_filter else None,
            "pairs_selected": selected_pairs,
            "top_pairs": top_pairs,
            "pair_activity": base_data["pair_activity"],
            "time_slots": sorted({p["time_slot"] for p in filtered_points}),
            "points": filtered_points,
            "meta": {
                "log_paths": [
                    str(PROJECT_ROOT / "logs" / "signals.log"),
                    str(PROJECT_ROOT / "logs" / "bot.log"),
                ],
                "total_raw_signals": base_data["total_raw_signals"],
                "total_points": len(filtered_points),
                "max_lines_used": max_lines,
            },
        }

    class ClosePositionBody(BaseModel):
        symbol: str

    @app.post("/api/positions/close", dependencies=[Depends(verify_api_key)])
    def post_close_position(body: ClosePositionBody):
        sym = body.symbol.upper()
        if not bybit_client:
            raise HTTPException(status_code=501, detail="Exchange client not available")
        
        try:
            # Get position info to know side and size
            pos_info = bybit_client.get_position_info(symbol=sym)
            if pos_info.get("retCode") != 0:
                raise HTTPException(status_code=400, detail="Failed to get position info")
            
            closed = False
            for p in pos_info.get("result", {}).get("list", []):
                size = _safe_float(p.get("size"), 0)
                if size <= 0:
                    continue
                side = p.get("side")
                close_side = "Sell" if side == "Buy" else "Buy"
                
                resp = bybit_client.place_order(
                    symbol=sym, side=close_side, qty=size, order_type="Market", reduce_only=True
                )
                
                if resp.get("retCode") == 0:
                    closed = True
                    if tg_bot:
                        _send_telegram_sync(f"⚠️ Position {sym} closed manually via Mobile App")
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to close: {resp.get('retMsg')}")
            
            if not closed:
                 raise HTTPException(status_code=400, detail="No open position found to close")
                 
            return {"ok": True, "symbol": sym, "message": "Position closed"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error closing position {sym}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/kline", dependencies=[Depends(verify_api_key)])
    def get_kline(symbol: str, interval: str = "15", limit: int = 100):
        """Возвращает свечные данные (OHLCV)."""
        symbol = symbol.upper()
        if not bybit_client:
            raise HTTPException(status_code=501, detail="Exchange client not available")
        
        # Map interval: '15m' -> '15', '1h' -> '60'
        mapped_interval = interval
        if interval == "15m": mapped_interval = "15"
        elif interval == "1h": mapped_interval = "60"
        elif interval == "4h": mapped_interval = "240"
        elif interval == "1d": mapped_interval = "D"
            
        try:
            # Bybit V5 API: category=linear for USDT perps
            # kline returns list of [startTime, open, high, low, close, volume, turnover]
            # startTime in ms
            resp = bybit_client.get_kline(symbol=symbol, interval=mapped_interval, limit=limit)
            
            if resp.get("retCode") != 0:
                raise HTTPException(status_code=400, detail=f"Bybit error: {resp.get('retMsg')}")
                
            result = resp.get("result", {})
            list_data = result.get("list", [])
            
            # Convert to more friendly format
            candles = []
            for item in list_data:
                # item: [startTime, open, high, low, close, volume, turnover]
                # We need to return them in chronological order (Bybit returns reverse chronological)
                candles.append({
                    "time": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                })
            
            # Sort by time ascending
            candles.sort(key=lambda x: x["time"])
            
            return {"symbol": symbol, "interval": interval, "candles": candles}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting kline for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/notifications/unread", dependencies=[Depends(verify_api_key)])
    def get_unread_notifications():
        """Возвращает список непрочитанных уведомлений."""
        return {"notifications": state.get_unread_notifications()}

    # --- Emergency ---
    @app.post("/api/emergency/stop_all", dependencies=[Depends(verify_api_key)])
    def post_emergency_stop_all():
        """Остановить бота и закрыть все позиции."""
        state.set_running(False)
        if tg_bot:
            _send_telegram_sync("🚨 EMERGENCY STOP triggered via Mobile App! All positions closed.")
        closed = []
        if bybit_client:
            for symbol in state.active_symbols:
                try:
                    pos_info = bybit_client.get_position_info(symbol=symbol)
                    if pos_info.get("retCode") != 0:
                        continue
                    for p in pos_info.get("result", {}).get("list", []):
                        size = _safe_float(p.get("size"), 0)
                        if size <= 0:
                            continue
                        side = p.get("side")
                        close_side = "Sell" if side == "Buy" else "Buy"
                        resp = bybit_client.place_order(
                            symbol=symbol, side=close_side, qty=size, order_type="Market", reduce_only=True
                        )
                        if resp.get("retCode") == 0:
                            closed.append(symbol)
                except Exception as e:
                    logger.error(f"Emergency close {symbol}: {e}")
        return {"ok": True, "closed_positions": closed, "message": "Bot stopped and positions closed"}

    class ChatBody(BaseModel):
        message: str

    class _ChatToolExecutor:
        def __init__(self):
            self.contract = self._load_contract()
            tools = self.contract.get("tools") if isinstance(self.contract, dict) else []
            if not isinstance(tools, list):
                tools = []
            self.tools_map: Dict[str, Dict[str, Any]] = {}
            for row in tools:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                self.tools_map[name] = row
            conf = self.contract.get("confirmation_policy") if isinstance(self.contract.get("confirmation_policy"), dict) else {}
            self.confirm_phrase = str(conf.get("explicit_confirm_phrase") or "ПОДТВЕРЖДАЮ").strip().upper()
            self.pending_ttl_seconds = self._resolve_pending_ttl_seconds()
            self.rate_limit_per_minute = self._resolve_rate_limit_per_minute()
            self.idempotency_ttl_seconds = self._resolve_idempotency_ttl_seconds()
            self._rate_hits: Dict[str, List[float]] = {}
            self._idempotency_cache: Dict[str, Dict[str, Any]] = {}
            self.audit_jsonl_path = str(PROJECT_ROOT / "logs" / "tool_execution_log.jsonl")
            self.audit_text_path = PROJECT_ROOT / "logs" / "tool_execution.log"

        def _load_contract(self) -> Dict[str, Any]:
            path = PROJECT_ROOT / "docs" / "ai_chat_tools.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to load ai_chat_tools.json: {e}")
            return {}

        def list_manifest(self) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for name, row in self.tools_map.items():
                out.append(
                    {
                        "name": name,
                        "risk_tier": row.get("risk_tier"),
                        "goal": row.get("goal"),
                        "input_schema": row.get("input_schema") if isinstance(row.get("input_schema"), dict) else {},
                    }
                )
            return out

        def _resolve_pending_ttl_seconds(self) -> int:
            execution = self.contract.get("global_policy", {}).get("execution", {}) if isinstance(self.contract, dict) else {}
            raw = execution.get("pending_confirmation_ttl_seconds")
            try:
                ttl = int(raw)
            except Exception:
                ttl = 300
            return max(30, min(3600, ttl))

        def _resolve_rate_limit_per_minute(self) -> int:
            execution = self.contract.get("global_policy", {}).get("execution", {}) if isinstance(self.contract, dict) else {}
            raw = execution.get("chat_tool_rate_limit_per_minute")
            try:
                value = int(raw)
            except Exception:
                value = 20
            return max(3, min(300, value))

        def _resolve_idempotency_ttl_seconds(self) -> int:
            execution = self.contract.get("global_policy", {}).get("execution", {}) if isinstance(self.contract, dict) else {}
            raw = execution.get("idempotency_ttl_seconds")
            try:
                value = int(raw)
            except Exception:
                value = 600
            return max(30, min(7200, value))

        def _prune_rate_state(self, now: float) -> None:
            border = now - 60.0
            keys_to_delete: List[str] = []
            for key, arr in self._rate_hits.items():
                arr[:] = [x for x in arr if x >= border]
                if not arr:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self._rate_hits.pop(key, None)

        def check_and_track_rate_limit(self, actor_key: str) -> Dict[str, Any]:
            key = str(actor_key or "anonymous").strip() or "anonymous"
            now = time.time()
            self._prune_rate_state(now)
            hits = self._rate_hits.get(key)
            if hits is None:
                hits = []
                self._rate_hits[key] = hits
            if len(hits) >= self.rate_limit_per_minute:
                retry_after = int(max(1, 60 - (now - hits[0])))
                return {
                    "allowed": False,
                    "retry_after_seconds": retry_after,
                    "remaining": 0,
                    "limit_per_minute": self.rate_limit_per_minute,
                }
            hits.append(now)
            remaining = max(0, self.rate_limit_per_minute - len(hits))
            return {
                "allowed": True,
                "retry_after_seconds": 0,
                "remaining": remaining,
                "limit_per_minute": self.rate_limit_per_minute,
            }

        def _prune_idempotency_cache(self, now: float) -> None:
            keys_to_delete: List[str] = []
            for key, row in self._idempotency_cache.items():
                ts = _safe_float(row.get("created_at_ts"), 0.0)
                if now - ts > self.idempotency_ttl_seconds:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self._idempotency_cache.pop(key, None)

        def _is_mutating_tool(self, tool: Dict[str, Any]) -> bool:
            method = str(tool.get("method") or "GET").strip().upper()
            return method in {"POST", "PUT", "PATCH", "DELETE"}

        def _extract_request_id(self, arguments: Dict[str, Any]) -> Optional[str]:
            if not isinstance(arguments, dict):
                return None
            for k in ("request_id", "idempotency_key"):
                v = arguments.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None

        def _normalized_arguments_for_idempotency(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(arguments, dict):
                return {}
            result: Dict[str, Any] = {}
            for k, v in arguments.items():
                if k in {"request_id", "idempotency_key"}:
                    continue
                result[k] = v
            return result

        def build_idempotency_key(self, tool_name: str, arguments: Dict[str, Any], risk_tier: str) -> Optional[str]:
            request_id = self._extract_request_id(arguments)
            normalized = self._normalized_arguments_for_idempotency(arguments)
            if request_id:
                source = f"req:{tool_name}:{request_id}"
            elif risk_tier in {"high", "critical"}:
                source = f"auto:{tool_name}:{json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)}"
            else:
                return None
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return f"{tool_name}:{digest}"

        def get_idempotent_result(self, idem_key: Optional[str]) -> Optional[Dict[str, Any]]:
            if not idem_key:
                return None
            now = time.time()
            self._prune_idempotency_cache(now)
            row = self._idempotency_cache.get(idem_key)
            if not isinstance(row, dict):
                return None
            execution = row.get("execution")
            if not isinstance(execution, dict):
                return None
            return execution

        def store_idempotent_result(self, idem_key: Optional[str], execution: Dict[str, Any]) -> None:
            if not idem_key or not isinstance(execution, dict):
                return
            self._prune_idempotency_cache(time.time())
            self._idempotency_cache[idem_key] = {
                "created_at_ts": time.time(),
                "execution": execution,
            }

        def get_limits_status(self, actor_key: str) -> Dict[str, Any]:
            key = str(actor_key or "anonymous").strip() or "anonymous"
            now = time.time()
            self._prune_rate_state(now)
            hits = self._rate_hits.get(key) or []
            remaining = max(0, self.rate_limit_per_minute - len(hits))
            self._prune_idempotency_cache(now)
            return {
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "rate_remaining": remaining,
                "idempotency_ttl_seconds": self.idempotency_ttl_seconds,
                "idempotency_cache_size": len(self._idempotency_cache),
            }

        def _validate_against_schema(self, value: Any, schema: Dict[str, Any], path: str = "input") -> List[str]:
            errors: List[str] = []
            stype = schema.get("type")
            if stype == "object":
                if not isinstance(value, dict):
                    return [f"{path} must be object"]
                required = schema.get("required") if isinstance(schema.get("required"), list) else []
                for key in required:
                    if key not in value:
                        errors.append(f"{path}.{key} is required")
                props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
                allow_additional = bool(schema.get("additionalProperties", True))
                if not allow_additional:
                    for key in value.keys():
                        if key not in props:
                            errors.append(f"{path}.{key} is not allowed")
                min_props = schema.get("minProperties")
                if isinstance(min_props, int) and len(value) < min_props:
                    errors.append(f"{path} must contain at least {min_props} properties")
                for key, sub_schema in props.items():
                    if key not in value:
                        continue
                    if isinstance(sub_schema, dict):
                        errors.extend(
                            self._validate_against_schema(value[key], sub_schema, f"{path}.{key}")
                        )
                return errors
            if stype == "string":
                if not isinstance(value, str):
                    return [f"{path} must be string"]
                min_len = schema.get("minLength")
                if isinstance(min_len, int) and len(value) < min_len:
                    errors.append(f"{path} length must be >= {min_len}")
                pattern = schema.get("pattern")
                if isinstance(pattern, str):
                    import re
                    if re.match(pattern, value) is None:
                        errors.append(f"{path} does not match pattern")
                enum = schema.get("enum")
                if isinstance(enum, list) and value not in enum:
                    errors.append(f"{path} must be one of {enum}")
                return errors
            if stype == "integer":
                if not isinstance(value, int):
                    return [f"{path} must be integer"]
                minimum = schema.get("minimum")
                maximum = schema.get("maximum")
                if isinstance(minimum, (int, float)) and value < minimum:
                    errors.append(f"{path} must be >= {minimum}")
                if isinstance(maximum, (int, float)) and value > maximum:
                    errors.append(f"{path} must be <= {maximum}")
                return errors
            if stype == "number":
                if not isinstance(value, (int, float)):
                    return [f"{path} must be number"]
                minimum = schema.get("minimum")
                maximum = schema.get("maximum")
                n = float(value)
                if isinstance(minimum, (int, float)) and n < float(minimum):
                    errors.append(f"{path} must be >= {minimum}")
                if isinstance(maximum, (int, float)) and n > float(maximum):
                    errors.append(f"{path} must be <= {maximum}")
                return errors
            if stype == "boolean":
                if not isinstance(value, bool):
                    return [f"{path} must be boolean"]
                return errors
            enum = schema.get("enum")
            if isinstance(enum, list) and value not in enum:
                errors.append(f"{path} must be one of {enum}")
            return errors

        def parse_manual_tool_command(self, message: str) -> Optional[Dict[str, Any]]:
            text = str(message or "").strip()
            if not text.lower().startswith("/tool "):
                return None
            payload = text[6:].strip()
            if not payload:
                return None
            tool_name = payload
            args: Dict[str, Any] = {}
            if " " in payload:
                tool_name, args_raw = payload.split(" ", 1)
                args_raw = args_raw.strip()
                if args_raw:
                    try:
                        parsed = json.loads(args_raw)
                        if isinstance(parsed, dict):
                            args = parsed
                    except Exception:
                        return None
            return {
                "intent": "tool_call",
                "tool_name": tool_name.strip(),
                "arguments": args,
                "goal": "manual_tool_command",
            }

        def is_confirmation_message(self, message: str) -> bool:
            text = str(message or "").strip().upper()
            return bool(text) and text == self.confirm_phrase

        def is_cancel_message(self, message: str) -> bool:
            text = str(message or "").strip().lower()
            return text in {"/cancel", "cancel", "отмена", "отменить", "отбой"}

        def get_pending_action(self) -> Optional[Dict[str, Any]]:
            pending = ai_agent.pending_chat_action
            if not isinstance(pending, dict):
                return None
            return pending

        def clear_pending_action(self) -> None:
            ai_agent.pending_chat_action = None

        def expire_pending_if_needed(self) -> Optional[Dict[str, Any]]:
            pending = self.get_pending_action()
            if not pending:
                return None
            expires_at = pending.get("expires_at_ts")
            try:
                expires = float(expires_at)
            except Exception:
                expires = 0.0
            now = time.time()
            if expires > 0 and now > expires:
                self.log_tool_event(
                    event_type="pending_expired",
                    tool_name=str(pending.get("tool_name") or ""),
                    goal=str(pending.get("goal") or ""),
                    risk_tier=str(pending.get("risk_tier") or ""),
                    ok=False,
                    summary=f"Pending action expired after {self.pending_ttl_seconds}s",
                    arguments=pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {},
                )
                self.clear_pending_action()
                return pending
            return None

        def pending_summary_text(self, pending: Dict[str, Any]) -> str:
            now = time.time()
            expires_at = pending.get("expires_at_ts")
            try:
                ttl_left = int(max(0, float(expires_at) - now))
            except Exception:
                ttl_left = self.pending_ttl_seconds
            return (
                f"Ожидается подтверждение действия {pending.get('tool_name')} "
                f"(уровень {pending.get('risk_tier')}, осталось {ttl_left}с). "
                f"Для выполнения: {self.confirm_phrase}. Для отмены: /cancel"
            )

        def _sanitize_result(self, payload: Any) -> Any:
            text = json.dumps(payload, ensure_ascii=False, default=str) if not isinstance(payload, str) else payload
            if len(text) > 4000:
                text = text[:4000] + "...(truncated)"
            return text

        async def execute(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
            goal: str = "",
            confirmed: bool = False,
            actor_key: str = "anonymous",
        ) -> Dict[str, Any]:
            tool = self.tools_map.get(str(tool_name or "").strip())
            if not tool:
                return {"ok": False, "error": f"Tool not allowed: {tool_name}", "tool_name": tool_name}
            if not isinstance(arguments, dict):
                arguments = {}
            rate = self.check_and_track_rate_limit(actor_key)
            if not rate.get("allowed"):
                self.log_tool_event(
                    event_type="tool_rate_limited",
                    tool_name=tool_name,
                    goal=goal,
                    risk_tier="n/a",
                    ok=False,
                    summary=f"retry_after={rate.get('retry_after_seconds')}",
                    arguments=arguments,
                )
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error": "Rate limit exceeded for chat tools",
                    "rate_limited": True,
                    "retry_after_seconds": rate.get("retry_after_seconds"),
                    "rate_limit_per_minute": rate.get("limit_per_minute"),
                }
            schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object"}
            validation_arguments = arguments if isinstance(arguments, dict) else {}
            if isinstance(validation_arguments, dict):
                validation_arguments = dict(validation_arguments)
                validation_arguments.pop("request_id", None)
                validation_arguments.pop("idempotency_key", None)
            validation_errors = self._validate_against_schema(validation_arguments, schema, "arguments")
            if validation_errors:
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error": "Invalid tool arguments",
                    "validation_errors": validation_errors,
                }
            risk_tier = str(tool.get("risk_tier") or "low").lower()
            requires_confirmation = risk_tier in {"high", "critical"}
            idem_key = self.build_idempotency_key(tool_name, arguments, risk_tier)
            if confirmed and self._is_mutating_tool(tool):
                cached = self.get_idempotent_result(idem_key)
                if cached:
                    out = dict(cached)
                    out["idempotent_reused"] = True
                    self.log_tool_event(
                        event_type="tool_idempotent_reused",
                        tool_name=tool_name,
                        goal=goal,
                        risk_tier=risk_tier,
                        ok=bool(out.get("ok")),
                        summary="Reused cached execution result",
                        arguments=arguments,
                    )
                    return out
            if requires_confirmation and not confirmed:
                created_at = time.time()
                ai_agent.pending_chat_action = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "goal": goal,
                    "risk_tier": risk_tier,
                    "idempotency_key": idem_key,
                    "created_at_ts": created_at,
                    "expires_at_ts": created_at + self.pending_ttl_seconds,
                }
                preview = self._preview_impact(tool_name, arguments)
                self.log_tool_event(
                    event_type="pending_created",
                    tool_name=tool_name,
                    goal=goal,
                    risk_tier=risk_tier,
                    ok=False,
                    summary=preview,
                    arguments=arguments,
                )
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "risk_tier": risk_tier,
                    "requires_confirmation": True,
                    "confirmation_phrase": self.confirm_phrase,
                    "preview": preview,
                    "ttl_seconds": self.pending_ttl_seconds,
                }
            result = await self._run_tool(tool_name, arguments)
            ai_agent.pending_chat_action = None
            result["tool_name"] = tool_name
            result["risk_tier"] = risk_tier
            if self._is_mutating_tool(tool) and result.get("ok"):
                self.store_idempotent_result(idem_key, result)
            self.log_tool_event(
                event_type="tool_executed",
                tool_name=tool_name,
                goal=goal,
                risk_tier=risk_tier,
                ok=bool(result.get("ok")),
                summary=self._sanitize_result(result.get("result") if result.get("ok") else result.get("error")),
                arguments=arguments,
            )
            return result

        def log_tool_event(
            self,
            event_type: str,
            tool_name: str,
            goal: str,
            risk_tier: str,
            ok: bool,
            summary: Any,
            arguments: Dict[str, Any],
        ) -> None:
            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            summary_str = str(summary or "")
            if len(summary_str) > 1200:
                summary_str = summary_str[:1200] + "...(truncated)"
            record = {
                "event_type": event_type,
                "timestamp_utc": timestamp_utc,
                "tool_name": tool_name,
                "goal": goal,
                "risk_tier": risk_tier,
                "ok": ok,
                "summary": summary_str,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
            append_jsonl(self.audit_jsonl_path, record)
            try:
                self.audit_text_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.audit_text_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{timestamp_utc} | {event_type.upper()} | tool={tool_name} | risk={risk_tier} | ok={ok} | goal={goal} | {summary_str}\n"
                    )
            except Exception:
                pass

        def _preview_impact(self, tool_name: str, arguments: Dict[str, Any]) -> str:
            if tool_name == "update_risk_settings":
                updates = arguments.get("updates") if isinstance(arguments.get("updates"), dict) else {}
                before = _risk_to_dict(settings.risk)
                changed = []
                for k, v in updates.items():
                    old_v = before.get(k)
                    if old_v != v:
                        changed.append(f"{k}: {old_v} -> {v}")
                if not changed:
                    return "Изменений не обнаружено."
                return "; ".join(changed)
            if tool_name == "update_ml_settings":
                updates = arguments.get("updates") if isinstance(arguments.get("updates"), dict) else {}
                before = _ml_to_dict(settings.ml_strategy)
                changed = []
                for k, v in updates.items():
                    old_v = before.get(k)
                    if old_v != v:
                        changed.append(f"{k}: {old_v} -> {v}")
                if not changed:
                    return "Изменений не обнаружено."
                return "; ".join(changed)
            if tool_name == "apply_research_experiment":
                return f"Будет применён эксперимент {arguments.get('experiment_id')} в рабочую стратегию."
            if tool_name == "stop_bot":
                return "Будет остановлена торговля бота."
            if tool_name == "emergency_stop_all":
                return "Бот будет остановлен, открытые позиции будут закрыты."
            if tool_name == "apply_best_model_for_symbol":
                return f"Будет применена лучшая MTF комбинация моделей для {arguments.get('symbol')}."
            if tool_name == "apply_model_for_symbol":
                return f"Будет применена модель для {arguments.get('symbol')} (mode={arguments.get('mode')})."
            if tool_name == "retrain_models":
                return f"Будет запущено переобучение моделей для {arguments.get('symbol')}."
            if tool_name == "run_strategy_optimization":
                return "Будет запущена оптимизация стратегий."
            return f"Будет выполнено действие {tool_name}."

        async def _run_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if tool_name == "get_bot_status":
                    return {"ok": True, "result": _get_status_data(state, bybit_client, settings, trading_loop)}
                if tool_name == "get_bot_stats":
                    return {"ok": True, "result": get_stats()}
                if tool_name == "get_trade_history":
                    limit = int(arguments.get("limit") or 50)
                    limit = max(1, min(200, limit))
                    return {"ok": True, "result": get_history_trades(limit=limit)}
                if tool_name == "list_api_routes":
                    routes: List[Dict[str, Any]] = []
                    for r in getattr(app, "routes", []) or []:
                        path = getattr(r, "path", None)
                        methods = getattr(r, "methods", None)
                        name = getattr(r, "name", None)
                        if not path:
                            continue
                        m = []
                        if isinstance(methods, (set, list, tuple)):
                            m = sorted([str(x) for x in methods if x])
                        routes.append({"path": str(path), "methods": m, "name": str(name) if name else None})
                    routes = sorted(routes, key=lambda x: (x.get("path") or ""))
                    return {"ok": True, "result": {"routes": routes, "count": len(routes)}}
                if tool_name == "get_settings_snapshot":
                    return {"ok": True, "result": get_settings()}
                if tool_name == "get_risk_snapshot":
                    return {"ok": True, "result": get_risk()}
                if tool_name == "get_ml_settings":
                    return {"ok": True, "result": get_ml()}
                if tool_name == "get_ai_research_status":
                    return {"ok": True, "result": get_research_status()}
                if tool_name == "get_experiment_health":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    return {"ok": True, "result": get_research_experiment_health(experiment_id)}
                if tool_name == "get_experiment_report":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    return {"ok": True, "result": get_experiment_report(experiment_id)}
                if tool_name == "start_research_experiment":
                    symbol = str(arguments.get("symbol") or "").upper()
                    experiment_type = str(arguments.get("type") or "balanced")
                    metadata = arguments.get("metadata") if isinstance(arguments.get("metadata"), dict) else None
                    allow_duplicate = bool(arguments.get("allow_duplicate", False))
                    safe_mode = bool(arguments.get("safe_mode", True))
                    result = ai_agent.start_research_experiment(
                        symbol=symbol,
                        experiment_type=experiment_type,
                        metadata=metadata,
                        allow_duplicate=allow_duplicate,
                        safe_mode=safe_mode,
                    )
                    if not result.get("ok"):
                        return {"ok": False, "error": result.get("error", "Failed to start research"), "result": result}
                    return {"ok": True, "result": result}
                if tool_name == "control_research_campaign":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    action = str(arguments.get("action") or "")
                    return {
                        "ok": True,
                        "result": control_research_campaign(
                            experiment_id,
                            CampaignControlBody(action=action),
                        ),
                    }
                if tool_name == "start_paper_trading":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    if not paper_trading_manager:
                        return {"ok": False, "error": "Paper trading manager not available"}
                    session = paper_trading_manager.start_session(experiment_id)
                    if not session:
                        return {"ok": False, "error": f"Failed to start paper trading for {experiment_id}"}
                    return {"ok": True, "result": {"experiment_id": experiment_id, "symbol": session.symbol}}
                if tool_name == "stop_paper_trading":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    if not paper_trading_manager:
                        return {"ok": False, "error": "Paper trading manager not available"}
                    success = paper_trading_manager.stop_session(experiment_id)
                    if not success:
                        return {"ok": False, "error": f"Paper trading session not found for {experiment_id}"}
                    return {"ok": True, "result": {"experiment_id": experiment_id, "stopped": True}}
                if tool_name == "get_paper_metrics":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    return {"ok": True, "result": get_paper_metrics(experiment_id=experiment_id)}
                if tool_name == "get_paper_realtime_chart":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    return {"ok": True, "result": await get_paper_realtime_chart(experiment_id)}
                if tool_name == "update_risk_settings":
                    updates = arguments.get("updates") if isinstance(arguments.get("updates"), dict) else {}
                    return {"ok": True, "result": put_risk(updates)}
                if tool_name == "update_ml_settings":
                    updates = arguments.get("updates") if isinstance(arguments.get("updates"), dict) else {}
                    return {"ok": True, "result": put_ml(updates)}
                if tool_name == "get_logs_tail":
                    log_type = str(arguments.get("log_type") or "bot").strip().lower() or "bot"
                    lines = int(arguments.get("lines") or 200)
                    lines = max(1, min(500, lines))
                    return {"ok": True, "result": get_logs(log_type=log_type, lines=lines)}
                if tool_name == "get_tool_execution_log":
                    n = max(1, min(500, int(arguments.get("limit") or 50)))
                    path = PROJECT_ROOT / "logs" / "tool_execution_log.jsonl"
                    if not path.exists():
                        return {"ok": True, "result": {"events": [], "path": str(path), "limit": n}}
                    rows: List[Dict[str, Any]] = []
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        et = str(arguments.get("event_type") or "").strip().lower()
                        tn = str(arguments.get("tool_name") or "").strip().lower()
                        rt = str(arguments.get("risk_tier") or "").strip().lower()
                        ok_filter = arguments.get("ok")
                        ok_present = isinstance(ok_filter, bool)
                        for line in reversed(lines):
                            s = line.strip()
                            if not s:
                                continue
                            try:
                                rec = json.loads(s)
                            except Exception:
                                continue
                            if not isinstance(rec, dict):
                                continue
                            if et and str(rec.get("event_type") or "").strip().lower() != et:
                                continue
                            if tn and str(rec.get("tool_name") or "").strip().lower() != tn:
                                continue
                            if rt and str(rec.get("risk_tier") or "").strip().lower() != rt:
                                continue
                            if ok_present and bool(rec.get("ok")) != bool(ok_filter):
                                continue
                            rows.append(rec)
                            if len(rows) >= n:
                                break
                    except Exception as e:
                        return {"ok": False, "error": f"Failed to read tool execution log: {e}"}
                    return {
                        "ok": True,
                        "result": {
                            "events": rows,
                            "path": str(path),
                            "limit": n,
                            "filters": {
                                "event_type": arguments.get("event_type"),
                                "tool_name": arguments.get("tool_name"),
                                "risk_tier": arguments.get("risk_tier"),
                                "ok": arguments.get("ok"),
                            },
                        },
                    }
                if tool_name == "list_docs":
                    docs_dir = PROJECT_ROOT / "docs"
                    if not docs_dir.exists():
                        return {"ok": True, "result": {"files": [], "path": str(docs_dir)}}
                    files = []
                    for p in docs_dir.iterdir():
                        if p.is_file():
                            files.append(p.name)
                    files = sorted(files)
                    return {"ok": True, "result": {"files": files, "path": str(docs_dir)}}
                if tool_name == "read_doc":
                    filename = str(arguments.get("filename") or "").strip()
                    docs_dir = PROJECT_ROOT / "docs"
                    base = docs_dir.resolve()
                    target = (docs_dir / filename).resolve()
                    if base != target and base not in target.parents:
                        return {"ok": False, "error": "Invalid path"}
                    if not target.exists() or not target.is_file():
                        return {"ok": False, "error": f"File not found: {filename}"}
                    lines_out: List[str] = []
                    truncated = False
                    try:
                        with open(target, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f):
                                if i >= 200:
                                    truncated = True
                                    break
                                lines_out.append(line.rstrip("\n"))
                    except Exception as e:
                        return {"ok": False, "error": str(e)}
                    return {
                        "ok": True,
                        "result": {
                            "filename": filename,
                            "path": str(target),
                            "lines": lines_out,
                            "truncated": truncated,
                        },
                    }
                if tool_name == "list_log_files":
                    logs_dir = PROJECT_ROOT / "logs"
                    if not logs_dir.exists():
                        return {"ok": True, "result": {"files": [], "path": str(logs_dir)}}
                    files = []
                    for p in logs_dir.iterdir():
                        if p.is_file():
                            files.append(p.name)
                    files = sorted(files)
                    return {"ok": True, "result": {"files": files, "path": str(logs_dir)}}
                if tool_name == "read_log_tail":
                    filename = str(arguments.get("filename") or "").strip()
                    logs_dir = PROJECT_ROOT / "logs"
                    base = logs_dir.resolve()
                    target = (logs_dir / filename).resolve()
                    if base != target and base not in target.parents:
                        return {"ok": False, "error": "Invalid path"}
                    if not target.exists() or not target.is_file():
                        return {"ok": False, "error": f"File not found: {filename}"}
                    n = int(arguments.get("lines") or 200)
                    n = max(1, min(1000, n))
                    try:
                        with open(target, "r", encoding="utf-8", errors="ignore") as f:
                            raw = f.readlines()
                        last = raw[-n:] if len(raw) > n else raw
                        return {
                            "ok": True,
                            "result": {
                                "filename": filename,
                                "path": str(target),
                                "lines": [l.rstrip("\n") for l in last],
                            },
                        }
                    except Exception as e:
                        return {"ok": False, "error": str(e)}
                if tool_name == "get_models_overview":
                    return {"ok": True, "result": get_models_list()}
                if tool_name == "get_models_for_symbol":
                    sym = str(arguments.get("symbol") or "").strip().upper()
                    return {"ok": True, "result": get_models_for_symbol(sym)}
                if tool_name == "apply_best_model_for_symbol":
                    sym = str(arguments.get("symbol") or "").strip().upper()
                    return {"ok": True, "result": post_apply_best_mtf(sym)}
                if tool_name == "apply_model_for_symbol":
                    sym = str(arguments.get("symbol") or "").strip().upper()
                    mode = str(arguments.get("mode") or "single").strip().lower()
                    model_name = str(arguments.get("model_name") or "").strip()
                    model_1h_name = str(arguments.get("model_1h_name") or "").strip()
                    model_15m_name = str(arguments.get("model_15m_name") or "").strip()
                    inventory = get_models_for_symbol(sym)
                    models = inventory.get("models") if isinstance(inventory, dict) else []
                    if not isinstance(models, list):
                        models = []
                    def _find_path_by_name(name: str) -> Optional[str]:
                        if not name:
                            return None
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("name") or "") == name:
                                p = row.get("path")
                                return str(p) if isinstance(p, str) and p.strip() else None
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("name") or "").lower() == name.lower():
                                p = row.get("path")
                                return str(p) if isinstance(p, str) and p.strip() else None
                        return None
                    if mode == "mtf":
                        p1h = _find_path_by_name(model_1h_name)
                        p15m = _find_path_by_name(model_15m_name)
                        if not p1h or not p15m:
                            return {"ok": False, "error": "MTF requires model_1h_name and model_15m_name from get_models_for_symbol"}
                        body = ApplyModelBody(mode="mtf", model_1h_path=p1h, model_15m_path=p15m)
                        return {"ok": True, "result": post_apply_model(sym, body)}
                    p = _find_path_by_name(model_name)
                    if not p:
                        return {"ok": False, "error": "Single mode requires model_name from get_models_for_symbol"}
                    body = ApplyModelBody(mode="single", model_path=p)
                    return {"ok": True, "result": post_apply_model(sym, body)}
                if tool_name == "cleanup_old_inactive_models":
                    return {"ok": True, "result": post_cleanup_old_inactive_models(ModelsCleanupBody())}
                if tool_name == "test_model_background":
                    import threading
                    sym = str(arguments.get("symbol") or "").strip().upper()
                    mode = str(arguments.get("mode") or "single").strip().lower()
                    model_name = str(arguments.get("model_name") or "").strip()
                    model_1h_name = str(arguments.get("model_1h_name") or "").strip()
                    model_15m_name = str(arguments.get("model_15m_name") or "").strip()
                    days = 14
                    if not model_manager:
                        return {"ok": False, "error": "Model manager not available"}
                    inventory = get_models_for_symbol(sym)
                    models = inventory.get("models") if isinstance(inventory, dict) else []
                    if not isinstance(models, list):
                        models = []
                    def _find_path_by_name(name: str) -> Optional[str]:
                        if not name:
                            return None
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("name") or "") == name:
                                p = row.get("path")
                                return str(p) if isinstance(p, str) and p.strip() else None
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("name") or "").lower() == name.lower():
                                p = row.get("path")
                                return str(p) if isinstance(p, str) and p.strip() else None
                        return None
                    paths: List[str] = []
                    if mode == "mtf":
                        p1h = _find_path_by_name(model_1h_name)
                        p15m = _find_path_by_name(model_15m_name)
                        if not p1h or not p15m:
                            return {"ok": False, "error": "MTF requires model_1h_name and model_15m_name from get_models_for_symbol"}
                        paths = [p1h, p15m]
                    else:
                        p = _find_path_by_name(model_name) if model_name else state.symbol_models.get(sym)
                        if not p:
                            return {"ok": False, "error": "Single mode requires model_name or active model for symbol"}
                        paths = [str(p)]
                    def _run_test():
                        try:
                            if tg_bot:
                                _send_telegram_sync(f"🧪 Started testing model(s) for {sym}...")
                            state.add_notification(f"Started testing {sym}", "info")
                            for mp in paths:
                                results = model_manager.test_model(mp, sym, days=days)
                                if results:
                                    model_manager.save_model_test_result(sym, mp, results)
                            state.add_notification(f"Testing finished for {sym}", "success")
                            if tg_bot:
                                _send_telegram_sync(f"✅ Testing finished for {sym}")
                        except Exception as e:
                            logger.error(f"Error in background test for {sym}: {e}")
                            if tg_bot:
                                _send_telegram_sync(f"❌ Testing failed for {sym}: {e}")
                            state.add_notification(f"Testing failed for {sym}", "error")
                    threading.Thread(target=_run_test, daemon=True).start()
                    return {"ok": True, "result": {"ok": True, "symbol": sym, "models": paths, "message": "Test started in background"}}
                if tool_name == "retrain_models":
                    import threading
                    sym = str(arguments.get("symbol") or "").strip().upper()
                    script = PROJECT_ROOT / "retrain_ml_optimized.py"
                    if not script.exists():
                        return {"ok": False, "error": "retrain_ml_optimized.py not found"}
                    def _run_retrain_task():
                        try:
                            if tg_bot:
                                _send_telegram_sync(f"🔄 Retraining started for {sym} via Chat Agent...")
                            state.add_notification(f"Started retraining {sym}", "info")
                            result = subprocess.run(
                                [sys.executable, str(script), "--symbol", sym],
                                cwd=str(PROJECT_ROOT),
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                            )
                            if result.returncode == 0:
                                logger.info(f"Retrain finished for {sym}")
                                if tg_bot:
                                    _send_telegram_sync(f"✅ Retraining finished for {sym} successfully.")
                                state.add_notification(f"Retraining finished for {sym}", "success")
                            else:
                                logger.error(f"Retrain failed for {sym}: {result.stderr}")
                                if tg_bot:
                                    err_msg = result.stderr[:200] if result.stderr else "Unknown error"
                                    _send_telegram_sync(f"❌ Retraining failed for {sym}.\nError: {err_msg}")
                                state.add_notification(f"Retraining failed for {sym}", "error")
                        except Exception as e:
                            logger.error(f"Retrain exception: {e}")
                            if tg_bot:
                                _send_telegram_sync(f"❌ Retraining error for {sym}: {e}")
                            state.add_notification(f"Retraining error for {sym}", "error")
                    threading.Thread(target=_run_retrain_task, daemon=True).start()
                    return {"ok": True, "result": {"ok": True, "symbol": sym, "message": "Retrain started in background"}}
                if tool_name == "run_strategy_optimization":
                    import threading
                    script = PROJECT_ROOT / "auto_strategy_optimizer.py"
                    if not script.exists():
                        return {"ok": False, "error": "auto_strategy_optimizer.py not found"}
                    symbols_csv = str(arguments.get("symbols_csv") or "").strip()
                    days = int(arguments.get("days") or 30)
                    days = max(1, min(3650, days))
                    skip_training = bool(arguments.get("skip_training", False))
                    skip_comparison = bool(arguments.get("skip_comparison", False))
                    skip_mtf = bool(arguments.get("skip_mtf", False))
                    def _run_optimizer_task():
                        try:
                            if tg_bot:
                                _send_telegram_sync("🚀 Optimization started via Chat Agent...")
                            state.add_notification("Started strategy optimization", "info")
                            cmd = [sys.executable, str(script), "--days", str(days)]
                            if symbols_csv:
                                cmd.extend(["--symbols", symbols_csv])
                            if skip_training:
                                cmd.append("--skip-training")
                            if skip_comparison:
                                cmd.append("--skip-comparison")
                            if skip_mtf:
                                cmd.append("--skip-mtf-testing")
                            result = subprocess.run(
                                cmd,
                                cwd=str(PROJECT_ROOT),
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                            )
                            if result.returncode == 0:
                                logger.info("Optimization finished successfully")
                                if tg_bot:
                                    _send_telegram_sync("✅ Strategy optimization completed successfully.")
                                state.add_notification("Optimization completed", "success")
                            else:
                                logger.error(f"Optimization failed: {result.stderr}")
                                if tg_bot:
                                    err_msg = result.stderr[:200] if result.stderr else "Unknown error"
                                    _send_telegram_sync(f"❌ Optimization failed.\nError: {err_msg}")
                                state.add_notification("Optimization failed", "error")
                        except Exception as e:
                            logger.error(f"Optimization exception: {e}")
                            state.add_notification("Optimization error", "error")
                    threading.Thread(target=_run_optimizer_task, daemon=True).start()
                    return {"ok": True, "result": {"ok": True, "message": "Optimization started in background"}}
                if tool_name == "get_latest_optimization_results":
                    return {"ok": True, "result": get_latest_optimization()}
                if tool_name == "get_latest_optimization_chart_info":
                    opt_dir = PROJECT_ROOT / "optimization_results"
                    if not opt_dir.exists():
                        return {"ok": True, "result": {"found": False}}
                    files = sorted(opt_dir.glob("comparison_chart_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if not files:
                        return {"ok": True, "result": {"found": False}}
                    p = files[0]
                    return {
                        "ok": True,
                        "result": {
                            "found": True,
                            "filename": p.name,
                            "path": str(p),
                            "timestamp": p.stat().st_mtime,
                        },
                    }
                if tool_name == "apply_research_experiment":
                    experiment_id = str(arguments.get("experiment_id") or "")
                    return {"ok": True, "result": await post_apply_experiment(ApplyExperimentBody(experiment_id=experiment_id))}
                if tool_name == "stop_bot":
                    return {"ok": True, "result": await post_stop()}
                if tool_name == "emergency_stop_all":
                    return {"ok": True, "result": post_emergency_stop_all()}
                return {"ok": False, "error": f"Tool handler not implemented: {tool_name}"}
            except HTTPException as e:
                return {"ok": False, "error": f"HTTP {e.status_code}: {e.detail}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        def render_response_text(self, execution: Dict[str, Any]) -> str:
            if execution.get("requires_confirmation"):
                return (
                    f"Это действие уровня {execution.get('risk_tier')}. "
                    f"Предпросмотр: {execution.get('preview')}\n"
                    f"TTL: {execution.get('ttl_seconds')}с. "
                    f"Для выполнения напишите: {execution.get('confirmation_phrase')}. "
                    f"Для отмены: /cancel"
                )
            if execution.get("rate_limited"):
                return (
                    "Превышен лимит вызовов инструментов.\n"
                    f"Повторите через {execution.get('retry_after_seconds')}с "
                    f"(лимит {execution.get('rate_limit_per_minute')} в минуту)."
                )
            if execution.get("ok"):
                payload = execution.get("result")
                if execution.get("idempotent_reused"):
                    return (
                        f"Повторный запрос для {execution.get('tool_name')} обнаружен. "
                        "Возвращён кэшированный результат.\n"
                        f"{self._sanitize_result(payload)}"
                    )
                return f"Инструмент {execution.get('tool_name')} выполнен успешно.\n{self._sanitize_result(payload)}"
            return f"Не удалось выполнить {execution.get('tool_name')}: {execution.get('error')}"

    chat_tool_executor = _ChatToolExecutor()

    @app.get("/api/ai/chat/history", dependencies=[Depends(verify_api_key)])
    async def get_chat_history(limit: int = 50):
        """Возвращает историю чата."""
        try:
            logger.info(f"Fetching chat history (limit={limit})...")
            history = await ai_agent._get_chat_history(limit)
            logger.info(f"Chat history fetched: {len(history)} messages")
            return {"history": history}
        except Exception as e:
            logger.error(f"Chat history error: {e}", exc_info=True)
            return {"history": []}

    @app.delete("/api/ai/chat/message/{message_id}", dependencies=[Depends(verify_api_key)])
    async def delete_chat_message(message_id: str):
        try:
            result = await ai_agent._delete_chat_message(message_id)
            if result.get("ok"):
                return result
            raise HTTPException(status_code=404, detail=result.get("detail") or "Message not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete chat message error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/ai/chat/tools", dependencies=[Depends(verify_api_key)])
    def get_chat_tools(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        pending = chat_tool_executor.get_pending_action()
        actor_key = str(x_api_key or "anonymous").strip() or "anonymous"
        return {
            "contract_version": chat_tool_executor.contract.get("contract_version"),
            "tools": chat_tool_executor.list_manifest(),
            "pending_confirmation_ttl_seconds": chat_tool_executor.pending_ttl_seconds,
            "confirmation_phrase": chat_tool_executor.confirm_phrase,
            "pending_action": pending,
            "limits": chat_tool_executor.get_limits_status(actor_key),
        }

    @app.get("/api/ai/chat/limits", dependencies=[Depends(verify_api_key)])
    def get_chat_limits(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        actor_key = str(x_api_key or "anonymous").strip() or "anonymous"
        return chat_tool_executor.get_limits_status(actor_key)

    @app.get("/api/ai/chat/tool_execution_log", dependencies=[Depends(verify_api_key)])
    def get_chat_tool_execution_log(
        limit: int = 50,
        event_type: Optional[str] = None,
        tool_name: Optional[str] = None,
        risk_tier: Optional[str] = None,
        ok: Optional[bool] = None,
    ):
        n = max(1, min(500, int(limit or 50)))
        path = PROJECT_ROOT / "logs" / "tool_execution_log.jsonl"
        if not path.exists():
            return {"events": [], "path": str(path), "limit": n}
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            et = str(event_type or "").strip().lower()
            tn = str(tool_name or "").strip().lower()
            rt = str(risk_tier or "").strip().lower()
            for line in reversed(lines):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if et and str(rec.get("event_type") or "").strip().lower() != et:
                    continue
                if tn and str(rec.get("tool_name") or "").strip().lower() != tn:
                    continue
                if rt and str(rec.get("risk_tier") or "").strip().lower() != rt:
                    continue
                if ok is not None and bool(rec.get("ok")) != bool(ok):
                    continue
                rows.append(rec)
                if len(rows) >= n:
                    break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read tool execution log: {e}")
        return {
            "events": rows,
            "path": str(path),
            "limit": n,
            "filters": {
                "event_type": event_type,
                "tool_name": tool_name,
                "risk_tier": risk_tier,
                "ok": ok,
            },
        }

    @app.get("/api/ai/chat/planner_policy_stats", dependencies=[Depends(verify_api_key)])
    def get_chat_planner_policy_stats(
        since_hours: int = 24,
        limit: int = 5000,
        bucket_minutes: Optional[int] = None,
        active_profile: Optional[str] = None,
        min_actionable_for_alert: Optional[int] = None,
        min_conversion_actionable: Optional[float] = None,
        daily_days: int = 14,
        weekly_weeks: int = 12,
    ):
        n = max(100, min(20000, int(limit or 5000)))
        hours = max(1, min(24 * 30, int(since_hours or 24)))
        cfg = _load_planner_policy_config()
        bucket_mins = max(
            5,
            min(24 * 60, int((bucket_minutes if bucket_minutes is not None else 60) or 60)),
        )
        profile = str(active_profile if active_profile is not None else cfg.get("active_profile")).strip().lower()
        if profile not in {"conservative", "balanced", "aggressive"}:
            profile = str(cfg.get("active_profile") or "balanced")
        min_actionable = max(
            1,
            min(
                1000,
                int(
                    (min_actionable_for_alert if min_actionable_for_alert is not None else cfg.get("min_actionable_for_alert"))
                    or 5
                ),
            ),
        )
        min_conv = max(
            0.0,
            min(
                100.0,
                float(
                    (min_conversion_actionable if min_conversion_actionable is not None else cfg.get("min_conversion_actionable"))
                    or 35.0
                ),
            ),
        )
        keep_days = max(3, min(120, int(daily_days or 14)))
        keep_weeks = max(2, min(104, int(weekly_weeks or 12)))
        path = PROJECT_ROOT / "logs" / "tool_execution_log.jsonl"
        if not path.exists():
            return {
                "window_hours": hours,
                "bucket_minutes": bucket_mins,
                "active_profile": profile,
                "totals": {
                    "blocked": 0,
                    "executed": 0,
                    "skipped": 0,
                    "all": 0,
                },
                "by_scenario": {},
                "by_profile": {},
                "by_event_type": {},
                "conversion": {"by_profile": {}, "by_scenario": {}},
                "time_buckets": [],
                "long_term": {"daily": [], "weekly": []},
                "suggested_profile": {
                    "profile": profile,
                    "reason": "insufficient_data",
                    "confidence": 0.0,
                },
                "alerts": [],
                "alert_emit": {"emitted": 0, "suppressed": 0, "cooldown_seconds": 0},
                "alert_config": {
                    "active_profile": profile,
                    "min_actionable_for_alert": min_actionable,
                    "min_conversion_actionable": min_conv,
                    "alert_cooldown_minutes": cfg.get("alert_cooldown_minutes"),
                },
                "path": str(path),
            }

        event_keys = {
            "planner_policy_blocked": "blocked",
            "planner_auto_apply_executed": "executed",
            "planner_auto_apply_skipped": "skipped",
        }

        def _parse_event_ts(v: Any) -> Optional[float]:
            if not isinstance(v, str):
                return None
            s = v.strip()
            if not s:
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                return None

        cutoff = time.time() - (hours * 3600)
        totals = {"blocked": 0, "executed": 0, "skipped": 0, "all": 0}
        by_scenario: Dict[str, Dict[str, int]] = {}
        by_profile: Dict[str, Dict[str, int]] = {}
        by_event_type: Dict[str, int] = {}
        bucket_span = bucket_mins * 60
        time_buckets: Dict[int, Dict[str, Any]] = {}
        daily_agg: Dict[str, Dict[str, int]] = {}
        weekly_agg: Dict[str, Dict[str, int]] = {}
        scanned = 0
        matched = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if scanned >= n:
                    break
                scanned += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                ev = str(rec.get("event_type") or "").strip()
                bucket = event_keys.get(ev)
                if not bucket:
                    continue
                ts_value = _parse_event_ts(rec.get("timestamp_utc"))
                if ts_value is not None and ts_value < cutoff:
                    continue
                matched += 1
                totals["all"] += 1
                totals[bucket] += 1
                by_event_type[ev] = int(by_event_type.get(ev) or 0) + 1
                args = rec.get("arguments")
                scenario = "unknown"
                profile = "unknown"
                if isinstance(args, dict):
                    scenario = str(args.get("scenario") or "unknown").strip() or "unknown"
                    profile = str(args.get("auto_apply_profile") or "unknown").strip() or "unknown"
                sc = by_scenario.get(scenario)
                if not sc:
                    sc = {"blocked": 0, "executed": 0, "skipped": 0, "all": 0}
                    by_scenario[scenario] = sc
                sc["all"] += 1
                sc[bucket] += 1
                pr = by_profile.get(profile)
                if not pr:
                    pr = {"blocked": 0, "executed": 0, "skipped": 0, "all": 0}
                    by_profile[profile] = pr
                pr["all"] += 1
                pr[bucket] += 1
                if ts_value is None:
                    ts_value = time.time()
                bucket_ts = int(ts_value // bucket_span) * bucket_span
                b = time_buckets.get(bucket_ts)
                if not b:
                    bucket_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(bucket_ts))
                    b = {
                        "bucket_start_utc": bucket_start_utc,
                        "bucket_start_ts": bucket_ts,
                        "blocked": 0,
                        "executed": 0,
                        "skipped": 0,
                        "all": 0,
                    }
                    time_buckets[bucket_ts] = b
                b["all"] += 1
                b[bucket] += 1
                dt = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                day_key = dt.strftime("%Y-%m-%d")
                week_key = dt.strftime("%G-W%V")
                d = daily_agg.get(day_key)
                if not d:
                    d = {"blocked": 0, "executed": 0, "skipped": 0, "all": 0}
                    daily_agg[day_key] = d
                d["all"] += 1
                d[bucket] += 1
                w = weekly_agg.get(week_key)
                if not w:
                    w = {"blocked": 0, "executed": 0, "skipped": 0, "all": 0}
                    weekly_agg[week_key] = w
                w["all"] += 1
                w[bucket] += 1
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read planner policy stats: {e}")

        def _build_conversion(source: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for key, row in source.items():
                blocked = int(row.get("blocked") or 0)
                executed = int(row.get("executed") or 0)
                actionable = blocked + executed
                out[key] = {
                    "blocked": blocked,
                    "executed": executed,
                    "actionable": actionable,
                    "conversion_actionable": (executed * 100.0 / actionable) if actionable > 0 else 0.0,
                    "conversion_blocked_to_executed": (executed * 100.0 / blocked) if blocked > 0 else 0.0,
                }
            return out

        bucket_rows = [time_buckets[k] for k in sorted(time_buckets.keys())]
        for b in bucket_rows:
            blocked = int(b.get("blocked") or 0)
            executed = int(b.get("executed") or 0)
            actionable = blocked + executed
            b["actionable"] = actionable
            b["conversion_actionable"] = (executed * 100.0 / actionable) if actionable > 0 else 0.0

        conversion_by_profile = _build_conversion(by_profile)
        conversion_by_scenario = _build_conversion(by_scenario)
        daily_rows: List[Dict[str, Any]] = []
        for k in sorted(daily_agg.keys())[-keep_days:]:
            row = daily_agg[k]
            blocked = int(row.get("blocked") or 0)
            executed = int(row.get("executed") or 0)
            actionable = blocked + executed
            daily_rows.append(
                {
                    "day_utc": k,
                    "blocked": blocked,
                    "executed": executed,
                    "skipped": int(row.get("skipped") or 0),
                    "all": int(row.get("all") or 0),
                    "actionable": actionable,
                    "conversion_actionable": (executed * 100.0 / actionable) if actionable > 0 else 0.0,
                }
            )
        weekly_rows: List[Dict[str, Any]] = []
        for k in sorted(weekly_agg.keys())[-keep_weeks:]:
            row = weekly_agg[k]
            blocked = int(row.get("blocked") or 0)
            executed = int(row.get("executed") or 0)
            actionable = blocked + executed
            weekly_rows.append(
                {
                    "week_utc": k,
                    "blocked": blocked,
                    "executed": executed,
                    "skipped": int(row.get("skipped") or 0),
                    "all": int(row.get("all") or 0),
                    "actionable": actionable,
                    "conversion_actionable": (executed * 100.0 / actionable) if actionable > 0 else 0.0,
                }
            )

        suggested_profile = {
            "profile": profile,
            "reason": "insufficient_data",
            "confidence": 0.0,
        }
        candidate_rows: List[Dict[str, Any]] = []
        for p, row in conversion_by_profile.items():
            if not isinstance(row, dict):
                continue
            actionable = int(row.get("actionable") or 0)
            conv = _safe_float(row.get("conversion_actionable"), 0.0)
            candidate_rows.append(
                {
                    "profile": p,
                    "actionable": actionable,
                    "conversion_actionable": conv,
                }
            )
        eligible = [x for x in candidate_rows if int(x.get("actionable") or 0) >= min_actionable]
        base = eligible if eligible else candidate_rows
        if base:
            base.sort(
                key=lambda x: (
                    _safe_float(x.get("conversion_actionable"), 0.0),
                    int(x.get("actionable") or 0),
                ),
                reverse=True,
            )
            top = base[0]
            target_profile = str(top.get("profile") or profile)
            reason = "best_conversion"
            confidence = min(1.0, _safe_float(top.get("conversion_actionable"), 0.0) / 100.0)
            suggested_profile = {
                "profile": target_profile,
                "reason": reason,
                "confidence": round(confidence, 4),
            }
        alerts: List[Dict[str, Any]] = []
        active_profile_row = conversion_by_profile.get(profile)
        if isinstance(active_profile_row, dict):
            actionable = int(active_profile_row.get("actionable") or 0)
            conv = _safe_float(active_profile_row.get("conversion_actionable"), 0.0)
            if actionable >= min_actionable and conv < min_conv:
                alerts.append(
                    {
                        "type": "active_profile_conversion_low",
                        "severity": "warning",
                        "profile": profile,
                        "actionable": actionable,
                        "conversion_actionable": conv,
                        "threshold": min_conv,
                    }
                )
        if len(bucket_rows) >= 2:
            prev = bucket_rows[-2]
            last = bucket_rows[-1]
            prev_conv = _safe_float(prev.get("conversion_actionable"), 0.0)
            last_conv = _safe_float(last.get("conversion_actionable"), 0.0)
            prev_actionable = int(prev.get("actionable") or 0)
            last_actionable = int(last.get("actionable") or 0)
            if prev_actionable >= min_actionable and last_actionable >= min_actionable and last_conv + 10.0 < prev_conv:
                alerts.append(
                    {
                        "type": "conversion_drop_last_bucket",
                        "severity": "warning",
                        "previous_bucket_start_utc": str(prev.get("bucket_start_utc") or ""),
                        "last_bucket_start_utc": str(last.get("bucket_start_utc") or ""),
                        "previous_conversion_actionable": prev_conv,
                        "last_conversion_actionable": last_conv,
                    }
                )
        has_drop_alert = any(str(a.get("type") or "") == "conversion_drop_last_bucket" for a in alerts)
        has_low_alert = any(str(a.get("type") or "") == "active_profile_conversion_low" for a in alerts)
        if has_low_alert:
            if profile == "aggressive":
                suggested_profile = {
                    "profile": "balanced",
                    "reason": "degrade_from_aggressive_alert",
                    "confidence": max(0.7, _safe_float(suggested_profile.get("confidence"), 0.0)),
                }
            elif profile == "balanced":
                suggested_profile = {
                    "profile": "conservative",
                    "reason": "degrade_from_balanced_alert",
                    "confidence": max(0.7, _safe_float(suggested_profile.get("confidence"), 0.0)),
                }
        elif has_drop_alert and profile == "aggressive":
            suggested_profile = {
                "profile": "balanced",
                "reason": "bucket_drop_alert",
                "confidence": max(0.65, _safe_float(suggested_profile.get("confidence"), 0.0)),
            }
        emit_cfg = dict(cfg)
        emit_cfg["active_profile"] = profile
        emit_result = _emit_policy_alerts(alerts, emit_cfg) if alerts else {"emitted": 0, "suppressed": 0, "cooldown_seconds": int(_safe_float(cfg.get("alert_cooldown_minutes"), 60) * 60)}

        return {
            "window_hours": hours,
            "bucket_minutes": bucket_mins,
            "active_profile": profile,
            "totals": totals,
            "by_scenario": by_scenario,
            "by_profile": by_profile,
            "by_event_type": by_event_type,
            "conversion": {
                "by_profile": conversion_by_profile,
                "by_scenario": conversion_by_scenario,
            },
            "time_buckets": bucket_rows,
            "long_term": {
                "daily": daily_rows,
                "weekly": weekly_rows,
            },
            "suggested_profile": suggested_profile,
            "alerts": alerts,
            "alert_emit": emit_result,
            "alert_config": {
                "active_profile": profile,
                "min_actionable_for_alert": min_actionable,
                "min_conversion_actionable": min_conv,
                "alert_cooldown_minutes": cfg.get("alert_cooldown_minutes"),
            },
            "scanned": scanned,
            "matched": matched,
            "path": str(path),
        }

    @app.get("/api/ai/chat/planner_policy_config", dependencies=[Depends(verify_api_key)])
    def get_chat_planner_policy_config():
        return _load_planner_policy_config()

    @app.put("/api/ai/chat/planner_policy_config", dependencies=[Depends(verify_api_key)])
    def put_chat_planner_policy_config(body: Dict[str, Any] = Body(...)):
        saved = _save_planner_policy_config(body if isinstance(body, dict) else {})
        return {"ok": True, "config": saved}

    @app.get("/api/ai/chat/planner_policy_export", dependencies=[Depends(verify_api_key)])
    def get_chat_planner_policy_export(
        since_hours: int = 24,
        limit: int = 5000,
        bucket_minutes: int = 60,
        active_profile: str = "balanced",
        daily_days: int = 14,
        weekly_weeks: int = 12,
    ):
        payload = get_chat_planner_policy_stats(
            since_hours=since_hours,
            limit=limit,
            bucket_minutes=bucket_minutes,
            active_profile=active_profile,
            daily_days=daily_days,
            weekly_weeks=weekly_weeks,
        )
        lines: List[str] = []
        lines.append("section,key,metric,value")
        totals = payload.get("totals") if isinstance(payload, dict) else {}
        if isinstance(totals, dict):
            for k in ("all", "blocked", "executed", "skipped"):
                lines.append(f"totals,all,{k},{totals.get(k, 0)}")
        conversion = payload.get("conversion") if isinstance(payload, dict) else {}
        by_profile = conversion.get("by_profile") if isinstance(conversion, dict) else {}
        if isinstance(by_profile, dict):
            for profile_key, row in by_profile.items():
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"profile,{profile_key},conversion_actionable,{_safe_float(row.get('conversion_actionable'), 0.0):.4f}"
                )
                lines.append(
                    f"profile,{profile_key},conversion_blocked_to_executed,{_safe_float(row.get('conversion_blocked_to_executed'), 0.0):.4f}"
                )
        by_scenario = conversion.get("by_scenario") if isinstance(conversion, dict) else {}
        if isinstance(by_scenario, dict):
            for scenario_key, row in by_scenario.items():
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"scenario,{scenario_key},conversion_actionable,{_safe_float(row.get('conversion_actionable'), 0.0):.4f}"
                )
        buckets = payload.get("time_buckets") if isinstance(payload, dict) else []
        if isinstance(buckets, list):
            for b in buckets:
                if not isinstance(b, dict):
                    continue
                key = str(b.get("bucket_start_utc") or "")
                lines.append(f"bucket,{key},blocked,{int(b.get('blocked') or 0)}")
                lines.append(f"bucket,{key},executed,{int(b.get('executed') or 0)}")
                lines.append(
                    f"bucket,{key},conversion_actionable,{_safe_float(b.get('conversion_actionable'), 0.0):.4f}"
                )
        long_term = payload.get("long_term") if isinstance(payload, dict) else {}
        daily_rows = long_term.get("daily") if isinstance(long_term, dict) else []
        if isinstance(daily_rows, list):
            for row in daily_rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("day_utc") or "")
                lines.append(
                    f"daily,{key},conversion_actionable,{_safe_float(row.get('conversion_actionable'), 0.0):.4f}"
                )
                lines.append(f"daily,{key},actionable,{int(row.get('actionable') or 0)}")
        weekly_rows = long_term.get("weekly") if isinstance(long_term, dict) else []
        if isinstance(weekly_rows, list):
            for row in weekly_rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("week_utc") or "")
                lines.append(
                    f"weekly,{key},conversion_actionable,{_safe_float(row.get('conversion_actionable'), 0.0):.4f}"
                )
                lines.append(f"weekly,{key},actionable,{int(row.get('actionable') or 0)}")
        suggested = payload.get("suggested_profile") if isinstance(payload, dict) else {}
        if isinstance(suggested, dict):
            lines.append(f"suggested_profile,active,profile,{str(suggested.get('profile') or '')}")
            lines.append(
                f"suggested_profile,active,confidence,{_safe_float(suggested.get('confidence'), 0.0):.4f}"
            )
            lines.append(f"suggested_profile,active,reason,{str(suggested.get('reason') or '')}")
        alerts = payload.get("alerts") if isinstance(payload, dict) else []
        if isinstance(alerts, list):
            for i, row in enumerate(alerts, start=1):
                if not isinstance(row, dict):
                    continue
                lines.append(f"alert,{i},type,{str(row.get('type') or '')}")
                lines.append(f"alert,{i},severity,{str(row.get('severity') or '')}")
        csv_body = "\n".join(lines) + "\n"
        return Response(
            content=csv_body,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="planner_policy_export.csv"'
            },
        )

    @app.post("/api/ai/chat", dependencies=[Depends(verify_api_key)])
    async def post_chat(body: ChatBody, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        try:
            log_lines = []
            log_path = PROJECT_ROOT / "logs" / "bot.log"
            if log_path.exists():
                try:
                    import collections
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_lines = collections.deque(f, maxlen=20)
                    log_lines = list(log_lines)
                except Exception:
                    pass

            from dataclasses import asdict
            active_positions_count = len([t for t in state.trades if t.status == "open"])
            open_orders_count = 0 
            recent_trades_data = []
            for t in state.trades[-5:]:
                try:
                    recent_trades_data.append(asdict(t))
                except Exception:
                    recent_trades_data.append(str(t))

            context = {
                "risk_settings": _risk_to_dict(settings.risk),
                "active_positions": active_positions_count,
                "open_orders": open_orders_count,
                "recent_trades": recent_trades_data,
                "bot_status": "RUNNING" if state.is_running else "STOPPED",
                "last_notification": asdict(state.notifications[-1]) if state.notifications else "None"
            }

            user_message = str(body.message or "").strip()
            actor_key = str(x_api_key or "anonymous").strip() or "anonymous"
            if not user_message:
                response = await ai_agent.chat_with_user(user_message, context, log_lines)
                return {"response": response}

            if user_message.lower() in {"/tools", "tools", "инструменты"}:
                manifest = chat_tool_executor.list_manifest()
                limits = chat_tool_executor.get_limits_status(actor_key)
                preview = []
                for row in manifest[:20]:
                    preview.append(
                        f"- {row.get('name')} [{row.get('risk_tier')}] — {row.get('goal')}"
                    )
                response_text = (
                    "Доступные инструменты:\n"
                    + ("\n".join(preview) if preview else "нет инструментов")
                    + "\n\n"
                    + f"Rate limit: {limits.get('rate_remaining')}/{limits.get('rate_limit_per_minute')} в текущем окне."
                )
                await ai_agent._save_chat_message("user", user_message)
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "tools": manifest, "limits": limits}

            if user_message.lower() in {"/limits", "limits", "лимиты"}:
                limits = chat_tool_executor.get_limits_status(actor_key)
                response_text = (
                    "Лимиты чат-инструментов:\n"
                    f"- Rate limit/min: {limits.get('rate_limit_per_minute')}\n"
                    f"- Remaining: {limits.get('rate_remaining')}\n"
                    f"- Idempotency TTL: {limits.get('idempotency_ttl_seconds')}с\n"
                    f"- Idempotency cache size: {limits.get('idempotency_cache_size')}"
                )
                await ai_agent._save_chat_message("user", user_message)
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "limits": limits}

            expired = chat_tool_executor.expire_pending_if_needed()
            if expired:
                response_text = (
                    f"Ожидание подтверждения для {expired.get('tool_name')} истекло. "
                    f"Сформулируйте команду заново, если действие всё ещё актуально."
                )
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text}

            pending = chat_tool_executor.get_pending_action()
            if user_message.lower() in {"/pending", "pending", "ожидающее"}:
                await ai_agent._save_chat_message("user", user_message)
                if pending:
                    response_text = chat_tool_executor.pending_summary_text(pending)
                else:
                    response_text = "Ожидающих подтверждений нет."
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "pending_action": pending}
            if pending and chat_tool_executor.is_cancel_message(user_message):
                await ai_agent._save_chat_message("user", user_message)
                chat_tool_executor.log_tool_event(
                    event_type="pending_cancelled",
                    tool_name=str(pending.get("tool_name") or ""),
                    goal=str(pending.get("goal") or ""),
                    risk_tier=str(pending.get("risk_tier") or ""),
                    ok=False,
                    summary="Pending action cancelled by user",
                    arguments=pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {},
                )
                chat_tool_executor.clear_pending_action()
                response_text = "Ожидающее действие отменено."
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text}

            if pending and chat_tool_executor.is_confirmation_message(user_message):
                await ai_agent._save_chat_message("user", user_message)
                execution = await chat_tool_executor.execute(
                    tool_name=str(pending.get("tool_name") or ""),
                    arguments=pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {},
                    goal=str(pending.get("goal") or ""),
                    confirmed=True,
                    actor_key=actor_key,
                )
                response_text = chat_tool_executor.render_response_text(execution)
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "tool_execution": execution}
            if pending:
                response_text = chat_tool_executor.pending_summary_text(pending)
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text}

            if user_message.lower().startswith("/market"):
                await ai_agent._save_chat_message("user", user_message)
                parts = user_message.split()
                selected_symbol = None
                if len(parts) >= 2:
                    selected_symbol = str(parts[1]).upper().strip()
                if not selected_symbol:
                    active = [s for s in state.active_symbols if isinstance(s, str) and s]
                    selected_symbol = active[0] if active else None
                if not selected_symbol:
                    response_text = "Нет активных символов. Добавьте пару и повторите /market SYMBOL."
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}
                try:
                    insight = await get_ai_market_insight(symbol=selected_symbol, interval="60")
                    status_data = _get_status_data(state, bybit_client, settings, trading_loop)
                    open_positions = status_data.get("positions") if isinstance(status_data, dict) else []
                    open_count = len(open_positions) if isinstance(open_positions, list) else 0
                    trend = insight.get("trend") if isinstance(insight, dict) else None
                    volatility = insight.get("volatility") if isinstance(insight, dict) else None
                    confidence = insight.get("confidence") if isinstance(insight, dict) else None
                    advice = insight.get("advice") if isinstance(insight, dict) else None
                    conf_text = f"{float(confidence) * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
                    response_text = (
                        f"Маркет-срез {selected_symbol}\n"
                        f"- Trend: {trend or 'N/A'}\n"
                        f"- Volatility: {volatility or 'N/A'}\n"
                        f"- Confidence: {conf_text}\n"
                        f"- Open positions: {open_count}\n"
                        f"- Advice: {advice or 'N/A'}"
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text, "market_insight": insight}
                except HTTPException as e:
                    response_text = f"Не удалось выполнить /market: {e.detail}"
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}
                except Exception as e:
                    response_text = f"Не удалось выполнить /market: {e}"
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}

            if user_message.lower().startswith("/riskcheck"):
                await ai_agent._save_chat_message("user", user_message)
                try:
                    history_payload = get_history_trades(limit=10)
                    trades = history_payload.get("trades") if isinstance(history_payload, dict) else []
                    if not isinstance(trades, list):
                        trades = []
                    wins = 0
                    losses = 0
                    total_pnl = 0.0
                    for t in trades:
                        if not isinstance(t, dict):
                            continue
                        pnl = _safe_float(t.get("pnl_usd"), 0.0)
                        total_pnl += pnl
                        if pnl > 0:
                            wins += 1
                        elif pnl < 0:
                            losses += 1
                    avg_pnl = (total_pnl / len(trades)) if trades else 0.0
                    ai_risk = await get_ai_risk_analysis()
                    suggestions = ai_risk.get("suggestions") if isinstance(ai_risk, dict) else []
                    if not isinstance(suggestions, list):
                        suggestions = []
                    top = []
                    apply_updates: Dict[str, Any] = {}
                    allowed_risk_keys = {
                        "base_order_usd",
                        "max_position_usd",
                        "stop_loss_pct",
                        "take_profit_pct",
                        "margin_pct_balance",
                    }
                    for item in suggestions[:3]:
                        if not isinstance(item, dict):
                            continue
                        key = str(item.get("setting_key") or "").strip()
                        cur = item.get("current_value")
                        sug = item.get("suggested_value")
                        reason = item.get("reason")
                        if key in allowed_risk_keys and isinstance(sug, (int, float)):
                            apply_updates[key] = float(sug)
                        top.append(f"- {key}: {cur} -> {sug} ({reason})")
                    suggestions_text = "\n".join(top) if top else "- Нет явных изменений по риску"
                    risk_score = ai_risk.get("risk_score") if isinstance(ai_risk, dict) else None
                    analysis = ai_risk.get("analysis") if isinstance(ai_risk, dict) else None
                    apply_command = ""
                    if apply_updates:
                        apply_payload = {
                            "request_id": f"req_{int(time.time() * 1000)}_riskcheck",
                            "updates": apply_updates,
                        }
                        apply_command = "/tool update_risk_settings " + json.dumps(
                            apply_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        )
                    response_text = (
                        "Risk-check по последним 10 сделкам\n"
                        f"- Trades: {len(trades)}\n"
                        f"- Wins/Losses: {wins}/{losses}\n"
                        f"- Total PnL: {total_pnl:.2f} USD\n"
                        f"- Avg PnL: {avg_pnl:.2f} USD\n"
                        f"- AI risk score: {risk_score if risk_score is not None else 'N/A'}\n"
                        f"- AI analysis: {analysis if analysis else 'N/A'}\n"
                        "Рекомендации:\n"
                        f"{suggestions_text}\n"
                        + (
                            f"Apply command:\n{apply_command}"
                            if apply_command
                            else "Для применения используйте /tool update_risk_settings {\"request_id\":\"req_...\",\"updates\":{...}}"
                        )
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text, "risk_check": {"trades": trades, "ai_risk": ai_risk}}
                except HTTPException as e:
                    response_text = f"Не удалось выполнить /riskcheck: {e.detail}"
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}
                except Exception as e:
                    response_text = f"Не удалось выполнить /riskcheck: {e}"
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}

            if user_message.lower().startswith("/runbook"):
                await ai_agent._save_chat_message("user", user_message)
                parts = user_message.split()
                mode = parts[1].lower().strip() if len(parts) >= 2 else "help"
                if mode in {"help", "?", "list"}:
                    response_text = (
                        "Доступные runbook-сценарии:\n"
                        "- /runbook planner paper_auto_validation [experiment_id] [window_minutes] [auto_apply=true|false] [auto_profile=conservative|balanced|aggressive]\n"
                        "- /runbook paper_validate_apply [experiment_id]\n"
                        "- /runbook incident_response [symbol]\n"
                        "- /runbook campaign_watchdog\n"
                        "- /runbook paper_auto_validation [experiment_id] [window_minutes]\n"
                        "Сценарий проверяет completed/health/drift/OOS/stress и формирует безопасную команду apply."
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {"response": response_text}

                if mode in {"planner", "plan"}:
                    scenario = str(parts[2]).lower().strip() if len(parts) >= 3 else "paper_auto_validation"
                    if scenario not in {"paper_auto_validation", "incident_response", "campaign_watchdog"}:
                        response_text = "Planner поддерживает scenarios: paper_auto_validation, incident_response, campaign_watchdog."
                        await ai_agent._save_chat_message("assistant", response_text)
                        return {"response": response_text}

                    auto_apply = False
                    auto_apply_max_risk = "medium"
                    auto_apply_profile = "balanced"
                    profile_to_risk = {
                        "conservative": "low",
                        "balanced": "medium",
                        "aggressive": "high",
                    }
                    for token in parts[3:]:
                        t = str(token).strip().lower()
                        if t in {"auto", "apply", "auto_apply=true", "auto=true", "yes"}:
                            auto_apply = True
                        if t.startswith("auto_profile="):
                            val = t.split("=", 1)[1].strip()
                            if val in profile_to_risk:
                                auto_apply_profile = val
                                auto_apply_max_risk = profile_to_risk[val]
                        if t.startswith("auto_risk="):
                            val = t.split("=", 1)[1].strip()
                            if val in {"low", "medium", "high", "critical"}:
                                auto_apply_max_risk = val

                    def _parse_ts_planner(v: Any) -> Optional[float]:
                        if isinstance(v, (int, float)):
                            return float(v)
                        if not isinstance(v, str):
                            return None
                        s = v.strip()
                        if not s:
                            return None
                        if s.endswith("Z"):
                            s = s[:-1] + "+00:00"
                        try:
                            dt = datetime.fromisoformat(s)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt.timestamp()
                        except Exception:
                            return None

                    def _risk_rank(r: str) -> int:
                        return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(str(r or "").lower(), 99)

                    def _recommended_tool_name(command: str) -> str:
                        parsed = chat_tool_executor.parse_manual_tool_command(command)
                        if isinstance(parsed, dict):
                            return str(parsed.get("tool_name") or "")
                        return ""

                    def _recommended_tool_risk(command: str) -> str:
                        name = _recommended_tool_name(command)
                        tool_row = chat_tool_executor.tools_map.get(name) if name else None
                        if isinstance(tool_row, dict):
                            return str(tool_row.get("risk_tier") or "unknown").lower()
                        return "unknown"

                    experiment_id = ""
                    window_minutes = 30
                    symbol = ""
                    if scenario == "paper_auto_validation":
                        arg1 = str(parts[3]).strip() if len(parts) >= 4 else ""
                        arg2 = str(parts[4]).strip() if len(parts) >= 5 else ""
                        if arg1 and not arg1.startswith("auto_"):
                            if arg1.isdigit():
                                window_minutes = int(arg1)
                            else:
                                experiment_id = arg1
                        if arg2 and arg2.isdigit():
                            window_minutes = int(arg2)
                        window_minutes = max(5, min(240, window_minutes))
                        if not experiment_id:
                            status_payload = get_paper_status()
                            sessions = status_payload.get("sessions") if isinstance(status_payload, dict) else []
                            if isinstance(sessions, list) and sessions:
                                first = sessions[0]
                                if isinstance(first, dict):
                                    experiment_id = str(first.get("experiment_id") or "")
                        if not experiment_id:
                            response_text = "Planner: не найден experiment_id. Укажите /runbook planner paper_auto_validation exp_... 30 auto_apply=true"
                            await ai_agent._save_chat_message("assistant", response_text)
                            return {"response": response_text}
                    elif scenario == "incident_response":
                        symbol = str(parts[3]).upper().strip() if len(parts) >= 4 and not str(parts[3]).startswith("auto_") else ""

                    plan_steps = [
                        f"PLAN: scenario={scenario} auto_apply={auto_apply} auto_profile={auto_apply_profile} auto_apply_max_risk={auto_apply_max_risk}",
                        "EXECUTE: собрать данные по сценарию",
                        "VERIFY: определить recommendation и risk команды",
                        "SUMMARY: вернуть решение и (опционально) результат auto_apply",
                    ]

                    recommendation = "none"
                    recommendation_reason = "N/A"
                    recommended_command = ""
                    summary_line = ""
                    verification = "skip"
                    execution = None
                    verify_payload: Dict[str, Any] = {}

                    if scenario == "paper_auto_validation":
                        metrics_payload = get_paper_metrics(experiment_id=experiment_id)
                        metrics = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else {}
                        if not isinstance(metrics, dict):
                            metrics = {}
                        total_trades = int(metrics.get("total_trades") or 0)
                        win_rate = _safe_float(metrics.get("win_rate"), 0.0)
                        total_pnl = _safe_float(metrics.get("total_pnl"), 0.0)
                        max_dd_pct = _safe_float(metrics.get("max_drawdown_pct"), 0.0)
                        status_text = str(metrics_payload.get("status") or "")
                        trades_payload = get_paper_trades(experiment_id=experiment_id, limit=300)
                        trades_rows = trades_payload.get("trades") if isinstance(trades_payload, dict) else []
                        if not isinstance(trades_rows, list):
                            trades_rows = []
                        cutoff = datetime.now(timezone.utc).timestamp() - (window_minutes * 60)
                        window_pnls: List[float] = []
                        window_trades = 0
                        window_wins = 0
                        window_pnl = 0.0
                        for row in trades_rows:
                            if not isinstance(row, dict):
                                continue
                            exit_ts = _parse_ts_planner(row.get("exit_time"))
                            if exit_ts is None or exit_ts < cutoff:
                                continue
                            pnl = _safe_float(row.get("pnl"), 0.0)
                            window_pnls.append(pnl)
                            window_trades += 1
                            window_pnl += pnl
                            if pnl > 0:
                                window_wins += 1
                        window_win_rate = (window_wins * 100.0 / window_trades) if window_trades > 0 else 0.0
                        rolling_volatility = float(statistics.pstdev(window_pnls)) if len(window_pnls) >= 2 else 0.0
                        consecutive_losses_window = 0
                        for pnl in reversed(window_pnls):
                            if pnl < 0:
                                consecutive_losses_window += 1
                            else:
                                break
                        recommendation = "continue_paper"
                        recommendation_reason = "Окно наблюдения стабильно."
                        if status_text in {"error", "interrupted"}:
                            recommendation = "start_paper"
                            recommendation_reason = "Paper session в ошибке/прервана."
                            recommended_command = '/tool start_paper_trading {"experiment_id":"' + experiment_id + '"}'
                        elif window_trades >= 3 and (
                            window_pnl <= -20
                            or window_win_rate < 35.0
                            or rolling_volatility > 25.0
                            or consecutive_losses_window >= 4
                        ):
                            recommendation = "stop_paper"
                            recommendation_reason = "Негативная динамика в окне."
                            recommended_command = '/tool stop_paper_trading {"experiment_id":"' + experiment_id + '"}'
                        elif total_trades >= 5 and win_rate >= 45.0 and total_pnl > 0 and max_dd_pct <= 20.0:
                            recommendation = "apply_candidate"
                            recommendation_reason = "Базовые гейты пройдены."
                            recommended_command = (
                                '/tool apply_research_experiment {"request_id":"req_'
                                + str(int(time.time() * 1000))
                                + '","experiment_id":"'
                                + experiment_id
                                + '"}'
                            )
                        summary_line = (
                            f"window={window_minutes}m, trades={window_trades}, win_rate={window_win_rate:.2f}, pnl={window_pnl:.2f}, "
                            f"volatility={rolling_volatility:.2f}, consecutive_losses={consecutive_losses_window}"
                        )
                    elif scenario == "incident_response":
                        stats = get_stats()
                        history_payload = get_history_trades(limit=10)
                        trades = history_payload.get("trades") if isinstance(history_payload, dict) else []
                        if not isinstance(trades, list):
                            trades = []
                        negatives = 0
                        total_pnl = 0.0
                        for t in trades:
                            if not isinstance(t, dict):
                                continue
                            pnl = _safe_float(t.get("pnl_usd"), 0.0)
                            total_pnl += pnl
                            if pnl < 0:
                                negatives += 1
                        week_pnl = _safe_float(stats.get("week_pnl"), 0.0) if isinstance(stats, dict) else 0.0
                        drawdown_hint = "high" if week_pnl < -50 or negatives >= 7 else ("medium" if negatives >= 5 else "low")
                        incidents_24h = _incident_stats(24)
                        if drawdown_hint == "high" and int(incidents_24h.get("count_high_or_critical") or 0) >= 2:
                            drawdown_hint = "critical"
                        recommendation = "continue_monitoring"
                        recommendation_reason = "Стабильная ситуация."
                        if drawdown_hint == "critical":
                            recommendation = "emergency_stop"
                            recommendation_reason = "Критическая серия инцидентов."
                            recommended_command = '/tool emergency_stop_all {"request_id":"req_' + str(int(time.time() * 1000)) + '"}'
                        elif drawdown_hint in {"high", "medium"}:
                            recommendation = "reduce_risk"
                            recommendation_reason = "Повышенный риск по инцидентам."
                            recommended_command = '/tool update_risk_settings {"request_id":"req_' + str(int(time.time() * 1000)) + '","updates":{"base_order_usd":15,"stop_loss_pct":0.02}}'
                        summary_line = (
                            f"symbol={symbol or 'all'}, level={drawdown_hint}, negatives_last10={negatives}, total_pnl_last10={total_pnl:.2f}, "
                            f"week_pnl={week_pnl:.2f}, incidents_24h={incidents_24h.get('count_high_or_critical')}"
                        )
                    elif scenario == "campaign_watchdog":
                        payload = get_research_status()
                        experiments = payload.get("experiments") if isinstance(payload, dict) else []
                        if not isinstance(experiments, list):
                            experiments = []
                        active_rows = []
                        stale_rows = []
                        for row in experiments:
                            if not isinstance(row, dict):
                                continue
                            status_text = str(row.get("status") or "").lower()
                            if status_text in {"running", "queued", "pending"}:
                                active_rows.append(row)
                        for row in active_rows[:20]:
                            exp_id = str(row.get("id") or "")
                            if not exp_id:
                                continue
                            try:
                                health = get_research_experiment_health(exp_id)
                                if bool(health.get("stale")):
                                    stale_rows.append(
                                        {
                                            "experiment_id": exp_id,
                                            "symbol": row.get("symbol"),
                                            "status_reason": health.get("status_reason"),
                                        }
                                    )
                            except Exception:
                                continue
                        recommendation = "continue_campaigns"
                        recommendation_reason = "Состояние кампаний штатное."
                        if stale_rows:
                            target = stale_rows[0]
                            recommendation = "pause_stale_campaign"
                            recommendation_reason = "Обнаружены stale кампании."
                            recommended_command = (
                                '/tool control_research_campaign {"experiment_id":"'
                                + str(target.get("experiment_id") or "")
                                + '","action":"pause"}'
                            )
                        summary_line = f"active={len(active_rows)}, stale={len(stale_rows)}"

                    recommended_risk = _recommended_tool_risk(recommended_command) if recommended_command else "none"
                    if auto_apply and recommended_command:
                        if _risk_rank(recommended_risk) > _risk_rank(auto_apply_max_risk):
                            verification = f"blocked_by_risk_policy(risk={recommended_risk}, max={auto_apply_max_risk})"
                            chat_tool_executor.log_tool_event(
                                event_type="planner_policy_blocked",
                                tool_name=_recommended_tool_name(recommended_command) or "unknown",
                                goal=f"planner:{scenario}",
                                risk_tier=recommended_risk,
                                ok=False,
                                summary=f"Blocked by policy profile={auto_apply_profile}, max_risk={auto_apply_max_risk}",
                                arguments={
                                    "scenario": scenario,
                                    "recommended_command": recommended_command,
                                    "recommended_risk": recommended_risk,
                                    "auto_apply_profile": auto_apply_profile,
                                    "auto_apply_max_risk": auto_apply_max_risk,
                                },
                            )
                        else:
                            execution_plan = chat_tool_executor.parse_manual_tool_command(recommended_command)
                            if isinstance(execution_plan, dict):
                                execution = await chat_tool_executor.execute(
                                    tool_name=str(execution_plan.get("tool_name") or ""),
                                    arguments=execution_plan.get("arguments") if isinstance(execution_plan.get("arguments"), dict) else {},
                                    goal="runbook_planner_auto_apply",
                                    confirmed=False,
                                    actor_key=actor_key,
                                )
                                verification = "executed"
                                chat_tool_executor.log_tool_event(
                                    event_type="planner_auto_apply_executed",
                                    tool_name=str(execution_plan.get("tool_name") or "unknown"),
                                    goal=f"planner:{scenario}",
                                    risk_tier=recommended_risk,
                                    ok=bool(execution.get("ok")) if isinstance(execution, dict) else False,
                                    summary=f"profile={auto_apply_profile}, max_risk={auto_apply_max_risk}",
                                    arguments={
                                        "scenario": scenario,
                                        "recommended_command": recommended_command,
                                        "recommended_risk": recommended_risk,
                                    },
                                )
                            else:
                                verification = "parse_failed"
                    elif auto_apply and not recommended_command:
                        verification = "no_recommended_command"
                        chat_tool_executor.log_tool_event(
                            event_type="planner_auto_apply_skipped",
                            tool_name="none",
                            goal=f"planner:{scenario}",
                            risk_tier="none",
                            ok=True,
                            summary="Auto-apply enabled but no recommended command",
                            arguments={"scenario": scenario, "recommendation": recommendation},
                        )

                    if verification == "executed":
                        if scenario == "paper_auto_validation":
                            verify_payload = get_paper_status()
                        elif scenario == "incident_response":
                            verify_payload = get_status()
                        elif scenario == "campaign_watchdog":
                            verify_payload = get_research_status()

                    response_text = (
                        "Runbook planner\n"
                        + "\n".join(f"- {x}" for x in plan_steps)
                        + "\n"
                        + f"SUMMARY: recommendation={recommendation} reason={recommendation_reason}\n"
                        + f"metrics: {summary_line}\n"
                        + f"recommended_command={recommended_command if recommended_command else 'N/A'}\n"
                        + f"recommended_risk={recommended_risk}\n"
                        + f"auto_apply={auto_apply}, auto_apply_profile={auto_apply_profile}, auto_apply_max_risk={auto_apply_max_risk}, verification={verification}"
                    )
                    if isinstance(execution, dict):
                        response_text += "\n" + chat_tool_executor.render_response_text(execution)
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {
                        "response": response_text,
                        "runbook_planner": {
                            "scenario": scenario,
                            "auto_apply": auto_apply,
                            "auto_apply_profile": auto_apply_profile,
                            "auto_apply_max_risk": auto_apply_max_risk,
                            "recommendation": recommendation,
                            "recommendation_reason": recommendation_reason,
                            "recommended_command": recommended_command,
                            "recommended_risk": recommended_risk,
                            "execution": execution,
                            "verification": verification,
                            "verify_payload": verify_payload,
                        },
                    }

                if mode in {"paper_validate_apply", "paper", "validate_apply"}:
                    experiment_id = str(parts[2]).strip() if len(parts) >= 3 else ""
                    experiments_payload = get_research_status()
                    experiments = experiments_payload.get("experiments") if isinstance(experiments_payload, dict) else []
                    if not isinstance(experiments, list):
                        experiments = []
                    selected = None
                    if experiment_id:
                        for row in experiments:
                            if isinstance(row, dict) and str(row.get("id") or "") == experiment_id:
                                selected = row
                                break
                    else:
                        for row in experiments:
                            if isinstance(row, dict) and str(row.get("status") or "").lower() == "completed":
                                selected = row
                                break
                        if selected is None and experiments:
                            selected = experiments[0] if isinstance(experiments[0], dict) else None
                    if not selected:
                        response_text = "Runbook: эксперименты не найдены. Сначала запустите /tools -> start_research_experiment."
                        await ai_agent._save_chat_message("assistant", response_text)
                        return {"response": response_text}
                    experiment_id = str(selected.get("id") or experiment_id or "").strip()
                    health_ok = False
                    health_payload = {}
                    report_payload = {}
                    try:
                        health_payload = get_research_experiment_health(experiment_id)
                        health_ok = bool(health_payload.get("ok")) and not bool(health_payload.get("stale"))
                    except Exception:
                        health_ok = False
                    try:
                        report_payload = get_experiment_report(experiment_id)
                    except Exception:
                        report_payload = {}
                    status_text = str(selected.get("status") or "").lower()
                    completed = status_text == "completed" or int(selected.get("progress") or 0) >= 100
                    oos_gate = selected.get("oos_gates")
                    if oos_gate is None:
                        oos_gate = selected.get("oos_validation", {}).get("pass") if isinstance(selected.get("oos_validation"), dict) else None
                    drift_level = "unknown"
                    drift = selected.get("drift_signals")
                    if isinstance(drift, dict):
                        drift_level = str(drift.get("drift_level") or drift.get("level") or "unknown")
                    stress_level = "unknown"
                    stress = selected.get("stress_results")
                    if isinstance(stress, dict):
                        stress_level = str(stress.get("risk_level") or stress.get("severity") or "unknown")
                    apply_ready = bool(completed and health_ok)
                    if isinstance(oos_gate, bool) and not oos_gate:
                        apply_ready = False
                    if drift_level.lower() in {"high", "critical"}:
                        apply_ready = False
                    if stress_level.lower() in {"high", "critical"}:
                        apply_ready = False
                    req_id = f"req_{int(time.time() * 1000)}"
                    apply_cmd = (
                        "/tool apply_research_experiment "
                        f"{{\"request_id\":\"{req_id}\",\"experiment_id\":\"{experiment_id}\"}}"
                    )
                    stop_reason = report_payload.get("stop_reason") if isinstance(report_payload, dict) else None
                    response_text = (
                        f"Runbook paper_validate_apply for {experiment_id}\n"
                        f"1) Completed: {completed}\n"
                        f"2) Health OK: {health_ok}\n"
                        f"3) OOS gate: {oos_gate if oos_gate is not None else 'unknown'}\n"
                        f"4) Drift: {drift_level}\n"
                        f"5) Stress: {stress_level}\n"
                        f"6) Stop reason: {stop_reason if stop_reason else 'N/A'}\n"
                        f"APPLY_READY: {'YES' if apply_ready else 'NO'}\n"
                        "Safe apply command:\n"
                        f"{apply_cmd}"
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {
                        "response": response_text,
                        "runbook": {
                            "name": "paper_validate_apply",
                            "experiment_id": experiment_id,
                            "apply_ready": apply_ready,
                            "apply_command": apply_cmd,
                            "checks": {
                                "completed": completed,
                                "health_ok": health_ok,
                                "oos_gate": oos_gate,
                                "drift_level": drift_level,
                                "stress_level": stress_level,
                            },
                        },
                    }

                if mode in {"incident_response", "incident"}:
                    symbol = str(parts[2]).upper().strip() if len(parts) >= 3 else ""
                    status_data = _get_status_data(state, bybit_client, settings, trading_loop)
                    stats = get_stats()
                    history_payload = get_history_trades(limit=10)
                    trades = history_payload.get("trades") if isinstance(history_payload, dict) else []
                    if not isinstance(trades, list):
                        trades = []
                    negatives = 0
                    total_pnl = 0.0
                    for t in trades:
                        if not isinstance(t, dict):
                            continue
                        pnl = _safe_float(t.get("pnl_usd"), 0.0)
                        total_pnl += pnl
                        if pnl < 0:
                            negatives += 1
                    recent_consecutive_losses = 0
                    for t in trades:
                        if not isinstance(t, dict):
                            continue
                        pnl = _safe_float(t.get("pnl_usd"), 0.0)
                        if pnl < 0:
                            recent_consecutive_losses += 1
                        else:
                            break
                    week_pnl = _safe_float(stats.get("week_pnl"), 0.0) if isinstance(stats, dict) else 0.0
                    drawdown_hint = "high" if week_pnl < -50 or negatives >= 7 else ("medium" if negatives >= 5 else "low")
                    last24_stats = _incident_stats(24)
                    if drawdown_hint == "high" and int(last24_stats.get("count_high_or_critical") or 0) >= 2:
                        drawdown_hint = "critical"
                    elif drawdown_hint == "medium" and int(last24_stats.get("count_high_or_critical") or 0) >= 3:
                        drawdown_hint = "high"
                    after_record_stats = _record_incident(
                        drawdown_hint,
                        symbol or None,
                        {
                            "week_pnl": week_pnl,
                            "negatives_last10": negatives,
                            "consecutive_losses_last10": recent_consecutive_losses,
                            "total_pnl_last10": total_pnl,
                        },
                    )
                    risk_cmd = '/tool update_risk_settings {"request_id":"req_' + str(int(time.time() * 1000)) + '","updates":{"base_order_usd":15,"stop_loss_pct":0.02}}'
                    stop_cmd = '/tool stop_bot {"request_id":"req_' + str(int(time.time() * 1000)) + '"}'
                    emergency_cmd = '/tool emergency_stop_all {"request_id":"req_' + str(int(time.time() * 1000)) + '"}'
                    escalation_action = "continue_monitoring"
                    escalation_cmd = ""
                    if drawdown_hint == "critical":
                        escalation_action = "consider_emergency_stop"
                        escalation_cmd = emergency_cmd
                    elif drawdown_hint == "high" or recent_consecutive_losses >= 4:
                        escalation_action = "consider_stop_bot"
                        escalation_cmd = stop_cmd
                    response_text = (
                        f"Runbook incident_response {symbol or '(all symbols)'}\n"
                        f"- Bot status: {status_data.get('bot_status') if isinstance(status_data, dict) else 'N/A'}\n"
                        f"- Active positions: {len(status_data.get('positions', [])) if isinstance(status_data, dict) and isinstance(status_data.get('positions'), list) else 0}\n"
                        f"- Last 10 trades negative: {negatives}\n"
                        f"- Last 10 consecutive losses: {recent_consecutive_losses}\n"
                        f"- Last 10 total PnL: {total_pnl:.2f} USD\n"
                        f"- Week PnL: {week_pnl:.2f} USD\n"
                        f"- Incident level: {drawdown_hint}\n"
                        f"- Incidents 24h (high/critical): {after_record_stats.get('count_high_or_critical')}\n"
                        "Рекомендованные действия:\n"
                        "1) /riskcheck\n"
                        "2) При medium/high снизить риск\n"
                        f"3) Suggested command: {risk_cmd}\n"
                        f"4) Escalation action: {escalation_action}\n"
                        f"5) Escalation command: {escalation_cmd if escalation_cmd else 'N/A'}"
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {
                        "response": response_text,
                        "runbook": {
                            "name": "incident_response",
                            "symbol": symbol or None,
                            "incident_level": drawdown_hint,
                            "negative_trades_last10": negatives,
                            "consecutive_losses_last10": recent_consecutive_losses,
                            "total_pnl_last10": total_pnl,
                            "week_pnl": week_pnl,
                            "incidents_24h": after_record_stats,
                            "escalation_action": escalation_action,
                            "escalation_command": escalation_cmd,
                            "suggested_risk_command": risk_cmd,
                        },
                    }

                if mode in {"campaign_watchdog", "watchdog"}:
                    payload = get_research_status()
                    experiments = payload.get("experiments") if isinstance(payload, dict) else []
                    if not isinstance(experiments, list):
                        experiments = []
                    active_rows = []
                    stale_rows = []
                    for row in experiments:
                        if not isinstance(row, dict):
                            continue
                        status_text = str(row.get("status") or "").lower()
                        if status_text in {"running", "queued", "pending"}:
                            active_rows.append(row)
                    for row in active_rows[:20]:
                        exp_id = str(row.get("id") or "")
                        if not exp_id:
                            continue
                        try:
                            health = get_research_experiment_health(exp_id)
                            if bool(health.get("stale")):
                                stale_rows.append(
                                    {
                                        "experiment_id": exp_id,
                                        "symbol": row.get("symbol"),
                                        "status": row.get("status"),
                                        "status_reason": health.get("status_reason"),
                                    }
                                )
                        except Exception:
                            continue
                    response_lines = [
                        "Runbook campaign_watchdog",
                        f"- Active campaigns: {len(active_rows)}",
                        f"- Stale campaigns: {len(stale_rows)}",
                    ]
                    for item in stale_rows[:8]:
                        response_lines.append(
                            f"  • {item.get('experiment_id')} {item.get('symbol')} reason={item.get('status_reason')}"
                        )
                    if stale_rows:
                        response_lines.append(
                            "Рекомендация: используйте /tool control_research_campaign {\"experiment_id\":\"...\",\"action\":\"pause\"} или stop."
                        )
                    else:
                        response_lines.append("Проблемных кампаний не обнаружено.")
                    response_text = "\n".join(response_lines)
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {
                        "response": response_text,
                        "runbook": {
                            "name": "campaign_watchdog",
                            "active_count": len(active_rows),
                            "stale_count": len(stale_rows),
                            "stale_campaigns": stale_rows,
                        },
                    }

                if mode in {"paper_auto_validation", "paper_auto", "paper_validate"}:
                    experiment_id = ""
                    window_minutes = 30
                    arg1 = str(parts[2]).strip() if len(parts) >= 3 else ""
                    arg2 = str(parts[3]).strip() if len(parts) >= 4 else ""
                    if arg1:
                        if arg1.isdigit():
                            window_minutes = int(arg1)
                        else:
                            experiment_id = arg1
                    if arg2 and arg2.isdigit():
                        window_minutes = int(arg2)
                    window_minutes = max(5, min(240, window_minutes))
                    if not experiment_id:
                        status_payload = get_paper_status()
                        sessions = status_payload.get("sessions") if isinstance(status_payload, dict) else []
                        if isinstance(sessions, list) and sessions:
                            first = sessions[0]
                            if isinstance(first, dict):
                                experiment_id = str(first.get("experiment_id") or "")
                    if not experiment_id:
                        response_text = "Runbook: не найден experiment_id для paper validation. Укажите /runbook paper_auto_validation exp_..."
                        await ai_agent._save_chat_message("assistant", response_text)
                        return {"response": response_text}
                    try:
                        metrics_payload = get_paper_metrics(experiment_id=experiment_id)
                    except Exception as e:
                        response_text = f"Runbook paper_auto_validation failed: {e}"
                        await ai_agent._save_chat_message("assistant", response_text)
                        return {"response": response_text}
                    metrics = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else {}
                    if not isinstance(metrics, dict):
                        metrics = {}
                    total_trades = int(metrics.get("total_trades") or 0)
                    win_rate = _safe_float(metrics.get("win_rate"), 0.0)
                    total_pnl = _safe_float(metrics.get("total_pnl"), 0.0)
                    max_dd_pct = _safe_float(metrics.get("max_drawdown_pct"), 0.0)
                    status_text = str(metrics_payload.get("status") or "")
                    timestamps = metrics.get("timestamps") if isinstance(metrics.get("timestamps"), list) else []
                    equity_curve = metrics.get("equity_curve") if isinstance(metrics.get("equity_curve"), list) else []
                    cutoff = datetime.now(timezone.utc).timestamp() - (window_minutes * 60)

                    def _parse_ts(v: Any) -> Optional[float]:
                        if isinstance(v, (int, float)):
                            return float(v)
                        if not isinstance(v, str):
                            return None
                        s = v.strip()
                        if not s:
                            return None
                        if s.endswith("Z"):
                            s = s[:-1] + "+00:00"
                        try:
                            dt = datetime.fromisoformat(s)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt.timestamp()
                        except Exception:
                            return None

                    window_indices: List[int] = []
                    for i, ts in enumerate(timestamps):
                        ts_value = _parse_ts(ts)
                        if ts_value is None:
                            continue
                        if ts_value >= cutoff:
                            window_indices.append(i)
                    window_return_pct = 0.0
                    window_points = len(window_indices)
                    if window_points >= 2 and len(equity_curve) > window_indices[0]:
                        first_idx = window_indices[0]
                        last_idx = window_indices[-1]
                        if len(equity_curve) > last_idx:
                            first_equity = _safe_float(equity_curve[first_idx], 0.0)
                            last_equity = _safe_float(equity_curve[last_idx], 0.0)
                            if first_equity > 0:
                                window_return_pct = ((last_equity - first_equity) / first_equity) * 100.0

                    trades_payload = get_paper_trades(experiment_id=experiment_id, limit=300)
                    trades_rows = trades_payload.get("trades") if isinstance(trades_payload, dict) else []
                    if not isinstance(trades_rows, list):
                        trades_rows = []
                    window_trades = 0
                    window_wins = 0
                    window_pnl = 0.0
                    window_pnls: List[float] = []
                    window_trade_rows: List[Dict[str, Any]] = []
                    for row in trades_rows:
                        if not isinstance(row, dict):
                            continue
                        exit_ts = _parse_ts(row.get("exit_time"))
                        if exit_ts is None or exit_ts < cutoff:
                            continue
                        pnl = _safe_float(row.get("pnl"), 0.0)
                        window_trades += 1
                        window_pnl += pnl
                        window_pnls.append(pnl)
                        window_trade_rows.append(row)
                        if pnl > 0:
                            window_wins += 1
                    window_win_rate = (window_wins * 100.0 / window_trades) if window_trades > 0 else 0.0
                    rolling_volatility = 0.0
                    if len(window_pnls) >= 2:
                        try:
                            rolling_volatility = float(statistics.pstdev(window_pnls))
                        except Exception:
                            rolling_volatility = 0.0
                    window_trade_rows.sort(
                        key=lambda r: (_parse_ts(r.get("exit_time")) or 0.0),
                        reverse=True,
                    )
                    consecutive_losses_window = 0
                    for row in window_trade_rows:
                        pnl = _safe_float(row.get("pnl"), 0.0)
                        if pnl < 0:
                            consecutive_losses_window += 1
                        else:
                            break

                    gates = {
                        "min_trades": total_trades >= 5,
                        "win_rate": win_rate >= 45.0,
                        "pnl_positive": total_pnl > 0,
                        "drawdown_ok": max_dd_pct <= 20.0,
                        "status_ok": status_text in {"active", "completed", "stopped"},
                        "rolling_volatility_ok": rolling_volatility <= 25.0,
                        "consecutive_losses_ok": consecutive_losses_window <= 3,
                    }
                    pass_all = all(bool(v) for v in gates.values())
                    recommendation = "continue_paper"
                    recommendation_reason = "Окно наблюдения стабильно."
                    recommended_command = ""
                    if status_text in {"error", "interrupted"}:
                        recommendation = "start_paper"
                        recommendation_reason = "Paper session в ошибке/прервана, рекомендуется перезапуск."
                        recommended_command = '/tool start_paper_trading {"experiment_id":"' + experiment_id + '"}'
                    elif window_trades >= 3 and (
                        window_pnl <= -20
                        or window_win_rate < 35.0
                        or window_return_pct <= -1.5
                        or rolling_volatility > 25.0
                        or consecutive_losses_window >= 4
                    ):
                        recommendation = "stop_paper"
                        recommendation_reason = "Негативная динамика в окне наблюдения."
                        recommended_command = '/tool stop_paper_trading {"experiment_id":"' + experiment_id + '"}'
                    elif pass_all and window_return_pct >= 0.5 and window_win_rate >= 50.0:
                        recommendation = "apply_candidate"
                        recommendation_reason = "Гейты и окно наблюдения подтверждают кандидата."

                    apply_cmd = (
                        '/tool apply_research_experiment {"request_id":"req_'
                        + str(int(time.time() * 1000))
                        + '","experiment_id":"'
                        + experiment_id
                        + '"}'
                    )
                    response_text = (
                        f"Runbook paper_auto_validation {experiment_id}\n"
                        f"- status: {status_text}\n"
                        f"- total_trades: {total_trades}\n"
                        f"- win_rate: {win_rate:.2f}\n"
                        f"- total_pnl: {total_pnl:.2f}\n"
                        f"- max_drawdown_pct: {max_dd_pct:.2f}\n"
                        f"- window_minutes: {window_minutes}\n"
                        f"- window_points: {window_points}\n"
                        f"- window_trades: {window_trades}\n"
                        f"- window_win_rate: {window_win_rate:.2f}\n"
                        f"- window_pnl: {window_pnl:.2f}\n"
                        f"- window_return_pct: {window_return_pct:.2f}\n"
                        f"- rolling_volatility: {rolling_volatility:.2f}\n"
                        f"- consecutive_losses_window: {consecutive_losses_window}\n"
                        f"- gates: {gates}\n"
                        f"AUTO_PASS: {'YES' if pass_all else 'NO'}\n"
                        f"Recommendation: {recommendation} ({recommendation_reason})\n"
                        f"Recommended command: {recommended_command if recommended_command else 'N/A'}\n"
                        "Safe apply command:\n"
                        f"{apply_cmd}"
                    )
                    await ai_agent._save_chat_message("assistant", response_text)
                    return {
                        "response": response_text,
                        "runbook": {
                            "name": "paper_auto_validation",
                            "experiment_id": experiment_id,
                            "auto_pass": pass_all,
                            "gates": gates,
                            "window_minutes": window_minutes,
                            "window_trades": window_trades,
                            "window_win_rate": window_win_rate,
                            "window_pnl": window_pnl,
                            "window_return_pct": window_return_pct,
                            "rolling_volatility": rolling_volatility,
                            "consecutive_losses_window": consecutive_losses_window,
                            "recommendation": recommendation,
                            "recommendation_reason": recommendation_reason,
                            "recommended_command": recommended_command,
                            "apply_command": apply_cmd,
                        },
                    }

                response_text = "Неизвестный runbook. Используйте /runbook help."
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text}

            plan = chat_tool_executor.parse_manual_tool_command(user_message)
            if plan is None:
                plan = await ai_agent.plan_chat_tool_call(
                    message=user_message,
                    tools_manifest=chat_tool_executor.list_manifest(),
                    has_pending_action=bool(pending),
                )

            if isinstance(plan, dict) and str(plan.get("intent") or "").lower() == "tool_call":
                await ai_agent._save_chat_message("user", user_message)
                execution = await chat_tool_executor.execute(
                    tool_name=str(plan.get("tool_name") or ""),
                    arguments=plan.get("arguments") if isinstance(plan.get("arguments"), dict) else {},
                    goal=str(plan.get("goal") or ""),
                    confirmed=False,
                    actor_key=actor_key,
                )
                if execution.get("ok"):
                    response_text = await ai_agent.synthesize_tool_results(
                        user_message=user_message,
                        executions=[execution],
                        context=context,
                    )
                else:
                    response_text = chat_tool_executor.render_response_text(execution)
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "tool_execution": execution}

            if isinstance(plan, dict) and str(plan.get("intent") or "").lower() == "tool_chain":
                steps = plan.get("steps")
                if not isinstance(steps, list) or not steps:
                    response = await ai_agent.chat_with_user(user_message, context, log_lines)
                    return {"response": response}
                await ai_agent._save_chat_message("user", user_message)
                executions: List[Dict[str, Any]] = []
                for row in steps[:4]:
                    if not isinstance(row, dict):
                        continue
                    execution = await chat_tool_executor.execute(
                        tool_name=str(row.get("tool_name") or ""),
                        arguments=row.get("arguments") if isinstance(row.get("arguments"), dict) else {},
                        goal=str(row.get("goal") or ""),
                        confirmed=False,
                        actor_key=actor_key,
                    )
                    executions.append(execution)
                    if (
                        execution.get("requires_confirmation")
                        or execution.get("rate_limited")
                        or (not execution.get("ok"))
                    ):
                        break
                all_ok = bool(executions) and all(bool(x.get("ok")) for x in executions if isinstance(x, dict))
                if all_ok:
                    response_text = await ai_agent.synthesize_tool_results(
                        user_message=user_message,
                        executions=executions,
                        context=context,
                    )
                else:
                    chunks: List[str] = []
                    for ex in executions:
                        if not isinstance(ex, dict):
                            continue
                        chunks.append(chat_tool_executor.render_response_text(ex))
                    response_text = "\n\n".join(chunks).strip() or "Не удалось выполнить цепочку инструментов."
                await ai_agent._save_chat_message("assistant", response_text)
                return {"response": response_text, "tool_executions": executions}

            response = await ai_agent.chat_with_user(user_message, context, log_lines)
            return {"response": response}
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return {"response": f"Error processing request: {str(e)}"}

# --- AI Agent Endpoints ---
    def _parse_trades_from_log(limit: int = 50) -> List[Dict[str, Any]]:
        """Parses trades.log for historical trade data."""
        log_path = PROJECT_ROOT / "logs" / "trades.log"
        if not log_path.exists():
            return []
            
        trades = []
        import re
        # Regex for TRADE CLOSE lines:
        # 2026-02-18 14:06:25 - trades - INFO - TRADE CLOSE: BNBUSDT | Exit: 611.0 | PnL: $-0.16 (-3.31%) | Reason: SL
        close_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*TRADE CLOSE: (\w+) \| Exit: ([\d.]+) \| PnL: \$([-?\d.]+) \(([-?\d.]+)%\) \| Reason: (.*)")
        
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                # Read from end efficiently would be better, but for <10MB file readlines is fine
                lines = f.readlines()
                
            for line in reversed(lines):
                if len(trades) >= limit:
                    break
                match = close_pattern.search(line)
                if match:
                    timestamp, symbol, exit_price, pnl_usd, pnl_pct, reason = match.groups()
                    trades.append({
                        "symbol": symbol,
                        "pnl_usd": float(pnl_usd),
                        "pnl_pct": float(pnl_pct),
                        "exit_reason": reason.strip(),
                        "exit_time": timestamp,
                        "status": "closed",
                        "side": "Unknown" # Side is in OPEN log, simplified here
                    })
        except Exception as e:
            logger.error(f"Error parsing trades.log: {e}")
            
        return trades

    @app.get("/api/ai/analyze_risks", dependencies=[Depends(verify_api_key)])
    async def get_ai_risk_analysis():
        """
        AI анализирует последние 50 сделок и предлагает изменения в risk_settings.
        """
        from dataclasses import asdict
        try:
            closed_trades = [t for t in state.trades if getattr(t, "status", None) == "closed"]
            if len(closed_trades) < 5:
                logger.info("Few trades in memory, parsing trades.log for AI analysis...")
                log_trades = _parse_trades_from_log(limit=50)
                if log_trades:
                    if not closed_trades:
                        trades_data = log_trades
                    else:
                        trades_data = [asdict(t) for t in closed_trades] + log_trades
                else:
                    trades_data = [asdict(t) for t in closed_trades]
            else:
                trades_data = [asdict(t) for t in closed_trades]

            if not trades_data:
                return {"analysis": "Нет закрытых сделок для анализа (ни в памяти, ни в логах).", "suggestions": [], "risk_score": 100}

            current_risk = _risk_to_dict(settings.risk)
            payload = {
                "trades": trades_data,
                "current_risk": current_risk,
            }
            result = await ai_agent.analyze_risk_settings(payload)
            if not isinstance(result, dict):
                result = {
                    "analysis": "Некорректный ответ AI-анализа",
                    "suggestions": [],
                    "risk_score": 0,
                }
            suggestions = result.get("suggestions")
            normalized_suggestions: List[Dict[str, Any]] = []
            if isinstance(suggestions, list):
                for item in suggestions:
                    if not isinstance(item, dict):
                        continue
                    key = _normalize_risk_key(item.get("setting_key"))
                    if not key or not hasattr(settings.risk, key):
                        continue
                    current_value = current_risk.get(key, getattr(settings.risk, key, None))
                    suggested_value = item.get("suggested_value")
                    if current_value == suggested_value:
                        continue
                    if isinstance(current_value, (int, float)) and isinstance(suggested_value, (int, float)):
                        if abs(float(current_value) - float(suggested_value)) < 1e-12:
                            continue
                    row = dict(item)
                    row["setting_key"] = key
                    row["current_value"] = current_value
                    normalized_suggestions.append(row)
            result["suggestions"] = normalized_suggestions
            result["current_risk"] = current_risk
            return result
        except Exception as e:
            logger.error(f"AI risk endpoint failed: {e}", exc_info=True)
            return {
                "analysis": f"Ошибка анализа риска: {str(e)}",
                "suggestions": [],
                "risk_score": 0,
            }

    @app.get("/api/ai/market_insight", dependencies=[Depends(verify_api_key)])
    async def get_ai_market_insight(symbol: str, interval: str = "60"):
        """
        AI дает комментарий по рынку на основе свечей и индикаторов.
        """
        symbol = symbol.upper()
        if not bybit_client:
             raise HTTPException(status_code=501, detail="Exchange client not available")
             
        try:
            # 1. Get Klines
            kline_resp = bybit_client.get_kline(symbol=symbol, interval=interval, limit=100)
            if kline_resp.get("retCode") != 0:
                 raise HTTPException(status_code=400, detail="Failed to get kline data")
                 
            klines = kline_resp.get("result", {}).get("list", [])
            # Convert to OHLCV dicts (sorted old -> new)
            ohlcv = []
            for k in reversed(klines): # Bybit returns new -> old
                ohlcv.append({
                    "time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
                
            # 2. Calculate Basic Indicators (using pandas_ta if available or simple math)
            # For simplicity, let's just pass raw OHLCV to AI, it's good at it.
            # But adding SMA/RSI helps.
            import pandas as pd
            try:
                import pandas_ta as ta
                has_ta = True
            except ImportError:
                has_ta = False
                logger.warning("pandas_ta not installed, skipping indicators")
            
            df = pd.DataFrame(ohlcv)
            # Calculate simple indicators
            if has_ta and len(df) > 50:
                try:
                    df['rsi'] = ta.rsi(df['close'], length=14)
                    df['sma_20'] = ta.sma(df['close'], length=20)
                    df['sma_50'] = ta.sma(df['close'], length=50)
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                except Exception as e:
                     logger.warning(f"Error calculating indicators: {e}")
            
            latest = df.iloc[-1]
            
            def _safe_val(s, key):
                v = s.get(key, 0)
                if pd.isna(v): return 0.0
                return float(v)

            indicators = {
                "rsi": _safe_val(latest, 'rsi'),
                "sma_20": _safe_val(latest, 'sma_20'),
                "sma_50": _safe_val(latest, 'sma_50'),
                "atr": _safe_val(latest, 'atr'),
                "close_price": float(latest['close'])
            }
            
            # 3. Ask AI
            insight = await ai_agent.analyze_market_sentiment(symbol, ohlcv, indicators)
            return insight
            
        except Exception as e:
            logger.error(f"AI Market Insight error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    class ResearchBody(BaseModel):
        symbol: str
        type: str = "balanced" # aggressive, conservative, balanced
        metadata: Optional[Dict[str, Any]] = None
        allow_duplicate: bool = False
        safe_mode: bool = True

    @app.post("/api/ai/research/start", dependencies=[Depends(verify_api_key)])
    async def post_start_research(body: ResearchBody):
        """Запускает эксперимент (Research Agent)."""
        symbol = body.symbol.upper()
        try:
            res = ai_agent.start_research_experiment(
                symbol,
                body.type,
                metadata=body.metadata,
                allow_duplicate=body.allow_duplicate,
                safe_mode=body.safe_mode,
            )
            if not res.get("ok"):
                 error_msg = res.get("error", "Unknown error")
                 logger.error(f"Research start failed for {symbol}: {error_msg}")
                 if "Duplicate" in error_msg:
                     raise HTTPException(status_code=409, detail=error_msg)
                 raise HTTPException(status_code=500, detail=error_msg)
            
            if tg_bot:
                 await tg_bot.send_notification(f"🧪 Research Experiment ({body.type}) started for {symbol}")
                 
            return res
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Research endpoint error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    class CampaignControlBody(BaseModel):
        action: str

    @app.post("/api/ai/research/campaign/{experiment_id}/control", dependencies=[Depends(verify_api_key)])
    def control_research_campaign(experiment_id: str, body: CampaignControlBody):
        from .experiment_management import ExperimentStore
        from datetime import datetime, timezone

        action = str(body.action or "").strip().lower()
        if action not in {"pause", "resume", "stop"}:
            raise HTTPException(status_code=400, detail="action must be one of: pause, resume, stop")

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        target = store.get(experiment_id)
        if not target:
            raise HTTPException(status_code=404, detail="Experiment not found")

        target_campaign = target.get("ai_campaign") if isinstance(target.get("ai_campaign"), dict) else {}
        root_id = str(target_campaign.get("root_experiment_id") or target.get("id") or experiment_id)

        data = store.read_all()
        changed_ids: List[str] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for eid, exp in data.items():
            if not isinstance(exp, dict):
                continue
            campaign = exp.get("ai_campaign") if isinstance(exp.get("ai_campaign"), dict) else {}
            exp_root = str(campaign.get("root_experiment_id") or exp.get("id") or "")
            if exp_root != root_id:
                continue

            if not campaign:
                campaign = {"root_experiment_id": root_id}
            if action == "resume":
                campaign["auto_chain"] = True
                exp["campaign_control"] = {"state": "running", "updated_at": now_iso}
            else:
                campaign["auto_chain"] = False
                exp["campaign_control"] = {
                    "state": "paused" if action == "pause" else "stopped",
                    "updated_at": now_iso,
                }
            exp["ai_campaign"] = campaign
            exp["campaign_status"] = "paused" if action == "pause" else ("stopped" if action == "stop" else exp.get("campaign_status"))
            exp["updated_at"] = now_iso
            changed_ids.append(str(eid))

        store.write_all(data)
        return {
            "ok": True,
            "root_experiment_id": root_id,
            "action": action,
            "updated_experiments": changed_ids,
        }

    @app.get("/api/ai/experiments/insights", dependencies=[Depends(verify_api_key)])
    def get_experiment_insights(symbol: Optional[str] = None, type: Optional[str] = None):
        from pathlib import Path
        from .experiment_management import (
            ExperimentAnalyzer,
            ExperimentCriteria,
            ExperimentStore,
            HypothesisGenerator,
        )

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        experiments = store.list()
        analyzer = ExperimentAnalyzer(experiments)
        summary = analyzer.summarize(symbol=symbol, experiment_type=type)
        impact = analyzer.compute_param_impact(symbol.upper()) if symbol else None
        hypotheses = HypothesisGenerator(ExperimentCriteria()).propose(summary)
        return {
            "symbol": symbol.upper() if symbol else None,
            "type": type,
            "summary": summary,
            "impact": impact,
            "hypotheses": hypotheses,
        }

    @app.get("/api/ai/experiments/report/{experiment_id}", dependencies=[Depends(verify_api_key)])
    def get_experiment_report(experiment_id: str):
        from pathlib import Path
        from .experiment_management import ExperimentAnalyzer, ExperimentStore

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            project_root = Path(__file__).resolve().parent.parent
            removed_paths: List[str] = []
            meta_path = project_root / "experiment_meta" / f"{experiment_id}.json"
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    removed_paths.append(str(meta_path))
                except Exception as e:
                    logger.error(f"Failed to remove meta file {meta_path}: {e}", exc_info=True)
            artifacts_dir = project_root / "experiment_artifacts" / experiment_id
            if artifacts_dir.exists():
                try:
                    shutil.rmtree(artifacts_dir)
                    removed_paths.append(str(artifacts_dir))
                except Exception as e:
                    logger.error(f"Failed to remove artifacts dir {artifacts_dir}: {e}", exc_info=True)
            if paper_trading_manager:
                try:
                    paper_trading_manager.delete_session(experiment_id)
                except Exception as e:
                    logger.error(f"Failed to delete paper trading session for {experiment_id}: {e}", exc_info=True)
            if removed_paths:
                return {
                    "ok": True,
                    "experiment_id": experiment_id,
                    "force": force,
                    "strategy_reset": False,
                    "removed_paths": removed_paths,
                    "removed_model_paths": [],
                    "orphan": True,
                }
            raise HTTPException(status_code=404, detail="Experiment not found")
        analyzer = ExperimentAnalyzer(store.list())
        impact = None
        if exp.get("symbol"):
            impact = analyzer.compute_param_impact(str(exp.get("symbol")))
        markdown_builder = None
        try:
            from .experiment_management import ExperimentReportBuilder

            markdown_builder = ExperimentReportBuilder()
        except Exception:
            markdown_builder = None
        if markdown_builder is not None:
            md = markdown_builder.build_markdown(exp, impact=impact)
        else:
            results = exp.get("results") if isinstance(exp.get("results"), dict) else {}
            lines = [
                f"# Experiment Report: {exp.get('id') or experiment_id}",
                "",
                "## Overview",
                f"- Symbol: {exp.get('symbol')}",
                f"- Type: {exp.get('type')}",
                f"- Status: {exp.get('status')}",
                "",
                "## Performance",
                f"- Total PnL (%): {results.get('total_pnl_pct')}",
                f"- Win rate (%): {results.get('win_rate')}",
                f"- Max drawdown (%): {results.get('max_drawdown_pct') or results.get('max_drawdown')}",
                f"- Total trades: {results.get('total_trades')}",
                f"- Recommended tactic: {results.get('recommended_tactic')}",
            ]
            if isinstance(impact, dict) and impact:
                lines.extend(["", "## Parameter Impact"])
                for key, value in impact.items():
                    lines.append(f"- {key}: {value}")
            md = "\n".join(lines)
        return {"experiment_id": experiment_id, "markdown": md}

    @app.get("/api/ai/research/experiment/{experiment_id}/health", dependencies=[Depends(verify_api_key)])
    def get_research_experiment_health(experiment_id: str):
        from .experiment_management import ExperimentStore
        from datetime import datetime

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            project_root = Path(__file__).resolve().parent.parent
            removed_paths: List[str] = []
            meta_path = project_root / "experiment_meta" / f"{experiment_id}.json"
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    removed_paths.append(str(meta_path))
                except Exception as e:
                    logger.error(f"Failed to remove meta file {meta_path}: {e}", exc_info=True)
            artifacts_dir = project_root / "experiment_artifacts" / experiment_id
            if artifacts_dir.exists():
                try:
                    shutil.rmtree(artifacts_dir)
                    removed_paths.append(str(artifacts_dir))
                except Exception as e:
                    logger.error(f"Failed to remove artifacts dir {artifacts_dir}: {e}", exc_info=True)
            if paper_trading_manager:
                try:
                    paper_trading_manager.delete_session(experiment_id)
                except Exception as e:
                    logger.error(f"Failed to delete paper trading session for {experiment_id}: {e}", exc_info=True)
            if removed_paths:
                return {
                    "ok": True,
                    "experiment_id": experiment_id,
                    "force": force,
                    "strategy_reset": False,
                    "removed_paths": removed_paths,
                    "removed_model_paths": [],
                    "orphan": True,
                }
            raise HTTPException(status_code=404, detail="Experiment not found")

        heartbeat_at = exp.get("heartbeat_at") or exp.get("updated_at")
        last_output_at = exp.get("last_output_at")
        seconds_since = None
        seconds_since_output = None
        if isinstance(heartbeat_at, str):
            try:
                dt = datetime.fromisoformat(heartbeat_at)
                seconds_since = int((datetime.now(dt.tzinfo) - dt).total_seconds())
            except Exception:
                seconds_since = None
        if isinstance(last_output_at, str):
            try:
                dt2 = datetime.fromisoformat(last_output_at)
                seconds_since_output = int((datetime.now(dt2.tzinfo) - dt2).total_seconds())
            except Exception:
                seconds_since_output = None

        runner_pid = exp.get("runner_pid")
        alive = None
        exit_code = None
        try:
            proc = getattr(ai_agent, "research_processes", {}).get(experiment_id) if ai_agent else None
            if proc is not None:
                exit_code = proc.poll()
                alive = exit_code is None
        except Exception:
            alive = None

        if alive is None and isinstance(runner_pid, int):
            try:
                import os
                os.kill(runner_pid, 0)
                alive = True
            except Exception:
                alive = False

        stale = seconds_since is not None and seconds_since > 60 and exp.get("status") in {"starting", "training", "backtesting"}
        no_output_warning = (
            alive is True
            and seconds_since_output is not None
            and seconds_since_output > 1200
            and exp.get("status") in {"training", "backtesting"}
        )

        child_count = None
        zombie_child_count = None
        no_child_processes = None
        child_probe_error = None
        if isinstance(runner_pid, int):
            try:
                res = subprocess.run(
                    ["/usr/bin/ps", "--ppid", str(runner_pid), "-o", "stat="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                out = res.stdout or ""
                stats = [s.strip() for s in out.splitlines() if s.strip()]
                child_count = len(stats)
                zombie_child_count = sum(1 for s in stats if s.startswith("Z"))
                no_child_processes = child_count == 0
                if res.returncode not in (0, 1) and (res.stderr or "").strip():
                    child_probe_error = f"ps rc={res.returncode}: {(res.stderr or '').strip()}"
            except Exception as e:
                child_count = None
                zombie_child_count = None
                no_child_processes = None
                child_probe_error = str(e)

        return {
            "experiment_id": experiment_id,
            "status": exp.get("status"),
            "runner_pid": runner_pid,
            "alive": alive,
            "exit_code": exit_code,
            "heartbeat_at": heartbeat_at,
            "seconds_since_heartbeat": seconds_since,
            "last_output_at": last_output_at,
            "seconds_since_output": seconds_since_output,
            "stale": stale,
            "no_output_warning": no_output_warning,
            "runner_idle_seconds": seconds_since_output,
            "child_count": child_count,
            "zombie_child_count": zombie_child_count,
            "no_child_processes": no_child_processes,
            "child_probe_error": child_probe_error,
            "runner_phase": exp.get("runner_phase"),
            "runner_step": exp.get("runner_step"),
        }

    @app.post("/api/ai/research/experiment/{experiment_id}/stop", dependencies=[Depends(verify_api_key)])
    def stop_research_experiment(experiment_id: str):
        from .experiment_management import ExperimentStore
        from datetime import datetime, timezone

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            project_root = Path(__file__).resolve().parent.parent
            removed_paths: List[str] = []
            meta_path = project_root / "experiment_meta" / f"{experiment_id}.json"
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    removed_paths.append(str(meta_path))
                except Exception as e:
                    logger.error(f"Failed to remove meta file {meta_path}: {e}", exc_info=True)
            artifacts_dir = project_root / "experiment_artifacts" / experiment_id
            if artifacts_dir.exists():
                try:
                    shutil.rmtree(artifacts_dir)
                    removed_paths.append(str(artifacts_dir))
                except Exception as e:
                    logger.error(f"Failed to remove artifacts dir {artifacts_dir}: {e}", exc_info=True)
            if paper_trading_manager:
                try:
                    paper_trading_manager.delete_session(experiment_id)
                except Exception as e:
                    logger.error(f"Failed to delete paper trading session for {experiment_id}: {e}", exc_info=True)
            if removed_paths:
                return {
                    "ok": True,
                    "experiment_id": experiment_id,
                    "force": force,
                    "strategy_reset": False,
                    "removed_paths": removed_paths,
                    "removed_model_paths": [],
                    "orphan": True,
                }
            raise HTTPException(status_code=404, detail="Experiment not found")

        status = str(exp.get("status") or "unknown")
        if status in {"completed", "failed", "interrupted"}:
            return {
                "ok": True,
                "experiment_id": experiment_id,
                "already_stopped": True,
                "status": status,
                "killed": False,
            }

        proc = getattr(ai_agent, "research_processes", {}).get(experiment_id) if ai_agent else None
        pid = exp.get("runner_pid")
        if not isinstance(pid, int):
            pid = exp.get("pid")
        if not isinstance(pid, int) and proc is not None:
            try:
                pid = int(proc.pid)
            except Exception:
                pid = None

        killed = False
        try:
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                killed = proc.poll() is not None
        except Exception as e:
            logger.error(f"Failed to stop experiment via handle: {e}", exc_info=True)

        if not killed:
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
            except Exception:
                killed = False

        if not killed:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                killed = True
            except Exception:
                killed = False

        patch = {
            "status": "interrupted",
            "status_reason": "stopped_by_user",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(pid, int):
            patch["stopped_pid"] = pid
        try:
            store.upsert(experiment_id, patch)
        except Exception as e:
            logger.error(f"Failed to persist interrupted status for {experiment_id}: {e}", exc_info=True)

        return {"ok": True, "experiment_id": experiment_id, "pid": pid, "killed": killed}

    @app.delete("/api/ai/research/experiment/{experiment_id}", dependencies=[Depends(verify_api_key)])
    def delete_research_experiment(experiment_id: str, force: bool = False):
        from .experiment_management import ExperimentStore

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            project_root = Path(__file__).resolve().parent.parent
            removed_paths: List[str] = []
            meta_path = project_root / "experiment_meta" / f"{experiment_id}.json"
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    removed_paths.append(str(meta_path))
                except Exception as e:
                    logger.error(f"Failed to remove meta file {meta_path}: {e}", exc_info=True)
            artifacts_dir = project_root / "experiment_artifacts" / experiment_id
            if artifacts_dir.exists():
                try:
                    shutil.rmtree(artifacts_dir)
                    removed_paths.append(str(artifacts_dir))
                except Exception as e:
                    logger.error(f"Failed to remove artifacts dir {artifacts_dir}: {e}", exc_info=True)
            if paper_trading_manager:
                try:
                    paper_trading_manager.delete_session(experiment_id)
                except Exception as e:
                    logger.error(f"Failed to delete paper trading session for {experiment_id}: {e}", exc_info=True)
            if removed_paths:
                return {
                    "ok": True,
                    "experiment_id": experiment_id,
                    "force": force,
                    "strategy_reset": False,
                    "removed_paths": removed_paths,
                    "removed_model_paths": [],
                    "orphan": True,
                }
            raise HTTPException(status_code=404, detail="Experiment not found")

        status = str(exp.get("status") or "unknown")
        if status == "active" and not force:
            raise HTTPException(status_code=400, detail="Active experiment cannot be deleted")

        project_root = Path(__file__).resolve().parent.parent

        pid = exp.get("runner_pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:
                    pass

        if paper_trading_manager:
            try:
                paper_trading_manager.delete_session(experiment_id)
            except Exception as e:
                logger.error(f"Failed to delete paper trading session for {experiment_id}: {e}", exc_info=True)

        removed_model_paths: List[str] = []
        strategy_reset = False
        try:
            results = exp.get("results") if isinstance(exp.get("results"), dict) else {}
            models = results.get("models") if isinstance(results.get("models"), dict) else {}
            model_candidates: List[str] = []
            for v in models.values():
                if isinstance(v, str) and v:
                    model_candidates.append(v)

            if exp.get("symbol") and hasattr(state, "get_strategy_config"):
                current_cfg = state.get_strategy_config(str(exp.get("symbol")))
                if isinstance(current_cfg, dict):
                    in_use = []
                    for p in model_candidates:
                        if p and (
                            current_cfg.get("model_path") == p
                            or current_cfg.get("model_1h_path") == p
                            or current_cfg.get("model_15m_path") == p
                        ):
                            in_use.append(p)
                    if in_use:
                        if not force:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Experiment models are currently in use: {', '.join(in_use)}. Use ?force=true to detach and delete.",
                            )
                        try:
                            state.set_strategy_config(str(exp.get("symbol")), {"mode": "single"})
                            strategy_reset = True
                        except Exception as e:
                            logger.error(f"Failed to reset strategy config for symbol {exp.get('symbol')}: {e}", exc_info=True)
                        try:
                            symbol_key = str(exp.get("symbol")).upper()
                            if hasattr(state, "symbol_models") and isinstance(state.symbol_models, dict):
                                current_model_path = current_cfg.get("model_path")
                                if isinstance(current_model_path, str) and current_model_path in in_use:
                                    state.symbol_models.pop(symbol_key, None)
                                    state.save()
                        except Exception as e:
                            logger.error(f"Failed to clear symbol_models for symbol {exp.get('symbol')}: {e}", exc_info=True)

            for p in model_candidates:
                fp = (project_root / p) if not Path(p).is_absolute() else Path(p)
                if "ml_models" not in fp.parts:
                    continue
                if fp.exists() and fp.is_file():
                    try:
                        fp.unlink()
                        removed_model_paths.append(str(fp))
                    except Exception as e:
                        logger.error(f"Failed to remove model file {fp}: {e}", exc_info=True)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to remove model files for {experiment_id}: {e}", exc_info=True)

        data = store.read_all()
        if experiment_id in data:
            del data[experiment_id]
            store.write_all(data)

        removed_paths: List[str] = []
        meta_path = project_root / "experiment_meta" / f"{experiment_id}.json"
        if meta_path.exists():
            try:
                meta_path.unlink()
                removed_paths.append(str(meta_path))
            except Exception as e:
                logger.error(f"Failed to remove meta file {meta_path}: {e}", exc_info=True)

        artifacts_dir = project_root / "experiment_artifacts" / experiment_id
        if artifacts_dir.exists():
            try:
                shutil.rmtree(artifacts_dir)
                removed_paths.append(str(artifacts_dir))
            except Exception as e:
                logger.error(f"Failed to remove artifacts dir {artifacts_dir}: {e}", exc_info=True)

        return {
            "ok": True,
            "experiment_id": experiment_id,
            "force": force,
            "strategy_reset": strategy_reset,
            "removed_paths": removed_paths,
            "removed_model_paths": removed_model_paths,
        }

    @app.get("/api/ai/research/status", dependencies=[Depends(verify_api_key)])
    def get_research_status():
        """Возвращает список активных и прошлых экспериментов с сравнением с текущей стратегией."""
        from datetime import datetime, timedelta
        
        experiments = ai_agent.get_research_experiments()
        
        # Добавляем информацию о текущей стратегии и сравнение для каждого эксперимента
        for experiment in experiments:
            symbol = experiment.get("symbol")
            if not symbol:
                continue
            results_payload = experiment.get("results") if isinstance(experiment.get("results"), dict) else {}
            ai_plan = experiment.get("ai_plan") if isinstance(experiment.get("ai_plan"), dict) else {}
            if not ai_plan and isinstance(results_payload.get("ai_plan"), dict):
                ai_plan = results_payload.get("ai_plan")
            experiment["ai_plan_brief"] = {
                "interval": ai_plan.get("interval") if isinstance(ai_plan, dict) else None,
                "use_mtf": ai_plan.get("use_mtf") if isinstance(ai_plan, dict) else None,
                "safe_mode": ai_plan.get("safe_mode") if isinstance(ai_plan, dict) else None,
                "backtest_days": ai_plan.get("backtest_days") if isinstance(ai_plan, dict) else None,
                "hypothesis": results_payload.get("hypothesis") or experiment.get("hypothesis"),
                "expected_outcome": results_payload.get("expected_outcome") or experiment.get("expected_outcome"),
            }
            c = experiment.get("ai_campaign") if isinstance(experiment.get("ai_campaign"), dict) else {}
            experiment["campaign"] = {
                "auto_chain": c.get("auto_chain"),
                "iteration": c.get("iteration"),
                "remaining_steps": c.get("remaining_steps"),
                "max_steps": c.get("max_steps"),
                "root_experiment_id": c.get("root_experiment_id"),
                "parent_experiment_id": c.get("parent_experiment_id"),
                "status": experiment.get("campaign_status"),
                "next_experiment_id": results_payload.get("next_experiment_id") or experiment.get("next_experiment_id"),
                "control": experiment.get("campaign_control"),
            }
            experiment["next_actions"] = (
                results_payload.get("next_experiments")
                if isinstance(results_payload.get("next_experiments"), list)
                else experiment.get("next_experiments")
            )
            experiment["analysis_summary"] = results_payload.get("analysis_summary") or experiment.get("analysis_summary")
            experiment["oos_metrics"] = (
                results_payload.get("oos_metrics")
                if isinstance(results_payload.get("oos_metrics"), dict)
                else None
            )
            experiment["drift_signals"] = (
                results_payload.get("drift_signals")
                if isinstance(results_payload.get("drift_signals"), dict)
                else None
            )
            experiment["stress_results"] = (
                results_payload.get("stress_results")
                if isinstance(results_payload.get("stress_results"), dict)
                else None
            )
                
            # Получаем текущую стратегию для символа
            current_strategy = None
            if trading_loop and getattr(trading_loop, "strategies", None):
                current_strategy = trading_loop.strategies.get(symbol)
            
            # Получаем текущие модели из конфигурации
            current_models = {}
            if hasattr(state, "get_strategy_config"):
                config = state.get_strategy_config(symbol)
                if config:
                    if config.get("mode") == "mtf":
                        current_models = {
                            "1h": config.get("model_1h_path"),
                            "15m": config.get("model_15m_path")
                        }
                    else:
                        current_models = {
                            "single": config.get("model_path")
                        }
            
            # Собираем метрики текущей стратегии из реальных сделок за последнюю неделю
            current_metrics = {}
            if hasattr(state, "trades"):
                week_ago = datetime.now() - timedelta(days=7)
                week_trades = []
                for trade in state.trades:
                    if trade.symbol == symbol and trade.status == "closed" and trade.exit_time:
                        try:
                            exit_dt = datetime.fromisoformat(trade.exit_time)
                            if exit_dt >= week_ago:
                                week_trades.append(trade)
                        except (ValueError, TypeError):
                            pass
                
                if week_trades:
                    total_pnl = sum(t.pnl_usd for t in week_trades)
                    total_pnl_pct = sum(t.pnl_pct for t in week_trades if hasattr(t, 'pnl_pct')) if hasattr(week_trades[0], 'pnl_pct') else 0
                    winning_trades = [t for t in week_trades if t.pnl_usd > 0]
                    win_rate = (len(winning_trades) / len(week_trades) * 100) if week_trades else 0
                    
                    current_metrics = {
                        "total_pnl": total_pnl,
                        "total_pnl_pct": total_pnl_pct,
                        "win_rate": win_rate,
                        "total_trades": len(week_trades),
                        "winning_trades": len(winning_trades),
                        "losing_trades": len(week_trades) - len(winning_trades)
                    }
            
            # Сравниваем с результатами эксперимента
            experiment_results = experiment.get("results", {})
            experiment_metrics = experiment_results.get("metrics", {})
            # Если experiment_metrics пустые, используем experiment_results напрямую
            if not experiment_metrics:
                experiment_metrics = experiment_results
            
            # Добавляем информацию о текущей стратегии в эксперимент
            experiment["current_strategy"] = {
                "has_strategy": current_strategy is not None,
                "models": current_models,
                "metrics": current_metrics
            }
            
            # Рекомендация: если эксперимент лучше текущей стратегии, предлагаем замену
            experiment_pnl = experiment_metrics.get("total_pnl_pct", 0)
            experiment_winrate = experiment_metrics.get("win_rate", 0)
            current_pnl = current_metrics.get("total_pnl_pct", 0)
            current_winrate = current_metrics.get("win_rate", 0)
            
            # Простая логика рекомендации (можно улучшить)
            recommendation = "keep_current"
            if experiment_pnl > current_pnl + 2 and experiment_winrate > current_winrate + 5:
                recommendation = "replace"
            elif experiment_pnl > current_pnl + 5:
                recommendation = "replace"
            elif experiment_pnl < current_pnl - 5:
                recommendation = "discard"
            
            experiment["recommendation"] = recommendation
            experiment["comparison"] = {
                "experiment_pnl": experiment_pnl,
                "experiment_winrate": experiment_winrate,
                "current_pnl": current_pnl,
                "current_winrate": current_winrate,
                "improvement_pnl": experiment_pnl - current_pnl,
                "improvement_winrate": experiment_winrate - current_winrate
            }
            
            # Add paper trading status if available
            if paper_trading_manager:
                paper_status = paper_trading_manager.get_session_status(experiment.get("id"))
                if paper_status:
                    experiment["paper_status"] = paper_status
                    paper_metrics = paper_trading_manager.get_metrics(experiment.get("id"))
                    if paper_metrics:
                        experiment["paper_metrics"] = paper_metrics
        
        return {"experiments": experiments}

    class ApplyExperimentBody(BaseModel):
        experiment_id: str

    @app.post("/api/ai/research/apply", dependencies=[Depends(verify_api_key)])
    async def post_apply_experiment(body: ApplyExperimentBody):
        """Применяет экспериментальную стратегию (заменяет текущую)."""
        from datetime import datetime
        try:
            # Получаем эксперимент
            experiments = ai_agent.get_research_experiments()
            experiment = None
            for exp in experiments:
                if exp.get("id") == body.experiment_id:
                    experiment = exp
                    break
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            symbol = experiment.get("symbol")
            if not symbol:
                raise HTTPException(status_code=400, detail="Experiment missing symbol")
            
            # Получаем модели из эксперимента
            results = experiment.get("results") if isinstance(experiment.get("results"), dict) else {}
            model_sources: List[Dict[str, Any]] = []
            if isinstance(results.get("models"), dict):
                model_sources.append(results.get("models"))
            if isinstance(experiment.get("models"), dict):
                model_sources.append(experiment.get("models"))
            if isinstance(experiment.get("applied_models"), dict):
                model_sources.append(experiment.get("applied_models"))
            merged_models: Dict[str, Any] = {}
            for src in model_sources:
                if not isinstance(src, dict):
                    continue
                for k, v in src.items():
                    if v in (None, "", "null"):
                        continue
                    merged_models[str(k)] = v
            model_15m_path = (
                merged_models.get("15m")
                or merged_models.get("model_15m")
                or merged_models.get("tf_15m")
                or merged_models.get("m15")
            )
            model_1h_path = (
                merged_models.get("1h")
                or merged_models.get("model_1h")
                or merged_models.get("tf_1h")
                or merged_models.get("h1")
            )
            recommended_tactic = results.get("recommended_tactic")
            if not recommended_tactic and isinstance((results.get("selection") or {}).get("recommended_tactic"), str):
                recommended_tactic = (results.get("selection") or {}).get("recommended_tactic")
            
            if not model_15m_path and not model_1h_path:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Experiment has no models",
                        "experiment_id": body.experiment_id,
                        "available_model_keys": sorted(list(merged_models.keys())),
                    },
                )
            
            # Применяем стратегию
            if not recommended_tactic:
                if model_1h_path and model_15m_path:
                    recommended_tactic = "mtf"
                elif model_1h_path:
                    recommended_tactic = "single_1h"
                else:
                    recommended_tactic = "single_15m"

            if recommended_tactic == "single_1h" and not model_1h_path and model_15m_path:
                recommended_tactic = "single_15m"
            if recommended_tactic == "single_15m" and not model_15m_path and model_1h_path:
                recommended_tactic = "single_1h"
            if recommended_tactic == "mtf" and not (model_1h_path and model_15m_path):
                recommended_tactic = "single_15m" if model_15m_path else "single_1h"

            if recommended_tactic == "single_1h":
                model_path = model_1h_path
                if not model_path:
                    raise HTTPException(status_code=400, detail="No 1h model available for single_1h tactic")
                if not model_manager:
                    raise HTTPException(status_code=501, detail="Model manager not available")
                model_manager.apply_model(symbol, model_path)
                state.symbol_models[symbol] = model_path
                config = {"mode": "single", "model_path": model_path}
                state.set_strategy_config(symbol, config)
                experiment["applied_mode"] = "single_1h"
                experiment["applied_models"] = {"single": model_path}
                logger.info(f"Applied single_1h experiment {body.experiment_id} for {symbol}: model={model_path}")
            elif recommended_tactic == "single_15m":
                model_path = model_15m_path
                if not model_path:
                    raise HTTPException(status_code=400, detail="No 15m model available for single_15m tactic")
                if not model_manager:
                    raise HTTPException(status_code=501, detail="Model manager not available")
                model_manager.apply_model(symbol, model_path)
                state.symbol_models[symbol] = model_path
                config = {"mode": "single", "model_path": model_path}
                state.set_strategy_config(symbol, config)
                experiment["applied_mode"] = "single_15m"
                experiment["applied_models"] = {"single": model_path}
                logger.info(f"Applied single_15m experiment {body.experiment_id} for {symbol}: model={model_path}")
            elif model_1h_path and model_15m_path:
                # MTF стратегия
                config = {
                    "mode": "mtf",
                    "model_1h_path": model_1h_path,
                    "model_15m_path": model_15m_path
                }
                state.set_strategy_config(symbol, config)
                experiment["applied_mode"] = "mtf"
                experiment["applied_models"] = {"1h": model_1h_path, "15m": model_15m_path}
                logger.info(f"Applied MTF experiment {body.experiment_id} for {symbol}: 1h={model_1h_path}, 15m={model_15m_path}")
            else:
                # Single стратегия (используем 15m модель, если есть)
                model_path = model_15m_path or model_1h_path
                if not model_path:
                    raise HTTPException(status_code=400, detail="No valid model path found")
                
                if not model_manager:
                    raise HTTPException(status_code=501, detail="Model manager not available")
                
                model_manager.apply_model(symbol, model_path)
                state.symbol_models[symbol] = model_path
                config = {"mode": "single", "model_path": model_path}
                state.set_strategy_config(symbol, config)
                experiment["applied_mode"] = "single"
                experiment["applied_models"] = {"single": model_path}
                logger.info(f"Applied single experiment {body.experiment_id} for {symbol}: model={model_path}")
            
            # Обновляем эксперимент с пометкой о применении
            experiment["applied"] = True
            experiment["applied_at"] = datetime.now().isoformat()
            
            # Сохраняем обновленный эксперимент
            file_path = Path(__file__).resolve().parent.parent / "experiments.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data[body.experiment_id] = experiment
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
            
            if tg_bot:
                await tg_bot.send_notification(f"🔄 Experiment applied for {symbol}")
            
            return {"ok": True, "symbol": symbol, "experiment_id": body.experiment_id}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to apply experiment: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # --- Paper Trading Endpoints ---
    class PaperStartBody(BaseModel):
        experiment_id: str

    @app.post("/api/paper/start", dependencies=[Depends(verify_api_key)])
    async def post_paper_start(body: PaperStartBody):
        """Start paper trading session for an experiment."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        try:
            session = paper_trading_manager.start_session(body.experiment_id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Failed to start paper trading session for experiment {body.experiment_id}")
            
            if tg_bot:
                await tg_bot.send_notification(f"📊 Paper trading started for experiment {body.experiment_id}")
            
            return {"ok": True, "experiment_id": body.experiment_id, "symbol": session.symbol}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start paper trading: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    class PaperStopBody(BaseModel):
        experiment_id: str

    @app.post("/api/paper/stop", dependencies=[Depends(verify_api_key)])
    async def post_paper_stop(body: PaperStopBody):
        """Stop paper trading session for an experiment."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        try:
            success = paper_trading_manager.stop_session(body.experiment_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Paper trading session not found for experiment {body.experiment_id}")
            
            if tg_bot:
                await tg_bot.send_notification(f"📊 Paper trading stopped for experiment {body.experiment_id}")
            
            return {"ok": True, "experiment_id": body.experiment_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to stop paper trading: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/status", dependencies=[Depends(verify_api_key)])
    def get_paper_status():
        """Get status of all active paper trading sessions."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        try:
            sessions = paper_trading_manager.get_all_sessions_status()
            return {"sessions": sessions}
        except Exception as e:
            logger.error(f"Failed to get paper trading status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/trades", dependencies=[Depends(verify_api_key)])
    def get_paper_trades(experiment_id: str = None, limit: int = 100):
        """Get paper trading trades for a specific experiment."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id parameter is required")
        
        try:
            trades = paper_trading_manager.get_trades(experiment_id, limit)
            return {"experiment_id": experiment_id, "trades": trades}
        except Exception as e:
            logger.error(f"Failed to get paper trading trades: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/metrics", dependencies=[Depends(verify_api_key)])
    def get_paper_metrics(experiment_id: str = None):
        """Get paper trading metrics for a specific experiment."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id parameter is required")
        
        try:
            metrics = paper_trading_manager.get_metrics(experiment_id)
            if not metrics:
                raise HTTPException(status_code=404, detail=f"Paper trading session not found for experiment {experiment_id}")
            
            return metrics
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get paper trading metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/realtime_chart", dependencies=[Depends(verify_api_key)])
    def get_paper_realtime_chart(experiment_id: str = None):
        """Get real-time chart data for paper trading."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id parameter is required")
        
        try:
            chart_data = paper_trading_manager.get_realtime_chart_data(experiment_id)
            if not chart_data:
                raise HTTPException(status_code=404, detail=f"Paper trading session not found for experiment {experiment_id}")
            
            return chart_data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get real-time chart data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/chart", dependencies=[Depends(verify_api_key)])
    def get_paper_chart(experiment_id: str = None, days: int = 7):
        """Get chart data for paper trading comparison."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id parameter is required")
        
        try:
            # Get paper trading metrics
            paper_metrics = paper_trading_manager.get_metrics(experiment_id)
            if not paper_metrics:
                raise HTTPException(status_code=404, detail=f"Paper trading session not found for experiment {experiment_id}")
            
            # Get working model metrics from state
            working_metrics = {}
            if trading_loop and hasattr(trading_loop, 'state'):
                state = trading_loop.state
                symbol = paper_metrics.get('symbol')
                if symbol and hasattr(state, 'trades'):
                    from datetime import datetime, timedelta
                    week_ago = datetime.now() - timedelta(days=days)
                    week_trades = []
                    for trade in state.trades:
                        if trade.symbol == symbol and trade.status == "closed" and trade.exit_time:
                            try:
                                exit_dt = datetime.fromisoformat(trade.exit_time)
                                if exit_dt >= week_ago:
                                    week_trades.append(trade)
                            except (ValueError, TypeError):
                                pass
                    
                    if week_trades:
                        # Calculate equity curve for working model
                        equity = 10000.0  # Starting balance
                        equity_curve = [equity]
                        timestamps = [datetime.now()]
                        
                        for trade in week_trades:
                            equity += trade.pnl_usd
                            equity_curve.append(equity)
                            timestamps.append(datetime.fromisoformat(trade.exit_time))
                        
                        working_metrics = {
                            "equity_curve": equity_curve,
                            "timestamps": [ts.isoformat() for ts in timestamps],
                            "total_pnl": sum(t.pnl_usd for t in week_trades),
                            "win_rate": (len([t for t in week_trades if t.pnl_usd > 0]) / len(week_trades) * 100) if week_trades else 0,
                            "total_trades": len(week_trades)
                        }
            
            # Prepare chart data
            chart_data = {
                "experiment_id": experiment_id,
                "symbol": paper_metrics.get('symbol'),
                "paper_trading": {
                    "equity_curve": paper_metrics.get('metrics', {}).get('equity_curve', []),
                    "timestamps": paper_metrics.get('metrics', {}).get('timestamps', []),
                    "total_pnl": paper_metrics.get('metrics', {}).get('total_pnl', 0),
                    "win_rate": paper_metrics.get('metrics', {}).get('win_rate', 0),
                    "total_trades": paper_metrics.get('metrics', {}).get('total_trades', 0)
                },
                "working_model": working_metrics,
                "comparison": {
                    "paper_pnl": paper_metrics.get('metrics', {}).get('total_pnl', 0),
                    "working_pnl": working_metrics.get('total_pnl', 0),
                    "paper_winrate": paper_metrics.get('metrics', {}).get('win_rate', 0),
                    "working_winrate": working_metrics.get('win_rate', 0),
                    "improvement_pnl": paper_metrics.get('metrics', {}).get('total_pnl', 0) - working_metrics.get('total_pnl', 0),
                    "improvement_winrate": paper_metrics.get('metrics', {}).get('win_rate', 0) - working_metrics.get('win_rate', 0)
                }
            }
            
            return chart_data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get chart data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/paper/final_stats", dependencies=[Depends(verify_api_key)])
    def get_paper_final_stats(experiment_id: str = None):
        """Get final statistics after paper trading tests."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id parameter is required")
        
        try:
            # Get paper trading metrics
            paper_metrics = paper_trading_manager.get_metrics(experiment_id)
            if not paper_metrics:
                raise HTTPException(status_code=404, detail=f"Paper trading session not found for experiment {experiment_id}")
            
            # Get session status
            session_status = paper_trading_manager.get_session_status(experiment_id)
            
            # Prepare final statistics
            final_stats = {
                "experiment_id": experiment_id,
                "symbol": paper_metrics.get('symbol'),
                "session_status": session_status,
                "paper_trading_metrics": paper_metrics.get('metrics', {}),
                "summary": {
                    "total_trades": paper_metrics.get('metrics', {}).get('total_trades', 0),
                    "win_rate": paper_metrics.get('metrics', {}).get('win_rate', 0),
                    "total_pnl": paper_metrics.get('metrics', {}).get('total_pnl', 0),
                    "total_pnl_pct": paper_metrics.get('metrics', {}).get('total_pnl_pct', 0),
                    "profit_factor": paper_metrics.get('metrics', {}).get('profit_factor', 0),
                    "max_drawdown_pct": paper_metrics.get('metrics', {}).get('max_drawdown_pct', 0),
                    "sharpe_ratio": paper_metrics.get('metrics', {}).get('sharpe_ratio', 0),
                    "avg_win": paper_metrics.get('metrics', {}).get('avg_win', 0),
                    "avg_loss": paper_metrics.get('metrics', {}).get('avg_loss', 0),
                    "consecutive_wins": paper_metrics.get('metrics', {}).get('consecutive_wins', 0),
                    "consecutive_losses": paper_metrics.get('metrics', {}).get('consecutive_losses', 0),
                    "largest_win": paper_metrics.get('metrics', {}).get('largest_win', 0),
                    "largest_loss": paper_metrics.get('metrics', {}).get('largest_loss', 0),
                },
                "recommendation": "replace" if paper_metrics.get('metrics', {}).get('total_pnl', 0) > 0 else "discard"
            }
            
            return final_stats
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get final statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    class ReplaceModelBody(BaseModel):
        experiment_id: str
        symbol: str

    @app.post("/api/paper/replace_model", dependencies=[Depends(verify_api_key)])
    async def post_replace_model(body: ReplaceModelBody):
        """Replace working model with experimental model."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        try:
            # Load experiment data
            experiment_data = paper_trading_manager.load_experiment(body.experiment_id)
            if not experiment_data:
                raise HTTPException(status_code=404, detail=f"Experiment {body.experiment_id} not found")
            
            # Get model paths
            models = experiment_data.get('results', {}).get('models', {})
            model_15m = models.get('15m')
            model_1h = models.get('1h')
            
            if not model_15m and not model_1h:
                raise HTTPException(status_code=400, detail="No valid models found in experiment")
            
            # Update the working model configuration
            # This would typically involve updating the settings or state
            # For now, we'll just return success and log the action
            logger.info(f"Replacing model for {body.symbol} with experiment {body.experiment_id}")
            logger.info(f"  15m model: {model_15m}")
            logger.info(f"  1h model: {model_1h}")
            
            # In a real implementation, you would:
            # 1. Update the trading loop's strategy
            # 2. Save the new configuration
            # 3. Restart the trading loop if needed
            
            if tg_bot:
                await tg_bot.send_notification(f"🔄 Model replacement requested for {body.symbol} with experiment {body.experiment_id}")
            
            return {
                "ok": True,
                "experiment_id": body.experiment_id,
                "symbol": body.symbol,
                "models": models,
                "message": "Model replacement requested. In a full implementation, this would update the working model configuration."
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to replace model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # REST API endpoint for real-time chart updates
    @app.get("/api/paper/realtime_chart/{experiment_id}", dependencies=[Depends(verify_api_key)])
    async def get_paper_realtime_chart(experiment_id: str):
        """Get real-time chart data for paper trading session."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        try:
            chart_data = paper_trading_manager.get_realtime_chart_data(experiment_id)
            if not chart_data:
                raise HTTPException(status_code=404, detail=f"No chart data found for experiment {experiment_id}")
            
            return chart_data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get realtime chart data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Get bot settings and current strategy for realistic simulation
    @app.get("/api/bot/settings", dependencies=[Depends(verify_api_key)])
    def get_bot_settings():
        """Get current bot settings for realistic paper trading simulation."""
        if not trading_loop:
            raise HTTPException(status_code=501, detail="Trading loop not available")
        
        try:
            settings = trading_loop.settings
            state = trading_loop.state
            
            # Get current strategy configuration
            strategy_config = {}
            if hasattr(settings, 'ml_strategy'):
                strategy_config = {
                    "use_mtf_strategy": settings.ml_strategy.use_mtf_strategy,
                    "confidence_threshold": settings.ml_strategy.confidence_threshold,
                    "min_signal_strength": settings.ml_strategy.min_signal_strength,
                    "mtf_confidence_threshold_1h": settings.ml_strategy.mtf_confidence_threshold_1h,
                    "mtf_confidence_threshold_15m": settings.ml_strategy.mtf_confidence_threshold_15m,
                    "mtf_alignment_mode": settings.ml_strategy.mtf_alignment_mode,
                    "mtf_require_alignment": settings.ml_strategy.mtf_require_alignment,
                    "follow_btc_filter_enabled": getattr(settings.ml_strategy, "follow_btc_filter_enabled", True),
                    "follow_btc_override_confidence": getattr(settings.ml_strategy, "follow_btc_override_confidence", 0.80),
                }
            
            # Get risk management settings
            risk_config = {}
            if hasattr(settings, 'risk'):
                risk_config = {
                    "base_order_usd": settings.risk.base_order_usd,
                    "max_position_usd": settings.risk.max_position_usd,
                    "reverse_min_confidence": settings.risk.reverse_min_confidence,
                }
            
            # Get current balance and positions
            current_balance = state.balance if hasattr(state, 'balance') else 10000.0
            current_positions = []
            if hasattr(state, 'positions'):
                for pos in state.positions:
                    current_positions.append({
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                    })
            
            return {
                "ok": True,
                "settings": {
                    "strategy": strategy_config,
                    "risk": risk_config,
                    "current_balance": current_balance,
                    "current_positions": current_positions,
                    "is_running": state.is_running if hasattr(state, 'is_running') else False,
                }
            }
        except Exception as e:
            logger.error(f"Failed to get bot settings: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Get real-time trading data for comparison
    @app.post("/api/bot/update_settings", dependencies=[Depends(verify_api_key)])
    async def update_bot_settings():
        """Update bot settings in paper trading manager."""
        if not paper_trading_manager:
            raise HTTPException(status_code=501, detail="Paper trading manager not available")
        
        if not trading_loop:
            raise HTTPException(status_code=501, detail="Trading loop not available")
        
        try:
            # Get current settings from trading loop
            settings = trading_loop.settings
            state = trading_loop.state
            
            bot_settings = {
                "settings": {
                    "strategy": {
                        "use_mtf_strategy": settings.ml_strategy.use_mtf_strategy,
                        "confidence_threshold": settings.ml_strategy.confidence_threshold,
                        "min_signal_strength": settings.ml_strategy.min_signal_strength,
                        "mtf_confidence_threshold_1h": settings.ml_strategy.mtf_confidence_threshold_1h,
                        "mtf_confidence_threshold_15m": settings.ml_strategy.mtf_confidence_threshold_15m,
                        "mtf_alignment_mode": settings.ml_strategy.mtf_alignment_mode,
                        "mtf_require_alignment": settings.ml_strategy.mtf_require_alignment,
                        "follow_btc_filter_enabled": getattr(settings.ml_strategy, "follow_btc_filter_enabled", True),
                        "follow_btc_override_confidence": getattr(settings.ml_strategy, "follow_btc_override_confidence", 0.80),
                    },
                    "risk": {
                        "base_order_usd": settings.risk.base_order_usd,
                        "max_position_usd": settings.risk.max_position_usd,
                        "reverse_min_confidence": settings.risk.reverse_min_confidence,
                    },
                    "current_balance": state.balance if hasattr(state, 'balance') else 10000.0,
                    "is_running": state.is_running if hasattr(state, 'is_running') else False,
                }
            }
            
            # Update paper trading manager settings
            paper_trading_manager.update_settings(bot_settings)
            
            return {"ok": True, "message": "Settings updated successfully"}
        except Exception as e:
            logger.error(f"Failed to update settings: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/bot/realtime_data", dependencies=[Depends(verify_api_key)])
    def get_realtime_data(symbol: str = None):
        """Get real-time trading data for comparison with paper trading."""
        if not trading_loop:
            raise HTTPException(status_code=501, detail="Trading loop not available")
        
        try:
            # Get recent trades for the symbol
            recent_trades = []
            if hasattr(trading_loop.state, 'trades'):
                from datetime import datetime, timedelta
                hour_ago = datetime.now() - timedelta(hours=1)
                
                for trade in trading_loop.state.trades:
                    if symbol and trade.symbol != symbol:
                        continue
                    
                    try:
                        exit_time = datetime.fromisoformat(trade.exit_time) if trade.exit_time else None
                        if exit_time and exit_time >= hour_ago:
                            recent_trades.append({
                                "symbol": trade.symbol,
                                "action": trade.action,
                                "entry_price": trade.entry_price,
                                "exit_price": trade.exit_price,
                                "pnl_usd": trade.pnl_usd,
                                "pnl_percent": trade.pnl_percent,
                                "entry_time": trade.entry_time,
                                "exit_time": trade.exit_time,
                            })
                    except (ValueError, TypeError):
                        pass
            
            # Calculate equity curve for real trading
            equity_curve = []
            timestamps = []
            if recent_trades:
                equity = 10000.0  # Starting balance
                equity_curve.append(equity)
                timestamps.append(datetime.now().isoformat())
                
                for trade in recent_trades:
                    equity += trade["pnl_usd"]
                    equity_curve.append(equity)
                    timestamps.append(trade.get("exit_time", datetime.now().isoformat()))
            
            return {
                "ok": True,
                "symbol": symbol,
                "recent_trades": recent_trades,
                "equity_curve": equity_curve,
                "timestamps": timestamps,
                "total_trades": len(recent_trades),
                "total_pnl": sum(t["pnl_usd"] for t in recent_trades),
                "win_rate": (len([t for t in recent_trades if t["pnl_usd"] > 0]) / len(recent_trades) * 100) if recent_trades else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get realtime data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def run_api_server(state, bybit_client, settings, trading_loop=None, model_manager=None, tg_bot=None, host: str = "0.0.0.0", port: int = 8765):
    """Запускает API сервер в текущем event loop (для asyncio.gather с ботом)."""
    logger.info(f"[Mobile API] run_api_server вызван: host={host}, port={port}")
    try:
        import uvicorn
        from uvicorn import Config, Server
        logger.info("[Mobile API] uvicorn импортирован")
    except ImportError as e:
        logger.warning(f"[Mobile API] uvicorn не установлен: {e}. Установите: pip install uvicorn")
        return

    try:
        logger.info("[Mobile API] Создание FastAPI приложения...")
        app = create_app(state, bybit_client, settings, trading_loop, model_manager, tg_bot)
        logger.info("[Mobile API] Config и Server...")
        config = Config(app=app, host=host, port=port, log_level="info")
        server = Server(config)
        logger.info(f"[Mobile API] Запуск Uvicorn на http://{host}:{port} (снаружи: http://5.101.179.47:{port}/api/health)")
        await server.serve()
        logger.info("[Mobile API] server.serve() завершён")
    except asyncio.CancelledError:
        logger.info("[Mobile API] Сервер остановлен (CancelledError)")
        raise
    except Exception as e:
        logger.error(f"[Mobile API] Ошибка при запуске/работе сервера: {e}", exc_info=True)
        raise
