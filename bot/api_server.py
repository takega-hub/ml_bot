"""
REST API для мобильного приложения (iPhone).
Полный функционал как в Telegram-админке: статус, пары, риск, ML, модели, история, экстренные действия.
Аутентификация: заголовок X-API-Key (MOBILE_API_KEY в .env).
"""
import json
import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
                    leverage = _safe_float(p.get("leverage", settings.leverage), settings.leverage)
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
        if trading_loop and getattr(trading_loop, "strategies", None):
            strategy = trading_loop.strategies.get(symbol)
            if strategy and getattr(strategy, "predict_combined", None):
                strategy_info["mtf"] = True
                strategy_info["model_1h"] = getattr(strategy, "model_1h_path", None)
                strategy_info["model_15m"] = getattr(strategy, "model_15m_path", None)
            elif strategy:
                strategy_info["model"] = getattr(strategy, "model_path", None) or state.symbol_models.get(symbol)
        else:
            strategy_info["model"] = state.symbol_models.get(symbol)
        cooldown = state.get_cooldown_info(symbol) if hasattr(state, "get_cooldown_info") else None
        if cooldown and cooldown.get("active"):
            strategy_info["cooldown"] = {"hours_left": cooldown.get("hours_left"), "reason": cooldown.get("reason", "")}
        strategies.append(strategy_info)

    stats = state.get_stats()

    return {
        "is_running": state.is_running,
        "wallet_balance": round(wallet_balance, 2),
        "available_balance": round(available, 2),
        "total_margin": round(total_margin, 2),
        "positions": open_positions,
        "strategies": strategies,
        "active_symbols": list(state.active_symbols),
        "total_pnl": round(stats["total_pnl"], 2),
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


def create_app(state, bybit_client, settings, trading_loop=None, model_manager=None):
    """Создаёт FastAPI приложение с инжектированными зависимостями."""
    try:
        from fastapi import FastAPI, Depends, HTTPException, Header, Body
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Установите fastapi и uvicorn: pip install fastapi uvicorn")

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

    def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if not api_key:
            raise HTTPException(status_code=500, detail="MOBILE_API_KEY not configured")
        if not x_api_key or x_api_key.strip() != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/status", dependencies=[Depends(verify_api_key)])
    def get_status():
        return _get_status_data(state, bybit_client, settings, trading_loop)

    @app.get("/api/dashboard", dependencies=[Depends(verify_api_key)])
    def get_dashboard():
        return _get_dashboard_data(state, bybit_client, settings)

    @app.post("/api/start", dependencies=[Depends(verify_api_key)])
    def post_start():
        state.set_running(True)
        return {"ok": True, "is_running": True}

    @app.post("/api/stop", dependencies=[Depends(verify_api_key)])
    def post_stop():
        state.set_running(False)
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
        """Список известных и активных пар с cooldown."""
        all_possible = sorted(set(
            s for s in state.known_symbols
            if isinstance(s, str) and s.endswith("USDT")
        ) + list(state.active_symbols))
        cooldowns = {}
        for s in all_possible:
            info = state.get_cooldown_info(s) if hasattr(state, "get_cooldown_info") else None
            if info and info.get("active"):
                cooldowns[s] = {"hours_left": info.get("hours_left"), "reason": info.get("reason", "")}
        return {
            "known_symbols": list(state.known_symbols),
            "active_symbols": list(state.active_symbols),
            "max_active": getattr(state, "max_active_symbols", 5),
            "cooldowns": cooldowns,
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

    # --- Risk (read + write to file and in-memory) ---
    def _risk_to_dict(r):
        return {
            "margin_pct_balance": r.margin_pct_balance,
            "base_order_usd": r.base_order_usd,
            "stop_loss_pct": r.stop_loss_pct,
            "take_profit_pct": r.take_profit_pct,
            "enable_trailing_stop": r.enable_trailing_stop,
            "trailing_stop_activation_pct": r.trailing_stop_activation_pct,
            "trailing_stop_distance_pct": r.trailing_stop_distance_pct,
            "enable_partial_close": r.enable_partial_close,
            "enable_breakeven": r.enable_breakeven,
            "breakeven_level1_activation_pct": r.breakeven_level1_activation_pct,
            "breakeven_level1_sl_pct": r.breakeven_level1_sl_pct,
            "breakeven_level2_activation_pct": r.breakeven_level2_activation_pct,
            "breakeven_level2_sl_pct": r.breakeven_level2_sl_pct,
            "enable_loss_cooldown": r.enable_loss_cooldown,
            "fee_rate": r.fee_rate,
            "mid_term_tp_pct": r.mid_term_tp_pct,
            "long_term_tp_pct": r.long_term_tp_pct,
            "long_term_sl_pct": r.long_term_sl_pct,
            "long_term_ignore_reverse": r.long_term_ignore_reverse,
            "dca_enabled": r.dca_enabled,
            "dca_drawdown_pct": r.dca_drawdown_pct,
            "dca_max_adds": r.dca_max_adds,
            "dca_min_confidence": r.dca_min_confidence,
            "reverse_on_strong_signal": r.reverse_on_strong_signal,
            "reverse_min_confidence": r.reverse_min_confidence,
            "reverse_min_strength": getattr(r, "reverse_min_strength", "сильное"),
        }

    @app.get("/api/risk", dependencies=[Depends(verify_api_key)])
    def get_risk():
        return _risk_to_dict(settings.risk)

    @app.put("/api/risk", dependencies=[Depends(verify_api_key)])
    def put_risk(body: Dict[str, Any] = Body(...)):
        risk = settings.risk
        for key, val in body.items():
            if hasattr(risk, key):
                if key in ("margin_pct_balance", "stop_loss_pct", "take_profit_pct", "trailing_stop_activation_pct",
                           "trailing_stop_distance_pct", "breakeven_level1_activation_pct", "breakeven_level1_sl_pct",
                           "breakeven_level2_activation_pct", "breakeven_level2_sl_pct", "fee_rate",
                           "mid_term_tp_pct", "long_term_tp_pct", "long_term_sl_pct", "dca_drawdown_pct",
                           "dca_min_confidence", "reverse_min_confidence") and isinstance(val, (int, float)):
                    if val >= 1:
                        val = val / 100.0
                setattr(risk, key, val)
        risk_file = PROJECT_ROOT / "risk_settings.json"
        with open(risk_file, "w", encoding="utf-8") as f:
            json.dump(_risk_to_dict(risk), f, indent=2, ensure_ascii=False)
        return {"ok": True, "risk": _risk_to_dict(risk)}

    # --- ML settings (read + write) ---
    def _ml_to_dict(m):
        return {
            "use_mtf_strategy": getattr(m, "use_mtf_strategy", False),
            "mtf_confidence_threshold_1h": getattr(m, "mtf_confidence_threshold_1h", 0.5),
            "mtf_confidence_threshold_15m": getattr(m, "mtf_confidence_threshold_15m", 0.35),
            "mtf_alignment_mode": getattr(m, "mtf_alignment_mode", "strict"),
            "atr_filter_enabled": getattr(m, "atr_filter_enabled", False),
            "auto_optimize_strategies": getattr(m, "auto_optimize_strategies", False),
            "auto_optimize_day": getattr(m, "auto_optimize_day", "sunday"),
            "auto_optimize_hour": getattr(m, "auto_optimize_hour", 3),
            "use_fixed_sl_from_risk": getattr(m, "use_fixed_sl_from_risk", False),
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
        for key in ("use_mtf_strategy", "atr_filter_enabled", "auto_optimize_strategies", "use_fixed_sl_from_risk"):
            if key in body:
                data[key] = bool(body[key])
        for key in ("mtf_confidence_threshold_1h", "mtf_confidence_threshold_15m", "confidence_threshold", "min_confidence_for_trade"):
            if key in body:
                v = float(body[key])
                if v >= 1:
                    v = v / 100.0
                data[key] = v
                setattr(m, key, v)
        for key in ("mtf_alignment_mode", "auto_optimize_day"):
            if key in body:
                data[key] = body[key]
                setattr(m, key, body[key])
        if "auto_optimize_hour" in body:
            data["auto_optimize_hour"] = int(body["auto_optimize_hour"])
            m.auto_optimize_hour = data["auto_optimize_hour"]
        for k, v in data.items():
            if hasattr(m, k):
                if k in ("confidence_threshold", "min_confidence_for_trade", "mtf_confidence_threshold_1h", "mtf_confidence_threshold_15m") and isinstance(v, (int, float)) and v >= 1:
                    v = v / 100.0
                setattr(m, k, v)
        with open(ml_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"ok": True, "ml": _ml_to_dict(m)}

    # --- Models ---
    @app.get("/api/models", dependencies=[Depends(verify_api_key)])
    def get_models_list():
        """Текущие модели по активным парам."""
        out = []
        for symbol in state.active_symbols:
            path = state.symbol_models.get(symbol)
            name = Path(path).stem if path and Path(path).exists() else None
            out.append({"symbol": symbol, "model_path": path, "model_name": name})
        return {"symbols": out}

    @app.get("/api/models/{symbol}", dependencies=[Depends(verify_api_key)])
    def get_models_for_symbol(symbol: str):
        symbol = symbol.upper()
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")
        models = model_manager.find_models_for_symbol(symbol)
        results = model_manager.get_model_test_results(symbol)
        current = state.symbol_models.get(symbol)
        list_models = []
        for i, mp in enumerate(models):
            mp_str = str(mp)
            res = results.get(mp_str, {})
            list_models.append({
                "index": i,
                "path": mp_str,
                "name": Path(mp).stem,
                "current": mp_str == current,
                "test": res,
            })
        return {"symbol": symbol, "models": list_models, "current": current}

    class ApplyModelBody(BaseModel):
        model_path: str

    @app.post("/api/models/{symbol}/apply", dependencies=[Depends(verify_api_key)])
    def post_apply_model(symbol: str, body: ApplyModelBody):
        symbol = symbol.upper()
        if not model_manager:
            raise HTTPException(status_code=501, detail="Model manager not available")
        model_manager.apply_model(symbol, body.model_path)
        return {"ok": True, "symbol": symbol, "model_path": body.model_path}

    class RetrainBody(BaseModel):
        symbol: str

    @app.post("/api/models/retrain", dependencies=[Depends(verify_api_key)])
    def post_retrain(body: RetrainBody):
        """Запуск переобучения в фоне (subprocess)."""
        sym = body.symbol.upper()
        script = PROJECT_ROOT / "retrain_ml_optimized.py"
        if not script.exists():
            raise HTTPException(status_code=501, detail="retrain_ml_optimized.py not found")
        try:
            subprocess.Popen(
                [sys.executable, str(script), "--symbol", sym],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.exception("Retrain start failed")
            raise HTTPException(status_code=500, detail=str(e))
        return {"ok": True, "symbol": sym, "message": "Retrain started in background"}

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
        """log_type: bot, trades, signals, errors."""
        path_map = {"bot": "logs/bot.log", "trades": "logs/trades.log", "signals": "logs/signals.log", "errors": "logs/errors.log"}
        path = PROJECT_ROOT / path_map.get(log_type, path_map["bot"])
        if not path.exists():
            return {"lines": [], "path": str(path)}
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.readlines()
            last = raw[-lines:] if len(raw) > lines else raw
            return {"lines": [l.rstrip() for l in last], "path": str(path)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

    # --- Emergency ---
    @app.post("/api/emergency/stop_all", dependencies=[Depends(verify_api_key)])
    def post_emergency_stop_all():
        """Остановить бота и закрыть все позиции."""
        state.set_running(False)
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

    return app


async def run_api_server(state, bybit_client, settings, trading_loop=None, model_manager=None, host: str = "0.0.0.0", port: int = 8765):
    """Запускает API сервер в текущем event loop (для asyncio.gather с ботом)."""
    try:
        import uvicorn
        from uvicorn import Config, Server
    except ImportError:
        logger.warning("uvicorn not installed. Mobile API disabled. Install: pip install uvicorn")
        return

    app = create_app(state, bybit_client, settings, trading_loop, model_manager)
    config = Config(app=app, host=host, port=port, log_level="warning")
    server = Server(config)
    await server.serve()
