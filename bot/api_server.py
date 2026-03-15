"""
REST API для мобильного приложения (iPhone).
Полный функционал как в Telegram-админке: статус, пары, риск, ML, модели, история, экстренные действия.
Аутентификация: заголовок X-API-Key (MOBILE_API_KEY в .env).
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Any, List, Dict
import shutil
import signal

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
                    if unrealised_pnl == 0:
                        # Иногда Bybit не возвращает unrealisedPnl сразу, считаем сами
                        if side == "Buy":
                            unrealised_pnl = (mark_price - entry_price) * size
                        else:
                            unrealised_pnl = (entry_price - mark_price) * size
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
        from fastapi import FastAPI, Depends, HTTPException, Header, Body, BackgroundTasks
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
        # Capture old settings for history tracking
        old_settings = _risk_to_dict(settings.risk)
        
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
        
        # Capture new settings
        new_settings = _risk_to_dict(risk)
        
        risk_file = PROJECT_ROOT / "risk_settings.json"
        with open(risk_file, "w", encoding="utf-8") as f:
            json.dump(new_settings, f, indent=2, ensure_ascii=False)
            
        # Notify AI Agent about changes
        try:
             # Get total trade count to track observation period
             total_trades = len(state.trades)
             # Pass old and new settings to record history
             ai_agent.on_risk_settings_updated(total_trades, old_settings, new_settings)
        except Exception as e:
             logger.error(f"Failed to notify AI agent about risk update: {e}")
             
        return {"ok": True, "risk": new_settings}

    @app.get("/api/ai/risk_history", dependencies=[Depends(verify_api_key)])
    def get_ai_risk_history():
        """Returns the history of risk setting changes."""
        return {"history": ai_agent.get_risk_history()}

    # --- ML settings (read + write) ---
    def _ml_to_dict(m):
        return {
            "use_mtf_strategy": getattr(m, "use_mtf_strategy", False),
            "mtf_confidence_threshold_1h": getattr(m, "mtf_confidence_threshold_1h", 0.5),
            "mtf_confidence_threshold_15m": getattr(m, "mtf_confidence_threshold_15m", 0.35),
            "mtf_alignment_mode": getattr(m, "mtf_alignment_mode", "strict"),
            "atr_filter_enabled": getattr(m, "atr_filter_enabled", False),
            "follow_btc_filter_enabled": getattr(m, "follow_btc_filter_enabled", True),
            "follow_btc_override_confidence": getattr(m, "follow_btc_override_confidence", 0.80),
            "auto_optimize_strategies": getattr(m, "auto_optimize_strategies", False),
            "auto_optimize_day": getattr(m, "auto_optimize_day", "sunday"),
            "auto_optimize_hour": getattr(m, "auto_optimize_hour", 3),
            "use_fixed_sl_from_risk": getattr(m, "use_fixed_sl_from_risk", False),
            "ai_entry_confirmation_enabled": getattr(m, "ai_entry_confirmation_enabled", False),
            "ai_fallback_force_enabled": getattr(m, "ai_fallback_force_enabled", False),
            "ai_fallback_spread_reduce_pct": getattr(m, "ai_fallback_spread_reduce_pct", 0.10),
            "ai_fallback_spread_veto_pct": getattr(m, "ai_fallback_spread_veto_pct", 0.25),
            "ai_fallback_min_depth_usd_5": getattr(m, "ai_fallback_min_depth_usd_5", 0.0),
            "ai_fallback_imbalance_abs_reduce": getattr(m, "ai_fallback_imbalance_abs_reduce", 0.60),
            "ai_fallback_orderflow_ratio_low": getattr(m, "ai_fallback_orderflow_ratio_low", 0.40),
            "ai_fallback_orderflow_ratio_high": getattr(m, "ai_fallback_orderflow_ratio_high", 2.50),
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
            "follow_btc_filter_enabled",
            "auto_optimize_strategies",
            "use_fixed_sl_from_risk",
            "ai_entry_confirmation_enabled",
            "ai_fallback_force_enabled",
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
        ):
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
                                mult = resp.get("size_multiplier")
                                codes = resp.get("reason_codes")
                                latency = resp.get("latency_ms")
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
                    except Exception:
                        pass
                    out.append(s[:2000])
                return {"lines": out, "path": str(path)}
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

    @app.post("/api/ai/chat", dependencies=[Depends(verify_api_key)])
    async def post_chat(body: ChatBody):
        """
        Эндпоинт для чата с AI-агентом.
        Агент получает доступ к логам, состоянию и настройкам.
        """
        try:
            # Read last 20 lines of main log
            log_lines = []
            log_path = PROJECT_ROOT / "logs" / "bot.log"
            if log_path.exists():
                try:
                    # Efficiently read last lines
                    import collections
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_lines = collections.deque(f, maxlen=20)
                    log_lines = list(log_lines)
                except Exception:
                    pass
            
            # Collect context
            from dataclasses import asdict
            active_positions_count = len([t for t in state.trades if t.status == "open"])
            
            # BotState does not track open orders directly in this version
            open_orders_count = 0 
            
            recent_trades_data = []
            for t in state.trades[-5:]:
                try:
                    # Use asdict for dataclasses
                    recent_trades_data.append(asdict(t))
                except Exception:
                    # Fallback if asdict fails or it's not a dataclass
                    recent_trades_data.append(str(t))

            context = {
                "risk_settings": _risk_to_dict(settings.risk),
                "active_positions": active_positions_count,
                "open_orders": open_orders_count,
                "recent_trades": recent_trades_data,
                "bot_status": "RUNNING" if state.is_running else "STOPPED",
                "last_notification": asdict(state.notifications[-1]) if state.notifications else "None"
            }
            
            response = await ai_agent.chat_with_user(body.message, context, log_lines)
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

        # 1. Try to get from memory state first
        closed_trades = [t for t in state.trades if t.status == "closed"]
        
        # 2. If memory is empty or low, try to parse from logs
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
        
        result = await ai_agent.analyze_risk_settings(trades_data, current_risk)
        return result

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
            ExperimentReportBuilder,
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
        from .experiment_management import (
            ExperimentAnalyzer,
            ExperimentReportBuilder,
            ExperimentStore,
        )

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        analyzer = ExperimentAnalyzer(store.list())
        impact = None
        if exp.get("symbol"):
            impact = analyzer.compute_param_impact(str(exp.get("symbol")))
        md = ExperimentReportBuilder().build_markdown(exp, impact=impact)
        return {"experiment_id": experiment_id, "markdown": md}

    @app.get("/api/ai/research/experiment/{experiment_id}/health", dependencies=[Depends(verify_api_key)])
    def get_research_experiment_health(experiment_id: str):
        from .experiment_management import ExperimentStore
        from datetime import datetime

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
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
            raise HTTPException(status_code=404, detail="Experiment not found")

        status = str(exp.get("status") or "unknown")
        if status in {"completed", "failed", "interrupted"}:
            raise HTTPException(status_code=400, detail=f"Experiment is already {status}")

        pid = exp.get("runner_pid")
        if not isinstance(pid, int):
            raise HTTPException(status_code=400, detail="Experiment PID is missing")

        killed = False
        try:
            proc = getattr(ai_agent, "research_processes", {}).get(experiment_id) if ai_agent else None
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
        try:
            store.upsert(experiment_id, patch)
        except Exception as e:
            logger.error(f"Failed to persist interrupted status for {experiment_id}: {e}", exc_info=True)

        return {"ok": True, "experiment_id": experiment_id, "pid": pid, "killed": killed}

    @app.delete("/api/ai/research/experiment/{experiment_id}", dependencies=[Depends(verify_api_key)])
    def delete_research_experiment(experiment_id: str):
        from .experiment_management import ExperimentStore

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        status = str(exp.get("status") or "unknown")
        if status == "active":
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
                        raise HTTPException(
                            status_code=400,
                            detail=f"Experiment models are currently in use: {', '.join(in_use)}",
                        )

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
            results = experiment.get("results", {})
            models = results.get("models", {})
            model_15m_path = models.get("15m")
            model_1h_path = models.get("1h")
            recommended_tactic = results.get("recommended_tactic")
            
            if not model_15m_path and not model_1h_path:
                raise HTTPException(status_code=400, detail="Experiment has no models")
            
            # Применяем стратегию
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
