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

    from bot.ai_agent_service import AIAgentService
    ai_agent = AIAgentService()

    paper_trading_manager = getattr(trading_loop, 'paper_trading_manager', None) if trading_loop else None

    app = FastAPI(title="ML Bot Mobile API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    MOBILE_API_KEY = os.getenv("MOBILE_API_KEY") or os.getenv("API_KEY") or ""

    async def verify_api_key(x_api_key: Optional[str] = Header(None)):
        if not MOBILE_API_KEY:
            return True
        if x_api_key != MOBILE_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return True

    class ResearchBody(BaseModel):
        symbol: str
        type: str = "balanced"
        metadata: Optional[Dict[str, Any]] = None
        allow_duplicate: bool = False
        safe_mode: bool = True
        goal: Optional[str] = None
        constraints: Optional[Dict[str, Any]] = None
        budget: Optional[Dict[str, Any]] = None
        auto_iterate: bool = True
        max_steps: int = 3
        campaign_parallel_limit: int = 2

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
                goal=body.goal,
                constraints=body.constraints,
                budget=body.budget,
                auto_iterate=body.auto_iterate,
                max_steps=body.max_steps,
                campaign_parallel_limit=body.campaign_parallel_limit,
            )
            if not res.get("ok"):
                error_msg = res.get("error", "Unknown error")
                logger.error(f"Research start failed for {symbol}: {error_msg}")
                error_detail = {
                    "error": error_msg,
                    "experiment_id": res.get("experiment_id"),
                    "param_signature": res.get("param_signature"),
                    "effective_params": res.get("effective_params"),
                }
                if "Duplicate" in error_msg:
                    raise HTTPException(status_code=409, detail=error_detail)
                if "parallel limit" in error_msg.lower():
                    raise HTTPException(status_code=429, detail=error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if tg_bot:
                await tg_bot.send_notification(f"🧪 Research Experiment ({body.type}) started for {symbol}")

            return res
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Research start endpoint failed: {e}", exc_info=True)
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
        killed_pids: List[int] = []
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

            if action == "stop":
                status = str(exp.get("status") or "")
                if status in {"starting", "training", "training_completed", "backtesting"}:
                    pid = exp.get("runner_pid")
                    if isinstance(pid, int):
                        killed = False
                        proc = getattr(ai_agent, "research_processes", {}).get(eid) if ai_agent else None
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
                            killed = True
                        if not killed:
                            try:
                                os.kill(pid, signal.SIGTERM)
                                killed = True
                            except Exception:
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                    killed = True
                                except Exception:
                                    killed = False
                        if killed:
                            killed_pids.append(pid)
                    exp["status"] = "interrupted"
                    exp["status_reason"] = "campaign_stopped_by_user"
                    exp["stopped_at"] = now_iso

        store.write_all(data)
        return {
            "ok": True,
            "root_experiment_id": root_id,
            "action": action,
            "updated_experiments": changed_ids,
            "killed_pids": killed_pids,
        }

    @app.delete("/api/ai/research/experiment/{experiment_id}", dependencies=[Depends(verify_api_key)])
    def delete_research_experiment(experiment_id: str):
        from .experiment_management import ExperimentStore

        store = ExperimentStore(Path(__file__).resolve().parent.parent / "experiments.json")
        exp = store.get(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        data = store.read_all()
        removed = data.pop(experiment_id, None)
        store.write_all(data)

        removed_paths: List[str] = []
        removed_model_paths: List[str] = []
        if isinstance(removed, dict):
            meta_path = PROJECT_ROOT / "experiment_meta" / f"{experiment_id}.json"
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    removed_paths.append(str(meta_path))
                except Exception:
                    pass
            artifacts_path = PROJECT_ROOT / "artifacts" / experiment_id
            if artifacts_path.exists():
                try:
                    shutil.rmtree(artifacts_path)
                    removed_paths.append(str(artifacts_path))
                except Exception:
                    pass
            models = ((removed.get("results") or {}).get("models") if isinstance(removed.get("results"), dict) else {}) or {}
            for _, model_path in models.items():
                if isinstance(model_path, str) and model_path and Path(model_path).exists():
                    try:
                        Path(model_path).unlink()
                        removed_model_paths.append(model_path)
                    except Exception:
                        pass

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
        from .experiment_management import ExperimentCriteria, compute_unified_score

        experiments = ai_agent.get_research_experiments()

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
            experiment["market_regime"] = (
                results_payload.get("market_regime")
                if isinstance(results_payload.get("market_regime"), dict)
                else experiment.get("market_regime")
            )
            experiment["regime_memory"] = (
                results_payload.get("regime_memory")
                if isinstance(results_payload.get("regime_memory"), dict)
                else experiment.get("regime_memory")
            )
            experiment["regime_memory_summary"] = experiment.get("regime_memory_summary")
            experiment["research_notebook"] = experiment.get("research_notebook")
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
            experiment["risk_profile"] = (
                results_payload.get("risk_profile")
                if isinstance(results_payload.get("risk_profile"), dict)
                else experiment.get("risk_profile")
            )
            experiment["execution_realism"] = (
                results_payload.get("execution_realism")
                if isinstance(results_payload.get("execution_realism"), dict)
                else experiment.get("execution_realism")
            )

            current_strategy = None
            if trading_loop and getattr(trading_loop, "strategies", None):
                current_strategy = trading_loop.strategies.get(symbol)

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

                    current_metrics = {
                        "total_pnl": total_pnl,
                        "total_pnl_pct": total_pnl_pct,
                        "win_rate": (len(winning_trades) / len(week_trades) * 100) if week_trades else 0,
                        "total_trades": len(week_trades),
                        "winning_trades": len(winning_trades),
                        "losing_trades": len(week_trades) - len(winning_trades)
                    }

            experiment_results = experiment.get("results", {})
            experiment_metrics = experiment_results.get("metrics", {})
            if not experiment_metrics:
                experiment_metrics = experiment_results

            experiment["current_strategy"] = {
                "has_strategy": current_strategy is not None,
                "models": current_models,
                "metrics": current_metrics
            }

            experiment_pnl = experiment_metrics.get("total_pnl_pct", 0)
            experiment_winrate = experiment_metrics.get("win_rate", 0)
            current_pnl = current_metrics.get("total_pnl_pct", 0)
            current_winrate = current_metrics.get("win_rate", 0)

            criteria = ExperimentCriteria(min_total_trades=10, min_profit_factor=1.0, min_total_pnl_pct=-1.0, max_drawdown_pct=35.0)
            challenger_eval = compute_unified_score(experiment_metrics if isinstance(experiment_metrics, dict) else {}, criteria=criteria)
            champion_eval = compute_unified_score(current_metrics if isinstance(current_metrics, dict) else {}, criteria=criteria)
            challenger_score = float(challenger_eval.get("score") or 0.0)
            champion_score = float(champion_eval.get("score") or 0.0)
            challenger_gates = bool((challenger_eval.get("gates") or {}).get("passed"))
            champion_gates = bool((champion_eval.get("gates") or {}).get("passed"))
            recommended_tactic = experiment_results.get("recommended_tactic")
            oos_payload = experiment_results.get("oos_validation") if isinstance(experiment_results.get("oos_validation"), dict) else {}
            oos_for_recommended = oos_payload.get(recommended_tactic) if isinstance(oos_payload.get(recommended_tactic), dict) else {}
            oos_eval = oos_for_recommended.get("evaluation") if isinstance(oos_for_recommended.get("evaluation"), dict) else {}
            oos_score = float(oos_eval.get("score") or 0.0) if oos_eval else None
            oos_gates = bool((oos_eval.get("gates") or {}).get("passed")) if oos_eval else None
            wf_payload = experiment_results.get("walk_forward") if isinstance(experiment_results.get("walk_forward"), dict) else {}
            wf_for_recommended = wf_payload.get(recommended_tactic) if isinstance(wf_payload.get(recommended_tactic), dict) else {}
            wf_stability_pass = wf_for_recommended.get("stability_pass") if isinstance(wf_for_recommended, dict) else None

            recommendation = "keep_current"
            if (
                challenger_gates
                and (challenger_score >= champion_score + 1.5)
                and (oos_gates is not False)
                and (wf_stability_pass is not False)
            ):
                recommendation = "replace"
            elif (oos_gates is False) or (wf_stability_pass is False) or (not challenger_gates):
                recommendation = "discard"

            experiment["recommendation"] = recommendation
            experiment["champion_challenger"] = {
                "criteria": {
                    "min_total_trades": criteria.min_total_trades,
                    "min_profit_factor": criteria.min_profit_factor,
                    "min_total_pnl_pct": criteria.min_total_pnl_pct,
                    "max_drawdown_pct": criteria.max_drawdown_pct,
                },
                "challenger": {
                    "score": challenger_score,
                    "gates_passed": challenger_gates,
                    "evaluation": challenger_eval,
                    "oos_score": oos_score,
                    "oos_gates_passed": oos_gates,
                    "walk_forward_stability_pass": wf_stability_pass,
                },
                "champion": {
                    "score": champion_score,
                    "gates_passed": champion_gates,
                    "evaluation": champion_eval,
                },
            }
            experiment["comparison"] = {
                "experiment_pnl": experiment_pnl,
                "experiment_winrate": experiment_winrate,
                "current_pnl": current_pnl,
                "current_winrate": current_winrate,
                "improvement_pnl": experiment_pnl - current_pnl,
                "improvement_winrate": experiment_winrate - current_winrate,
                "experiment_score": challenger_score,
                "current_score": champion_score,
                "score_delta": challenger_score - champion_score,
            }

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

            results = experiment.get("results", {})
            models = results.get("models", {})
            model_15m_path = models.get("15m")
            model_1h_path = models.get("1h")
            recommended_tactic = results.get("recommended_tactic")

            if not model_15m_path and not model_1h_path:
                raise HTTPException(status_code=400, detail="Experiment has no trained models")

            if hasattr(state, "set_strategy_config"):
                if recommended_tactic == "mtf" and model_15m_path and model_1h_path:
                    state.set_strategy_config(symbol, {
                        "mode": "mtf",
                        "model_15m_path": model_15m_path,
                        "model_1h_path": model_1h_path,
                    })
                else:
                    chosen_model = model_1h_path if recommended_tactic == "single_1h" else model_15m_path
                    state.set_strategy_config(symbol, {
                        "mode": "single",
                        "model_path": chosen_model,
                    })

            if tg_bot:
                await tg_bot.send_notification(f"✅ Applied experiment {body.experiment_id} for {symbol}")

            return {
                "ok": True,
                "experiment_id": body.experiment_id,
                "symbol": symbol,
                "applied_at": datetime.now().isoformat(),
                "recommended_tactic": recommended_tactic,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Apply experiment failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app
