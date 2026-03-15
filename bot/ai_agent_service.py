from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except Exception:
    load_dotenv = None
    HAS_DOTENV = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    requests = None
    HAS_REQUESTS = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except Exception:
    create_client = None
    HAS_SUPABASE = False

from .experiment_management import (
    ExperimentAnalyzer,
    ExperimentCriteria,
    ExperimentStore,
    HypothesisGenerator,
    build_param_signature,
    build_regime_plan_context,
    derive_market_regime_from_metrics,
    get_code_version,
)

logger = logging.getLogger(__name__)


class AIAgentService:
    def __init__(self):
        if HAS_DOTENV:
            try:
                env_path = Path(__file__).resolve().parent.parent / ".env"
                load_dotenv(dotenv_path=env_path, override=False)
            except Exception:
                pass
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.research_processes: Dict[str, subprocess.Popen] = {}
        self.risk_state_file = Path(__file__).resolve().parent.parent / "ai_risk_state.json"
        self.chat_history_file = Path(__file__).resolve().parent.parent / "ai_chat_history.json"
        self.supabase = None
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if HAS_SUPABASE and supabase_url and supabase_key:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
            except Exception:
                self.supabase = None

    def _load_risk_state(self) -> Dict[str, Any]:
        try:
            if self.risk_state_file.exists():
                with open(self.risk_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _save_risk_state(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.risk_state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            logger.exception("Failed to save AI risk state")

    def _load_chat_history(self) -> List[Dict[str, Any]]:
        if self.supabase is not None:
            try:
                response = (
                    self.supabase.table("chat_messages")
                    .select("role,content,created_at")
                    .order("created_at", desc=True)
                    .limit(500)
                    .execute()
                )
                data = getattr(response, "data", None)
                if isinstance(data, list):
                    data = list(reversed(data))
                    out: List[Dict[str, Any]] = []
                    for row in data:
                        if not isinstance(row, dict):
                            continue
                        out.append(
                            {
                                "role": row.get("role"),
                                "content": row.get("content"),
                                "timestamp": row.get("created_at"),
                            }
                        )
                    return out
            except Exception:
                pass
        try:
            if self.chat_history_file.exists():
                with open(self.chat_history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
        except Exception:
            pass
        return []

    def _save_chat_history(self, history: List[Dict[str, Any]]) -> None:
        if self.supabase is not None:
            return
        try:
            with open(self.chat_history_file, "w", encoding="utf-8") as f:
                json.dump(history[-500:], f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            logger.exception("Failed to save chat history")

    async def _get_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            n = int(limit)
        except Exception:
            n = 50
        n = max(1, min(500, n))
        history = self._load_chat_history()
        return history[-n:]

    async def _save_chat_message(self, role: str, content: str) -> None:
        if self.supabase is not None:
            try:
                self.supabase.table("chat_messages").insert({"role": role, "content": content}).execute()
                return
            except Exception:
                pass
        history = self._load_chat_history()
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_chat_history(history)

    def _openrouter_chat(self, messages: List[Dict[str, str]], max_tokens: int = 800, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if not HAS_REQUESTS:
            raise RuntimeError("requests package is not available")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter returned invalid content")
        return content

    async def analyze_risk_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._load_risk_state()
        if not self.api_key:
            return {
                "analysis": "AI Agent не настроен: отсутствует OPENROUTER_API_KEY.",
                "suggestions": [],
                "risk_score": 0,
            }
        if not HAS_REQUESTS:
            return {
                "analysis": "AI Agent недоступен: пакет requests не установлен в окружении.",
                "suggestions": [],
                "risk_score": 0,
            }
        prompt = (
            "Ты AI риск-аналитик для крипто-бота. Оцени настройки риска и историю изменений. "
            "Верни строго JSON с полями analysis, suggestions, risk_score.\n\n"
            f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"STATE:\n{json.dumps(state, ensure_ascii=False)}"
        )
        try:
            loop = asyncio.get_event_loop()
            max_retries = 3
            last_exception = None
            for attempt in range(max_retries):
                try:
                    content = await loop.run_in_executor(
                        None,
                        lambda: self._openrouter_chat(
                            messages=[
                                {"role": "system", "content": "You are an expert risk analyst. Output valid JSON only."},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=600,
                            temperature=0.2,
                        ),
                    )
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].strip()
                    result = json.loads(content)
                    state["last_analysis"] = result
                    self._save_risk_state(state)
                    return result
                except Exception as e:
                    last_exception = e
                    logger.warning(f"AI Risk Analysis attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
            if "last_analysis" in state:
                cached = state["last_analysis"]
                cached["analysis"] = f"[OFFLINE MODE] {cached.get('analysis', '')} (Cached data)"
                return cached
            raise last_exception
        except Exception as e:
            logger.error(f"AI Risk Analysis failed after retries: {e}")
            return {
                "analysis": f"Ошибка анализа: {str(e)}. Проверьте настройки прокси/VPN.",
                "suggestions": [],
                "risk_score": 0,
            }

    def on_risk_settings_updated(self, total_trades: int, old_settings: Dict[str, Any], new_settings: Dict[str, Any]) -> None:
        state = self._load_risk_state()
        history = state.get("history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_trades": int(total_trades or 0),
                "old_settings": old_settings if isinstance(old_settings, dict) else {},
                "new_settings": new_settings if isinstance(new_settings, dict) else {},
            }
        )
        state["history"] = history[-200:]
        self._save_risk_state(state)

    def get_risk_history(self) -> List[Dict[str, Any]]:
        state = self._load_risk_state()
        history = state.get("history")
        if not isinstance(history, list):
            return []
        normalized: List[Dict[str, Any]] = []
        changed = False
        for item in history:
            if not isinstance(item, dict):
                changed = True
                continue
            row = dict(item)
            if not isinstance(row.get("timestamp"), str):
                row["timestamp"] = datetime.now(timezone.utc).isoformat()
                changed = True
            if not isinstance(row.get("total_trades"), int):
                try:
                    row["total_trades"] = int(row.get("total_trades") or 0)
                except Exception:
                    row["total_trades"] = 0
                changed = True
            if not isinstance(row.get("old_settings"), dict):
                row["old_settings"] = {}
                changed = True
            if not isinstance(row.get("new_settings"), dict):
                row["new_settings"] = {}
                changed = True
            normalized.append(row)
        if changed:
            state["history"] = normalized[-200:]
            self._save_risk_state(state)
        return normalized

    async def analyze_market_sentiment(self, symbol: str, kline_data: List[Dict[str, Any]], indicators: Dict[str, float]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": "AI Agent не настроен: отсутствует OPENROUTER_API_KEY.",
                "confidence": 0,
            }
        if not HAS_REQUESTS:
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": "AI Agent недоступен: пакет requests не установлен в окружении.",
                "confidence": 0,
            }
        prompt = f"""
        Ты — AI трейдер. Проанализируй рынок {symbol} на основе свечей и индикаторов.

        Последние 20 свечей (OHLCV):
        {json.dumps(kline_data[-20:], indent=2)}

        Текущие индикаторы:
        {json.dumps(indicators, indent=2)}

        Дай краткую сводку по рынку.
        Ответь строго в формате JSON:
        {{
            "trend": "bullish" | "bearish" | "sideways",
            "volatility": "high" | "low" | "normal",
            "advice": "Твой совет.",
            "confidence": 0.85
        }}
        """
        try:
            loop = asyncio.get_event_loop()
            max_retries = 3
            last_exception = None
            for attempt in range(max_retries):
                try:
                    content = await loop.run_in_executor(
                        None,
                        lambda: self._openrouter_chat(
                            messages=[
                                {"role": "system", "content": "You are an expert crypto trader. Output valid JSON only."},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=500,
                            temperature=0.2,
                        ),
                    )
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].strip()
                    return json.loads(content)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"AI Market Analysis attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
            raise last_exception
        except Exception as e:
            logger.error(f"AI Market Analysis failed: {e}")
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": f"Ошибка анализа: {str(e)}",
                "confidence": 0,
            }

    async def chat_with_user(self, message: str, context: Optional[Dict[str, Any]] = None, log_lines: Optional[List[str]] = None) -> str:
        user_message = str(message or "").strip()
        await self._save_chat_message("user", user_message)
        if not user_message:
            reply = "Пустое сообщение."
            await self._save_chat_message("assistant", reply)
            return reply
        if not self.api_key:
            reply = "AI Agent не настроен: отсутствует OPENROUTER_API_KEY."
            await self._save_chat_message("assistant", reply)
            return reply
        if not HAS_REQUESTS:
            reply = "AI Agent недоступен: пакет requests не установлен в окружении."
            await self._save_chat_message("assistant", reply)
            return reply
        payload = {
            "message": user_message,
            "context": context if isinstance(context, dict) else {},
            "logs_tail": log_lines[-20:] if isinstance(log_lines, list) else [],
        }
        prompt = (
            "Ты помощник торгового бота. Отвечай кратко и по делу на русском языке.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            loop = asyncio.get_event_loop()
            reply = await loop.run_in_executor(
                None,
                lambda: self._openrouter_chat(
                    messages=[
                        {"role": "system", "content": "You are a helpful trading bot assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=700,
                    temperature=0.3,
                ),
            )
        except Exception as e:
            reply = f"Ошибка AI: {str(e)}"
        await self._save_chat_message("assistant", str(reply))
        return str(reply)

    def validate_confirm_entry_request(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        for key in ("request_id", "timestamp_utc", "symbol", "timeframe", "signal", "bot_context", "market_context"):
            if key not in payload:
                raise ValueError(f"missing field: {key}")
        if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(payload.get("timestamp_utc"), str) or not payload["timestamp_utc"]:
            raise ValueError("timestamp_utc must be a non-empty string")
        if not isinstance(payload.get("symbol"), str) or not payload["symbol"]:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(payload.get("timeframe"), str) or not payload["timeframe"]:
            raise ValueError("timeframe must be a non-empty string")
        if not isinstance(payload.get("signal"), dict):
            raise ValueError("signal must be an object")
        if not isinstance(payload.get("bot_context"), dict):
            raise ValueError("bot_context must be an object")
        if not isinstance(payload.get("market_context"), dict):
            raise ValueError("market_context must be an object")
        s = payload["signal"]
        for k in ("action", "price"):
            if k not in s:
                raise ValueError(f"signal missing field: {k}")
        if not isinstance(s.get("action"), str) or s["action"] not in ("LONG", "SHORT"):
            raise ValueError("signal.action must be LONG or SHORT")
        if not isinstance(s.get("price"), (int, float)):
            raise ValueError("signal.price must be a number")

    async def confirm_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_confirm_entry_request(payload)
        if not self.api_key:
            return {
                "decision": "REJECT",
                "confidence": 0.0,
                "reason": "AI Agent не настроен: отсутствует OPENROUTER_API_KEY.",
                "risk_flags": ["ai_unavailable"],
                "tp_adjustment": None,
                "sl_adjustment": None,
            }
        if not HAS_REQUESTS:
            return {
                "decision": "REJECT",
                "confidence": 0.0,
                "reason": "AI Agent недоступен: пакет requests не установлен в окружении.",
                "risk_flags": ["ai_unavailable"],
                "tp_adjustment": None,
                "sl_adjustment": None,
            }
        prompt = (
            "Ты — AI фильтр входа для крипто-бота. Оцени вход и верни строго JSON с полями "
            "decision, confidence, reason, risk_flags, tp_adjustment, sl_adjustment.\n\n"
            f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: self._openrouter_chat(
                    messages=[
                        {"role": "system", "content": "You are an expert trading gatekeeper. Output valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.1,
                ),
            )
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            result = json.loads(content)
            decision = str(result.get("decision") or "REJECT").upper()
            if decision not in {"APPROVE", "REJECT", "WAIT"}:
                decision = "REJECT"
            result["decision"] = decision
            result["confidence"] = float(result.get("confidence") or 0.0)
            if not isinstance(result.get("risk_flags"), list):
                result["risk_flags"] = []
            return result
        except Exception as e:
            logger.error(f"AI confirm_entry failed: {e}")
            return {
                "decision": "REJECT",
                "confidence": 0.0,
                "reason": f"Ошибка AI анализа: {e}",
                "risk_flags": ["ai_error"],
                "tp_adjustment": None,
                "sl_adjustment": None,
            }

    def _build_research_ai_plan(
        self,
        symbol: str,
        experiment_type: str,
        summary: Dict[str, Any],
        impact: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        regime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        regime_context = regime_context if isinstance(regime_context, dict) else {}
        plan_defaults = regime_context.get("plan_defaults") if isinstance(regime_context.get("plan_defaults"), dict) else {}
        market_regime = regime_context.get("market_regime") if isinstance(regime_context.get("market_regime"), dict) else derive_market_regime_from_metrics(symbol)
        regime_memory = regime_context.get("regime_memory") if isinstance(regime_context.get("regime_memory"), dict) else None

        defaults: Dict[str, Any] = {
            "interval": str(plan_defaults.get("interval") or "15m"),
            "use_mtf": bool(plan_defaults.get("use_mtf", True)),
            "safe_mode": bool(plan_defaults.get("safe_mode", True)),
            "backtest_days": int(plan_defaults.get("backtest_days") or 7),
            "experiment_description": f"AI-driven optimization run for {symbol}",
            "hypothesis": "Уточнить параметры модели под текущую рыночную структуру с фокусом на рост PnL при контроле drawdown.",
            "expected_outcome": "Рост total_pnl_pct при сопоставимом или меньшем max_drawdown_pct.",
            "rationale": "План основан на предыдущих экспериментах и их метриках устойчивости.",
            "param_changes": dict(plan_defaults.get("param_changes") or {}),
            "hyperparams": dict(plan_defaults.get("hyperparams") or {}),
            "next_experiments": list(plan_defaults.get("next_experiments") or [
                "Вариант с более строгими порогами confidence",
                "Вариант с увеличенным окном бэктеста",
            ]),
            "tags": list(plan_defaults.get("tags") or ["ai_planner", "auto_optimization", experiment_type]),
            "market_regime": market_regime,
            "regime_memory": regime_memory,
        }
        if experiment_type == "aggressive":
            defaults.update({"interval": defaults.get("interval") or "15m", "use_mtf": False, "safe_mode": False, "backtest_days": min(int(defaults.get("backtest_days") or 7), 10)})
        elif experiment_type == "conservative":
            defaults.update({"interval": "1h" if market_regime.get("regime", "").startswith("sideways") else defaults.get("interval") or "1h", "use_mtf": True, "safe_mode": True, "backtest_days": max(int(defaults.get("backtest_days") or 14), 14)})
        else:
            defaults.update({"interval": defaults.get("interval") or "15m", "use_mtf": bool(defaults.get("use_mtf", True)), "safe_mode": bool(defaults.get("safe_mode", True)), "backtest_days": max(int(defaults.get("backtest_days") or 10), 10)})

        best = summary.get("best") if isinstance(summary, dict) else None
        if isinstance(best, dict):
            best_pnl = float(best.get("total_pnl_pct") or 0.0)
            best_dd = float(best.get("max_drawdown_pct") or 0.0)
            if best_pnl > 6 and best_dd < 15:
                defaults["expected_outcome"] = "Стабилизировать достигнутый профит и снизить волатильность кривой капитала."
            if best_pnl < 0:
                defaults["param_changes"].update({"regularization": "increase", "confidence_thresholds": "tighten"})

        if market_regime.get("regime"):
            defaults["rationale"] = (
                f"План учитывает режим рынка {market_regime.get('regime')}"
                + (" и накопленную память по этому режиму." if regime_memory else ".")
            )

        if hypotheses and isinstance(hypotheses[0], dict):
            h = hypotheses[0]
            if h.get("title"):
                defaults["hypothesis"] = str(h.get("title"))
            if h.get("description"):
                defaults["experiment_description"] = str(h.get("description"))

        prompt = f"""
Ты проектируешь следующий AI research experiment для торгового бота.

SYMBOL: {symbol}
EXPERIMENT_TYPE: {experiment_type}
SUMMARY:
{json.dumps(summary, ensure_ascii=False)}

PARAM_IMPACT:
{json.dumps(impact, ensure_ascii=False)}

HYPOTHESES:
{json.dumps(hypotheses, ensure_ascii=False)}

MARKET_REGIME:
{json.dumps(market_regime, ensure_ascii=False)}

REGIME_MEMORY:
{json.dumps(regime_memory, ensure_ascii=False)}

Верни строго JSON:
{{
  "interval": "15m|1h",
  "use_mtf": true,
  "safe_mode": true,
  "backtest_days": 7,
  "experiment_description": "краткое описание",
  "hypothesis": "гипотеза",
  "expected_outcome": "ожидаемый эффект",
  "rationale": "почему выбран такой план",
  "param_changes": {{}},
  "hyperparams": {{}},
  "next_experiments": ["...", "..."],
  "tags": ["...", "..."],
  "market_regime": {{}},
  "risk_profile": {{}},
  "execution_realism": {{}}
}}
Без текста вокруг JSON.
"""
        try:
            content = self._openrouter_chat(
                messages=[
                    {"role": "system", "content": "You are a quantitative research planner. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=900,
                temperature=0.2,
            )
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Planner returned non-object JSON")
            merged = {**defaults, **data}
            interval = str(merged.get("interval") or defaults["interval"]).lower()
            merged["interval"] = "1h" if interval in {"1h", "60m"} else "15m"
            merged["use_mtf"] = bool(merged.get("use_mtf"))
            merged["safe_mode"] = bool(merged.get("safe_mode"))
            days = int(merged.get("backtest_days") or defaults["backtest_days"])
            merged["backtest_days"] = max(3, min(120, days))
            if not isinstance(merged.get("param_changes"), dict):
                merged["param_changes"] = defaults["param_changes"]
            if not isinstance(merged.get("hyperparams"), dict):
                merged["hyperparams"] = defaults["hyperparams"]
            if not isinstance(merged.get("next_experiments"), list):
                merged["next_experiments"] = defaults["next_experiments"]
            if not isinstance(merged.get("tags"), list):
                merged["tags"] = defaults["tags"]
            if not isinstance(merged.get("market_regime"), dict):
                merged["market_regime"] = defaults["market_regime"]
            if not isinstance(merged.get("regime_memory"), dict) and defaults.get("regime_memory") is not None:
                merged["regime_memory"] = defaults["regime_memory"]
            return merged
        except Exception as e:
            logger.warning(f"AI planner fallback to heuristic defaults: {e}")
            return defaults

    def _build_risk_guard(
        self,
        symbol: str,
        experiment_type: str,
        ai_plan: Dict[str, Any],
        regime_context: Optional[Dict[str, Any]] = None,
        safe_mode_requested: bool = True,
        allow_regime_memory_soft_block: bool = False,
    ) -> Dict[str, Any]:
        regime_context = regime_context if isinstance(regime_context, dict) else {}
        market_regime = regime_context.get("market_regime") if isinstance(regime_context.get("market_regime"), dict) else ai_plan.get("market_regime")
        regime_memory = regime_context.get("regime_memory") if isinstance(regime_context.get("regime_memory"), dict) else ai_plan.get("regime_memory")
        if not isinstance(market_regime, dict):
            market_regime = derive_market_regime_from_metrics(symbol)
        risk_profile = ai_plan.get("risk_profile") if isinstance(ai_plan.get("risk_profile"), dict) else {}
        execution_realism = ai_plan.get("execution_realism") if isinstance(ai_plan.get("execution_realism"), dict) else {}

        interval = str(ai_plan.get("interval") or "15m").lower()
        interval = "1h" if interval in {"1h", "60m"} else "15m"
        use_mtf = bool(ai_plan.get("use_mtf"))
        safe_mode_effective = bool(ai_plan.get("safe_mode", safe_mode_requested))
        requested_backtest_days = int(ai_plan.get("backtest_days") or 7)
        backtest_days = max(3, min(120, requested_backtest_days))

        max_leverage = 2 if safe_mode_effective else 3
        max_drawdown_pct = 18.0 if safe_mode_effective else 22.0
        stop_after_losses = 3 if safe_mode_effective else 4
        min_trades = 30 if safe_mode_effective else 20
        slippage_bps = 8 if interval == "15m" else 5
        spread_bps = 4 if interval == "15m" else 3
        funding_bps_daily = 3
        blocked_reasons: List[str] = []
        warnings: List[str] = []

        regime_name = str((market_regime or {}).get("regime") or "unknown")
        volatility = str((market_regime or {}).get("volatility") or "normal")
        if volatility == "high":
            max_leverage = min(max_leverage, 2)
            max_drawdown_pct = min(max_drawdown_pct, 16.0 if safe_mode_effective else 18.0)
            stop_after_losses = min(stop_after_losses, 2 if safe_mode_effective else 3)
            backtest_days = max(backtest_days, 14)
            slippage_bps = max(slippage_bps, 12)
            spread_bps = max(spread_bps, 6)
            warnings.append("high_volatility_regime")
        if regime_name.startswith("sideways"):
            max_leverage = min(max_leverage, 2)
            backtest_days = max(backtest_days, 14)
            warnings.append("sideways_requires_longer_validation")
        if experiment_type == "aggressive" and volatility == "high":
            blocked_reasons.append("aggressive_mode_blocked_in_high_volatility")
        if experiment_type == "aggressive" and not safe_mode_effective:
            warnings.append("aggressive_without_safe_mode")

        if isinstance(regime_memory, dict):
            failed = int(regime_memory.get("failed") or 0)
            successful = int(regime_memory.get("successful") or 0)
            if failed > successful:
                max_drawdown_pct = min(max_drawdown_pct, 14.0)
                max_leverage = min(max_leverage, 2)
                backtest_days = max(backtest_days, 21)
                warnings.append("regime_memory_negative_bias")
            if failed >= max(3, successful + 2):
                if allow_regime_memory_soft_block and safe_mode_effective and experiment_type != "aggressive":
                    backtest_days = max(backtest_days, 30)
                    max_drawdown_pct = min(max_drawdown_pct, 12.0)
                    stop_after_losses = min(stop_after_losses, 2)
                    min_trades = max(min_trades, 45)
                    warnings.append("regime_memory_soft_block_applied")
                else:
                    blocked_reasons.append("regime_memory_shows_repeated_failures")

        if isinstance(risk_profile.get("max_leverage"), (int, float)):
            if float(risk_profile.get("max_leverage") or 0.0) > max_leverage:
                blocked_reasons.append("requested_leverage_above_guardrail")
        if isinstance(risk_profile.get("max_drawdown_pct"), (int, float)):
            if float(risk_profile.get("max_drawdown_pct") or 0.0) > max_drawdown_pct:
                blocked_reasons.append("requested_drawdown_limit_above_guardrail")
        if isinstance(risk_profile.get("stop_after_consecutive_losses"), (int, float)):
            if int(risk_profile.get("stop_after_consecutive_losses") or 0) > stop_after_losses:
                blocked_reasons.append("stop_after_losses_too_loose")

        normalized_risk_profile = {
            "safe_mode_required": bool(safe_mode_effective or blocked_reasons),
            "max_leverage": max_leverage,
            "max_drawdown_pct": max_drawdown_pct,
            "stop_after_consecutive_losses": stop_after_losses,
            "min_total_trades": min_trades,
            "regime": regime_name,
        }
        normalized_execution_realism = {
            "spread_bps": max(int(execution_realism.get("spread_bps") or 0), spread_bps),
            "slippage_bps": max(int(execution_realism.get("slippage_bps") or 0), slippage_bps),
            "funding_bps_daily": max(int(execution_realism.get("funding_bps_daily") or 0), funding_bps_daily),
            "latency_ms": max(int(execution_realism.get("latency_ms") or 0), 150 if interval == "15m" else 100),
        }

        return {
            "allowed": not blocked_reasons,
            "blocked_reasons": blocked_reasons,
            "warnings": warnings,
            "safe_mode": bool(safe_mode_effective or blocked_reasons),
            "backtest_days": backtest_days,
            "risk_profile": normalized_risk_profile,
            "execution_realism": normalized_execution_realism,
        }

    def start_research_experiment(
        self,
        symbol: str,
        experiment_type: str = "balanced",
        metadata: Optional[Dict[str, Any]] = None,
        allow_duplicate: bool = False,
        safe_mode: bool = True,
        goal: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, Any]] = None,
        auto_iterate: bool = True,
        max_steps: int = 3,
        campaign_parallel_limit: int = 2,
    ) -> Dict[str, Any]:
        try:
            project_root = Path(__file__).resolve().parent.parent
            store = ExperimentStore(project_root / "experiments.json")
            script_path = project_root / "run_research.py"
            if not script_path.exists():
                return {"ok": False, "error": f"run_research.py not found at {script_path}"}

            symbol = symbol.upper().strip()
            if not symbol:
                return {"ok": False, "error": "symbol is required"}
            experiment_type = (experiment_type or "balanced").strip().lower()
            if experiment_type not in {"balanced", "aggressive", "conservative"}:
                experiment_type = "balanced"

            meta = dict(metadata or {})
            if goal is not None and "goal" not in meta:
                meta["goal"] = goal
            if constraints is not None and "constraints" not in meta:
                meta["constraints"] = constraints
            if budget is not None and "budget" not in meta:
                meta["budget"] = budget
            if "auto_iterate" not in meta:
                meta["auto_iterate"] = bool(auto_iterate)

            all_experiments = store.list()
            requested_steps = meta.get("max_steps", max_steps)
            try:
                requested_steps = int(requested_steps)
            except Exception:
                requested_steps = 3
            requested_steps = max(1, min(50, requested_steps))
            meta["max_steps"] = requested_steps
            try:
                parallel_limit = int(campaign_parallel_limit)
            except Exception:
                parallel_limit = 2
            parallel_limit = max(1, min(20, parallel_limit))
            meta["campaign_parallel_limit"] = parallel_limit

            running_statuses = {"starting", "training", "training_completed", "backtesting"}
            root_ids = set()
            for e in all_experiments:
                if not isinstance(e, dict):
                    continue
                if str(e.get("status") or "") not in running_statuses:
                    continue
                c = e.get("ai_campaign") if isinstance(e.get("ai_campaign"), dict) else {}
                if not c:
                    continue
                if not bool(c.get("auto_chain", True)):
                    continue
                rid = str(c.get("root_experiment_id") or e.get("id") or "")
                if rid:
                    root_ids.add(rid)
            if bool(meta.get("auto_iterate", True)) and len(root_ids) >= parallel_limit:
                return {
                    "ok": False,
                    "error": f"Campaign parallel limit reached ({parallel_limit})",
                    "active_campaigns": len(root_ids),
                    "campaign_parallel_limit": parallel_limit,
                }

            experiment_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            analyzer = ExperimentAnalyzer(all_experiments)
            summary = analyzer.summarize(symbol=symbol)
            impact = analyzer.compute_param_impact(symbol=symbol)
            regime_memory = analyzer.summarize_regime_memory(symbol=symbol)
            hypotheses = HypothesisGenerator(ExperimentCriteria()).propose(summary)
            meta_market_regime = meta.get("market_regime") if isinstance(meta.get("market_regime"), dict) else None
            regime_context = build_regime_plan_context(
                symbol=symbol,
                experiment_type=experiment_type,
                summary=summary,
                regime_memory=regime_memory,
                market_regime=meta_market_regime,
            )
            ai_plan = meta.get("ai_plan") if isinstance(meta.get("ai_plan"), dict) else None
            allow_regime_memory_soft_block = bool(meta.get("allow_regime_memory_soft_block", False))
            if not ai_plan:
                ai_plan = self._build_research_ai_plan(
                    symbol=symbol,
                    experiment_type=experiment_type,
                    summary=summary,
                    impact=impact,
                    hypotheses=hypotheses,
                    regime_context=regime_context,
                )
            interval = str(ai_plan.get("interval") or "15m").lower()
            interval = "1h" if interval in {"1h", "60m"} else "15m"
            use_mtf = bool(ai_plan.get("use_mtf"))
            safe_mode_effective = bool(ai_plan.get("safe_mode", safe_mode))
            risk_guard = self._build_risk_guard(
                symbol=symbol,
                experiment_type=experiment_type,
                ai_plan=ai_plan,
                regime_context=regime_context,
                safe_mode_requested=safe_mode_effective,
                allow_regime_memory_soft_block=allow_regime_memory_soft_block,
            )
            if not risk_guard.get("allowed"):
                return {
                    "ok": False,
                    "error": "Experiment blocked by AI risk guard",
                    "experiment_id": experiment_id,
                    "blocked_reasons": risk_guard.get("blocked_reasons") or [],
                    "market_regime": regime_context.get("market_regime") or ai_plan.get("market_regime"),
                    "risk_profile": risk_guard.get("risk_profile"),
                }
            safe_mode_effective = bool(risk_guard.get("safe_mode", safe_mode_effective))
            backtest_days = int(risk_guard.get("backtest_days") or ai_plan.get("backtest_days") or 7)
            backtest_days = max(3, min(120, backtest_days))
            params = ["--interval", interval]
            if not use_mtf:
                params.append("--no-mtf")
            cmd = [
                sys.executable,
                str(script_path),
                "--symbol",
                symbol,
                "--type",
                experiment_type,
                "--experiment-id",
                experiment_id,
            ] + params

            meta["ai_enabled"] = True
            meta["ai_plan"] = ai_plan
            meta.setdefault("experiment_description", ai_plan.get("experiment_description"))
            meta.setdefault("hypothesis", ai_plan.get("hypothesis"))
            meta.setdefault("expected_outcome", ai_plan.get("expected_outcome"))
            meta.setdefault("rationale", ai_plan.get("rationale"))
            meta.setdefault("param_changes", ai_plan.get("param_changes"))
            if isinstance(ai_plan.get("hyperparams"), dict):
                meta.setdefault("hyperparams", ai_plan.get("hyperparams"))
            meta.setdefault("next_experiments", ai_plan.get("next_experiments"))
            meta.setdefault("tags", ai_plan.get("tags"))
            meta.setdefault("market_regime", ai_plan.get("market_regime") or regime_context.get("market_regime"))
            if ai_plan.get("regime_memory") is not None:
                meta.setdefault("regime_memory", ai_plan.get("regime_memory"))
            if regime_memory:
                meta.setdefault("regime_memory_summary", regime_memory)
            meta.setdefault("risk_profile", risk_guard.get("risk_profile"))
            meta.setdefault("execution_realism", risk_guard.get("execution_realism"))
            meta.setdefault("ai_risk_guard", {
                "allowed": bool(risk_guard.get("allowed")),
                "warnings": list(risk_guard.get("warnings") or []),
                "blocked_reasons": list(risk_guard.get("blocked_reasons") or []),
            })
            meta["backtest_days"] = backtest_days

            ai_campaign = meta.get("ai_campaign") if isinstance(meta.get("ai_campaign"), dict) else {}
            if not ai_campaign:
                ai_campaign = {
                    "auto_chain": bool(meta.get("auto_iterate", True)),
                    "remaining_steps": max(0, requested_steps - 1),
                    "iteration": 1,
                    "max_steps": requested_steps,
                }
            ai_campaign["auto_chain"] = bool(ai_campaign.get("auto_chain", True))
            try:
                ai_campaign["remaining_steps"] = max(0, min(20, int(ai_campaign.get("remaining_steps", 2))))
            except Exception:
                ai_campaign["remaining_steps"] = 2
            try:
                ai_campaign["iteration"] = max(1, min(50, int(ai_campaign.get("iteration", 1))))
            except Exception:
                ai_campaign["iteration"] = 1
            try:
                ai_campaign["max_steps"] = max(1, min(50, int(ai_campaign.get("max_steps", ai_campaign["iteration"] + ai_campaign["remaining_steps"]))))
            except Exception:
                ai_campaign["max_steps"] = ai_campaign["iteration"] + ai_campaign["remaining_steps"]
            ai_campaign["root_experiment_id"] = str(ai_campaign.get("root_experiment_id") or experiment_id)
            if ai_campaign["iteration"] > ai_campaign["max_steps"]:
                ai_campaign["iteration"] = ai_campaign["max_steps"]
            parent_experiment_id = ai_campaign.get("parent_experiment_id")
            if parent_experiment_id:
                for e in all_experiments:
                    if not isinstance(e, dict):
                        continue
                    c = e.get("ai_campaign") if isinstance(e.get("ai_campaign"), dict) else {}
                    if not c:
                        continue
                    if str(c.get("parent_experiment_id") or "") != str(parent_experiment_id):
                        continue
                    if int(c.get("iteration") or 0) != int(ai_campaign.get("iteration") or 0):
                        continue
                    existing_id = e.get("id")
                    if isinstance(existing_id, str) and existing_id:
                        return {
                            "ok": True,
                            "experiment_id": existing_id,
                            "symbol": symbol,
                            "type": experiment_type,
                            "idempotent_reused": True,
                            "ai_campaign": c,
                            "message": f"Reused existing campaign experiment {existing_id}",
                        }
            meta["ai_campaign"] = ai_campaign

            exp_params = {
                "primary_interval": interval,
                "train_intervals": ["15m", "1h"],
                "no_mtf": not use_mtf,
                "safe_mode": safe_mode_effective,
                "backtest_days": backtest_days,
                "risk_profile": meta.get("risk_profile") if isinstance(meta.get("risk_profile"), dict) else None,
                "execution_realism": meta.get("execution_realism") if isinstance(meta.get("execution_realism"), dict) else None,
            }
            signature = build_param_signature(
                symbol=symbol,
                experiment_type=experiment_type,
                params=exp_params,
                hyperparams=meta.get("hyperparams") if isinstance(meta.get("hyperparams"), dict) else None,
            )
            avoid_signatures = []
            regime_memory_row = meta.get("regime_memory") if isinstance(meta.get("regime_memory"), dict) else None
            if isinstance(regime_memory_row, dict) and isinstance(regime_memory_row.get("avoid_signatures"), list):
                avoid_signatures = [str(x) for x in regime_memory_row.get("avoid_signatures") if isinstance(x, str)]
            if signature in avoid_signatures and not allow_duplicate:
                return {
                    "ok": False,
                    "error": "Experiment blocked by regime memory",
                    "experiment_id": experiment_id,
                    "param_signature": signature,
                    "market_regime": meta.get("market_regime"),
                }

            existing = [e for e in store.list() if isinstance(e, dict) and e.get("param_signature") == signature]
            criteria = ExperimentCriteria()
            if existing and not allow_duplicate:
                def _is_ineffective(e: Dict[str, Any]) -> bool:
                    status = e.get("status")
                    if status in {"failed"}:
                        return True
                    if status != "completed":
                        return False
                    results = e.get("results") if isinstance(e.get("results"), dict) else {}
                    trades = int(results.get("total_trades") or 0)
                    pf = float(results.get("profit_factor") or 0.0)
                    pnl = float(results.get("total_pnl_pct") or 0.0)
                    dd = float(results.get("max_drawdown_pct") or results.get("max_drawdown") or 0.0)
                    return not (
                        trades >= criteria.min_total_trades
                        and pf >= criteria.min_profit_factor
                        and pnl >= criteria.min_total_pnl_pct
                        and dd <= criteria.max_drawdown_pct
                    )
                if any(_is_ineffective(e) for e in existing):
                    return {
                        "ok": False,
                        "error": "Duplicate ineffective experiment blocked",
                        "experiment_id": experiment_id,
                        "param_signature": signature,
                        "effective_params": exp_params,
                    }

            code_version = get_code_version(project_root)
            created_at = datetime.utcnow().isoformat() + "Z"
            record: Dict[str, Any] = {
                "id": experiment_id,
                "created_at": created_at,
                "updated_at": created_at,
                "status": "starting",
                "symbol": symbol,
                "type": experiment_type,
                "params": exp_params,
                "param_signature": signature,
                "code_version": code_version,
                "ai_enabled": True,
            }
            for k in [
                "baseline",
                "param_changes",
                "hypothesis",
                "expected_outcome",
                "rationale",
                "hyperparams",
                "hyperparameter_search",
                "criteria",
                "tags",
                "ai_plan",
                "next_experiments",
                "experiment_description",
                "backtest_days",
                "ai_enabled",
                "ai_campaign",
                "market_regime",
                "regime_memory",
                "regime_memory_summary",
                "risk_profile",
                "execution_realism",
                "ai_risk_guard",
                "goal",
                "constraints",
                "budget",
            ]:
                if k in meta:
                    record[k] = meta[k]
            store.upsert(experiment_id, record)

            meta_dir = project_root / "experiment_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / f"{experiment_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            cmd += ["--metadata-path", str(meta_path)]
            if safe_mode_effective:
                cmd.append("--safe-mode")
            logger.info(f"Starting research experiment: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.research_processes[experiment_id] = process
            store.upsert(
                experiment_id,
                {
                    "pid": process.pid,
                    "status": "starting",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return {
                "ok": True,
                "experiment_id": experiment_id,
                "symbol": symbol,
                "type": experiment_type,
                "pid": process.pid,
                "ai_plan": ai_plan,
                "market_regime": meta.get("market_regime"),
                "regime_memory": meta.get("regime_memory"),
                "risk_profile": meta.get("risk_profile"),
                "execution_realism": meta.get("execution_realism"),
                "ai_risk_guard": meta.get("ai_risk_guard"),
            }
        except Exception as e:
            logger.exception("Failed to start research experiment")
            return {"ok": False, "error": str(e)}

    def get_research_experiments(self) -> List[Dict[str, Any]]:
        project_root = Path(__file__).resolve().parent.parent
        store = ExperimentStore(project_root / "experiments.json")
        return store.list()

    def stop_research_experiment(self, experiment_id: str) -> Dict[str, Any]:
        process = self.research_processes.get(experiment_id)
        if process is None:
            return {"ok": False, "error": "Experiment process not found"}
        try:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
            project_root = Path(__file__).resolve().parent.parent
            store = ExperimentStore(project_root / "experiments.json")
            store.upsert(
                experiment_id,
                {
                    "status": "interrupted",
                    "stopped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return {"ok": True, "experiment_id": experiment_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}
