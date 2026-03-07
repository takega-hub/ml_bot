import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Explicitly load .env to ensure keys are available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    logger.warning(f".env file not found at {env_path}")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI library not installed. AI Agent features will be disabled. Run 'pip install openai'")

class AIAgentService:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = None
        
        # Debug logging for API Key (masked)
        if self.api_key:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
            logger.info(f"AI Agent: API Key found ({masked_key})")
        else:
            logger.warning("AI Agent: API Key NOT found in environment or arguments")

        if self.api_key and HAS_OPENAI:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                logger.info(f"AI Agent initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        elif not HAS_OPENAI:
             logger.warning("AI Agent disabled: openai module missing")
        else:
            logger.warning("AI Agent initialized without API Key. Analysis will be disabled.")

    def _load_risk_state(self) -> Dict[str, Any]:
        """Loads the last risk analysis state from file."""
        state_path = Path(__file__).resolve().parent.parent / "ai_risk_state.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load risk state: {e}")
        return {}

    def _save_risk_state(self, state: Dict[str, Any]):
        """Saves the risk analysis state to file."""
        state_path = Path(__file__).resolve().parent.parent / "ai_risk_state.json"
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def get_risk_history(self) -> List[Dict[str, Any]]:
        """Returns the history of risk setting changes."""
        state = self._load_risk_state()
        history = state.get("history", [])
        # Sort by timestamp descending (newest first)
        return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)

    def on_risk_settings_updated(self, current_trades_count: int, old_settings: Dict[str, Any] = None, new_settings: Dict[str, Any] = None):
        """Called when risk settings are updated by user."""
        from datetime import datetime
        state = self._load_risk_state()
        
        # Update last update info
        state["last_update_trade_count"] = current_trades_count
        state["last_update_time"] = datetime.now().isoformat()
        state["status"] = "monitoring"
        
        # Record history
        if old_settings and new_settings:
            changes = {}
            for k, v in new_settings.items():
                if k in old_settings and old_settings[k] != v:
                    changes[k] = {"old": old_settings[k], "new": v}
            
            if changes:
                history_entry = {
                    "timestamp": state["last_update_time"],
                    "trade_count": current_trades_count,
                    "changes": changes,
                    "full_snapshot": new_settings
                }
                
                if "history" not in state:
                    state["history"] = []
                
                # Keep last 20 changes
                state["history"].append(history_entry)
                if len(state["history"]) > 20:
                    state["history"] = state["history"][-20:]
                    
        self._save_risk_state(state)
        logger.info(f"AI Risk Agent: Entered monitoring mode at trade count {current_trades_count}. Changes recorded: {len(changes) if 'changes' in locals() else 0}")

    async def analyze_risk_settings(self, trades: List[Dict[str, Any]], current_risk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует историю сделок и текущие настройки риска.
        Предлагает оптимизации.
        """
        if not self.client:
            return {"error": "AI Agent not configured"}

        # Check monitoring status
        state = self._load_risk_state()
        last_count = state.get("last_update_trade_count", 0)
        current_count = len(trades)
        
        # Minimum trades required to evaluate new settings
        MIN_TRADES_TO_EVALUATE = 10 
        
        trades_diff = current_count - last_count
        
        if trades_diff < MIN_TRADES_TO_EVALUATE and state.get("status") == "monitoring":
            remaining = MIN_TRADES_TO_EVALUATE - trades_diff
            return {
                "analysis": f"Настройки были недавно изменены. Агент наблюдает за результатами. Необходимо еще {remaining} сделок для нового анализа.",
                "suggestions": [],
                "risk_score": 85, # Neutral score while waiting
                "status": "monitoring",
                "progress": trades_diff / MIN_TRADES_TO_EVALUATE
            }

        # Prepare context
        recent_trades = trades[-50:] if len(trades) > 50 else trades
        trades_summary = []
        for t in recent_trades:
            trades_summary.append({
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "pnl": t.get("pnl_usd"),
                "exit_reason": t.get("exit_reason", "unknown"),
                "duration": t.get("duration_minutes", 0)
            })
            
        # Prepare history context
        history_context = ""
        if "history" in state and state["history"]:
            history_context = "История изменений настроек (последние 5):\n"
            for entry in state["history"][-5:]:
                changes_str = ", ".join([f"{k}: {v['old']} -> {v['new']}" for k, v in entry["changes"].items()])
                history_context += f"- Trade #{entry['trade_count']}: {changes_str}\n"
        
        prompt = f"""
        Ты — профессиональный риск-менеджер в алгоритмическом трейдинге.
        
        Твоя задача: проанализировать последние 50 сделок бота и текущие настройки риска.
        Если ты видишь системные проблемы (серия убытков, слишком большие просадки, ранние выходы), предложи изменения в настройках.
        
        Текущие настройки риска (JSON):
        {json.dumps(current_risk, indent=2)}
        
        {history_context}
        
        История сделок (JSON, последние 50):
        {json.dumps(trades_summary, indent=2)}
        
        Ответь строго в формате JSON:
        {{
            "analysis": "Твое текстовое резюме анализа (макс 100 слов). Учитывай историю изменений, если она есть.",
            "suggestions": [
                {{
                    "setting_key": "stop_loss_pct",
                    "current_value": 0.015,
                    "suggested_value": 0.02,
                    "reason": "Высокая волатильность выбивает стопы слишком часто."
                }}
            ],
            "risk_score": 75 (0-100, где 100 - безопасно, 0 - критический риск)
        }}
        Не добавляй никакого текста до или после JSON.
        """

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert algorithmic trading risk manager. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            ))
            
            content = response.choices[0].message.content
            # Clean up potential markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"AI Risk Analysis failed: {e}")
            return {
                "analysis": f"Ошибка анализа: {str(e)}. Проверьте настройки прокси/VPN.",
                "suggestions": [],
                "risk_score": 0
            }

    async def analyze_market_sentiment(self, symbol: str, kline_data: List[Dict[str, Any]], indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Анализирует рыночные данные и дает комментарий по ситуации.
        """
        if not self.client:
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": "AI Agent not configured (OpenAI/OpenRouter key missing).",
                "confidence": 0
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
            "advice": "Твой совет (например: 'Лучше воздержаться от лонгов, ждем пробоя уровня').",
            "confidence": 0.85
        }}
        """

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert crypto trader. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            ))
            
            content = response.choices[0].message.content
             # Clean up potential markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            return json.loads(content)
            
        except Exception as e:
            logger.error(f"AI Market Analysis failed: {e}")
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": f"Ошибка анализа: {str(e)}",
                "confidence": 0
            }

    def start_research_experiment(self, symbol: str, experiment_type: str) -> Dict[str, Any]:
        """
        Запускает эксперимент по обучению модели с новыми параметрами.
        experiment_type: 'aggressive' | 'conservative' | 'balanced'
        """
        try:
            import sys
            import subprocess
            from pathlib import Path
            
            project_root = Path(__file__).resolve().parent.parent
            # Use the optimized retraining script
            script_path = project_root / "retrain_ml_optimized.py"
            
            if not script_path.exists():
                 logger.error(f"Experiment script not found: {script_path}")
                 return {"ok": False, "error": f"Script not found: {script_path.name}"}
            
            # Define parameters based on type
            params = []
            suffix = f"_{experiment_type}_exp"
            
            # Map types to supported arguments
            if experiment_type == 'aggressive':
                # 15m interval, standard optimized logic
                params = ["--interval", "15m"] 
            elif experiment_type == 'conservative':
                # 1h interval for more stable signals
                params = ["--interval", "1h"]
            else:
                # Balanced - default 15m
                params = ["--interval", "15m"]
            
            cmd = [sys.executable, str(script_path), "--symbol", symbol, "--model-suffix", suffix] + params
            
            logger.info(f"Starting experiment: {' '.join(cmd)}")
            
            # Run in background
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8', # Ensure encoding is set
                errors='replace'
            )
            
            return {
                "ok": True, 
                "pid": process.pid, 
                "symbol": symbol, 
                "type": experiment_type,
                "model_suffix": suffix,
                "message": f"Experiment {experiment_type} started for {symbol} (PID: {process.pid})"
            }
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}
