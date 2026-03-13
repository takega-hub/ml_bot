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

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    logger.warning("Supabase library not installed. Chat history will not be saved. Run 'pip install supabase'")

class AIAgentService:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = None
        
        # Supabase setup
        self.supabase: Optional[Client] = None
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        self._init_supabase()

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

    def _init_supabase(self):
        """Attempts to initialize Supabase client."""
        if self.supabase:
            return

        # Reload env vars to be sure
        self.supabase_url = self.supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = self.supabase_key or os.getenv("SUPABASE_KEY")

        if HAS_SUPABASE and self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
             # Only log if we have partial config to avoid noise if user doesn't want supabase
             if self.supabase_url or self.supabase_key:
                 logger.warning(f"Supabase init skipped: HAS_SUPABASE={HAS_SUPABASE}, URL={bool(self.supabase_url)}, KEY={bool(self.supabase_key)}")

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
            
            # Simple retry mechanism
            max_retries = 3
            last_exception = None
            
            for attempt in range(max_retries):
                try:
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
                        
                    result = json.loads(content)
                    
                    # Save successful analysis to state for caching
                    state["last_analysis"] = result
                    self._save_risk_state(state)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"AI Risk Analysis attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1) # Wait a bit before retry
            
            # If all retries failed, try to return cached analysis
            if "last_analysis" in state:
                logger.info("Returning cached risk analysis due to API failure")
                cached = state["last_analysis"]
                # Append a note about cache
                cached["analysis"] = f"[OFFLINE MODE] {cached.get('analysis', '')} (Cached data)"
                return cached
                
            raise last_exception
            
        except Exception as e:
            logger.error(f"AI Risk Analysis failed after retries: {e}")
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
            
            # Simple retry mechanism
            max_retries = 3
            last_exception = None
            
            for attempt in range(max_retries):
                try:
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
                    last_exception = e
                    logger.warning(f"AI Market Analysis attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)
            
            raise last_exception
            
        except Exception as e:
            logger.error(f"AI Market Analysis failed: {e}")
            return {
                "trend": "neutral",
                "volatility": "normal",
                "advice": f"Ошибка анализа: {str(e)}",
                "confidence": 0
            }

    async def _get_chat_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieves chat history from Supabase."""
        if not self.supabase:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: 
                self.supabase.table("chat_messages")
                .select("role, content")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            # Reverse to chronological order
            data = response.data
            return data[::-1] if data else []
        except Exception as e:
            logger.warning(f"Failed to fetch chat history: {e}")
            return []

    async def _save_chat_message(self, role: str, content: str):
        """Saves a chat message to Supabase."""
        if not self.supabase:
            self._init_supabase()
            
        if not self.supabase:
            logger.warning("Supabase client not initialized, skipping save.")
            return

        logger.info(f"Saving chat message to Supabase: {role} - {content[:20]}...")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: 
                self.supabase.table("chat_messages")
                .insert({"role": role, "content": content})
                .execute()
            )
            logger.info("Chat message saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}", exc_info=True)

    def _load_recent_candles(self, symbol: str, interval: str = "15m", limit: int = 24) -> str:
        """Loads recent candle data from CSV in ml_data directory."""
        try:
            # Map interval to filename format
            # Filename example: ADAUSDT_15_cache.csv
            interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
            file_interval = interval_map.get(interval, "15")
            
            project_root = Path(__file__).resolve().parent.parent
            ml_data_dir = project_root / "ml_data"
            
            # Try cache file first
            cache_file = ml_data_dir / f"{symbol}_{file_interval}_cache.csv"
            
            target_file = None
            if cache_file.exists():
                target_file = cache_file
            else:
                # Try to find any file matching pattern
                files = list(ml_data_dir.glob(f"{symbol}_{file_interval}_*.csv"))
                if files:
                    # Sort by modification time, newest first
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    target_file = files[0]
            
            if not target_file:
                return f"No historical data found for {symbol} ({interval})"
                
            # Read last N lines
            lines = []
            with open(target_file, "r", encoding="utf-8") as f:
                # Read header
                header = f.readline().strip()
                # Read all lines
                all_lines = f.readlines()
                
            last_lines = all_lines[-limit:] if len(all_lines) > limit else all_lines
            
            data_str = f"Market Data ({symbol} {interval}, last {len(last_lines)} candles):\n"
            data_str += header + "\n"
            data_str += "".join(last_lines)
            
            return data_str
            
        except Exception as e:
            logger.error(f"Error loading candles for {symbol}: {e}")
            return f"Error loading data: {str(e)}"

    def _get_models_info(self) -> str:
        """
        Collects information about available ML models and their performance.
        """
        try:
            project_root = Path(__file__).resolve().parent.parent
            models_dir = project_root / "ml_models"
            
            info = "=== ML Models Information ===\n"
            
            # 1. List available model files
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pkl"))
                info += f"Available Model Files ({len(model_files)}):\n"
                for mf in model_files[:10]: # Limit to 10 to avoid huge prompt
                    info += f"- {mf.name}\n"
                if len(model_files) > 10:
                    info += f"... and {len(model_files) - 10} more.\n"
            else:
                info += "Models directory (ml_models) not found.\n"
            
            # 2. Get active models from runtime state
            state_path = project_root / "runtime_state.json"
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                        active_models = state.get("symbol_models", {})
                        if active_models:
                            info += "\nActive Models (Currently Used):\n"
                            for sym, model_path in active_models.items():
                                info += f"- {sym}: {Path(model_path).name}\n"
                except Exception as e:
                    info += f"Error reading runtime state: {e}\n"
            
            # 3. Get performance metrics from analysis results
            analysis_path = project_root / "analysis_results.json"
            if analysis_path.exists():
                try:
                    with open(analysis_path, "r", encoding="utf-8") as f:
                        analysis = json.load(f)
                        recommendations = analysis.get("recommendations", {})
                        
                        info += "\nRecent Performance Analysis (Backtest/Live):\n"
                        for sym, data in recommendations.items():
                            best_mtf = data.get("best_mtf_actual", {})
                            if best_mtf:
                                pnl = best_mtf.get("pnl", 0)
                                wr = best_mtf.get("wr", 0)
                                info += f"- {sym}: Best MTF Model -> PnL: {pnl:.2f}%, WinRate: {wr:.1f}%\n"
                                info += f"  (Model 1H: {best_mtf.get('model_1h')}, Model 15M: {best_mtf.get('model_15m')})\n"
                            
                            # Also check single models if MTF not available or for context
                            best_single = data.get("best_single_15m", {})
                            if best_single:
                                info += f"  (Best Single 15m: {best_single.get('model')} -> PnL: {best_single.get('pnl',0):.2f}%, WR: {best_single.get('wr',0):.1f}%)\n"
                except Exception as e:
                    info += f"Error reading analysis results: {e}\n"
            
            return info + "\n"
        except Exception as e:
            logger.error(f"Error getting models info: {e}")
            return "Error retrieving models information.\n"

    async def chat_with_user(self, message: str, context: Dict[str, Any], logs: List[str]) -> str:
        """
        Handles chat interaction with the user.
        """
        if not self.client:
            return "AI Agent is not configured (API Key missing)."

        # 1. Save user message (fire and forget task to speed up response, or await if consistency needed)
        # We await it to ensure order if messages come fast
        await self._save_chat_message("user", message)

        # 2. Get history
        history = await self._get_chat_history(limit=10)
        
        # Format history for prompt
        history_str = ""
        if history:
            history_str = "История диалога:\n"
            for msg in history:
                history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Check for symbol in message (simple regex)
        import re
        symbol_match = re.search(r'\b([A-Z0-9]+USDT)\b', message.upper())
        market_context = ""
        
        if symbol_match:
            symbol = symbol_match.group(1)
            # Load 15m and 1h data
            data_15m = self._load_recent_candles(symbol, "15m", limit=15)
            data_1h = self._load_recent_candles(symbol, "1h", limit=15)
            market_context = f"\n=== Market Data for {symbol} (CSV History) ===\n{data_15m}\n{data_1h}\n"
        else:
            # If no symbol found but user asks about market/candles, give a hint
            keywords = ["СВЕЧ", "CANDLE", "MARKET", "РЫНОК", "ЦЕНА", "PRICE", "ПРОГНОЗ", "FORECAST", "ANALYSIS", "АНАЛИЗ"]
            if any(k in message.upper() for k in keywords):
                market_context = "\n[SYSTEM NOTE: User asked about market/candles but didn't provide a symbol. Tell them you can analyze it if they provide a pair like BTCUSDT.]\n"
        
        log_text = "".join(logs)
        context_str = json.dumps(context, indent=2, default=str)
        
        # Get models info
        models_info = self._get_models_info()

        prompt = f"""
        Ты — умный ассистент торгового бота.
        Твоя цель: помогать пользователю понимать, что происходит с ботом, анализировать логи и давать советы.
        
        ВАЖНО: У тебя ЕСТЬ доступ к историческим данным свечей (CSV), если пользователь укажет конкретную торговую пару (например, BTCUSDT).
        Если пользователь спрашивает, есть ли у тебя доступ к данным свечей, отвечай: "Да, я могу загрузить и проанализировать данные свечей, если вы укажете конкретную пару (например, BTCUSDT)".
        
        ТАКЖЕ ВАЖНО: У тебя есть информация о ML моделях и их эффективности.
        {models_info}
        
        Контекст бота:
        {context_str}
        
        Последние логи:
        {log_text}
        
        {market_context}
        
        {history_str}
        
        Вопрос пользователя:
        {message}
        
        Отвечай кратко, по делу и на русском языке. 
        Если пользователь спрашивает о рынке или прогнозе по конкретной паре, используй предоставленные исторические данные (Market Data) для анализа тренда и волатильности.
        Если видишь ошибку в логах, объясни её простыми словами и предложи решение.
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful trading bot assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            ))
            
            ai_response = response.choices[0].message.content
            
            # 3. Save AI response
            await self._save_chat_message("assistant", ai_response)
            
            return ai_response
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"Error: {str(e)}"

    def start_research_experiment(
        self,
        symbol: str,
        experiment_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        allow_duplicate: bool = False,
    ) -> Dict[str, Any]:
        """
        Starts a research experiment (Training + Backtest)
        experiment_type: 'aggressive' | 'conservative' | 'balanced'
        """
        try:
            import sys
            import subprocess
            from pathlib import Path
            import uuid
            import json
            from datetime import datetime
            from typing import Dict, Any
            from .experiment_management import (
                ExperimentCriteria,
                ExperimentStore,
                build_param_signature,
                get_code_version,
            )
            
            project_root = Path(__file__).resolve().parent.parent
            # Use the new research runner script
            script_path = project_root / "run_research.py"
            
            if not script_path.exists():
                 logger.error(f"Research script not found: {script_path}")
                 return {"ok": False, "error": f"Script not found: {script_path.name}"}
            
            experiment_id = f"exp_{int(asyncio.get_event_loop().time())}_{str(uuid.uuid4())[:8]}"
            
            # Define parameters based on type
            params = []
            
            # Map types to supported arguments
            if experiment_type == 'aggressive':
                params = ["--interval", "15m", "--no-mtf"] 
            elif experiment_type == 'conservative':
                # Conservative uses 1h models
                params = ["--interval", "1h", "--no-mtf"]
            else:
                # Balanced - default 15m
                params = ["--interval", "15m", "--no-mtf"]
            
            cmd = [
                sys.executable, str(script_path), 
                "--symbol", symbol, 
                "--type", experiment_type,
                "--experiment-id", experiment_id
            ] + params

            store = ExperimentStore(project_root / "experiments.json")
            meta = metadata if isinstance(metadata, dict) else {}
            exp_params = {
                "primary_interval": (params[1] if len(params) >= 2 and params[0] == "--interval" else "15m"),
                "train_intervals": ["15m", "1h"],
                "no_mtf": "--no-mtf" in params,
            }
            signature = build_param_signature(
                symbol=symbol,
                experiment_type=experiment_type,
                params=exp_params,
                hyperparams=meta.get("hyperparams") if isinstance(meta.get("hyperparams"), dict) else None,
            )

            existing = [
                e
                for e in store.list()
                if isinstance(e, dict) and e.get("param_signature") == signature
            ]
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
                    }

            code_version = get_code_version(project_root)
            created_at = datetime.now().isoformat()
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
            }
            for k in [
                "baseline",
                "param_changes",
                "hypothesis",
                "expected_outcome",
                "rationale",
                "hyperparams",
                "criteria",
                "tags",
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
            
            logger.info(f"Starting research experiment: {' '.join(cmd)}")
            
            # Run in background
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return {
                "ok": True, 
                "pid": process.pid, 
                "experiment_id": experiment_id,
                "symbol": symbol, 
                "type": experiment_type,
                "param_signature": signature,
                "message": f"Experiment {experiment_type} started for {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def get_research_experiments(self) -> List[Dict[str, Any]]:
        """Returns list of experiments from experiments.json"""
        try:
            file_path = Path(__file__).resolve().parent.parent / "experiments.json"
            if not file_path.exists():
                return []
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Convert dict to list and sort by created_at desc
            experiments = list(data.values())
            experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return []
