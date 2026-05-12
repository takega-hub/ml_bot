import time
import asyncio
import logging
import json
import math
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Union, TYPE_CHECKING
from bot.config import AppSettings
from bot.config import (
    get_trailing_activation_pct,
    get_breakeven_activation_pct,
    get_dca_drawdown_pct,
    get_partial_close_progress_pct,
    get_max_margin_usd,
    get_early_exit_profit_pct,
    calculate_margin_based_rr,
    recalculate_tp_sl_for_leverage,
    validate_leverage_settings,
    DEFAULT_LEVERAGE,
)
from bot.state import BotState, TradeRecord
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy, build_ml_signals
from bot.strategy import Action, Signal, Bias
from bot.indicators import prepare_with_indicators
from bot.notification_manager import NotificationManager, NotificationLevel
from bot.paper_trading import PaperTradingManager
from bot.ai_agent_service import AIAgentService
from bot.audit_logger import append_jsonl
from bot.decision_engine import DecisionEngineConfig, DecisionEngineThresholds, DecisionEngineWeights, SignalDecisionEngine

if TYPE_CHECKING:
    from bot.ml.mtf_strategy import MultiTimeframeMLStrategy

# Импортируем исключение для обработки ошибки недостатка средств
try:
    from pybit.exceptions import InvalidRequestError
except ImportError:
    InvalidRequestError = Exception  # Fallback если pybit не установлен

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")
signal_logger = logging.getLogger("signals")

class TradingLoop:
    def __init__(self, settings: AppSettings, state: BotState, bybit: BybitClient, tg_bot=None):
        self.settings = settings
        self.state = state
        self.bybit = bybit
        self.tg_bot = tg_bot
        self.notifier = NotificationManager(tg_bot, settings)
        self.strategies: Dict[str, Union[MLStrategy, 'MultiTimeframeMLStrategy']] = {}
        # Отслеживаем последнюю обработанную свечу для каждого символа и стратегии
        # Структура: {symbol: {strat_id: last_timestamp}}
        self.last_processed_candle: Dict[str, Dict[str, pd.Timestamp]] = {}
        # Кэш сигнала BTCUSDT для проверки направления других пар (обновляется каждые 5 минут)
        self._btc_signal_cache: Optional[Dict] = None
        self._btc_signal_cache_time: Optional[float] = None

        self._tp_cleared_for_symbol = set()
        
        # Ожидающие сигналы для входа по откату (pullback)
        # Структура: {symbol: [{'signal': Signal, 'signal_time': datetime, 'signal_high': float, 'signal_low': float, 'bars_waited': int}, ...]}
        self.pending_pullback_signals: Dict[str, List[Dict]] = {}
        self.pending_pullback_entry_orders: Dict[str, Dict] = {}
        
        # Paper trading manager for online testing of experimental models
        # Pass bot settings for realistic simulation
        # Get real balance from Bybit client
        current_balance = 10000.0  # Default balance
        try:
            if self.bybit:
                balance_info = self.bybit.get_wallet_balance()
                if balance_info.get("retCode") == 0:
                    result = balance_info.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        wallet = list_data[0].get("coin", [])
                        usdt = next((c for c in wallet if c.get("coin") == "USDT"), None)
                        if usdt:
                            current_balance = float(usdt.get("walletBalance", 10000.0))
        except Exception as e:
            logger.error(f"Failed to get balance from Bybit: {e}")
        
        bot_settings = {
            "settings": {
                "strategy": {
                    "use_mtf_strategy": self.settings.ml_strategy.use_mtf_strategy,
                    "confidence_threshold": self.settings.ml_strategy.confidence_threshold,
                    "min_signal_strength": self.settings.ml_strategy.min_signal_strength,
                    "mtf_confidence_threshold_1h": self.settings.ml_strategy.mtf_confidence_threshold_1h,
                    "mtf_confidence_threshold_15m": self.settings.ml_strategy.mtf_confidence_threshold_15m,
                    "mtf_alignment_mode": self.settings.ml_strategy.mtf_alignment_mode,
                    "mtf_require_alignment": self.settings.ml_strategy.mtf_require_alignment,
                },
                "risk": {
                    "base_order_usd": self.settings.risk.base_order_usd,
                    "max_position_usd": self.settings.risk.max_position_usd,
                    "reverse_min_confidence": self.settings.risk.reverse_min_confidence,
                },
                "current_balance": current_balance,
                "is_running": self.state.is_running if hasattr(self.state, 'is_running') else False,
            }
        }
        self.paper_trading_manager = PaperTradingManager(bot_settings)
        self._ai_agent: Optional[AIAgentService] = None
        self._decision_engine: Optional[SignalDecisionEngine] = None
        
        # Валидация моделей при старте
        if self.settings.ml_strategy.use_mtf_strategy or self.settings.ml_strategy.use_scalp_strategy:
            self._validate_mtf_models()

    def _get_ai_agent(self) -> Optional[AIAgentService]:
        if self._ai_agent is not None:
            return self._ai_agent
        try:
            self._ai_agent = AIAgentService()
            return self._ai_agent
        except Exception:
            self._ai_agent = None
            return None

    def _get_decision_engine(self) -> SignalDecisionEngine:
        root = Path(__file__).resolve().parent.parent
        cfg = DecisionEngineConfig(
            enabled=bool(getattr(self.settings.ml_strategy, "decision_engine_enabled", False)),
            mode=str(getattr(self.settings.ml_strategy, "decision_engine_mode", "shadow")),
            thresholds=DecisionEngineThresholds(
                allow_score=float(getattr(self.settings.ml_strategy, "decision_engine_allow_score", 0.35)),
                reduce_score=float(getattr(self.settings.ml_strategy, "decision_engine_reduce_score", 0.10)),
            ),
            weights=DecisionEngineWeights(
                w_ml_confidence=float(getattr(self.settings.ml_strategy, "decision_engine_w_ml_confidence", 1.2)),
                w_mtf_alignment=float(getattr(self.settings.ml_strategy, "decision_engine_w_mtf_alignment", 0.6)),
                w_atr_regime=float(getattr(self.settings.ml_strategy, "decision_engine_w_atr_regime", 0.6)),
                w_sr_proximity=float(getattr(self.settings.ml_strategy, "decision_engine_w_sr_proximity", 0.9)),
                w_trend_slope=float(getattr(self.settings.ml_strategy, "decision_engine_w_trend_slope", 0.3)),
                w_history_edge=float(getattr(self.settings.ml_strategy, "decision_engine_w_history_edge", 1.0)),
            ),
            atr_prefer_min_pct=float(getattr(self.settings.ml_strategy, "decision_engine_atr_prefer_min_pct", 0.35)),
            atr_prefer_max_pct=float(getattr(self.settings.ml_strategy, "decision_engine_atr_prefer_max_pct", 1.60)),
        )
        if self._decision_engine is None:
            self._decision_engine = SignalDecisionEngine(cfg, root)
        else:
            self._decision_engine.config = cfg
        return self._decision_engine

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    @staticmethod
    def _normalize_ai_confirm_entry_result(raw, decision_id: str) -> Dict:
        now_ts = datetime.now(timezone.utc).isoformat()
        if not isinstance(raw, dict):
            raw = {"decision": str(raw)}
        d = raw.get("decision")
        d_str = str(d) if d is not None else ""
        d_upper = d_str.upper()
        d_lower = d_str.lower()
        if d_lower in ("allow", "reduce", "veto"):
            decision = d_lower
        elif d_upper in ("APPROVE", "APPROVED", "ALLOW"):
            decision = "allow"
        elif d_upper in ("REJECT", "REJECTED", "DENY", "DENIED", "VETO", "BLOCK", "BLOCKED"):
            decision = "veto"
        elif d_upper in ("WAIT", "HOLD"):
            decision = "veto"
        else:
            decision = "veto"

        size_multiplier = raw.get("size_multiplier", 1.0)
        try:
            size_multiplier = float(size_multiplier)
        except Exception:
            size_multiplier = 1.0
        if size_multiplier <= 0:
            size_multiplier = 1.0
        if decision == "reduce":
            if size_multiplier not in (0.1, 0.25, 0.5):
                size_multiplier = 0.25
        if decision == "veto":
            size_multiplier = 1.0

        reason_codes = raw.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = raw.get("risk_flags")
        if not isinstance(reason_codes, list):
            reason_codes = []
        reason_codes = [str(x) for x in reason_codes]

        notes = raw.get("notes")
        if not isinstance(notes, str) or not notes:
            notes = raw.get("reason")
        if not isinstance(notes, str) or not notes:
            notes = ""

        confidence = raw.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        risk_score = raw.get("risk_score", 50)
        try:
            risk_score = int(risk_score)
        except Exception:
            risk_score = 50

        latency_ms = raw.get("latency_ms")
        try:
            latency_ms = int(latency_ms)
        except Exception:
            latency_ms = None

        return {
            "decision": decision,
            "confidence": confidence,
            "risk_score": risk_score,
            "size_multiplier": size_multiplier,
            "reason_codes": reason_codes,
            "notes": notes,
            "decision_id": decision_id,
            "timestamp_utc": raw.get("timestamp_utc") or now_ts,
            "latency_ms": latency_ms,
        }

    @classmethod
    def _orderbook_summary(cls, ob: Dict) -> Dict:
        result = ob.get("result") if isinstance(ob, dict) else None
        if not isinstance(result, dict):
            return {}
        bids = result.get("b") or []
        asks = result.get("a") or []

        def _sum_levels(levels, depth: int) -> float:
            total = 0.0
            if not isinstance(levels, list):
                return 0.0
            for level in levels[:depth]:
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    total += cls._safe_float(level[1], 0.0)
            return total

        def _imbalance(depth: int) -> float:
            bid_vol = _sum_levels(bids, depth)
            ask_vol = _sum_levels(asks, depth)
            tot = bid_vol + ask_vol
            if tot <= 0:
                return 0.0
            return (bid_vol - ask_vol) / tot

        best_bid = None
        best_ask = None
        if isinstance(bids, list) and bids:
            lvl = bids[0]
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                best_bid = cls._safe_float(lvl[0], 0.0)
        if isinstance(asks, list) and asks:
            lvl = asks[0]
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                best_ask = cls._safe_float(lvl[0], 0.0)
        spread = None
        spread_pct = None
        if best_bid and best_ask and best_ask > 0:
            spread = best_ask - best_bid
            mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else best_ask
            spread_pct = (spread / mid) * 100 if mid > 0 else None

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "imbalance_5": _imbalance(5),
            "imbalance_20": _imbalance(20),
            "imbalance_50": _imbalance(50),
            "bid_vol_5": _sum_levels(bids, 5),
            "ask_vol_5": _sum_levels(asks, 5),
            "bid_vol_20": _sum_levels(bids, 20),
            "ask_vol_20": _sum_levels(asks, 20),
        }

    @classmethod
    def _recent_trades_summary(cls, trades: List[Dict]) -> Dict:
        if not isinstance(trades, list) or not trades:
            return {}
        buy_vol = 0.0
        sell_vol = 0.0
        last_price = None
        last_ts = None
        for t in trades:
            if not isinstance(t, dict):
                continue
            side = str(t.get("side") or t.get("S") or "").lower()
            size = cls._safe_float(t.get("size") or t.get("v") or t.get("qty") or 0.0, 0.0)
            price = cls._safe_float(t.get("price") or t.get("p") or 0.0, 0.0)
            ts = t.get("time") or t.get("t")
            if last_price is None and price > 0:
                last_price = price
            if last_ts is None and ts is not None:
                last_ts = ts
            if "buy" in side:
                buy_vol += size
            elif "sell" in side:
                sell_vol += size
        total = buy_vol + sell_vol
        return {
            "trades_count": len(trades),
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "buy_sell_ratio": (buy_vol / sell_vol) if sell_vol > 0 else None,
            "total_volume": total,
            "last_price": last_price,
            "last_time": last_ts,
        }

    async def _confirm_entry_with_ai(self, symbol: str, side: str, signal: Signal, position_horizon: Optional[str]) -> Dict:
        decision_id = str(uuid.uuid4())
        now_utc = datetime.now(timezone.utc).isoformat()

        ob = {}
        trades = []
        ohlcv = []
        try:
            if self.bybit:
                ob = await asyncio.to_thread(self.bybit.get_orderbook, symbol, 50)
                trades = await asyncio.to_thread(self.bybit.get_recent_trades, symbol, 200)
                df = await asyncio.to_thread(self.bybit.get_kline_df, symbol, self.settings.timeframe, 60)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    ohlcv = [
                        {
                            "time": int(r.get("timestamp", 0)),
                            "open": float(r.get("open", 0.0)),
                            "high": float(r.get("high", 0.0)),
                            "low": float(r.get("low", 0.0)),
                            "close": float(r.get("close", 0.0)),
                            "volume": float(r.get("volume", 0.0)),
                        }
                        for r in df.tail(60).to_dict(orient="records")
                    ]
        except Exception as e:
            logger.warning(f"[{symbol}] AI confirm_entry market data error: {e}")

        indicators_info = signal.indicators_info if isinstance(signal.indicators_info, dict) else {}
        algo_gate = indicators_info.get("decision_engine_eval") if isinstance(indicators_info.get("decision_engine_eval"), dict) else None
        payload = {
            "request_id": decision_id,
            "timestamp_utc": now_utc,
            "symbol": symbol,
            "timeframe": self.settings.timeframe,
            "signal": {
                "action": signal.action.value,
                "reason": signal.reason or "",
                "price": float(signal.price),
                "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                "take_profit": float(signal.take_profit) if signal.take_profit else None,
                "signal_timestamp": str(signal.timestamp) if getattr(signal, "timestamp", None) is not None else None,
                "confidence": float(indicators_info.get("confidence", 0.0)) if isinstance(indicators_info, dict) else 0.0,
                "strength": str(indicators_info.get("strength", "")) if isinstance(indicators_info, dict) else "",
                "1h_pred": indicators_info.get("1h_pred") if isinstance(indicators_info, dict) else None,
                "1h_conf": indicators_info.get("1h_conf") if isinstance(indicators_info, dict) else None,
                "15m_pred": indicators_info.get("15m_pred") if isinstance(indicators_info, dict) else None,
                "15m_conf": indicators_info.get("15m_conf") if isinstance(indicators_info, dict) else None,
                "4h_pred": indicators_info.get("4h_pred") if isinstance(indicators_info, dict) else None,
                "4h_conf": indicators_info.get("4h_conf") if isinstance(indicators_info, dict) else None,
            },
            "bot_context": {
                "side": side,
                "algo_gate": algo_gate,
                "leverage": int(self.settings.get_leverage_for_symbol(symbol)),
                "position_horizon": position_horizon or "",
                "risk_settings": {
                    "base_order_usd": float(self.settings.risk.base_order_usd),
                    "margin_pct_balance": float(self.settings.risk.margin_pct_balance),
                    "stop_loss_pct": float(self.settings.risk.stop_loss_pct),
                    "take_profit_pct": float(self.settings.risk.take_profit_pct),
                    "max_position_usd": float(self.settings.risk.max_position_usd),
                },
                "ai_fallback_policy": {
                    "force_enabled": bool(getattr(self.settings.ml_strategy, "ai_fallback_force_enabled", False)),
                    "spread_reduce_pct": float(getattr(self.settings.ml_strategy, "ai_fallback_spread_reduce_pct", 0.10)),
                    "spread_veto_pct": float(getattr(self.settings.ml_strategy, "ai_fallback_spread_veto_pct", 0.25)),
                    "min_depth_usd_5": float(getattr(self.settings.ml_strategy, "ai_fallback_min_depth_usd_5", 0.0)),
                    "imbalance_abs_reduce": float(getattr(self.settings.ml_strategy, "ai_fallback_imbalance_abs_reduce", 0.60)),
                    "orderflow_ratio_low": float(getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_low", 0.40)),
                    "orderflow_ratio_high": float(getattr(self.settings.ml_strategy, "ai_fallback_orderflow_ratio_high", 2.50)),
                },
            },
            "market_context": {
                "ohlcv": ohlcv[-60:],
                "orderbook": self._orderbook_summary(ob),
                "recent_trades": self._recent_trades_summary(trades),
            },
        }

        agent = self._get_ai_agent()
        if not agent:
            result = {
                "decision": "allow",
                "confidence": 0.0,
                "risk_score": 50,
                "reason_codes": ["AI_UNAVAILABLE"],
                "notes": "AI agent init failed",
                "decision_id": decision_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "latency_ms": 0,
            }
            result_raw = result
        else:
            # Агент уже возвращает нормализованный результат
            result = await agent.confirm_entry(payload)
            result_raw = result

        root = Path(__file__).resolve().parent.parent
        if isinstance(result, dict) and "decision" in result:
            result = dict(result)
            if not result.get("decision_id"):
                result["decision_id"] = decision_id
            if not result.get("timestamp_utc"):
                result["timestamp_utc"] = now_utc
        else:
            result = self._normalize_ai_confirm_entry_result(result, decision_id)

        append_jsonl(
            str(root / "logs" / "ai_entry_audit.jsonl"),
            {
                "event_type": "confirm_entry",
                "symbol": symbol,
                "side": side,
                "decision_id": result.get("decision_id", decision_id),
                "timestamp_utc": result.get("timestamp_utc", now_utc),
                "latency_ms": result.get("latency_ms"),
                "request": payload,
                "response": result,
                "response_raw": result_raw,
                "decision_engine": algo_gate,
            },
        )
        return result
    
    def _validate_mtf_models(self):
        """Проверяет наличие MTF и Scalp моделей для активных символов при старте"""
        from bot.ml.model_selector import select_best_models, select_best_scalp_model
        
        logger.info("🔍 Валидация ML моделей для активных символов...")
        missing_mtf = []
        missing_scalp = []
        
        for symbol in self.state.active_symbols:
            # Валидация MTF
            if self.settings.ml_strategy.use_mtf_strategy:
                model_1h, model_15m, model_info = select_best_models(symbol=symbol)
                if not model_1h or not model_15m:
                    missing_mtf.append(symbol)
                    logger.warning(f"[{symbol}] ⚠️ MTF модели не найдены")
                else:
                    logger.info(f"[{symbol}] ✅ MTF модели найдены")

            # Валидация Scalp
            if self.settings.ml_strategy.use_scalp_strategy:
                m5m_path, m5m_info = select_best_scalp_model(symbol=symbol)
                if not m5m_path:
                    missing_scalp.append(symbol)
                    logger.warning(f"[{symbol}] ⚠️ Scalp (5m) модель не найдена")
                else:
                    logger.info(f"[{symbol}] ✅ Scalp (5m) модель найдена")
        
        if missing_mtf:
            logger.warning(f"⚠️ MTF стратегия включена, но модели не найдены для: {', '.join(missing_mtf)}")
        if missing_scalp:
            logger.warning(f"⚠️ Scalp стратегия включена, но модели не найдены для: {', '.join(missing_scalp)}")

    async def run(self):
        logger.info("Starting Trading Loop...")
        
        # Устанавливаем is_running = True при запуске (если еще не установлено)
        if not self.state.is_running:
            logger.info("Setting bot state to running...")
            self.state.set_running(True)
        
        # Синхронизируем позиции с биржей при старте
        await self.sync_positions_with_exchange()
        
        # Запускаем оба цикла параллельно с обработкой ошибок
        logger.info("Trading Loop: About to start both loops in parallel...")
        try:
            logger.info("Trading Loop: Starting asyncio.gather...")
            results = await asyncio.gather(
                self._signal_processing_loop(),
                self._position_monitoring_loop(),
                self._maintenance_loop(),
                return_exceptions=True  # Не останавливаемся при ошибке в одном из циклов
            )
            logger.info(f"Trading Loop: asyncio.gather completed with results: {results}")
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
            raise
    
    def _get_seconds_until_next_candle_close(self, timeframe: str) -> float:
        """
        Вычисляет количество секунд до закрытия следующей свечи.
        
        Args:
            timeframe: Таймфрейм ('15m', '1h', '4h', и т.д.)
        
        Returns:
            Количество секунд до закрытия следующей свечи
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # Парсим таймфрейм
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 24 * 60
        else:
            # Пытаемся распарсить как число (минуты)
            try:
                minutes = int(timeframe)
            except:
                minutes = 15  # По умолчанию 15 минут
        
        # Вычисляем время закрытия следующей свечи
        # Для 15m: закрытие в :00, :15, :30, :45
        # Для 1h: закрытие в :00 каждого часа
        # Для 4h: закрытие в 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        
        if minutes < 60:
            # Минутные свечи: округляем до ближайшего кратного minutes
            current_minute = now.minute
            next_close_minute = ((current_minute // minutes) + 1) * minutes
            if next_close_minute >= 60:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = now.replace(minute=next_close_minute, second=0, microsecond=0)
        elif minutes == 60:
            # Часовые свечи: закрытие в :00 каждого часа
            if now.minute == 0 and now.second < 5:
                # Свеча только что закрылась, следующая через час
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            # Многочасовые свечи (4h, 1d и т.д.)
            hours = minutes // 60
            current_hour = now.hour
            next_close_hour = ((current_hour // hours) + 1) * hours
            if next_close_hour >= 24:
                next_close = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                next_close = now.replace(hour=next_close_hour, minute=0, second=0, microsecond=0)
        
        seconds_until_close = (next_close - now).total_seconds()
        return max(0, seconds_until_close)
    
    def _get_seconds_since_last_candle_close(self, timeframe: str) -> float:
        """
        Вычисляет количество секунд с момента закрытия последней свечи.
        
        Args:
            timeframe: Таймфрейм ('15m', '1h', '4h', и т.д.)
        
        Returns:
            Количество секунд с момента закрытия последней свечи
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # Парсим таймфрейм
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 24 * 60
        else:
            try:
                minutes = int(timeframe)
            except:
                minutes = 15
        
        # Вычисляем время закрытия последней свечи
        if minutes < 60:
            current_minute = now.minute
            last_close_minute = (current_minute // minutes) * minutes
            last_close = now.replace(minute=last_close_minute, second=0, microsecond=0)
        elif minutes == 60:
            last_close = now.replace(minute=0, second=0, microsecond=0)
        else:
            hours = minutes // 60
            current_hour = now.hour
            last_close_hour = (current_hour // hours) * hours
            last_close = now.replace(hour=last_close_hour, minute=0, second=0, microsecond=0)
        
        seconds_since_close = (now - last_close).total_seconds()
        return max(0, seconds_since_close)

    async def _signal_processing_loop(self):
        """Основной цикл обработки сигналов с оптимизацией для немедленной обработки после закрытия свечи"""
        logger.info("Starting Signal Processing Loop...")
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.debug(f"Signal Processing Loop: Iteration {iteration}, is_running={self.state.is_running}")
                
                if not self.state.is_running:
                    logger.debug("Signal Processing Loop: Bot not running, sleeping...")
                    await asyncio.sleep(10)
                    continue

                logger.info(f"🔄 Signal Processing Loop: Processing {len(self.state.active_symbols)} symbols...")

                # Определяем минимальный таймфрейм среди активных стратегий для умной паузы
                min_timeframe = self.settings.timeframe
                for symbol in self.state.active_symbols:
                    configs = self.state.get_all_strategies_for_symbol(symbol)
                    if not configs and self.settings.ml_strategy.use_scalp_strategy:
                        # Если конфигов нет, но скальпинг включен глобально, предполагаем 5m
                        min_timeframe = "5m"
                        break
                    for cfg in configs:
                        if cfg.get("mode") == "scalp":
                            min_timeframe = "5m"
                            break
                    if min_timeframe == "5m": break

                for symbol in self.state.active_symbols:
                    logger.info(f"🎯 Signal Processing Loop: Starting to process {symbol}")
                    await self.process_symbol(symbol)
                    logger.info(f"✅ Signal Processing Loop: Completed processing {symbol}")
                    # Добавляем задержку между символами для снижения нагрузки на API
                    if len(self.state.active_symbols) > 1:
                        await asyncio.sleep(2)
                
                # УМНАЯ ПАУЗА: проверяем, когда закроется следующая свеча (используем минимальный ТФ)
                seconds_since_close = self._get_seconds_since_last_candle_close(min_timeframe)
                
                if seconds_since_close <= 30:
                    # Свеча только что закрылась, проверяем снова через 10 секунд для надежности
                    sleep_time = 10
                    logger.info(f"✅ Signal Processing Loop: Candle ({min_timeframe}) closed {seconds_since_close:.1f}s ago, checking again in {sleep_time}s...")
                else:
                    # Обычная пауза, но не больше времени до следующего закрытия
                    seconds_until_close = self._get_seconds_until_next_candle_close(min_timeframe)
                    # Используем минимум из обычной паузы и времени до закрытия (но не меньше 10 секунд)
                    sleep_time = min(self.settings.live_poll_seconds, max(10, seconds_until_close - 5))
                    logger.info(f"✅ Signal Processing Loop: Completed iteration {iteration}, sleeping for {sleep_time}s (next {min_timeframe} candle closes in {seconds_until_close:.1f}s)...")
                
                await asyncio.sleep(sleep_time)
                logger.debug(f"Signal Processing Loop: Woke up from sleep, starting next iteration...")
            except Exception as e:
                logger.error(f"[trading_loop] Error in signal processing loop: {e}")
                await asyncio.sleep(30)
    
    async def _position_monitoring_loop(self):
        """Цикл мониторинга открытых позиций для breakeven и trailing stop"""
        logger.info("Starting Position Monitoring Loop...")
        try:
            logger.info("Position Monitoring Loop: About to sleep for 10 seconds...")
            await asyncio.sleep(10)  # Даем время запуститься основному циклу
            logger.info("Position Monitoring Loop: Sleep completed, continuing...")
        except Exception as e:
            logger.error(f"Error in position monitoring loop initial sleep: {e}", exc_info=True)
            raise
        logger.info("Position Monitoring Loop: Initial delay completed, starting main loop...")
        
        cycle_count = 0
        while True:
            try:
                if not self.state.is_running:
                    logger.debug("Bot is not running, waiting...")
                    await asyncio.sleep(10)
                    continue
                
                cycle_count += 1
                # Логируем каждые 10 циклов (примерно каждые 4 минуты), чтобы видеть, что цикл работает
                if cycle_count % 10 == 0:
                    logger.info(f"📊 Position Monitoring Loop: Cycle {cycle_count} (checking positions every 25s)")
                
                # ОПТИМИЗАЦИЯ: получаем ВСЕ позиции одним запросом вместо отдельных для каждого символа
                # Это значительно снижает количество API запросов и предотвращает rate limit ошибки
                try:
                    logger.debug("Fetching all positions from exchange...")
                    # Добавляем таймаут для предотвращения зависания
                    all_positions = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.bybit.get_position_info,
                            settle_coin="USDT"  # Получаем все USDT позиции одним запросом
                        ),
                        timeout=30.0  # Таймаут 30 секунд
                    )
                    logger.debug(f"Received positions response: retCode={all_positions.get('retCode') if all_positions else 'None'}")
                    
                    if all_positions and all_positions.get("retCode") == 0:
                        result = all_positions.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            
                            # Логируем начало цикла мониторинга позиций
                            open_count = sum(1 for pos in list_data if pos and isinstance(pos, dict) and float(pos.get("size", 0)) > 0)
                            if open_count > 0:
                                logger.info(f"📊 Position Monitoring: Checking {open_count} open position(s)...")
                            
                            # Создаем словарь позиций по символам для быстрого доступа
                            positions_by_symbol = {}
                            for pos in list_data:
                                if pos and isinstance(pos, dict):
                                    symbol = pos.get("symbol")
                                    if symbol in self.state.active_symbols:
                                        positions_by_symbol[symbol] = pos
                            
                            # Обрабатываем позиции для каждого активного символа
                            for symbol in self.state.active_symbols:
                                try:
                                    position = positions_by_symbol.get(symbol)
                                    
                                    if position:
                                        size = float(position.get("size", 0))
                                        
                                        # Проверяем, закрылась ли позиция на бирже
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos and size == 0:
                                            # Позиция закрылась на бирже, но в state еще открыта
                                            await self.handle_position_closed(symbol, local_pos)
                                        elif size > 0:
                                            # Позиция открыта, проверяем частичное закрытие и обновляем стопы
                                            await self.check_partial_close(symbol, position)
                                            
                                            # Обновляем breakeven stop
                                            logger.debug(f"[{symbol}] Calling update_breakeven_stop for position size={size}")
                                            await self.update_breakeven_stop(symbol, position)
                                            
                                            # Обновляем trailing stop
                                            await self.update_trailing_stop(symbol, position)

                                    else:
                                        # Позиции нет в списке, проверяем локальное состояние
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos:
                                            # Позиция закрылась на бирже
                                            await self.handle_position_closed(symbol, local_pos)
                                
                                except Exception as e:
                                    logger.error(f"Error processing position for {symbol}: {e}")
                    else:
                        logger.warning(f"Failed to get positions: retCode={all_positions.get('retCode') if all_positions else 'None'}")
                
                except asyncio.TimeoutError:
                    logger.error("Timeout while fetching positions from exchange (30s)")
                except Exception as e:
                    logger.error(f"Error getting all positions: {e}", exc_info=True)
                
                # Проверяем позиции каждые 25 секунд (увеличено с 15 для снижения нагрузки на API)
                logger.debug("Position monitoring cycle completed, sleeping for 25 seconds...")
                await asyncio.sleep(25)
                logger.debug("Position Monitoring Loop: Woke up from sleep, starting next cycle...")
            
            except Exception as e:
                logger.error(f"[trading_loop] Error in position monitoring loop: {e}")
                await asyncio.sleep(30)

    async def process_symbol(self, symbol: str):
        try:
            logger.info(f"[{symbol}] 🚀 START process_symbol()")
            
            # 0. Проверяем cooldown
            if not self.settings.risk.enable_loss_cooldown:
                if self.state.cooldowns.get(symbol):
                    logger.info(f"[{symbol}] Cooldown found but disabled in settings. Removing.")
                    self.state.remove_cooldown(symbol)
                in_cooldown = False
            else:
                logger.info(f"[{symbol}] Checking cooldown...")
                try:
                    in_cooldown = await asyncio.wait_for(
                        asyncio.to_thread(self.state.is_symbol_in_cooldown, symbol),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{symbol}] Cooldown check timed out, assuming no cooldown")
                    in_cooldown = False
            
            if in_cooldown:
                logger.info(f"[{symbol}] In cooldown, returning")
                return

            # 0.5 Определение активных стратегий
            configs = self.state.get_all_strategies_for_symbol(symbol)
            from bot.ml.model_selector import select_best_models, select_best_scalp_model
            if configs:
                supported_modes = {"mtf", "scalp"}
                original_len = len(configs)
                configs = [
                    c for c in configs
                    if isinstance(c, dict) and str(c.get("mode", "")).strip().lower() in supported_modes
                ]
                if len(configs) < original_len:
                    logger.warning(
                        f"[{symbol}] Removed {original_len - len(configs)} legacy strategy configs "
                        f"(supported modes: mtf, scalp)"
                    )
                    try:
                        self.state.set_strategies_for_symbol(symbol, configs)
                        logger.info(f"[{symbol}] Persisted cleaned strategy config (legacy entries removed)")
                    except Exception as state_err:
                        logger.warning(f"[{symbol}] Failed to persist cleaned strategy config: {state_err}")
            if not configs:
                # Fallback to auto-select if no config found
                m1h, m15m, info = select_best_models(symbol=symbol, use_best_from_comparison=True)
                if m1h and m15m:
                    configs = [{
                        "mode": "mtf",
                        "model_1h_path": m1h,
                        "model_15m_path": m15m,
                        "name": "auto_mtf",
                        "confidence_threshold_1h": info.get('confidence_threshold_1h') or 0.50,
                        "confidence_threshold_15m": info.get('confidence_threshold_15m') or 0.35,
                    }]

            # Добавляем scalp-параллель всегда, если включен глобально и еще не добавлен.
            has_scalp_cfg = any(
                isinstance(c, dict) and str(c.get("mode", "")).strip().lower() == "scalp"
                for c in (configs or [])
            )
            if self.settings.ml_strategy.use_scalp_strategy and not has_scalp_cfg:
                m5m_path, m5m_info = select_best_scalp_model(symbol=symbol)
                if m5m_path:
                    scalp_config = {
                        "mode": "scalp",
                        "model_path": m5m_path,
                        "name": f"scalp_{Path(m5m_path).stem}",
                        "confidence_threshold": self.settings.ml_strategy.scalp_confidence_threshold,
                    }
                    if not configs:
                        configs = [scalp_config]
                    else:
                        configs.append(scalp_config)
                    logger.info(f"[{symbol}] ✅ Scalp strategy attached: {Path(m5m_path).name}")
                else:
                    logger.warning(f"[{symbol}] ⚠️ Scalp enabled but no 5m model found")

            if not configs:
                logger.warning(f"[{symbol}] No strategy configs found, skipping")
                return

            # 1. Получаем данные
            required_limit = 500
            
            # 5m data (scalping)
            df_5m = None
            has_scalp = any(c.get("mode") == "scalp" for c in configs)
            if has_scalp:
                try:
                    df_5m = await asyncio.wait_for(asyncio.to_thread(self._load_cached_5m_data, symbol), timeout=5.0)
                    needs_update_5m = df_5m is None or df_5m.empty
                    if not needs_update_5m:
                        last_5m = df_5m.index[-1]
                        if (pd.Timestamp.now() - last_5m).total_seconds() / 60 > 5:
                            needs_update_5m = True
                    if needs_update_5m:
                        df_5m = await asyncio.wait_for(asyncio.to_thread(self._fetch_and_cache_5m_data, symbol, df_5m), timeout=20.0)
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to fetch 5m data: {e}")

            # 15m data
            try:
                df = await asyncio.wait_for(
                    asyncio.to_thread(self._load_cached_15m_data, symbol),
                    timeout=5.0
                )
            except Exception:
                df = None
            
            needs_update = df is None or df.empty
            if not needs_update and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                last_candle_time = df.index[-1]
                current_time = pd.Timestamp.now()
                last_minute = last_candle_time.minute
                next_close_minute = ((last_minute // 15) + 1) * 15
                if next_close_minute >= 60:
                    next_close_time = last_candle_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
                else:
                    next_close_time = last_candle_time.replace(minute=next_close_minute, second=0, microsecond=0)

                if current_time >= next_close_time or (current_time - last_candle_time).total_seconds() / 60 > 20 or len(df) < required_limit:
                    needs_update = True
            
            if needs_update:
                try:
                    df = await asyncio.wait_for(
                        asyncio.to_thread(self._fetch_and_cache_15m_data, symbol, df, required_limit),
                        timeout=30.0
                    )
                except Exception:
                    return

            if df is None or df.empty:
                return

            # 1h data
            df_1h_cached = await asyncio.to_thread(self._load_cached_1h_data, symbol)
            needs_update_1h = df_1h_cached is None or df_1h_cached.empty
            if not needs_update_1h:
                last_1h = df_1h_cached.index[-1]
                if (pd.Timestamp.now() - last_1h).total_seconds() / 3600 > 2:
                    needs_update_1h = True
            
            if needs_update_1h:
                df_1h_cached = await asyncio.to_thread(self._fetch_and_cache_1h_data, symbol, df_1h_cached)

            # 1.5 Safety Guard: Check data sync
            if df is not None and not df.empty and df_1h_cached is not None and not df_1h_cached.empty:
                last_15m = df.index[-1]
                last_1h = df_1h_cached.index[-1]
                diff_seconds = abs((last_15m - last_1h).total_seconds())
                max_diff = getattr(self.settings.ml_strategy, "data_sync_max_diff_seconds", 7200)

                if diff_seconds > max_diff:
                    logger.warning(f"[{symbol}] 🛡️ SAFETY GUARD: Data streams out of sync! 15m: {last_15m}, 1h: {last_1h}, Diff: {diff_seconds/60:.1f}m. Skipping.")
                    return

            # 2. Инициализируем стратегии (Multi-strategy support)
            # Create strategy instances
            active_strats = []
            for cfg in configs:
                strat_id = f"{symbol}_{cfg.get('name', 'unnamed')}"
                instance = self.strategies.get(strat_id)
                
                # Check if needs reinit
                needs_init = instance is None
                if instance:
                    if cfg.get("mode") == "mtf" and not hasattr(instance, 'predict_combined'):
                        needs_init = True
                    elif cfg.get("mode") == "scalp" and hasattr(instance, 'predict_combined'):
                        needs_init = True

                if needs_init:
                    from bot.ml.mtf_strategy import MultiTimeframeMLStrategy
                    ms = self.settings.ml_strategy
                    if cfg.get("mode") == "mtf":
                        instance = MultiTimeframeMLStrategy(
                            model_1h_path=cfg["model_1h_path"],
                            model_15m_path=cfg["model_15m_path"],
                            confidence_threshold_1h=cfg.get("confidence_threshold_1h", 0.50),
                            confidence_threshold_15m=cfg.get("confidence_threshold_15m", 0.35),
                            require_alignment=cfg.get("require_alignment", True),
                            alignment_mode=cfg.get("alignment_mode", "strict"),
                            use_dynamic_ensemble_weights=getattr(ms, "use_dynamic_ensemble_weights", False),
                            use_fixed_sl_from_risk=getattr(ms, "use_fixed_sl_from_risk", False),
                        )
                    elif cfg.get("mode") == "scalp":
                        instance = MLStrategy(
                            model_path=cfg["model_path"],
                            confidence_threshold=cfg.get("confidence_threshold", 0.35),
                            min_signal_strength=getattr(ms, "scalp_min_signal_strength", "умеренное"),
                            use_dynamic_ensemble_weights=getattr(ms, "use_dynamic_ensemble_weights", False),
                            use_fixed_sl_from_risk=getattr(ms, "use_fixed_sl_from_risk", False),
                        )
                    else:
                        logger.warning(f"[{symbol}] Unsupported strategy mode: {cfg.get('mode')}, skipping")
                        continue
                    self.strategies[strat_id] = instance
                active_strats.append((cfg, instance))

            # 2.5 ОПТИМИЗАЦИЯ: Рассчитываем индикаторы один раз для всех стратегий
            from bot.ml.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()

            if df is not None and not df.empty:
                try:
                    df = fe.create_technical_indicators(df)
                except Exception as fe_err:
                    logger.error(f"[{symbol}] Error calculating 15m features: {fe_err}")

            if df_5m is not None and not df_5m.empty:
                try:
                    df_5m = fe.create_technical_indicators(df_5m)
                except Exception as fe_err:
                    logger.error(f"[{symbol}] Error calculating 5m features: {fe_err}")

            if df_1h_cached is not None and not df_1h_cached.empty:
                try:
                    # Проверяем, нужно ли считать фичи (если их мало, значит это "сырые" данные)
                    if len(df_1h_cached.columns) < 15:
                        df_1h_cached = fe.create_technical_indicators(df_1h_cached)
                except Exception as fe_err:
                    logger.error(f"[{symbol}] Error calculating 1h features: {fe_err}")

            # Состояние позиции нужно заранее: используется в generate_signal и при исполнении.
            has_pos = None
            try:
                pos_info = self.bybit.get_position_info(symbol=symbol)
                if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                    result = pos_info.get("result")
                    if result and isinstance(result, dict):
                        list_data = result.get("list", [])
                        if list_data and isinstance(list_data, list):
                            p = list_data[0]
                            if p and isinstance(p, dict):
                                size = float(p.get("size", 0) or 0)
                                if size > 0:
                                    side = p.get("side")
                                    if side == "Buy":
                                        has_pos = Bias.LONG
                                    elif side == "Sell":
                                        has_pos = Bias.SHORT
            except Exception as pos_err:
                logger.warning(f"[{symbol}] Failed to load position before signal generation: {pos_err}")

            if has_pos is None:
                local_pos = self.state.get_open_position(symbol)
                if local_pos and getattr(local_pos, "status", "") == "open":
                    local_side = str(getattr(local_pos, "side", "")).strip().lower()
                    if local_side == "buy":
                        has_pos = Bias.LONG
                    elif local_side == "sell":
                        has_pos = Bias.SHORT

            # 3. Генерируем сигналы от всех стратегий
            all_signals = []

            if symbol not in self.last_processed_candle:
                self.last_processed_candle[symbol] = {}

            for cfg, strat in active_strats:
                strat_id = f"{symbol}_{cfg.get('name', 'unnamed')}"
                try:
                    # Определяем текущий таймстемп и данные для данной стратегии
                    if cfg.get("mode") == "scalp" and df_5m is not None and not df_5m.empty:
                        strat_row = df_5m.iloc[-1]
                        ts = strat_row.get('timestamp') if 'timestamp' in strat_row else df_5m.index[-1]
                        strat_df = df_5m
                    else:
                        strat_row = df.iloc[-1]
                        ts = strat_row.get('timestamp') if 'timestamp' in strat_row else df.index[-1]
                        strat_df = df

                    # Проверяем, обрабатывали ли уже эту свечу для этой стратегии
                    if ts is not None and self.last_processed_candle[symbol].get(strat_id) == ts:
                        continue

                    if cfg.get("mode") == "scalp" and df_5m is not None and not df_5m.empty:
                        # Scalp strategy uses 5m data
                        sig = await asyncio.to_thread(
                            strat.generate_signal,
                            row=strat_row, df=strat_df,
                            has_position=has_pos, current_price=strat_row['close'],
                            leverage=self.settings.get_leverage_for_symbol(symbol),
                            skip_feature_creation=True,
                        )
                        if sig:
                            sig.indicators_info = sig.indicators_info or {}
                            sig.indicators_info["interval"] = 5
                            sig.indicators_info["strategy"] = "SCALP"
                    elif hasattr(strat, 'predict_combined'):
                        # MTF strategy
                        sig = await asyncio.to_thread(
                            strat.generate_signal,
                            row=strat_row, df_15m=df, df_1h=df_1h_cached,
                            has_position=has_pos, current_price=strat_row['close'],
                            leverage=self.settings.get_leverage_for_symbol(symbol),
                            skip_feature_creation=True,
                        )
                    else:
                        logger.warning(f"[{symbol}] Unsupported strategy instance for {strat_id}, skipping")
                        continue

                    if sig and sig.action != Action.HOLD:
                        sig.model_name = strat_id
                        all_signals.append(sig)

                    # Обновляем отметку времени для стратегии
                    if ts is not None:
                        self.last_processed_candle[symbol][strat_id] = ts

                except Exception as e:
                    logger.error(f"Error in strategy {strat_id}: {e}")

            if not all_signals:
                return

            # 4. Aggregation / Judge Logic
            mode = getattr(self.settings.ml_strategy, "signal_aggregation_mode", "highest_confidence")
            winning_signal = None

            if mode == "highest_confidence":
                all_signals.sort(key=lambda x: (x.indicators_info or {}).get("confidence", 0), reverse=True)
                winning_signal = all_signals[0]
            elif mode == "consensus":
                # Only take action if N strategies agree on direction
                longs = [s for s in all_signals if s.action == Action.LONG]
                shorts = [s for s in all_signals if s.action == Action.SHORT]
                min_count = getattr(self.settings.ml_strategy, "consensus_min_count", 2)

                if len(longs) >= min_count:
                    longs.sort(key=lambda x: (x.indicators_info or {}).get("confidence", 0), reverse=True)
                    winning_signal = longs[0]
                    winning_signal.reason = f"consensus_long_{len(longs)}_of_{len(all_signals)}"
                elif len(shorts) >= min_count:
                    shorts.sort(key=lambda x: (x.indicators_info or {}).get("confidence", 0), reverse=True)
                    winning_signal = shorts[0]
                    winning_signal.reason = f"consensus_short_{len(shorts)}_of_{len(all_signals)}"
                else:
                    logger.info(f"[{symbol}] No consensus: Longs={len(longs)}, Shorts={len(shorts)}, Min={min_count}")
                    return
            elif mode == "weighted_voting":
                # Dynamic weight: adapt by regime + signal strength
                long_score = 0.0
                short_score = 0.0
                regime = "neutral"
                try:
                    adx = float(df.iloc[-1].get("adx", np.nan))
                    atr_pct = float(df.iloc[-1].get("atr_pct", np.nan))
                    if np.isfinite(adx):
                        if adx >= 25:
                            regime = "trend"
                        elif adx <= 20:
                            regime = "sideways"
                    if np.isfinite(atr_pct) and atr_pct >= 1.2:
                        regime = "volatile"
                except Exception:
                    regime = "neutral"
                for s in all_signals:
                    model_name_lower = s.model_name.lower()
                    if "mtf" in model_name_lower:
                        weight = 1.0
                    elif "scalp" in model_name_lower:
                        weight = 0.8
                    else:
                        weight = 0.6

                    if regime == "trend" and "mtf" in model_name_lower:
                        weight *= 1.25
                    elif regime == "sideways" and "scalp" in model_name_lower:
                        weight *= 1.25
                    elif regime == "volatile" and "scalp" in model_name_lower:
                        weight *= 1.15

                    conf = (s.indicators_info or {}).get("confidence", 0)
                    strength = str((s.indicators_info or {}).get("strength", "")).strip().lower()
                    strength_mult = {
                        "очень_сильное": 1.15,
                        "сильное": 1.08,
                        "среднее": 1.03,
                        "умеренное": 1.0,
                        "слабое": 0.95,
                    }.get(strength, 1.0)
                    if s.action == Action.LONG:
                        long_score += conf * weight * strength_mult
                    else:
                        short_score += conf * weight * strength_mult

                if long_score > short_score and long_score > 0.5:
                    longs = [s for s in all_signals if s.action == Action.LONG]
                    longs.sort(key=lambda x: (x.indicators_info or {}).get("confidence", 0), reverse=True)
                    winning_signal = longs[0]
                    winning_signal.reason = f"weighted_vote_long_score_{long_score:.2f}"
                elif short_score > long_score and short_score > 0.5:
                    shorts = [s for s in all_signals if s.action == Action.SHORT]
                    shorts.sort(key=lambda x: (x.indicators_info or {}).get("confidence", 0), reverse=True)
                    winning_signal = shorts[0]
                    winning_signal.reason = f"weighted_vote_short_score_{short_score:.2f}"
                else:
                    logger.info(f"[{symbol}] Weighted vote inconclusive: Long={long_score:.2f}, Short={short_score:.2f}")
                    return

            if not winning_signal:
                return

            logger.info(f"[{symbol}] WINNING SIGNAL ({mode}): {winning_signal.action.value} from {winning_signal.model_name} (Conf: {(winning_signal.indicators_info or {}).get('confidence', 0):.2%})")

            # 5. Decision Engine & AI Agent Confirmation
            indicators_info = winning_signal.indicators_info or {}
            confidence = indicators_info.get("confidence", 0)
            current_price = float(getattr(winning_signal, "price", 0) or df.iloc[-1]["close"])

            # Keep legacy-compatible signal entries in signals.log for dashboard/API consumers.
            signal_logger.info(
                f"SIGNAL GEN: {symbol} {winning_signal.action.value} Conf={confidence:.2f} Price={current_price:.2f} "
                f"Strategy={indicators_info.get('strategy', 'UNKNOWN')} "
                f"Model={getattr(winning_signal, 'model_name', '') or indicators_info.get('model_name', 'UNKNOWN')} "
                f"Reason={winning_signal.reason}"
            )

            engine = self._get_decision_engine()
            ohlcv = [{"time": int(r.get("timestamp", 0)), "open": float(r.get("open", 0.0)), "high": float(r.get("high", 0.0)), "low": float(r.get("low", 0.0)), "close": float(r.get("close", 0.0)), "volume": float(r.get("volume", 0.0))} for r in df.tail(60).to_dict(orient="records")]

            eval_payload = {
                "action": winning_signal.action.value,
                "price": float(current_price),
                "confidence": float(confidence),
                "strength": str(indicators_info.get("strength", "")),
                "model_name": winning_signal.model_name
            }

            engine_eval = engine.evaluate(
                symbol=symbol,
                side="Buy" if winning_signal.action == Action.LONG else "Sell",
                signal_payload=eval_payload,
                ohlcv=ohlcv,
            )

            if isinstance(engine_eval, dict):
                engine_eval["decision_id"] = str(uuid.uuid4())
                indicators_info["decision_engine_eval"] = engine_eval
                indicators_info["engine_decision_id"] = engine_eval["decision_id"]
                winning_signal.indicators_info = indicators_info
                signal_logger.info(
                    f"ALGO GATE: {symbol} action={winning_signal.action.value} score={engine_eval.get('score')} "
                    f"decision={engine_eval.get('decision')} mult={engine_eval.get('size_multiplier')} "
                    f"codes={engine_eval.get('reason_codes')}"
                )
                signal_logger.info(
                    f"DECISION ENGINE: {symbol} {'Buy' if winning_signal.action == Action.LONG else 'Sell'} "
                    f"action={winning_signal.action.value} score={engine_eval.get('score')} "
                    f"decision={engine_eval.get('decision')} mult={engine_eval.get('size_multiplier')} "
                    f"codes={engine_eval.get('reason_codes')}"
                )
                if isinstance(engine_eval.get("engine"), dict):
                    signal_logger.info(
                        "ALGO GATE DETAILS: " + json.dumps(engine_eval.get("engine"), ensure_ascii=False, separators=(",", ":"))
                    )
                    signal_logger.info(
                        "DECISION ENGINE DETAILS: "
                        + json.dumps(engine_eval.get("engine"), ensure_ascii=False, separators=(",", ":"))
                    )
                engine_mode = str(getattr(self.settings.ml_strategy, "decision_engine_mode", "shadow"))
                engine_decision = str(engine_eval.get("decision", "")).lower()
                if engine_decision not in ("allow", "reduce", "veto"):
                    engine_decision = "veto"
                if bool(getattr(self.settings.ml_strategy, "decision_engine_enabled", False)) and engine_mode == "enforce" and engine_decision == "veto":
                    logger.info(
                        f"[{symbol}] 🚫 Signal blocked at stage=algo_gate "
                        f"decision_id={engine_eval.get('decision_id')} codes={engine_eval.get('reason_codes')}"
                    )
                    return

            # AI Confirmation if needed
            ai_agent = self._get_ai_agent()
            ai_confirmation_enabled = bool(
                getattr(self.settings.ml_strategy, "ai_entry_confirmation_enabled", False)
                or getattr(self.settings.ml_strategy, "ai_agent_enabled", False)  # legacy compatibility
            )
            if ai_agent and ai_confirmation_enabled:
                ai_resp = await self._confirm_entry_with_ai(
                    symbol=symbol,
                    side="Buy" if winning_signal.action == Action.LONG else "Sell",
                    signal=winning_signal,
                    position_horizon=self._classify_position_horizon(winning_signal),
                )
                if ai_resp:
                    ai_decision = (
                        ai_resp
                        if isinstance(ai_resp, dict) and "decision" in ai_resp
                        else self._normalize_ai_confirm_entry_result(ai_resp, str(uuid.uuid4()))
                    )
                    indicators_info["ai_entry_confirmation"] = ai_decision
                    winning_signal.indicators_info = indicators_info
                    signal_logger.info(
                        f"AI GATE: {symbol} action={winning_signal.action.value} decision={ai_decision.get('decision')} "
                        f"mult={ai_decision.get('size_multiplier')} codes={ai_decision.get('reason_codes')} "
                        f"decision_id={ai_decision.get('decision_id')}"
                    )
                    ai_mode = str(getattr(self.settings.ml_strategy, "ai_entry_confirmation_mode", "enforce"))
                    if ai_mode != "shadow" and ai_decision.get("decision") == "veto":
                        logger.info(f"[{symbol}] AI VETOED signal from {winning_signal.model_name}")
                        return

            # 6. Execution
            if has_pos is None:
                side_to_exec = "Buy" if winning_signal.action == Action.LONG else "Sell"
                await self.execute_trade(symbol, side_to_exec, winning_signal)

        except Exception as e:
            logger.error(f"Error in process_symbol for {symbol}: {e}", exc_info=True)
            strategy = self.strategies.get(symbol)
            if strategy is None:
                logger.warning(f"[{symbol}] No legacy strategy instance found for fallback path; skipping symbol.")
                return
            # Определяем, какая свеча закрыта и может быть использована для предсказания
            # ВАЖНО: Используем последнюю закрытую свечу (как в тесте), а не предпоследнюю
            # Последняя свеча считается закрытой, если прошло достаточно времени с момента её закрытия
            # Для 15m свечей: если прошло > 1 минуты с момента закрытия, свеча считается закрытой
            
            # Получаем timestamp последней свечи
            last_row = df.iloc[-1]
            last_timestamp = last_row.get('timestamp') if 'timestamp' in last_row else None
            if last_timestamp is None:
                last_timestamp = df.index[-1] if len(df.index) > 0 else None
            
            # ВАЖНО: В тесте всегда используется последняя свеча (которая уже закрыта в исторических данных)
            # В реальном боте мы должны использовать последнюю закрытую свечу
            # Для максимального соответствия тесту, используем последнюю свечу из данных
            # (она считается закрытой, так как мы обрабатываем её после закрытия)
            
            # В тесте: row = df_with_features.iloc[idx] - текущая свеча
            # В реальном боте: row = df.iloc[-1] - последняя свеча (аналог текущей в тесте)
            if len(df) >= 1:
                # Используем последнюю свечу (как в тесте используется текущая свеча)
                row = df.iloc[-1]
                current_price = row['close']
                candle_timestamp = last_timestamp
                use_last_candle = True  # Всегда используем последнюю свечу (как в тесте)
                logger.debug(f"[{symbol}] Using last candle for prediction (as in backtest)")
            else:
                row = df.iloc[-1]
                current_price = row['close']
                candle_timestamp = last_timestamp
                use_last_candle = True
                logger.debug(f"[{symbol}] Using only available candle for prediction")
            
            # Логируем время закрытия свечи и задержку обработки
            if candle_timestamp is not None:
                try:
                    from datetime import datetime
                    if isinstance(candle_timestamp, pd.Timestamp):
                        candle_close_time = candle_timestamp
                    elif isinstance(candle_timestamp, (int, float)):
                        candle_close_time = pd.Timestamp(candle_timestamp, unit='ms')
                    else:
                        candle_close_time = pd.Timestamp(candle_timestamp)
                    
                    now = pd.Timestamp.now()
                    delay_seconds = (now - candle_close_time).total_seconds()
                    delay_minutes = delay_seconds / 60
                    
                    logger.info(
                        f"[{symbol}] 📊 Candle info: closed at {candle_close_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"processing delay: {delay_seconds:.1f}s ({delay_minutes:.2f} min)"
                    )
                except Exception as e:
                    logger.debug(f"[{symbol}] Could not calculate candle delay: {e}")
            
            # Проверяем, не обрабатывали ли мы уже эту свечу
            # ВАЖНО: Проверяем только если timestamp валиден
            # Это предотвращает генерацию одинаковых сигналов для одной и той же закрытой свечи
            if candle_timestamp is not None:
                if symbol in self.last_processed_candle:
                    last_timestamp = self.last_processed_candle[symbol]
                    if last_timestamp is not None and last_timestamp == candle_timestamp:
                        # Эта свеча уже была обработана, пропускаем
                        logger.info(f"[{symbol}] ⏭️ Candle already processed: {candle_timestamp}, skipping signal generation")
                        logger.debug(f"[{symbol}] Last processed: {last_timestamp}, Current: {candle_timestamp}")
                        # Логируем, была ли свеча обработана с сигналом или без
                        logger.debug(f"[{symbol}] This candle was already processed in a previous iteration")
                        return
                
                # ВАЖНО: НЕ сохраняем timestamp здесь, а только после успешной обработки сигнала
                # Это позволит повторить обработку при ошибке
                logger.debug(f"[{symbol}] 📝 New candle detected: {candle_timestamp} (will save after successful processing)")
            else:
                logger.warning(f"[{symbol}] ⚠️ Warning: candle_timestamp is None, proceeding anyway...")
                # Если timestamp None, не сохраняем его, чтобы не блокировать следующие проверки
            
            # Проверяем позицию
            try:
                pos_info = self.bybit.get_position_info(symbol=symbol)
            except Exception as e:
                logger.error(f"Error getting position info for {symbol}: {e}")
                pos_info = None
            
            has_pos = None
            size = 0.0
            entry_price = 0.0
            
            if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                result = pos_info.get("result")
                if result and isinstance(result, dict):
                    list_data = result.get("list", [])
                    if list_data and len(list_data) > 0:
                        p = list_data[0]
                        if p and isinstance(p, dict):
                            size = float(p.get("size", 0))
                            if size > 0:
                                side = p.get("side")
                                has_pos = Bias.LONG if side == "Buy" else Bias.SHORT
                                entry_price = float(p.get("avgPrice", 0))
            elif pos_info is None:
                logger.warning(f"Position info is None for {symbol}")

            local_pos = self.state.get_open_position(symbol)
            
            if has_pos is not None and not local_pos:
                rec = self.pending_pullback_entry_orders.get(symbol)
                if rec and isinstance(rec, dict) and isinstance(rec.get("signal"), Signal):
                    s = rec.get("signal")
                    indicators_info = s.indicators_info if isinstance(s.indicators_info, dict) else {}
                    confidence = indicators_info.get("confidence", 0) if isinstance(indicators_info, dict) else 0
                    signal_strength = indicators_info.get("strength", "") if isinstance(indicators_info, dict) else ""
                    signal_parameters = {}
                    if isinstance(indicators_info, dict) and isinstance(indicators_info.get("decision_engine_eval"), dict):
                        signal_parameters["decision_engine_eval"] = indicators_info.get("decision_engine_eval")
                        if isinstance(indicators_info.get("engine_decision_id"), str):
                            signal_parameters["engine_decision_id"] = indicators_info.get("engine_decision_id")
                    ai_entry_confirmation = indicators_info.get("ai_entry_confirmation") if isinstance(indicators_info, dict) else None
                    if isinstance(ai_entry_confirmation, dict):
                        signal_parameters["ai_entry_confirmation"] = ai_entry_confirmation
                    trade = TradeRecord(
                        symbol=symbol,
                        side=side if side in ("Buy", "Sell") else ("Buy" if has_pos == Bias.LONG else "Sell"),
                        entry_price=float(entry_price),
                        qty=float(size),
                        status="open",
                        model_name=self.state.symbol_models.get(symbol, ""),
                        horizon=self._classify_position_horizon(s),
                        entry_reason=s.reason or "",
                        confidence=confidence,
                        take_profit=s.take_profit or indicators_info.get("take_profit"),
                        stop_loss=s.stop_loss or indicators_info.get("stop_loss"),
                        leverage=self.settings.get_leverage_for_symbol(symbol),
                        margin_usd=float(rec.get("expected_margin_usd") or 0.0),
                        signal_strength=str(signal_strength),
                        signal_parameters=signal_parameters,
                    )
                    self.state.add_trade(trade)
                    logger.info(f"[{symbol}] 🔄 Added filled pullback limit entry to state (qty={size}, avg={entry_price})")
                else:
                    trade = TradeRecord(
                        symbol=symbol,
                        side=side if side in ("Buy", "Sell") else ("Buy" if has_pos == Bias.LONG else "Sell"),
                        entry_price=float(entry_price),
                        qty=float(size),
                        status="open",
                        model_name=self.state.symbol_models.get(symbol, ""),
                        leverage=self.settings.get_leverage_for_symbol(symbol),
                    )
                    self.state.add_trade(trade)
                    logger.info(f"[{symbol}] 🔄 Added exchange position to state (qty={size}, avg={entry_price})")
                if symbol in self.pending_pullback_entry_orders:
                    await self._cancel_pullback_entry_order(symbol, reason="position_detected")
            
            # Обрабатываем pending сигналы (вход по откату) - проверяем ДО генерации нового сигнала
            if has_pos is None and self._pullback_limit_roll_enabled() and df is not None and not df.empty:
                try:
                    if candle_timestamp is not None:
                        await self._tick_pullback_entry_order(symbol, candle_timestamp)
                except Exception as e:
                    logger.warning(f"[{symbol}] ⚠️ Error ticking pullback limit order: {e}")
            elif has_pos is None and self.settings.ml_strategy.pullback_enabled and df is not None and not df.empty:
                try:
                    # Получаем текущие данные свечи
                    if len(df) > 0:
                        current_price = float(df['close'].iloc[-1])
                        high = float(df['high'].iloc[-1])
                        low = float(df['low'].iloc[-1])
                        
                        pullback_signal = await self._process_pending_pullback_signals(
                            symbol, current_price, high, low, df
                        )
                        if pullback_signal is not None:
                            # Условия отката выполнены - открываем позицию
                            if pullback_signal.action == Action.LONG:
                                logger.info(f"[{symbol}] ✅ Opening LONG position after pullback")
                                await self.execute_trade(symbol, "Buy", pullback_signal)
                            elif pullback_signal.action == Action.SHORT:
                                logger.info(f"[{symbol}] ✅ Opening SHORT position after pullback")
                                await self.execute_trade(symbol, "Sell", pullback_signal)
                            if candle_timestamp is not None:
                                self.last_processed_candle[symbol] = candle_timestamp
                            return  # Выходим, так как открыли позицию
                except Exception as e:
                    logger.error(f"[{symbol}] Error processing pending pullback signals (before signal generation): {e}")
                    import traceback
                    logger.error(f"[{symbol}] Traceback:\n{traceback.format_exc()}")
                    # Продолжаем обработку, не прерываем цикл

            # Генерация сигнала
            # КРИТИЧНО: generate_signal() выполняет долгие синхронные операции (feature engineering, model.predict)
            # Оборачиваем в to_thread() чтобы не блокировать event loop
            try:
                logger.info(f"[{symbol}] 🔄 Calling strategy.generate_signal() in thread...")
                
                # Подготавливаем данные для стратегии
                # ВАЖНО: В тесте используется df_window = df_with_features.iloc[:idx+1] - ВСЕ данные до текущего момента ВКЛЮЧИТЕЛЬНО
                # В реальном боте: df_for_strategy = df - ВСЕ данные, включая последнюю свечу (аналог [:idx+1] в тесте)
                df_for_strategy = df  # Используем все данные, включая последнюю свечу (как в тесте)
                logger.debug(f"[{symbol}] Using all data including last candle for strategy (as in backtest)")
                
                # Для MTF стратегии передаем df_15m (текущие данные) и df_1h (из кэша или None)
                # Для обычной стратегии передаем df как обычно
                if hasattr(strategy, 'predict_combined'):
                    # Это MTF стратегия - передаем df_15m и df_1h
                    # Логируем информацию о данных для диагностики
                    if df_1h_cached is not None and not df_1h_cached.empty:
                        logger.debug(f"[{symbol}] MTF: Using cached 1h data ({len(df_1h_cached)} candles)")
                        if isinstance(df_1h_cached.index, pd.DatetimeIndex) and len(df_1h_cached) > 0:
                            logger.debug(f"[{symbol}] MTF: 1h data range: {df_1h_cached.index[0]} to {df_1h_cached.index[-1]}")
                    else:
                        logger.debug(f"[{symbol}] MTF: No cached 1h data, will aggregate from 15m")
                    
                    logger.debug(f"[{symbol}] MTF: 15m data: {len(df_for_strategy)} candles")
                    
                    signal = await asyncio.to_thread(
                        strategy.generate_signal,
                        row=row,
                        df_15m=df_for_strategy,  # 15m данные
                        df_1h=df_1h_cached,  # 1h данные из кэша (если есть) или None (будет агрегировано)
                        has_position=has_pos,
                        current_price=current_price,
                        leverage=self.settings.get_leverage_for_symbol(symbol),
                        target_profit_pct_margin=self.settings.ml_strategy.target_profit_pct_margin,
                        max_loss_pct_margin=self.settings.ml_strategy.max_loss_pct_margin,
                        stop_loss_pct=self.settings.risk.stop_loss_pct,
                        take_profit_pct=self.settings.risk.take_profit_pct,
                    )
                else:
                    # Обычная стратегия - передаем df как обычно
                    signal = await asyncio.to_thread(
                        strategy.generate_signal,
                        row=row,
                        df=df_for_strategy,
                        has_position=has_pos,
                        current_price=current_price,
                        leverage=self.settings.get_leverage_for_symbol(symbol),
                        target_profit_pct_margin=self.settings.ml_strategy.target_profit_pct_margin,
                        max_loss_pct_margin=self.settings.ml_strategy.max_loss_pct_margin,
                        stop_loss_pct=self.settings.risk.stop_loss_pct,
                        take_profit_pct=self.settings.risk.take_profit_pct,
                    )
                logger.info(f"[{symbol}] ✅ strategy.generate_signal() completed")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
            
            if not signal:
                logger.warning(f"No signal generated for {symbol}")
                return
            
            # Сохраняем время получения сигнала для проверки "свежести"
            signal_received_time = pd.Timestamp.now()
            
            # Логируем каждый сигнал (для отладки)
            indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
            
            # Сохраняем время получения сигнала в indicators_info для использования в execute_trade
            if indicators_info is None:
                indicators_info = {}
            indicators_info['signal_received_time'] = signal_received_time.isoformat()
            if "strategy" not in indicators_info or not indicators_info.get("strategy"):
                if "scalp" in str(getattr(signal, "model_name", "")).lower():
                    indicators_info["strategy"] = "SCALP"
                elif "mtf" in str(getattr(signal, "model_name", "")).lower():
                    indicators_info["strategy"] = "MTF"
            indicators_info["model_name"] = str(getattr(signal, "model_name", "") or indicators_info.get("model_name", ""))
            signal.indicators_info = indicators_info
            
            confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
            # Для MTF стратегии показываем детальную информацию
            if isinstance(indicators_info, dict) and indicators_info.get('strategy') == 'MTF_ML':
                mtf_reason = indicators_info.get('reason', 'unknown')
                conf_1h = indicators_info.get('1h_conf', 0)
                conf_15m = indicators_info.get('15m_conf', 0)
                pred_1h = indicators_info.get('1h_pred', 0)
                pred_15m = indicators_info.get('15m_pred', 0)
                logger.info(
                    f"[{symbol}] Signal: {signal.action.value} | Reason: {signal.reason} | Price: {current_price:.2f} | "
                    f"Confidence: {confidence:.2%} | "
                    f"1h: {pred_1h}({conf_1h:.2%}) | 15m: {pred_15m}({conf_15m:.2%}) | "
                    f"Candle: {candle_timestamp}"
                )
            else:
                logger.info(f"[{symbol}] Signal: {signal.action.value} | Reason: {signal.reason} | Price: {current_price:.2f} | Confidence: {confidence:.2%} | Candle: {candle_timestamp}")
            
            # Log ALL non-HOLD signals to signals.log
            if signal.action != Action.HOLD:
                signal_logger.info(
                    f"SIGNAL GEN: {symbol} {signal.action.value} Conf={confidence:.2f} Price={current_price:.2f} "
                    f"Strategy={indicators_info.get('strategy', 'UNKNOWN')} "
                    f"Model={indicators_info.get('model_name', 'UNKNOWN')} "
                    f"Reason={signal.reason}"
                )
                try:
                    engine = self._get_decision_engine()
                    if engine:
                        ohlcv = [
                            {
                                "time": int(r.get("timestamp", 0)),
                                "open": float(r.get("open", 0.0)),
                                "high": float(r.get("high", 0.0)),
                                "low": float(r.get("low", 0.0)),
                                "close": float(r.get("close", 0.0)),
                                "volume": float(r.get("volume", 0.0)),
                            }
                            for r in df.tail(60).to_dict(orient="records")
                        ]
                        eval_payload = {
                            "action": signal.action.value,
                            "price": float(current_price),
                            "confidence": float(confidence),
                            "strength": str(indicators_info.get("strength", "")) if isinstance(indicators_info, dict) else "",
                            "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                            "take_profit": float(signal.take_profit) if signal.take_profit else None,
                            "1h_pred": indicators_info.get("1h_pred") if isinstance(indicators_info, dict) else None,
                            "1h_conf": indicators_info.get("1h_conf") if isinstance(indicators_info, dict) else None,
                            "15m_pred": indicators_info.get("15m_pred") if isinstance(indicators_info, dict) else None,
                            "15m_conf": indicators_info.get("15m_conf") if isinstance(indicators_info, dict) else None,
                            "4h_pred": indicators_info.get("4h_pred") if isinstance(indicators_info, dict) else None,
                            "4h_conf": indicators_info.get("4h_conf") if isinstance(indicators_info, dict) else None,
                        }
                        engine_eval = engine.evaluate(
                            symbol=symbol,
                            side="Buy" if signal.action == Action.LONG else "Sell",
                            signal_payload=eval_payload,
                            ohlcv=ohlcv,
                        )
                        if isinstance(engine_eval, dict) and not engine_eval.get("decision_id"):
                            engine_eval["decision_id"] = str(uuid.uuid4())
                        if isinstance(indicators_info, dict):
                            indicators_info["decision_engine_eval"] = engine_eval
                            if isinstance(engine_eval, dict) and isinstance(engine_eval.get("decision_id"), str):
                                indicators_info["engine_decision_id"] = engine_eval.get("decision_id")
                            signal.indicators_info = indicators_info
                        signal_logger.info(
                            f"ENGINE EVAL: {symbol} action={signal.action.value} score={engine_eval.get('score')} "
                            f"decision={engine_eval.get('decision')} mult={engine_eval.get('size_multiplier')} codes={engine_eval.get('reason_codes')}"
                        )
                        signal_logger.info(
                            f"DECISION ENGINE: {symbol} {'Buy' if signal.action == Action.LONG else 'Sell'} "
                            f"action={signal.action.value} score={engine_eval.get('score')} "
                            f"decision={engine_eval.get('decision')} mult={engine_eval.get('size_multiplier')} "
                            f"codes={engine_eval.get('reason_codes')}"
                        )
                        if isinstance(engine_eval, dict) and isinstance(engine_eval.get("engine"), dict):
                            signal_logger.info(
                                "ENGINE DETAILS: " + json.dumps(engine_eval.get("engine"), ensure_ascii=False, separators=(",", ":"))
                            )
                            signal_logger.info(
                                "DECISION ENGINE DETAILS: "
                                + json.dumps(engine_eval.get("engine"), ensure_ascii=False, separators=(",", ":"))
                            )
                        try:
                            root = Path(__file__).resolve().parent.parent
                            append_jsonl(
                                str(root / "logs" / "decision_engine_audit.jsonl"),
                                {
                                    "event_type": "engine_signal",
                                    "decision_id": engine_eval.get("decision_id") if isinstance(engine_eval, dict) else None,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "symbol": symbol,
                                    "timeframe": str(self.settings.timeframe),
                                    "candle_timestamp": candle_timestamp.isoformat() if candle_timestamp is not None else None,
                                    "signal": {
                                        "action": signal.action.value,
                                        "price": float(current_price),
                                        "reason": signal.reason,
                                        "confidence": float(confidence),
                                        "strength": str(indicators_info.get("strength", "")) if isinstance(indicators_info, dict) else "",
                                        "take_profit": float(signal.take_profit) if signal.take_profit else None,
                                        "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                                    },
                                    "engine_eval": engine_eval,
                                },
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
            
            logger.info(f"[{symbol}] ⏭️ Signal generated at {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, continuing processing...")

            # 4. Process paper trading for online testing of experimental models
            # Pass the same data to paper trading manager
            try:
                # Extract high and low from the current row
                high = float(row['high'])
                low = float(row['low'])
                
                self.paper_trading_manager.on_bar(
                    symbol=symbol,
                    row=row,
                    df=df,
                    current_price=current_price,
                    high=high,
                    low=low,
                    candle_timestamp=candle_timestamp
                )
                logger.debug(f"[{symbol}] ✅ Paper trading manager processed bar")
            except Exception as e:
                logger.error(f"[{symbol}] Error processing paper trading: {e}")

            # 5. Логируем сигнал в историю (только если уверенность >= reverse_min_confidence)
            # Это гарантирует, что в истории отображаются только сигналы с достаточной уверенностью
            min_confidence_for_history = self.settings.risk.reverse_min_confidence
            if signal.action != Action.HOLD:
                if confidence >= min_confidence_for_history:
                    logger.info(f"[{symbol}] 📝 Adding signal to history (confidence {confidence:.2%} >= {min_confidence_for_history:.2%})...")
                    self.state.add_signal(
                        symbol=symbol,
                        action=signal.action.value,
                        price=signal.price,
                        confidence=confidence,
                        reason=signal.reason,
                        indicators=indicators_info,
                        strategy=str(indicators_info.get("strategy", "") or ""),
                        model_name=str(indicators_info.get("model_name", "") or "")
                    )
                    logger.info(f"[{symbol}] ✅ Signal added to history, checking notification...")
                    
                    # Уведомление о сигнале высокой уверенности
                    if confidence > 0.7:
                        logger.info(f"[{symbol}] 📢 Sending notification...")
                        await self.notifier.medium(f"🔔 СИГНАЛ {signal.action.value} по {symbol}\nУверенность: {int(confidence*100)}%\nЦена: {signal.price}")
                        logger.info(f"[{symbol}] ✅ Notification sent")
                else:
                    logger.debug(f"[{symbol}] ⏭️ Signal skipped from history: confidence {confidence:.2%} < {min_confidence_for_history:.2%}")
            
            logger.info(f"[{symbol}] ✅ Signal processing completed, returning from process_symbol")

            # 6. Исполнение сделок
            # ВАЖНО: Проверяем уверенность перед открытием позиции
            # Для MTF стратегии используем специфичные пороги, для обычной - общий порог
            strategy = self.strategies.get(symbol)
            is_mtf_strategy = strategy and hasattr(strategy, 'predict_combined')
            
            if is_mtf_strategy:
                # Для MTF стратегии используем настройку min_confidence_for_trade
                # MTF уже проверила уверенность внутри, но для открытия сделки используем настройку
                mtf_threshold_1h = getattr(strategy, 'confidence_threshold_1h', 0.50)
                mtf_threshold_15m = getattr(strategy, 'confidence_threshold_15m', 0.35)
                # Используем настройку из конфигурации
                min_confidence_for_trade = self.settings.ml_strategy.min_confidence_for_trade
                logger.debug(f"[{symbol}] MTF strategy: using threshold {min_confidence_for_trade:.2%} (1h: {mtf_threshold_1h:.2%}, 15m: {mtf_threshold_15m:.2%})")
            else:
                # Для обычной стратегии используем настройку min_confidence_for_trade
                min_confidence_for_trade = self.settings.ml_strategy.min_confidence_for_trade
                logger.debug(f"[{symbol}] Single strategy: using threshold {min_confidence_for_trade:.2%}")
            
            if signal.action in (Action.LONG, Action.SHORT):
                # Проверяем уверенность перед открытием позиции
                if confidence < min_confidence_for_trade:
                    logger.info(
                        f"[{symbol}] ⏭️ Signal rejected for trade: confidence {confidence:.2%} < "
                        f"threshold {min_confidence_for_trade:.2%}"
                    )
                    # Сохраняем timestamp даже при отклонении сигнала, чтобы не обрабатывать эту свечу повторно
                    if candle_timestamp is not None:
                        self.last_processed_candle[symbol] = candle_timestamp
                        logger.debug(f"[{symbol}] ✅ Candle timestamp saved after signal rejection: {candle_timestamp}")
                    return  # Не открываем позицию, если уверенность ниже порога
                
                # КРИТИЧНО: Проверяем "свежесть" сигнала - открываем сделки только по свежим сигналам (не старше 15 минут)
                signal_age_seconds = (pd.Timestamp.now() - signal_received_time).total_seconds()
                signal_age_minutes = signal_age_seconds / 60
                max_signal_age_minutes = 15  # Максимальный возраст сигнала для открытия сделки
                
                if signal_age_minutes > max_signal_age_minutes:
                    logger.warning(
                        f"[{symbol}] ⏭️ Signal rejected: too old ({signal_age_minutes:.1f} minutes > {max_signal_age_minutes} minutes). "
                        f"Signal received at {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    return  # Не открываем позицию по устаревшему сигналу
                
                logger.info(
                    f"[{symbol}] ✅ Signal is fresh: {signal_age_minutes:.1f} minutes old (max: {max_signal_age_minutes} minutes)"
                )
                
                signal_side = Bias.LONG if signal.action == Action.LONG else Bias.SHORT
                
                # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ для диагностики
                indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
                signal_tp = signal.take_profit or indicators_info.get('take_profit')
                signal_sl = signal.stop_loss or indicators_info.get('stop_loss')
                tp_str = f"{signal_tp:.2f}" if signal_tp else "None"
                sl_str = f"{signal_sl:.2f}" if signal_sl else "None"
                logger.info(
                    f"[{symbol}] 🔍 TRADE DECISION: action={signal.action.value}, "
                    f"has_pos={has_pos}, local_pos={local_pos is not None}, "
                    f"signal_side={signal_side}, confidence={confidence:.2%} (>= {min_confidence_for_trade:.2%}), "
                    f"TP={tp_str}, SL={sl_str}, "
                    f"price={current_price:.2f}"
                )

                # Если позиция уже есть, решаем: игнорировать реверс или усреднять
                if has_pos is not None and local_pos:
                    # Проверяем, нужно ли реверсировать позицию по сильному сигналу
                    if has_pos != signal_side and self._is_strong_reverse_signal(signal, confidence):
                        logger.info(f"[{symbol}] Strong reverse signal detected, closing & reversing.")
                        if size > 0:
                            await self._close_position_market(symbol, has_pos, size)
                        await self.execute_trade(
                            symbol,
                            "Buy" if signal_side == Bias.LONG else "Sell",
                            signal,
                            position_horizon=self._classify_position_horizon(signal),
                        )
                        return

                    # Не закрываем средне/долгосрочные позиции по противоположному сигналу
                    if (
                        has_pos != signal_side
                        and local_pos.horizon in ("mid_term", "long_term")
                        and self.settings.risk.long_term_ignore_reverse
                    ):
                        logger.info(
                            f"[{symbol}] Opposite signal ignored for {local_pos.horizon} position."
                        )
                        return

                    # Усреднение при сигнале в ту же сторону и в минусе
                    if has_pos == signal_side:
                        if self._should_dca(local_pos, signal, current_price, confidence):
                            logger.info(f"[{symbol}] DCA conditions met, adding to position.")
                            await self.execute_trade(
                                symbol,
                                "Buy" if signal_side == Bias.LONG else "Sell",
                                signal,
                                is_add=True,
                                position_horizon=local_pos.horizon,
                            )
                        else:
                            logger.info(f"[{symbol}] ⏭️ Skipping trade: position already exists in same direction (DCA conditions not met)")
                        return

                if getattr(self.settings.ml_strategy, "follow_btc_filter_enabled", True) and symbol != "BTCUSDT":
                    override_conf = getattr(self.settings.ml_strategy, "follow_btc_override_confidence", 0.80)
                    indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
                    signal_strength = indicators_info.get("strength", "") if isinstance(indicators_info, dict) else ""
                    bypass = (signal_strength == "очень_сильное") or (override_conf is not None and confidence >= float(override_conf))
                    if not bypass:
                        btc_signal = await self._get_btc_signal()
                        if btc_signal and btc_signal.get("action") in (Action.LONG, Action.SHORT):
                            btc_action = btc_signal["action"]
                            if (btc_action == Action.LONG and signal.action == Action.SHORT) or \
                               (btc_action == Action.SHORT and signal.action == Action.LONG):
                                logger.info(
                                    f"[{symbol}] ⏭️ Signal ignored: BTCUSDT={btc_action.value}, "
                                    f"{symbol}={signal.action.value} (opposite direction, following BTC)"
                                )
                                return
                    else:
                        logger.info(
                            f"[{symbol}] 🟢 BTC follow bypassed: strength={signal_strength}, confidence={confidence:.2%}, threshold={float(override_conf):.2%}"
                        )

                guard = None
                if self.settings.risk.tp_reentry_enabled and has_pos is None and local_pos is None:
                    guard = self.state.get_tp_reentry_guard(symbol)

                if guard:
                    desired_side = "Buy" if signal.action == Action.LONG else "Sell"
                    if desired_side != guard.side:
                        self.state.clear_tp_reentry_guard(symbol)
                        guard = None

                if guard:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    exit_time = pd.Timestamp(guard.exit_time_utc)
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.tz_localize("UTC")
                    window_minutes = self._timeframe_minutes(self.settings.timeframe) * max(0, int(self.settings.risk.tp_reentry_window_candles))
                    if window_minutes > 0 and now_utc > (exit_time + pd.Timedelta(minutes=window_minutes)):
                        logger.info(
                            f"[{symbol}] TP reentry guard expired: skipped={guard.skipped_signals}, allowed={guard.allowed_reentries}, last={guard.last_decision}"
                        )
                        self.state.clear_tp_reentry_guard(symbol)
                        guard = None

                if guard:
                    wait_until = pd.Timestamp(guard.wait_until_utc)
                    if wait_until.tzinfo is None:
                        wait_until = wait_until.tz_localize("UTC")
                    ok, detail = self._eval_tp_reentry(symbol, guard, df_for_strategy, signal.action)
                    if isinstance(detail, str) and detail.startswith("tp_reentry_eval_error:"):
                        self.state.tp_reentry_record_allow(symbol, detail)
                        logger.warning(f"[{symbol}] ⚠️ TP reentry eval error (fail-open): {detail}")
                        self.state.clear_tp_reentry_guard(symbol)
                        guard = None
                    if now_utc < wait_until:
                        self.state.tp_reentry_record_skip(symbol, f"tp_reentry_wait {detail}", True)
                        logger.info(
                            f"[{symbol}] ⏭️ TP reentry wait: until={wait_until.isoformat()} {detail}"
                        )
                        return
                    if not ok:
                        self.state.tp_reentry_record_skip(symbol, f"tp_reentry_block {detail}", False)
                        logger.info(
                            f"[{symbol}] ⏭️ TP reentry blocked: {detail}"
                        )
                        return
                    self.state.tp_reentry_record_allow(symbol, detail)
                    logger.info(f"[{symbol}] ✅ TP reentry allowed: {detail}")
                    self.state.clear_tp_reentry_guard(symbol)
                
                # Фильтр по волатильности (ATR 1h): входить только когда «есть движение»
                if self.settings.ml_strategy.atr_filter_enabled and (signal.action == Action.LONG and has_pos != Bias.LONG or signal.action == Action.SHORT and has_pos != Bias.SHORT):
                    atr_pct_1h = await asyncio.to_thread(self._get_atr_pct_1h_sync, symbol)
                    if atr_pct_1h is not None:
                        min_pct = self.settings.ml_strategy.atr_min_pct
                        max_pct = self.settings.ml_strategy.atr_max_pct
                        if atr_pct_1h < min_pct or atr_pct_1h > max_pct:
                            logger.info(
                                f"[{symbol}] ⏭️ ATR filter: skip entry — ATR 1h={atr_pct_1h:.3f}% outside [{min_pct}, {max_pct}] (flat or panic)"
                            )
                            if candle_timestamp is not None:
                                self.last_processed_candle[symbol] = candle_timestamp
                            return
                    else:
                        logger.debug(f"[{symbol}] ATR 1h unavailable, allowing entry")
                
                # Открываем позицию, если ее нет или она в другую сторону (для short_term)
                if signal.action == Action.LONG and has_pos != Bias.LONG:
                    if self.settings.ml_strategy.pullback_enabled:
                        signal_high = float(row['high'])
                        signal_low = float(row['low'])
                        if self._pullback_limit_roll_enabled():
                            if symbol in self.pending_pullback_signals:
                                self.pending_pullback_signals[symbol] = []
                            await self._handle_pullback_limit_roll_signal(symbol, signal, candle_timestamp or pd.Timestamp.now(), signal_high, signal_low)
                        else:
                            self._add_pending_pullback_signal(symbol, signal, candle_timestamp or pd.Timestamp.now(), signal_high, signal_low)
                            logger.info(f"[{symbol}] 📋 Added LONG signal to pullback queue (waiting for pullback, signal_high={signal_high:.2f}, signal_low={signal_low:.2f})")
                    else:
                        logger.info(f"[{symbol}] ✅ Opening LONG position (no position or opposite)")
                        await self.execute_trade(symbol, "Buy", signal)
                elif signal.action == Action.SHORT and has_pos != Bias.SHORT:
                    if self.settings.ml_strategy.pullback_enabled:
                        signal_high = float(row['high'])
                        signal_low = float(row['low'])
                        if self._pullback_limit_roll_enabled():
                            if symbol in self.pending_pullback_signals:
                                self.pending_pullback_signals[symbol] = []
                            await self._handle_pullback_limit_roll_signal(symbol, signal, candle_timestamp or pd.Timestamp.now(), signal_high, signal_low)
                        else:
                            self._add_pending_pullback_signal(symbol, signal, candle_timestamp or pd.Timestamp.now(), signal_high, signal_low)
                            logger.info(f"[{symbol}] 📋 Added SHORT signal to pullback queue (waiting for pullback, signal_high={signal_high:.2f}, signal_low={signal_low:.2f})")
                    else:
                        logger.info(f"[{symbol}] ✅ Opening SHORT position (no position or opposite)")
                        await self.execute_trade(symbol, "Sell", signal)
                else:
                    logger.info(f"[{symbol}] ⏭️ Skipping trade: action={signal.action.value}, has_pos={has_pos}")
            
            # Сохраняем timestamp обработанной свечи ТОЛЬКО после успешной обработки сигнала
            # Это позволяет повторить обработку, если произошла ошибка при открытии позиции
            if candle_timestamp is not None:
                self.last_processed_candle[symbol] = candle_timestamp
                logger.debug(f"[{symbol}] ✅ Candle timestamp saved after successful processing: {candle_timestamp}")

        except Exception as e:
            logger.error(f"[trading_loop] Error processing {symbol}: {e}")
            # При ошибке НЕ сохраняем timestamp, чтобы можно было повторить обработку

    def _pullback_entry_mode(self) -> str:
        return str(getattr(self.settings.ml_strategy, "pullback_entry_mode", "pending")).strip().lower()

    def _pullback_limit_roll_enabled(self) -> bool:
        return bool(self.settings.ml_strategy.pullback_enabled) and self._pullback_entry_mode() == "limit_roll"

    def _to_timestamp(self, ts) -> Optional[pd.Timestamp]:
        if ts is None:
            return None
        if isinstance(ts, pd.Timestamp):
            return ts
        if isinstance(ts, (int, float)):
            try:
                return pd.Timestamp(ts, unit="ms")
            except Exception:
                return pd.Timestamp(ts)
        try:
            return pd.Timestamp(ts)
        except Exception:
            return None

    def _extract_open_orders_list(self, resp: Optional[dict]) -> List[dict]:
        if not resp or not isinstance(resp, dict):
            return []
        if resp.get("retCode") not in (0, "0", None):
            return []
        result = resp.get("result")
        if isinstance(result, dict):
            lst = result.get("list", [])
            if isinstance(lst, list):
                return [x for x in lst if isinstance(x, dict)]
        return []

    async def _prepare_entry(
        self,
        symbol: str,
        side: str,
        signal: Signal,
        is_add: bool,
        position_horizon: Optional[str],
    ) -> Optional[tuple]:
        indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
        signal_received_time = None
        if indicators_info and "signal_received_time" in indicators_info:
            try:
                signal_received_time = pd.Timestamp(indicators_info["signal_received_time"])
            except Exception:
                signal_received_time = None
        elif signal.timestamp:
            signal_received_time = signal.timestamp

        if signal_received_time and not is_add:
            try:
                signal_age_minutes = (pd.Timestamp.now() - signal_received_time).total_seconds() / 60
                if signal_age_minutes > 15:
                    logger.warning(
                        f"[{symbol}] ❌ Cannot open position: signal is too old ({signal_age_minutes:.1f} minutes > 15 minutes). "
                        f"Signal timestamp: {signal_received_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    logger.info(f"[{symbol}] 🚫 Entry blocked reason=stale_signal")
                    return None
            except Exception:
                pass

        signal_tp = signal.take_profit or indicators_info.get("take_profit")
        signal_sl = signal.stop_loss or indicators_info.get("stop_loss")
        if not is_add and (not signal_tp or not signal_sl):
            logger.warning(
                f"[{symbol}] ❌ Cannot open position: missing TP/SL! "
                f"TP={signal_tp}, SL={signal_sl}, signal.take_profit={signal.take_profit}, "
                f"signal.stop_loss={signal.stop_loss}, indicators_info={indicators_info}"
            )
            logger.info(f"[{symbol}] 🚫 Entry blocked reason=missing_tp_sl")
            return None

        if not is_add and bool(getattr(self.settings.ml_strategy, "decision_engine_enabled", False)):
            engine_mode = str(getattr(self.settings.ml_strategy, "decision_engine_mode", "shadow"))
            eng = indicators_info.get("decision_engine_eval") if isinstance(indicators_info, dict) else None
            if isinstance(eng, dict):
                eng_d = str(eng.get("decision", "")).lower()
                eng_mult = eng.get("size_multiplier")
                if eng_d not in ("allow", "reduce", "veto"):
                    eng_d = "veto"
                if engine_mode == "enforce" and eng_d == "veto":
                    logger.info(
                        f"[{symbol}] 🚫 Entry blocked reason=engine_veto "
                        f"decision_id={eng.get('decision_id') if isinstance(eng, dict) else None} "
                        f"codes={eng.get('reason_codes') if isinstance(eng, dict) else None}"
                    )
                    try:
                        root = Path(__file__).resolve().parent.parent
                        try:
                            append_jsonl(
                                str(root / "logs" / "decision_engine_audit.jsonl"),
                                {
                                    "event_type": "engine_blocked",
                                    "decision_id": eng.get("decision_id") if isinstance(eng, dict) else None,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "symbol": symbol,
                                    "timeframe": str(self.settings.timeframe),
                                    "side": side,
                                    "engine_eval": eng,
                                    "notes": "Blocked by decision engine",
                                },
                            )
                        except Exception:
                            pass
                        append_jsonl(
                            str(root / "logs" / "ai_entry_audit.jsonl"),
                            {
                                "event_type": "entry_blocked",
                                "symbol": symbol,
                                "side": side,
                                "decision_id": None,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "decision_source": "engine",
                                "ai": None,
                                "engine": eng,
                                "reason_codes": eng.get("reason_codes"),
                                "notes": "Blocked by decision engine",
                            },
                        )
                    except Exception:
                        pass
                    return None
                if engine_mode == "enforce" and eng_mult is not None:
                    try:
                        signal.indicators_info["ai_size_multiplier"] = float(eng_mult)
                    except Exception:
                        signal.indicators_info["ai_size_multiplier"] = 1.0
                    indicators_info = signal.indicators_info
                if engine_mode == "enforce" and eng_d == "reduce":
                    if "ai_size_multiplier" not in signal.indicators_info or signal.indicators_info.get("ai_size_multiplier", 1.0) >= 1.0:
                        signal.indicators_info["ai_size_multiplier"] = 0.25
                        indicators_info = signal.indicators_info

        if not is_add and getattr(self.settings.ml_strategy, "ai_entry_confirmation_enabled", False):
            decision = indicators_info.get("ai_entry_confirmation") if isinstance(indicators_info, dict) else None
            if not isinstance(decision, dict):
                decision = await self._confirm_entry_with_ai(symbol, side, signal, position_horizon)
            if not isinstance(signal.indicators_info, dict):
                signal.indicators_info = {}
            signal.indicators_info["ai_entry_confirmation"] = decision
            indicators_info = signal.indicators_info
            ai_mode = str(getattr(self.settings.ml_strategy, "ai_entry_confirmation_mode", "enforce"))

            ai_d = "veto"
            ai_mult = None
            if isinstance(decision, dict):
                ai_d = str(decision.get("decision", "")).lower()
                ai_mult = decision.get("size_multiplier")
                if ai_d not in ("allow", "reduce", "veto"):
                    ai_d = "veto"

            decision_id = decision.get("decision_id") if isinstance(decision, dict) else None
            codes = decision.get("reason_codes") if isinstance(decision, dict) else None
            logger.info(
                f"[{symbol}] 🤖 AI gate decision={ai_d} mult={ai_mult} codes={codes} decision_id={decision_id}"
            )

            if ai_mode == "shadow":
                logger.info(f"[{symbol}] 🤖 AI confirmation shadow mode: entry not blocked (AI decision: {ai_d})")
            elif ai_d == "veto":
                logger.info(
                    f"[{symbol}] 🚫 Entry blocked reason=ai_veto "
                    f"decision_id={decision_id} codes={codes}"
                )
                try:
                    root = Path(__file__).resolve().parent.parent
                    append_jsonl(
                        str(root / "logs" / "ai_entry_audit.jsonl"),
                        {
                            "event_type": "entry_blocked",
                            "symbol": symbol,
                            "side": side,
                            "decision_id": decision_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "decision_source": "ai",
                            "ai": decision if isinstance(decision, dict) else None,
                            "engine": indicators_info.get("decision_engine_eval") if isinstance(indicators_info, dict) else None,
                            "reason_codes": codes if isinstance(codes, list) else None,
                            "notes": decision.get("notes") if isinstance(decision, dict) else None,
                        },
                    )
                except Exception:
                    pass
                return None

            if ai_mode != "shadow" and ai_mult is not None:
                try:
                    signal.indicators_info["ai_size_multiplier"] = float(ai_mult)
                except Exception:
                    signal.indicators_info["ai_size_multiplier"] = 1.0
                indicators_info = signal.indicators_info
            if ai_mode != "shadow" and ai_d == "reduce":
                if "ai_size_multiplier" not in signal.indicators_info or signal.indicators_info.get("ai_size_multiplier", 1.0) >= 1.0:
                    signal.indicators_info["ai_size_multiplier"] = 0.25
                    indicators_info = signal.indicators_info

        tp_str = f"{signal_tp:.2f}" if signal_tp else "None"
        sl_str = f"{signal_sl:.2f}" if signal_sl else "None"
        logger.info(f"[{symbol}] ✅ TP/SL check passed: TP={tp_str}, SL={sl_str}")
        logger.info(f"[{symbol}] ✅ Entry precheck passed")
        return signal_tp, signal_sl, indicators_info

    async def _calc_order_qty(
        self,
        symbol: str,
        entry_price: float,
        indicators_info: dict,
        is_add: bool,
    ) -> Optional[tuple]:
        qty_step = self.bybit.get_qty_step(symbol)
        if qty_step <= 0:
            logger.error(f"Invalid qtyStep for {symbol}: {qty_step}")
            logger.info(f"[{symbol}] 🚫 Entry blocked reason=invalid_qty_step")
            return None

        qty_step_str = str(qty_step)
        precision = len(qty_step_str.split(".")[1]) if "." in qty_step_str else 0

        balance_info = await asyncio.to_thread(self.bybit.get_wallet_balance)
        balance = 0.0
        if balance_info and balance_info.get("retCode") == 0:
            result = balance_info.get("result")
            if result and isinstance(result, dict):
                list_data = result.get("list", [])
                if list_data and len(list_data) > 0:
                    wallet_item = list_data[0]
                    if wallet_item and isinstance(wallet_item, dict):
                        wallet = wallet_item.get("coin", [])
                        if wallet and isinstance(wallet, list):
                            usdt_coin = next((c for c in wallet if isinstance(c, dict) and c.get("coin") == "USDT"), None)
                            if usdt_coin:
                                balance_str = usdt_coin.get("walletBalance", "0")
                                balance = float(balance_str) if balance_str and balance_str != "" else 0.0

        if balance <= 0:
            logger.error(f"[{symbol}] ❌ Cannot get balance or balance is zero: {balance}")
            logger.info(f"[{symbol}] 🚫 Entry blocked reason=balance_unavailable_or_zero")
            return None

        logger.info(f"[{symbol}] ✅ Balance check passed: ${balance:.2f}")

        leverage = self.settings.get_leverage_for_symbol(symbol)
        
        if is_add:
            base_position_size_usd = self.settings.risk.base_order_usd * leverage
            position_size_usd = base_position_size_usd / 2.0
            fixed_margin_usd = position_size_usd / leverage
        else:
            size_multiplier = 1.0
            if isinstance(indicators_info, dict) and "ai_size_multiplier" in indicators_info:
                try:
                    size_multiplier = float(indicators_info.get("ai_size_multiplier", 1.0))
                except Exception:
                    size_multiplier = 1.0
            if size_multiplier <= 0:
                size_multiplier = 1.0
            if size_multiplier > 2.0:
                size_multiplier = 2.0
            fixed_margin_usd = self.settings.risk.base_order_usd * size_multiplier
            position_size_usd = fixed_margin_usd * leverage

            max_margin_usd = get_max_margin_usd(self.settings, leverage)
            if max_margin_usd and fixed_margin_usd > max_margin_usd:
                fixed_margin_usd = max_margin_usd
                position_size_usd = fixed_margin_usd * leverage
                logger.info(f"[{symbol}] Position limited by max_margin_usd: ${max_margin_usd:.2f}")

            max_position_usd = getattr(self.settings.risk, "max_position_usd", None)
            if isinstance(max_position_usd, (int, float)) and max_position_usd and max_position_usd > 0:
                if position_size_usd > float(max_position_usd):
                    position_size_usd = float(max_position_usd)
                    fixed_margin_usd = position_size_usd / leverage
                    logger.info(f"[{symbol}] Position limited by max_position_usd: ${max_position_usd:.2f}")

        if fixed_margin_usd > balance:
            logger.warning(
                f"[{symbol}] ⚠️ Fixed margin ${fixed_margin_usd:.2f} exceeds balance ${balance:.2f}, "
                f"using available balance"
            )
            fixed_margin_usd = balance
            position_size_usd = fixed_margin_usd * leverage

        if entry_price <= 0:
            logger.error(f"[{symbol}] ❌ Invalid entry_price: {entry_price}")
            logger.info(f"[{symbol}] 🚫 Entry blocked reason=invalid_entry_price")
            return None

        total_qty = position_size_usd / entry_price
        logger.info(
            f"Position size for {symbol}: "
            f"balance=${balance:.2f}, "
            f"margin=${fixed_margin_usd:.2f}, "
            f"position_size_usd=${position_size_usd:.2f}, "
            f"qty={total_qty:.6f}, leverage={leverage}x"
        )

        rounded_qty = math.floor(total_qty / qty_step) * qty_step
        if rounded_qty < qty_step:
            qty = qty_step
        else:
            qty = rounded_qty
        qty = float(f"{qty:.{precision}f}")
        if qty <= 0:
            logger.error(f"[{symbol}] ❌ Calculated qty is zero or negative: {qty}")
            logger.info(f"[{symbol}] 🚫 Entry blocked reason=qty_zero_after_rounding")
            return None
        return qty, fixed_margin_usd, position_size_usd, balance, qty_step, precision

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        signal: Signal,
        is_add: bool = False,
        position_horizon: Optional[str] = None,
    ):
        try:
            logger.info(f"[{symbol}] 🚀 execute_trade() called: side={side}, is_add={is_add}, price={signal.price:.2f}")
            
            # Проверяем наличие TP/SL в сигнале (критично для открытия позиции)
            prep = await self._prepare_entry(symbol, side, signal, is_add, position_horizon)
            if prep is None:
                logger.info(f"[{symbol}] 🚫 execute_trade aborted at stage=prepare_entry")
                return
            signal_tp, signal_sl, indicators_info = prep
            if not is_add:
                signal.take_profit = signal_tp
                signal.stop_loss = signal_sl

            qty_calc = await self._calc_order_qty(symbol, float(signal.price), indicators_info, is_add)
            if qty_calc is None:
                logger.info(f"[{symbol}] 🚫 execute_trade aborted at stage=calc_order_qty")
                return
            qty, fixed_margin_usd, position_size_usd, balance, qty_step, precision = qty_calc

            logger.info(f"[{symbol}] ✅ Position size calculated: qty={qty:.6f}, placing order...")

            def _trim_resp(x):
                if not isinstance(x, dict):
                    return x
                out = {}
                for k in ("retCode", "retMsg", "time", "result"):
                    if k in x:
                        out[k] = x.get(k)
                return out

            engine_eval = signal.indicators_info.get("decision_engine_eval") if isinstance(signal.indicators_info, dict) else None
            engine_id = None
            if isinstance(engine_eval, dict) and isinstance(engine_eval.get("decision_id"), str):
                engine_id = engine_eval.get("decision_id")
            ai = signal.indicators_info.get("ai_entry_confirmation") if isinstance(signal.indicators_info, dict) else None
            ai_id = ai.get("decision_id") if isinstance(ai, dict) else None
            try:
                root = Path(__file__).resolve().parent.parent
                append_jsonl(
                    str(root / "logs" / "trade_execution_audit.jsonl"),
                    {
                        "event_type": "order_attempt",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "timeframe": str(self.settings.timeframe),
                        "side": side,
                        "is_add": bool(is_add),
                        "qty": float(qty),
                        "signal_price": float(signal.price),
                        "tp": float(signal_tp) if signal_tp else None,
                        "sl": float(signal_sl) if signal_sl else None,
                        "use_take_profit": bool(getattr(self.settings.risk, "use_take_profit", True)),
                        "expected_margin_usd": float(fixed_margin_usd),
                        "expected_position_size_usd": float(position_size_usd),
                        "leverage": int(self.settings.get_leverage_for_symbol(symbol)),
                        "confidence": float(indicators_info.get("confidence", 0.0)) if isinstance(indicators_info, dict) else None,
                        "strength": str(indicators_info.get("strength", "")) if isinstance(indicators_info, dict) else None,
                        "engine_decision_id": engine_id,
                        "engine_score": engine_eval.get("score") if isinstance(engine_eval, dict) else None,
                        "engine_decision": engine_eval.get("decision") if isinstance(engine_eval, dict) else None,
                        "ai_decision_id": ai_id if isinstance(ai_id, str) else None,
                        "ai_decision": ai.get("decision") if isinstance(ai, dict) else None,
                    },
                )
            except Exception:
                pass
            
            try:
                tp_to_send = None
                if not is_add and bool(getattr(self.settings.risk, "use_take_profit", True)):
                    tp_to_send = signal_tp
                resp = self.bybit.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Market",
                    take_profit=tp_to_send,
                    stop_loss=None if is_add else signal_sl,
                )
            except InvalidRequestError as e:
                # Обрабатываем ошибку недостатка средств (код 110007)
                error_code = getattr(e, 'status_code', None) or getattr(e, 'ret_code', None)
                error_msg = str(e)
                
                # Проверяем, что это ошибка недостатка средств
                if error_code == 110007 or "not enough" in error_msg.lower() or "ab not enough" in error_msg.lower():
                    # Рассчитываем недостающую сумму
                    required_margin = fixed_margin_usd
                    shortfall = max(0, required_margin - balance)
                    
                    # Формируем детальное сообщение
                    message = (
                        f"⚠️ НЕДОСТАТОЧНО СРЕДСТВ ДЛЯ ОТКРЫТИЯ ПОЗИЦИИ\n\n"
                        f"📊 Параметры сделки:\n"
                        f"• Символ: {symbol}\n"
                        f"• Направление: {side}\n"
                        f"• Цена входа: ${signal.price:.6f}\n"
                        f"• Количество: {qty:.6f}\n"
                        f"• Размер позиции: ${position_size_usd:.2f}\n"
                        f"• Требуемая маржа: ${required_margin:.2f}\n"
                        f"• Плечо: {self.settings.get_leverage_for_symbol(symbol)}x\n"
                    )
                    
                    if signal.take_profit and signal.stop_loss:
                        message += (
                            f"• TP: ${signal.take_profit:.6f}\n"
                            f"• SL: ${signal.stop_loss:.6f}\n"
                        )
                    
                    message += (
                        f"\n💰 Баланс:\n"
                        f"• Доступно: ${balance:.2f}\n"
                        f"• Не хватает: ${shortfall:.2f}\n"
                        f"• Нужно всего: ${required_margin:.2f}"
                    )
                    
                    # Отправляем уведомление
                    await self.notifier.critical(message)
                    logger.error(
                        f"[{symbol}] ❌ Insufficient balance: required=${required_margin:.2f}, "
                        f"available=${balance:.2f}, shortfall=${shortfall:.2f}"
                    )
                    logger.info(f"[{symbol}] 🚫 execute_trade aborted reason=insufficient_balance_exception")
                    return
                else:
                    # Другая ошибка InvalidRequestError - пробрасываем дальше
                    raise
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ответа от биржи
            if resp:
                ret_code = resp.get("retCode") if isinstance(resp, dict) else None
                ret_msg = resp.get("retMsg", "") if isinstance(resp, dict) else ""
                logger.info(f"[{symbol}] 📡 Order response: retCode={ret_code}, retMsg={ret_msg}, full_response={resp}")
            else:
                logger.error(f"[{symbol}] ❌ Order response is None or empty!")

            try:
                root = Path(__file__).resolve().parent.parent
                append_jsonl(
                    str(root / "logs" / "trade_execution_audit.jsonl"),
                    {
                        "event_type": "order_response",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "timeframe": str(self.settings.timeframe),
                        "side": side,
                        "is_add": bool(is_add),
                        "engine_decision_id": engine_id,
                        "ai_decision_id": ai_id if isinstance(ai_id, str) else None,
                        "response": _trim_resp(resp),
                    },
                )
            except Exception:
                pass
            
            if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                if is_add:
                    # Log to trades.log
                    trade_logger.info(f"ORDER PLACED (ADD): {symbol} {side} Qty={qty} Price={signal.price}")
                    
                    logger.info(f"Successfully added to {side} for {symbol}")
                    await self.notifier.medium(
                        f"➕ ДОБАВЛЕНИЕ К ПОЗИЦИИ {side} {symbol}\n"
                        f"Цена: {signal.price}\n"
                        f"Объем: {qty}"
                    )
                    self.state.increment_dca(symbol)
                    # Обновляем среднюю цену и размер по бирже
                    pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
                    if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data:
                                position = list_data[0]
                                if position and isinstance(position, dict):
                                    size = float(position.get("size", 0))
                                    avg_price = float(position.get("avgPrice", 0))
                                    if size > 0 and avg_price > 0:
                                        self.state.update_position(symbol, size, avg_price)
                else:
                    # Log to trades.log
                    ai = None
                    if signal.indicators_info and isinstance(signal.indicators_info, dict):
                        ai = signal.indicators_info.get("ai_entry_confirmation")
                    ai_suffix = ""
                    if isinstance(ai, dict) and ai.get("decision_id"):
                        ai_suffix = f" AI={ai.get('decision')} AI_ID={ai.get('decision_id')}"
                    trade_logger.info(f"ORDER PLACED (OPEN): {symbol} {side} Qty={qty} Price={signal.price} TP={signal.take_profit} SL={signal.stop_loss}{ai_suffix}")
                    
                    logger.info(f"Successfully opened {side} for {symbol}")
                    
                    # Очищаем pending pullback сигналы для этого символа (позиция открыта)
                    if symbol in self.pending_pullback_signals:
                        cleared_count = len(self.pending_pullback_signals[symbol])
                        self.pending_pullback_signals[symbol] = []
                        if cleared_count > 0:
                            logger.info(f"[{symbol}] 🧹 Cleared {cleared_count} pending pullback signal(s) after opening position")
                    
                    if symbol in self.pending_pullback_entry_orders:
                        await self._cancel_pullback_entry_order(symbol, reason="position_opened")
                    
                    await self.notifier.high(
                        f"🚀 ОТКРЫТА ПОЗИЦИЯ {side} {symbol}\n"
                        f"Цена: {signal.price}\nTP: {signal.take_profit}\nSL: {signal.stop_loss}"
                    )
                    
                    # Добавляем в историю (пока как открытую)
                    indicators_info = signal.indicators_info if signal.indicators_info and isinstance(signal.indicators_info, dict) else {}
                    confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
                    signal_strength = indicators_info.get('strength', '') if isinstance(indicators_info, dict) else ''
                    
                    # Извлекаем TP/SL проценты из сигнала
                    signal_tp = signal.take_profit or indicators_info.get('take_profit')
                    signal_sl = signal.stop_loss or indicators_info.get('stop_loss')
                    tp_pct = None
                    sl_pct = None
                    if signal_tp and signal.price:
                        tp_pct = abs(signal_tp - signal.price) / signal.price
                    if signal_sl and signal.price:
                        sl_pct = abs(signal.price - signal_sl) / signal.price
                    
                    # Вычисляем маржу
                    margin_usd = fixed_margin_usd
                    
                    # Получаем плечо для расчета RR
                    leverage = self.settings.get_leverage_for_symbol(symbol)
                    if leverage is None or leverage <= 0:
                        leverage = DEFAULT_LEVERAGE
                    
                    # Параметры сигнала
                    rr_mode = getattr(self.settings.risk, 'trailing_activation_mode', 'price')
                    signal_parameters = {
                        'take_profit_pct': tp_pct,
                        'stop_loss_pct': sl_pct,
                        'risk_reward_ratio': (tp_pct / sl_pct) if (tp_pct and sl_pct and sl_pct > 0) else None,
                        'margin_based_rr': calculate_margin_based_rr(tp_pct, sl_pct, leverage, rr_mode),
                    }
                    if isinstance(indicators_info, dict) and isinstance(indicators_info.get("decision_engine_eval"), dict):
                        signal_parameters["decision_engine_eval"] = indicators_info.get("decision_engine_eval")
                        if isinstance(indicators_info.get("engine_decision_id"), str):
                            signal_parameters["engine_decision_id"] = indicators_info.get("engine_decision_id")
                    ai_entry_confirmation = indicators_info.get("ai_entry_confirmation") if isinstance(indicators_info, dict) else None
                    if isinstance(ai_entry_confirmation, dict):
                        signal_parameters["ai_entry_confirmation"] = ai_entry_confirmation
                    if isinstance(indicators_info, dict) and "ai_size_multiplier" in indicators_info:
                        try:
                            signal_parameters["ai_size_multiplier"] = float(indicators_info.get("ai_size_multiplier", 1.0))
                        except Exception:
                            signal_parameters["ai_size_multiplier"] = 1.0
                    
                    trade = TradeRecord(
                        symbol=symbol,
                        side=side,
                        entry_price=signal.price,
                        qty=qty,
                        status="open",
                        model_name=self.state.symbol_models.get(symbol, ""),
                        horizon=position_horizon or self._classify_position_horizon(signal),
                        entry_reason=signal.reason or "",
                        confidence=confidence,
                        take_profit=signal_tp,
                        stop_loss=signal_sl,
                        leverage=self.settings.get_leverage_for_symbol(symbol),
                        margin_usd=margin_usd,
                        signal_strength=signal_strength,
                        signal_parameters=signal_parameters,
                    )
                    self.state.add_trade(trade)
            else:
                ret_code = resp.get("retCode") if resp and isinstance(resp, dict) else "unknown"
                ret_msg = resp.get("retMsg", "") if resp and isinstance(resp, dict) else ""
                
                # Обрабатываем ошибку недостатка средств (код 110007)
                if ret_code == 110007 or (ret_msg and ("not enough" in ret_msg.lower() or "ab not enough" in ret_msg.lower())):
                    # Рассчитываем недостающую сумму
                    required_margin = fixed_margin_usd
                    shortfall = max(0, required_margin - balance)
                    
                    # Формируем детальное сообщение
                    message = (
                        f"⚠️ НЕДОСТАТОЧНО СРЕДСТВ ДЛЯ ОТКРЫТИЯ ПОЗИЦИИ\n\n"
                        f"📊 Параметры сделки:\n"
                        f"• Символ: {symbol}\n"
                        f"• Направление: {side}\n"
                        f"• Цена входа: ${signal.price:.6f}\n"
                        f"• Количество: {qty:.6f}\n"
                        f"• Размер позиции: ${position_size_usd:.2f}\n"
                        f"• Требуемая маржа: ${required_margin:.2f}\n"
                        f"• Плечо: {self.settings.get_leverage_for_symbol(symbol)}x\n"
                    )
                    
                    if signal.take_profit and signal.stop_loss:
                        message += (
                            f"• TP: ${signal.take_profit:.6f}\n"
                            f"• SL: ${signal.stop_loss:.6f}\n"
                        )
                    
                    message += (
                        f"\n💰 Баланс:\n"
                        f"• Доступно: ${balance:.2f}\n"
                        f"• Не хватает: ${shortfall:.2f}\n"
                        f"• Нужно всего: ${required_margin:.2f}"
                    )
                    
                    # Отправляем уведомление
                    await self.notifier.critical(message)
                    logger.error(
                        f"[{symbol}] ❌ Insufficient balance (retCode={ret_code}): required=${required_margin:.2f}, "
                        f"available=${balance:.2f}, shortfall=${shortfall:.2f}"
                    )
                    logger.info(f"[{symbol}] 🚫 execute_trade aborted reason=insufficient_balance_response")
                    return
                
                # Другие ошибки - просто логируем
                logger.error(
                    f"[{symbol}] ❌ Failed to open {side} position: "
                    f"retCode={ret_code}, retMsg={ret_msg}, "
                    f"qty={qty:.6f}, price={signal.price:.2f}, "
                    f"TP={signal.take_profit if not is_add else 'N/A'}, "
                    f"SL={signal.stop_loss if not is_add else 'N/A'}, "
                    f"full_response={resp}"
                )
                logger.info(f"[{symbol}] 🚫 execute_trade aborted reason=exchange_rejected_order retCode={ret_code}")
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Exception in execute_trade: {e}", exc_info=True)
            logger.info(f"[{symbol}] 🚫 execute_trade aborted reason=exception")
    
    async def update_breakeven_stop(self, symbol: str, position_info: dict):
        """Перемещает SL в безубыток при достижении порога прибыли"""
        try:
            # Проверяем, включен ли безубыток
            if not self.settings.risk.enable_breakeven:
                logger.debug(f"[{symbol}] Безубыток выключен, пропускаем обновление SL")
                return
            
            if not position_info or not isinstance(position_info, dict):
                logger.debug(f"[{symbol}] update_breakeven_stop: position_info is None or not dict")
                return
            
            if not position_info.get("size"):
                logger.debug(f"[{symbol}] update_breakeven_stop: position size is empty")
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                logger.debug(f"[{symbol}] update_breakeven_stop: position size is 0")
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            current_sl = position_info.get("stopLoss")
            
            if not entry_price or not mark_price:
                logger.debug(f"[{symbol}] update_breakeven_stop: entry_price={entry_price}, mark_price={mark_price}")
                return
            
            # Рассчитываем текущий PnL в процентах
            if side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price) * 100
            else:  # Sell
                pnl_pct = ((entry_price - mark_price) / entry_price) * 100
            
            logger.debug(f"[{symbol}] update_breakeven_stop: side={side}, entry={entry_price:.6f}, mark={mark_price:.6f}, pnl_pct={pnl_pct:.2f}%, current_sl={current_sl}")
            
            leverage = self.settings.get_leverage_for_symbol(symbol)
            if leverage is None or leverage <= 0:
                leverage = DEFAULT_LEVERAGE
            
            level1_activation = get_breakeven_activation_pct(self.settings, 1, leverage) * 100
            level2_activation = get_breakeven_activation_pct(self.settings, 2, leverage) * 100
            level1_sl_pct = self.settings.risk.breakeven_level1_sl_pct
            level2_sl_pct = self.settings.risk.breakeven_level2_sl_pct
            
            logger.debug(f"[{symbol}] Breakeven thresholds: level1_activation={level1_activation:.2f}%, level2_activation={level2_activation:.2f}%, level1_sl_pct={level1_sl_pct:.4f}, level2_sl_pct={level2_sl_pct:.4f}")
            
            # Определяем, какой уровень активировать
            new_sl = None
            level = None
            
            if pnl_pct >= level2_activation:
                # 2-я ступень: при прибыли >= level2_activation ставим SL на level2_sl_pct от входа
                level = "2-я ступень"
                if side == "Buy":
                    new_sl = entry_price * (1 + level2_sl_pct)
                else:
                    new_sl = entry_price * (1 - level2_sl_pct)
            elif pnl_pct >= level1_activation:
                # 1-я ступень: при прибыли >= level1_activation ставим SL на level1_sl_pct от входа
                level = "1-я ступень"
                if side == "Buy":
                    new_sl = entry_price * (1 + level1_sl_pct)
                else:
                    new_sl = entry_price * (1 - level1_sl_pct)
            else:
                # Прибыль недостаточна для активации безубытка
                logger.debug(f"[{symbol}] PnL {pnl_pct:.2f}% < level1_activation {level1_activation:.2f}%, безубыток не активируется")
                return
            
            # Округляем до tick size
            new_sl = self.bybit.round_price(new_sl, symbol)
            tick_size = self.bybit.get_price_step(symbol)
            
            # Проверяем, нужно ли обновлять SL
            should_update = False
            if current_sl:
                current_sl_float = float(current_sl)
                # Если новый SL совпадает с текущим (с учетом шага цены), не обновляем
                if tick_size > 0 and abs(new_sl - current_sl_float) < (tick_size / 2):
                    should_update = False
                elif side == "Buy" and new_sl > current_sl_float:
                    should_update = True
                elif side == "Sell" and new_sl < current_sl_float:
                    should_update = True
            else:
                should_update = True
            
            if should_update:
                logger.info(f"Moving {symbol} SL to breakeven ({level}): {new_sl} (PnL: {pnl_pct:.2f}%)")
                resp = await asyncio.to_thread(
                    self.bybit.set_trading_stop,
                    symbol=symbol,
                    stop_loss=new_sl
                )
                
                if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                    await self.notifier.medium(
                        f"🛡️ БЕЗУБЫТОК АКТИВИРОВАН ({level})\n{symbol} SL → ${new_sl:.2f}\nТекущий PnL: +{pnl_pct:.2f}%"
                    )
        
        except Exception as e:
            # Bybit возвращает "not modified" если стоп-лосс уже равен текущему
            if "ErrCode: 34040" in str(e) or "not modified" in str(e).lower():
                logger.info(f"{symbol} breakeven SL not modified (already set): {e}")
                return
            logger.error(f"Error updating breakeven stop for {symbol}: {e}")

    def _calculate_fees_usd(self, entry_price: float, exit_price: float, qty: float) -> float:
        """Считает комиссию биржи в USD (per side) по notional на входе и выходе."""
        fee_rate = self.settings.risk.fee_rate
        if fee_rate <= 0:
            return 0.0
        notional = (entry_price + exit_price) * qty
        return notional * fee_rate

    def _classify_position_horizon(self, signal: Signal) -> str:
        """Категоризирует позицию по расстоянию до TP/SL."""
        if not signal.take_profit or not signal.stop_loss or not signal.price:
            return "unknown"
        
        tp_dist = abs(signal.take_profit - signal.price) / signal.price
        sl_dist = abs(signal.stop_loss - signal.price) / signal.price
        
        if tp_dist >= self.settings.risk.long_term_tp_pct or sl_dist >= self.settings.risk.long_term_sl_pct:
            return "long_term"
        elif tp_dist >= self.settings.risk.mid_term_tp_pct:
            return "mid_term"
        else:
            return "short_term"

    def _get_atr_pct_1h_sync(self, symbol: str) -> Optional[float]:
        """Синхронно загружает 1h свечи, считает ATR(14) в % от цены. Для фильтра волатильности."""
        try:
            df = self.bybit.get_kline_df(symbol, "1h", limit=30)
            if df is None or df.empty or len(df) < 15:
                return None
            df = prepare_with_indicators(df)
            if "atr_pct" not in df.columns:
                return None
            val = df["atr_pct"].iloc[-1]
            if pd.isna(val) or (isinstance(val, (int, float)) and (val != val or val < 0)):  # NaN or negative
                return None
            return float(val)
        except Exception as e:
            logger.debug(f"[{symbol}] ATR 1h fetch failed: {e}")
            return None
    
    async def _check_pullback_condition(
        self, 
        pending_signal: Dict, 
        symbol: str, 
        current_price: float, 
        high: float, 
        low: float,
        df: pd.DataFrame
    ) -> bool:
        """
        Проверяет условия отката для pending сигнала.
        
        Args:
            pending_signal: Словарь с информацией о pending сигнале
            symbol: Торговая пара
            current_price: Текущая цена закрытия
            high: High текущей свечи
            low: Low текущей свечи
            df: DataFrame с данными (для получения EMA)
        
        Returns:
            True если условия отката выполнены, False иначе
        """
        signal = pending_signal['signal']
        signal_high = pending_signal['signal_high']
        signal_low = pending_signal['signal_low']
        
        pullback_enabled = self.settings.ml_strategy.pullback_enabled
        pullback_ema_period = self.settings.ml_strategy.pullback_ema_period
        pullback_pct = self.settings.ml_strategy.pullback_pct
        pullback_enter_on_continuation = bool(getattr(self.settings.ml_strategy, "pullback_enter_on_continuation", True))
        pullback_max_bars = self.settings.ml_strategy.pullback_max_bars
        
        if not pullback_enabled:
            return False
        
        try:
            # Получаем EMA значение, если доступно
            ema_value = None
            if len(df) > 0:
                if pullback_ema_period == 9:
                    # Используем ema_short (9)
                    if 'ema_short' in df.columns:
                        ema_value = df['ema_short'].iloc[-1]
                elif pullback_ema_period == 20 or pullback_ema_period == 21:
                    # Используем ema_long (21) для периода 20
                    if 'ema_long' in df.columns:
                        ema_value = df['ema_long'].iloc[-1]
                
                if pd.isna(ema_value) or ema_value is None:
                    ema_value = None
            
            if signal.action == Action.LONG:
                if pullback_enter_on_continuation and int(pending_signal.get("bars_waited", 0)) <= max(1, int(pullback_max_bars)):
                    continuation_level = signal_high * (1 + pullback_pct)
                    if high >= continuation_level:
                        logger.info(
                            f"[{symbol}] ✅ Pullback bypass: continuation up to {continuation_level:.2f} (high={high:.2f})"
                        )
                        return True
                # LONG: ждем откат к EMA или к уровню -0.3% от high сигнальной свечи
                pullback_level = signal_high * (1 - pullback_pct)
                
                # Проверяем откат к EMA (если доступно)
                if ema_value is not None and not pd.isna(ema_value):
                    if low <= ema_value <= high:
                        logger.info(f"[{symbol}] ✅ Pullback condition met: price touched EMA{pullback_ema_period} at {ema_value:.2f}")
                        return True  # Цена коснулась EMA
                
                # Проверяем откат к уровню (low текущей свечи <= pullback_level)
                if low <= pullback_level:
                    logger.info(f"[{symbol}] ✅ Pullback condition met: price reached pullback level {pullback_level:.2f} (low={low:.2f})")
                    return True
            else:  # SHORT
                if pullback_enter_on_continuation and int(pending_signal.get("bars_waited", 0)) <= max(1, int(pullback_max_bars)):
                    continuation_level = signal_low * (1 - pullback_pct)
                    if low <= continuation_level:
                        logger.info(
                            f"[{symbol}] ✅ Pullback bypass: continuation down to {continuation_level:.2f} (low={low:.2f})"
                        )
                        return True
                # SHORT: ждем откат вверх к EMA или к уровню +0.3% от low сигнальной свечи
                pullback_level = signal_low * (1 + pullback_pct)
                
                # Проверяем откат к EMA (если доступно)
                if ema_value is not None and not pd.isna(ema_value):
                    if low <= ema_value <= high:
                        logger.info(f"[{symbol}] ✅ Pullback condition met: price touched EMA{pullback_ema_period} at {ema_value:.2f}")
                        return True  # Цена коснулась EMA
                
                # Проверяем откат к уровню (high текущей свечи >= pullback_level)
                if high >= pullback_level:
                    logger.info(f"[{symbol}] ✅ Pullback condition met: price reached pullback level {pullback_level:.2f} (high={high:.2f})")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"[{symbol}] Error checking pullback condition: {e}")
            import traceback
            logger.error(f"[{symbol}] Traceback:\n{traceback.format_exc()}")
            return False
    
    async def _process_pending_pullback_signals(
        self, 
        symbol: str, 
        current_price: float, 
        high: float, 
        low: float,
        df: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Обрабатывает pending сигналы и проверяет условия отката.
        
        Returns:
            Signal для открытия позиции, если условия выполнены, None иначе
        """
        pullback_enabled = self.settings.ml_strategy.pullback_enabled
        pullback_max_bars = self.settings.ml_strategy.pullback_max_bars
        
        if not pullback_enabled or symbol not in self.pending_pullback_signals:
            return None
        
        pending_list = self.pending_pullback_signals.get(symbol, [])
        if not pending_list:
            return None
        
        # Удаляем устаревшие сигналы (превысили максимальную задержку)
        self.pending_pullback_signals[symbol] = [
            ps for ps in pending_list 
            if ps['bars_waited'] < pullback_max_bars
        ]
        
        if not self.pending_pullback_signals[symbol]:
            return None
        
        # Проверяем каждый pending сигнал
        for pending_signal in self.pending_pullback_signals[symbol][:]:  # Копируем список для безопасной итерации
            pending_signal['bars_waited'] += 1
            
            # Проверяем условия отката
            if await self._check_pullback_condition(pending_signal, symbol, current_price, high, low, df):
                # Условия выполнены - возвращаем сигнал для открытия позиции
                signal = pending_signal['signal']
                try:
                    indicators_info = signal.indicators_info if isinstance(signal.indicators_info, dict) else {}
                    indicators_info['signal_received_time'] = pd.Timestamp.now().isoformat()
                    signal.indicators_info = indicators_info
                except Exception:
                    pass
                self.pending_pullback_signals[symbol].remove(pending_signal)
                logger.info(f"[{symbol}] ✅ Pullback condition met after {pending_signal['bars_waited']} bars, opening position")
                return signal
        
        return None
    
    def _add_pending_pullback_signal(self, symbol: str, signal: Signal, signal_time: pd.Timestamp, signal_high: float, signal_low: float):
        """Добавляет сигнал в список ожидающих отката."""
        pullback_enabled = self.settings.ml_strategy.pullback_enabled
        if not pullback_enabled:
            return
        
        if symbol not in self.pending_pullback_signals:
            self.pending_pullback_signals[symbol] = []
        
        self.pending_pullback_signals[symbol].append({
            'signal': signal,
            'signal_time': signal_time,
            'signal_high': signal_high,
            'signal_low': signal_low,
            'bars_waited': 0,
        })
        
        logger.info(f"[{symbol}] 📋 Added signal to pending pullback queue: {signal.action.value} @ {signal.price:.2f}")

    async def update_leverage_for_symbol(self, symbol: str, new_leverage: int):
        try:
            is_valid, error_msg = validate_leverage_settings(new_leverage)
            if not is_valid:
                logger.error(f"[{symbol}] Invalid leverage {new_leverage}: {error_msg}")
                await self.notifier.error(f"❌ Ошибка: {error_msg}")
                return
            
            old_leverage = self.settings.get_leverage_for_symbol(symbol)
            logger.info(f"[{symbol}] Updating leverage from {old_leverage}x to {new_leverage}x")
            
            resp = await asyncio.to_thread(self.bybit.set_leverage, symbol, new_leverage)
            if resp and resp.get("retCode") == 0:
                logger.info(f"[{symbol}] Leverage updated successfully on exchange")
            else:
                logger.warning(f"[{symbol}] set_leverage returned: {resp}")
            
            pos = None
            for p in self.state.trades:
                if p.symbol == symbol and p.status == "open":
                    pos = p
                    break
            
            if not pos:
                logger.info(f"[{symbol}] No open position, leverage settings updated")
                return
            
            logger.info(f"[{symbol}] Found open position, recalculating TP/SL for new leverage")
            
            tick_size = self.bybit.get_tick_size(symbol)
            if tick_size is None:
                tick_size = 0.01
            
            tp, sl = recalculate_tp_sl_for_leverage(
                entry_price=pos.entry_price,
                side=pos.side,
                leverage=new_leverage,
                target_profit_pct_margin=self.settings.ml_strategy.target_profit_pct_margin,
                max_loss_pct_margin=self.settings.ml_strategy.max_loss_pct_margin,
                tick_size=tick_size
            )
                
            logger.info(f"[{symbol}] New SL: {sl}, New TP: {tp} (leverage: {old_leverage}x -> {new_leverage}x)")
            
            update_resp = await asyncio.to_thread(
                self.bybit.set_trading_stop,
                symbol=symbol,
                take_profit=tp,
                stop_loss=sl
            )
            
            if update_resp and update_resp.get("retCode") == 0:
                logger.info(f"[{symbol}] TP/SL successfully updated on exchange")
                pos.stop_loss = sl
                pos.take_profit = tp
                pos.leverage = new_leverage
                self.state.save()
                await self.notifier.info(
                    f"✅ Плечо для {symbol} обновлено: {old_leverage}x → {new_leverage}x\n"
                    f"TP и SL пересчитаны:\n"
                    f"TP: ${tp:.2f}\n"
                    f"SL: ${sl:.2f}"
                )
            else:
                logger.warning(f"[{symbol}] Failed to update TP/SL: {update_resp}")
                
        except Exception as e:
            if "not modified" in str(e).lower() or "ErrCode: 34036" in str(e) or "ErrCode: 34040" in str(e) or "110043" in str(e):
                logger.info(f"[{symbol}] Leverage or TP/SL not modified (already set)")
            else:
                logger.error(f"[{symbol}] Error updating leverage and TP/SL: {e}", exc_info=True)

    async def _cancel_pullback_entry_order(self, symbol: str, reason: str = "") -> None:
        rec = self.pending_pullback_entry_orders.get(symbol)
        if not rec or not isinstance(rec, dict):
            return
        order_link_id = rec.get("order_link_id")
        if not order_link_id:
            self.pending_pullback_entry_orders.pop(symbol, None)
            return
        try:
            open_resp = await asyncio.to_thread(self.bybit.get_open_orders, symbol=symbol, order_link_id=order_link_id)
            open_orders = self._extract_open_orders_list(open_resp)
            if not open_orders:
                self.pending_pullback_entry_orders.pop(symbol, None)
                return
            cancel_resp = await asyncio.to_thread(self.bybit.cancel_order, symbol=symbol, order_link_id=order_link_id)
            ok = isinstance(cancel_resp, dict) and cancel_resp.get("retCode") == 0
            if not ok:
                order_id = open_orders[0].get("orderId") if isinstance(open_orders[0], dict) else None
                if order_id:
                    cancel_resp = await asyncio.to_thread(self.bybit.cancel_order, symbol=symbol, order_id=order_id)
                    ok = isinstance(cancel_resp, dict) and cancel_resp.get("retCode") == 0
            if ok:
                logger.info(f"[{symbol}] 🧹 Pullback limit order canceled ({reason}) link={order_link_id}")
            else:
                logger.warning(f"[{symbol}] ⚠️ Pullback limit order cancel failed ({reason}) link={order_link_id} resp={cancel_resp}")
        except Exception as e:
            logger.warning(f"[{symbol}] ⚠️ Pullback limit order cancel exception ({reason}) link={order_link_id}: {e}")
        finally:
            self.pending_pullback_entry_orders.pop(symbol, None)

    def _signal_confidence(self, signal: Signal) -> float:
        try:
            info = signal.indicators_info if isinstance(signal.indicators_info, dict) else {}
            v = info.get("confidence", 0)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    async def _tick_pullback_entry_order(self, symbol: str, candle_timestamp) -> None:
        rec = self.pending_pullback_entry_orders.get(symbol)
        if not rec or not isinstance(rec, dict):
            return
        order_link_id = rec.get("order_link_id")
        if not order_link_id:
            self.pending_pullback_entry_orders.pop(symbol, None)
            return
        try:
            open_resp = await asyncio.to_thread(self.bybit.get_open_orders, symbol=symbol, order_link_id=order_link_id)
            open_orders = self._extract_open_orders_list(open_resp)
            if not open_orders:
                self.pending_pullback_entry_orders.pop(symbol, None)
                return
        except Exception:
            return

        try:
            rec["bars_waited"] = int(rec.get("bars_waited", 0)) + 1
        except Exception:
            rec["bars_waited"] = 1

        max_bars = int(getattr(self.settings.ml_strategy, "pullback_max_bars", 3) or 3)
        if rec["bars_waited"] >= max(1, max_bars):
            await self._cancel_pullback_entry_order(symbol, reason="max_bars")

    def _calc_pullback_limit_price(self, signal: Signal, symbol: str, signal_high: float, signal_low: float) -> Optional[float]:
        pullback_pct = float(getattr(self.settings.ml_strategy, "pullback_pct", 0.0))
        if pullback_pct <= 0:
            return None
        if signal.action == Action.LONG:
            limit_price = float(signal_high) * (1.0 - pullback_pct)
        elif signal.action == Action.SHORT:
            limit_price = float(signal_low) * (1.0 + pullback_pct)
        else:
            return None
        limit_price = float(self.bybit.round_price(float(limit_price), symbol))
        if limit_price <= 0:
            return None
        return limit_price

    async def _handle_pullback_limit_roll_signal(
        self,
        symbol: str,
        signal: Signal,
        candle_timestamp,
        signal_high: float,
        signal_low: float,
    ) -> None:
        if not self._pullback_limit_roll_enabled():
            return
        rec = self.pending_pullback_entry_orders.get(symbol)

        if signal.action not in (Action.LONG, Action.SHORT):
            if rec:
                await self._cancel_pullback_entry_order(symbol, reason="hold")
            return

        limit_price = self._calc_pullback_limit_price(signal, symbol, signal_high, signal_low)
        if limit_price is None:
            if rec:
                await self._cancel_pullback_entry_order(symbol, reason="bad_price")
            return

        side = "Buy" if signal.action == Action.LONG else "Sell"
        position_horizon = self._classify_position_horizon(signal)
        prep = await self._prepare_entry(symbol, side, signal, False, position_horizon)
        if prep is None:
            if rec:
                await self._cancel_pullback_entry_order(symbol, reason="entry_veto")
            return
        signal_tp, signal_sl, indicators_info = prep
        signal.take_profit = signal_tp
        signal.stop_loss = signal_sl

        if rec and isinstance(rec, dict):
            order_link_id = rec.get("order_link_id")
            if order_link_id:
                try:
                    open_resp = await asyncio.to_thread(self.bybit.get_open_orders, symbol=symbol, order_link_id=order_link_id)
                    if not self._extract_open_orders_list(open_resp):
                        self.pending_pullback_entry_orders.pop(symbol, None)
                        rec = None
                except Exception:
                    pass

        if rec and isinstance(rec, dict):
            old_action = rec.get("action")
            if old_action and str(old_action) != signal.action.value:
                await self._cancel_pullback_entry_order(symbol, reason="side_change")
                rec = None
            else:
                old_conf = float(rec.get("confidence") or 0.0)
                new_conf = self._signal_confidence(signal)
                drop = float(getattr(self.settings.ml_strategy, "pullback_limit_roll_conf_drop_pct", 0.05) or 0.05)
                if new_conf < (old_conf - drop):
                    await self._cancel_pullback_entry_order(symbol, reason="confidence_drop")
                    return

                old_price = float(rec.get("limit_price") or 0.0)
                min_rq = float(getattr(self.settings.ml_strategy, "pullback_limit_roll_min_requote_pct", 0.001) or 0.001)
                price_move = abs(float(limit_price) - old_price) / max(1e-12, old_price) if old_price > 0 else 1.0

                old_signal = rec.get("signal") if isinstance(rec.get("signal"), Signal) else None
                old_tp = float(getattr(old_signal, "take_profit", 0.0) or 0.0) if old_signal else 0.0
                old_sl = float(getattr(old_signal, "stop_loss", 0.0) or 0.0) if old_signal else 0.0
                tp_move = abs(float(signal_tp) - old_tp) / max(1e-12, old_tp) if old_tp > 0 and signal_tp else 0.0
                sl_move = abs(float(signal_sl) - old_sl) / max(1e-12, old_sl) if old_sl > 0 and signal_sl else 0.0

                if price_move < min_rq and tp_move < min_rq and sl_move < min_rq:
                    rec["bars_waited"] = 0
                    try:
                        if new_conf > old_conf:
                            rec["confidence"] = float(new_conf)
                    except Exception:
                        pass
                    return

                await self._cancel_pullback_entry_order(symbol, reason="requote")

        await self._place_pullback_limit_entry_order(symbol, signal, candle_timestamp, signal_high, signal_low)

    async def _place_pullback_limit_entry_order(
        self,
        symbol: str,
        signal: Signal,
        candle_timestamp,
        signal_high: float,
        signal_low: float,
    ) -> None:
        if signal.action not in (Action.LONG, Action.SHORT):
            return
        if not self._pullback_limit_roll_enabled():
            return
        if symbol in self.pending_pullback_entry_orders:
            return

        limit_price = self._calc_pullback_limit_price(signal, symbol, signal_high, signal_low)
        if limit_price is None:
            return

        side = "Buy" if signal.action == Action.LONG else "Sell"

        position_horizon = self._classify_position_horizon(signal)
        prep = await self._prepare_entry(symbol, side, signal, False, position_horizon)
        if prep is None:
            return
        signal_tp, signal_sl, indicators_info = prep
        signal.take_profit = signal_tp
        signal.stop_loss = signal_sl

        qty_calc = await self._calc_order_qty(symbol, float(limit_price), indicators_info, False)
        if qty_calc is None:
            return
        qty, fixed_margin_usd, position_size_usd, balance, qty_step, precision = qty_calc

        order_link_id = f"PB_{symbol}_{uuid.uuid4().hex[:12]}"
        try:
            resp = await asyncio.to_thread(
                self.bybit.place_order,
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Limit",
                price=limit_price,
                time_in_force="GoodTillCancel",
                order_link_id=order_link_id,
                take_profit=signal_tp,
                stop_loss=signal_sl,
            )
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Pullback limit order exception: {e}")
            return

        if not resp or not isinstance(resp, dict) or resp.get("retCode") != 0:
            logger.error(f"[{symbol}] ❌ Pullback limit order failed: resp={resp}")
            return

        placed_at = self._to_timestamp(candle_timestamp) or pd.Timestamp.now()
        self.pending_pullback_entry_orders[symbol] = {
            "order_link_id": order_link_id,
            "side": side,
            "limit_price": float(limit_price),
            "qty": float(qty),
            "placed_at": placed_at,
            "bars_waited": 0,
            "action": signal.action.value,
            "confidence": float(self._signal_confidence(signal)),
            "signal_high": float(signal_high),
            "signal_low": float(signal_low),
            "signal": signal,
            "expected_margin_usd": float(fixed_margin_usd),
            "expected_position_usd": float(position_size_usd),
        }
        logger.info(
            f"[{symbol}] 📌 Pullback limit order placed: {side} qty={qty:.{precision}f} price={limit_price:.6f} "
            f"link={order_link_id}"
        )

    def _timeframe_minutes(self, timeframe: str) -> int:
        tf = str(timeframe).strip().lower()
        if tf.endswith("m"):
            try:
                return int(tf[:-1])
            except ValueError:
                return 15
        if tf.endswith("h"):
            try:
                return int(tf[:-1]) * 60
            except ValueError:
                return 60
        try:
            return int(tf)
        except ValueError:
            return 15

    def _next_candle_close_utc(self, timeframe: str) -> pd.Timestamp:
        minutes = self._timeframe_minutes(timeframe)
        now = pd.Timestamp.now(tz="UTC")
        if minutes < 60:
            base = now.floor(f"{minutes}min")
            return base + pd.Timedelta(minutes=minutes)
        if minutes == 60:
            base = now.floor("h")
            return base + pd.Timedelta(hours=1)
        hours = max(1, minutes // 60)
        base = now.floor(f"{hours}h")
        return base + pd.Timedelta(hours=hours)

    def _eval_tp_reentry(self, symbol: str, guard, df_15m: pd.DataFrame, action: Action) -> tuple[bool, str]:
        try:
            r = self.settings.risk
            now = pd.Timestamp.now(tz="UTC")
            exit_time = pd.Timestamp(guard.exit_time_utc)
            if exit_time.tzinfo is None:
                exit_time = exit_time.tz_localize("UTC")

            window = df_15m
            if hasattr(df_15m, "index") and len(df_15m.index) > 0:
                try:
                    if isinstance(df_15m.index, pd.DatetimeIndex):
                        idx = df_15m.index
                        if idx.tz is None:
                            idx = idx.tz_localize("UTC")
                        window = df_15m.loc[idx >= exit_time]
                except Exception:
                    window = df_15m

            if window is None or window.empty:
                window = df_15m.tail(max(5, r.tp_reentry_sr_lookback))

            lb = max(5, int(r.tp_reentry_sr_lookback))
            tlb = max(5, int(r.tp_reentry_trend_lookback))
            w = window.tail(max(lb, tlb))

            if not {"high", "low", "close", "volume"}.issubset(set(w.columns)):
                return True, "tp_reentry_no_ohlcv"

            current_price = float(w["close"].iloc[-1])
            exit_price = float(guard.exit_price)

            if action == Action.LONG:
                pullback = (exit_price - float(w["low"].min())) / exit_price if exit_price > 0 else 0.0
                breakout_level = float(w["high"].iloc[:-1].max()) if len(w) > 1 else float(w["high"].max())
                breakout_ok = current_price >= breakout_level * (1.0 + r.tp_reentry_breakout_buffer_pct)
                trend_series = w["close"].tail(tlb).astype(float)
                x = np.arange(len(trend_series))
                slope = float(np.polyfit(x, trend_series.values, 1)[0]) / max(1e-12, float(trend_series.values[-1]))
                trend_ok = slope >= r.tp_reentry_min_trend_slope
            else:
                pullback = (float(w["high"].max()) - exit_price) / exit_price if exit_price > 0 else 0.0
                breakout_level = float(w["low"].iloc[:-1].min()) if len(w) > 1 else float(w["low"].min())
                breakout_ok = current_price <= breakout_level * (1.0 - r.tp_reentry_breakout_buffer_pct)
                trend_series = w["close"].tail(tlb).astype(float)
                x = np.arange(len(trend_series))
                slope = float(np.polyfit(x, trend_series.values, 1)[0]) / max(1e-12, float(trend_series.values[-1]))
                trend_ok = slope <= -r.tp_reentry_min_trend_slope

            pullback_ok = (pullback >= r.tp_reentry_min_pullback_pct) and (pullback <= r.tp_reentry_max_pullback_pct)

            vol_series = w["volume"].astype(float)
            avg_vol = float(vol_series.tail(min(20, len(vol_series))).mean())
            cur_vol = float(vol_series.iloc[-1])
            vol_ok = (avg_vol <= 0) or (cur_vol >= avg_vol * r.tp_reentry_volume_factor)

            ok = pullback_ok and breakout_ok and trend_ok and vol_ok
            reason = (
                f"tp_reentry ok={ok} pullback={pullback*100:.2f}% "
                f"[{r.tp_reentry_min_pullback_pct*100:.2f}-{r.tp_reentry_max_pullback_pct*100:.2f}] "
                f"breakout_ok={breakout_ok} trend_slope={slope:.6f} vol={cur_vol:.2f}/{avg_vol:.2f}x{r.tp_reentry_volume_factor}"
            )
            return ok, reason
        except Exception as e:
            return True, f"tp_reentry_eval_error:{e}"

    async def close_position(self, symbol: str, reason: str):
        try:
            logger.info(f"[{symbol}] 🚨 Initiating close position: {reason}")
            # Get current position info to be sure about size
            pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
            
            if not pos_info or pos_info.get("retCode") != 0:
                logger.error(f"[{symbol}] Failed to get position info for closing: {pos_info}")
                return

            result = pos_info.get("result", {}).get("list", [])
            if not result:
                logger.warning(f"[{symbol}] No position found to close")
                return

            position = result[0]
            size = float(position.get("size", 0))
            side = position.get("side")
            
            if size <= 0:
                logger.warning(f"[{symbol}] Position size is 0, nothing to close")
                return

            close_side = "Sell" if side == "Buy" else "Buy"
            
            logger.info(f"[{symbol}] Closing {side} position size {size} via Market order...")
            
            resp = await asyncio.to_thread(
                self.bybit.place_order,
                symbol=symbol,
                side=close_side,
                qty=size,
                order_type="Market",
                reduce_only=True
            )
            
            if resp and resp.get("retCode") == 0:
                logger.info(f"[{symbol}] ✅ Position closed successfully: {reason}")
                
                # Log to trades.log
                trade_logger.info(f"POSITION CLOSED: {symbol} {close_side} Size={size} Reason={reason}")
                
                await self.notifier.medium(f"🚫 ПОЗИЦИЯ ЗАКРЫТА ({reason})\n{symbol}\nSize: {size}")
            else:
                logger.error(f"[{symbol}] Failed to close position: {resp}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def check_time_stop(self, symbol: str, position: dict):
        try:
            raw_created_time = position.get("createdTime", 0)
            raw_updated_time = position.get("updatedTime", 0)

            candidates = []
            for raw in (raw_created_time, raw_updated_time):
                try:
                    v = float(raw)
                    if v > 0:
                        candidates.append(v)
                except (TypeError, ValueError):
                    continue

            if not candidates:
                return

            base_ts = max(candidates)
            unit = "ms" if base_ts > 1e11 else "s"

            open_time = pd.Timestamp(base_ts, unit=unit, tz="UTC")
            now = pd.Timestamp.now(tz="UTC")

            if open_time > now:
                duration_minutes = 0.0
                logger.warning(f"[{symbol}] Position open time {open_time} is in the future relative to {now}. createdTime={raw_created_time}, updatedTime={raw_updated_time}")
            else:
                duration_minutes = (now - open_time).total_seconds() / 60

            if duration_minutes > 60 * 24 * 30:
                logger.warning(f"[{symbol}] Unusually long duration: {duration_minutes:.1f} min. createdTime={raw_created_time}, updatedTime={raw_updated_time}, open_time={open_time}, now={now}")

            max_minutes = self.settings.risk.time_stop_minutes
            
            if max_minutes <= 0:
                return

            if duration_minutes > max_minutes:
                logger.info(f"[{symbol}] ⏰ Time Stop triggered! Duration: {duration_minutes:.1f} min > {max_minutes} min")
                await self.close_position(symbol, f"Time Stop > {max_minutes} min")
                
        except Exception as e:
            logger.error(f"[{symbol}] Error in check_time_stop: {e}")

    async def check_early_exit(self, symbol: str, position: dict):
        try:
            raw_created_time = position.get("createdTime", 0)
            raw_updated_time = position.get("updatedTime", 0)
            
            candidates = []
            for raw in (raw_created_time, raw_updated_time):
                try:
                    v = float(raw)
                    if v > 0:
                        candidates.append(v)
                except (TypeError, ValueError):
                    continue

            if not candidates:
                return

            base_ts = max(candidates)
            unit = "ms" if base_ts > 1e11 else "s"

            open_time = pd.Timestamp(base_ts, unit=unit, tz="UTC")
            now = pd.Timestamp.now(tz="UTC")

            if open_time > now:
                duration_minutes = 0.0
                logger.warning(f"[{symbol}] Position open time {open_time} is in the future relative to {now}. createdTime={raw_created_time}, updatedTime={raw_updated_time}")
            else:
                duration_minutes = (now - open_time).total_seconds() / 60

            if duration_minutes > 60 * 24 * 30:
                logger.warning(f"[{symbol}] Unusually long duration: {duration_minutes:.1f} min. createdTime={raw_created_time}, updatedTime={raw_updated_time}, open_time={open_time}, now={now}")

            early_exit_minutes = self.settings.risk.early_exit_minutes
            
            if early_exit_minutes <= 0:
                return

            try:
                leverage_raw = position.get("leverage", DEFAULT_LEVERAGE)
                leverage = int(leverage_raw) if leverage_raw else DEFAULT_LEVERAGE
                if leverage <= 0:
                    leverage = DEFAULT_LEVERAGE
            except (TypeError, ValueError):
                leverage = DEFAULT_LEVERAGE
            
            try:
                min_profit_pct = get_early_exit_profit_pct(self.settings, leverage)
            except Exception as e:
                logger.warning(f"[{symbol}] Error getting early_exit profit pct: {e}, using default")
                min_profit_pct = self.settings.risk.early_exit_min_profit_pct
            
            if duration_minutes > early_exit_minutes:
                entry_price = float(position.get("avgPrice", 0))
                mark_price = float(position.get("markPrice", entry_price))
                side = position.get("side")
                
                if entry_price == 0:
                    return

                if side == "Buy":
                    pnl_pct = (mark_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - mark_price) / entry_price
                
                if pnl_pct < min_profit_pct:
                    mode = self.settings.risk.early_exit_mode
                    margin_pct = self.settings.risk.early_exit_min_profit_pct_margin
                    logger.info(f"[{symbol}] 📉 Early Exit triggered! Duration: {duration_minutes:.1f} min > {early_exit_minutes} min, PnL: {pnl_pct*100:.2f}% < {min_profit_pct*100:.2f}% (mode: {mode}, margin: {margin_pct}%, leverage: {leverage}x)")
                    await self.close_position(symbol, f"Early Exit (Low PnL < {min_profit_pct*100:.1f}%)")
                    
        except Exception as e:
            logger.error(f"[{symbol}] Error in check_early_exit: {e}")

    def _should_dca(self, local_pos: TradeRecord, signal: Signal, current_price: float, confidence: float) -> bool:
        """Проверяет условия для усреднения позиции."""
        if not self.settings.risk.dca_enabled:
            return False
        if local_pos.horizon not in ("mid_term", "long_term"):
            return False
        if local_pos.dca_count >= self.settings.risk.dca_max_adds:
            return False
        if confidence < self.settings.risk.dca_min_confidence:
            return False
        if not current_price or not local_pos.entry_price:
            return False

        leverage = local_pos.leverage if local_pos.leverage else DEFAULT_LEVERAGE
        
        dca_drawdown_pct = get_dca_drawdown_pct(self.settings, leverage)
        
        if local_pos.side == "Buy":
            drawdown_pct = (local_pos.entry_price - current_price) / local_pos.entry_price
        else:
            drawdown_pct = (current_price - local_pos.entry_price) / local_pos.entry_price

        return drawdown_pct >= dca_drawdown_pct

    def _is_strong_reverse_signal(self, signal: Signal, confidence: float) -> bool:
        """Определяет, является ли обратный сигнал сильным для реверса."""
        if not self.settings.risk.reverse_on_strong_signal:
            return False
        if confidence < self.settings.risk.reverse_min_confidence:
            return False
        # Проверяем силу сигнала, если доступна
        strength = None
        if signal.indicators_info and isinstance(signal.indicators_info, dict):
            strength = signal.indicators_info.get("strength")
        if strength is None and signal.reason:
            # Пытаемся вытащить силу из текста причины (ml_..._сила_сильное_..)
            parts = str(signal.reason).split("_сила_")
            if len(parts) == 2:
                strength = parts[1].split("_")[0]
        if strength:
            order = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
            try:
                if order.index(strength) < order.index(self.settings.risk.reverse_min_strength):
                    return False
            except ValueError:
                # неизвестная сила — не блокируем, но логируем
                logger.warning(f"Unknown signal strength '{strength}', allowing reverse by confidence only.")
        return True
    
    def _load_cached_1h_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загружает кэшированные 1h данные из ml_data/{symbol}_60_cache.csv
        
        Returns:
            DataFrame с 1h данными или None если файл не найден
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            cache_file = ml_data_dir / f"{symbol}_60_cache.csv"
            
            if not cache_file.exists():
                return None
            
            df = pd.read_csv(cache_file)
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"[{symbol}] Cached 1h data missing required columns")
                return None
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps in cache, removing them")
                    df = df[~invalid_dates].copy()
                
                if len(df) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing")
                    return None
                
                # Сортируем по времени и удаляем дубликаты
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            
            # Берем последние 500 свечей (достаточно для 1h модели)
            if len(df) > 500:
                df = df.tail(500).reset_index(drop=True)
            
            # Устанавливаем timestamp как индекс для совместимости с MTF стратегией
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                # Проверяем, что индекс корректный
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from timestamp column")
                    # Удаляем некорректный кэш
                    try:
                        cache_file.unlink()
                        logger.info(f"[{symbol}] Removed invalid cache file (no DatetimeIndex)")
                    except Exception as e:
                        logger.debug(f"[{symbol}] Could not remove invalid cache: {e}")
                    return None
                
                # Проверяем, что даты разумные (не 1970 год)
                if len(df) > 0:
                    first_date = df.index[0]
                    last_date = df.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates in cache: first={first_date}, last={last_date}")
                        # Удаляем некорректный кэш и возвращаем None для пересоздания
                        try:
                            cache_file.unlink()
                            logger.info(f"[{symbol}] Removed invalid cache file (invalid dates), will recreate")
                        except Exception as e:
                            logger.warning(f"[{symbol}] Failed to remove invalid cache: {e}")
                        return None
            
            logger.debug(f"[{symbol}] Loaded {len(df)} cached 1h candles from {cache_file}")
            if len(df) > 0:
                logger.debug(f"[{symbol}] Cache date range: {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load cached 1h data: {e}")
            return None
    
    def _fetch_and_cache_1h_data(self, symbol: str, existing_cache: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Запрашивает 1h данные с биржи и сохраняет в кэш.
        Если есть существующий кэш, подгружает только новые данные.
        
        Args:
            symbol: Торговая пара
            existing_cache: Существующий кэш (если есть) для подгрузки только новых данных
        
        Returns:
            DataFrame с 1h данными или None при ошибке
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            ml_data_dir.mkdir(exist_ok=True)
            cache_file = ml_data_dir / f"{symbol}_60_cache.csv"
            
            # Определяем, с какого времени нужно подгружать данные
            start_from = None
            if existing_cache is not None and not existing_cache.empty:
                if isinstance(existing_cache.index, pd.DatetimeIndex) and len(existing_cache) > 0:
                    # Берем последнюю свечу из кэша и запрашиваем данные начиная с неё
                    last_candle_time = existing_cache.index[-1]
                    # Запрашиваем немного больше, чтобы перекрыть последнюю свечу (она может быть незакрыта)
                    start_from = last_candle_time - pd.Timedelta(hours=1)
                    logger.debug(f"[{symbol}] Updating cache from {last_candle_time} (will fetch from {start_from})")
            
            # Запрашиваем 500 свечей 1h с биржи
            logger.info(f"[{symbol}] Fetching 1h data from exchange (limit=500)...")
            df_new = self.bybit.get_kline_df(symbol, "60", 500)
            
            if df_new.empty:
                logger.warning(f"[{symbol}] No 1h data received from exchange")
                return existing_cache  # Возвращаем существующий кэш, если новый запрос пуст
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df_new.columns for col in required_cols):
                logger.warning(f"[{symbol}] 1h data from exchange missing required columns")
                return existing_cache
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df_new.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df_new["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df_new["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps from exchange, removing them")
                    df_new = df_new[~invalid_dates].copy()
                
                if len(df_new) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing from exchange")
                    return existing_cache
                
                df_new = df_new.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                df_new = df_new.set_index("timestamp")
                
                # Проверяем, что индекс корректный и даты разумные
                if not isinstance(df_new.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from exchange data")
                    return existing_cache
                
                if len(df_new) > 0:
                    first_date = df_new.index[0]
                    last_date = df_new.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates from exchange: first={first_date}, last={last_date}")
                        return existing_cache
                    logger.debug(f"[{symbol}] Exchange data date range: {first_date} to {last_date}")
            elif not isinstance(df_new.index, pd.DatetimeIndex):
                logger.warning(f"[{symbol}] Could not set timestamp index for new 1h data")
                return existing_cache
            
            # Объединяем с существующим кэшем, если есть
            if existing_cache is not None and not existing_cache.empty:
                # Объединяем данные, удаляя дубликаты
                df_combined = pd.concat([existing_cache, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # Оставляем последние значения для дубликатов
                df_combined = df_combined.sort_index()
                
                # Берем последние 500 свечей
                if len(df_combined) > 500:
                    df_combined = df_combined.tail(500)
                
                df_final = df_combined
                logger.info(f"[{symbol}] Merged cache: {len(existing_cache)} old + {len(df_new)} new = {len(df_final)} total candles")
            else:
                df_final = df_new
                logger.info(f"[{symbol}] Created new cache with {len(df_final)} candles")
            
            # Сохраняем в кэш
            try:
                df_to_save = df_final.reset_index() if isinstance(df_final.index, pd.DatetimeIndex) else df_final.copy()
                
                # Убеждаемся, что timestamp в правильном формате для сохранения
                if "timestamp" in df_to_save.columns:
                    # Проверяем, что timestamp - это datetime, а не что-то другое
                    if not pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        # Если это не datetime, пытаемся преобразовать
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        # Удаляем некорректные даты
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                    
                    # Сохраняем timestamp в формате строки для читаемости
                    # Проверяем, что timestamp - это datetime перед форматированием
                    if pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Если это не datetime, пытаемся преобразовать сначала
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                        if len(df_to_save) > 0:
                            df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"[{symbol}] ✅ Saved {len(df_final)} 1h candles to cache: {cache_file}")
                if len(df_to_save) > 0:
                    logger.debug(f"[{symbol}] Saved cache date range: {df_to_save['timestamp'].iloc[0]} to {df_to_save['timestamp'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to save 1h cache: {e}", exc_info=True)
            
            return df_final
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch and cache 1h data: {e}", exc_info=True)
            return existing_cache
    
    def _load_cached_5m_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загружает кэшированные 5m данные из ml_data/{symbol}_5_cache.csv
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            cache_file = ml_data_dir / f"{symbol}_5_cache.csv"

            if not cache_file.exists():
                return None

            df = pd.read_csv(cache_file)

            if "timestamp" in df.columns:
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
                df = df.set_index("timestamp")

            if len(df) > 500:
                df = df.tail(500)

            logger.debug(f"[{symbol}] Loaded {len(df)} cached 5m candles")
            return df
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load cached 5m data: {e}")
            return None

    def _fetch_and_cache_5m_data(self, symbol: str, existing_cache: Optional[pd.DataFrame] = None, required_limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Запрашивает 5m данные с биржи и сохраняет в кэш.
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            ml_data_dir.mkdir(exist_ok=True)
            cache_file = ml_data_dir / f"{symbol}_5_cache.csv"

            limit = 100 if existing_cache is not None and not existing_cache.empty else required_limit

            logger.info(f"[{symbol}] Fetching 5m data from exchange (limit={limit})...")
            df_new = self.bybit.get_kline_df(symbol, "5", limit)

            if df_new.empty:
                return existing_cache

            if "timestamp" in df_new.columns:
                if pd.api.types.is_numeric_dtype(df_new["timestamp"]):
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit='ms', errors='coerce')
                else:
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors='coerce')
                df_new = df_new.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
                df_new = df_new.set_index("timestamp")

            if existing_cache is not None and not existing_cache.empty:
                df_combined = pd.concat([existing_cache, df_new])
                df_final = df_combined[~df_combined.index.duplicated(keep='last')].sort_index().tail(required_limit)
            else:
                df_final = df_new

            try:
                df_to_save = df_final.reset_index()
                if "timestamp" in df_to_save.columns:
                    df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"[{symbol}] ✅ Saved {len(df_final)} 5m candles to cache")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to save 5m cache: {e}")

            return df_final
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch and cache 5m data: {e}", exc_info=True)
            return existing_cache

    def _load_cached_15m_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загружает кэшированные 15m данные из ml_data/{symbol}_15_cache.csv
        
        Returns:
            DataFrame с 15m данными или None если файл не найден
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            cache_file = ml_data_dir / f"{symbol}_15_cache.csv"
            
            if not cache_file.exists():
                return None
            
            df = pd.read_csv(cache_file)
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"[{symbol}] Cached 15m data missing required columns")
                return None
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps in cache, removing them")
                    df = df[~invalid_dates].copy()
                
                if len(df) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing")
                    return None
                
                # Сортируем по времени и удаляем дубликаты
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            
            # Устанавливаем timestamp как индекс
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                # Проверяем, что индекс корректный
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from timestamp column")
                    # Удаляем некорректный кэш
                    try:
                        cache_file.unlink()
                        logger.info(f"[{symbol}] Removed invalid cache file (no DatetimeIndex)")
                    except Exception as e:
                        logger.debug(f"[{symbol}] Could not remove invalid cache: {e}")
                    return None
                
                # Проверяем, что даты разумные (не 1970 год)
                if len(df) > 0:
                    first_date = df.index[0]
                    last_date = df.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates in cache: first={first_date}, last={last_date}")
                        # Удаляем некорректный кэш и возвращаем None для пересоздания
                        try:
                            cache_file.unlink()
                            logger.info(f"[{symbol}] Removed invalid cache file (invalid dates), will recreate")
                        except Exception as e:
                            logger.warning(f"[{symbol}] Failed to remove invalid cache: {e}")
                        return None
            
            logger.debug(f"[{symbol}] Loaded {len(df)} cached 15m candles from {cache_file}")
            if len(df) > 0:
                logger.debug(f"[{symbol}] Cache date range: {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load cached 15m data: {e}")
            return None
    
    def _fetch_and_cache_15m_data(self, symbol: str, existing_cache: Optional[pd.DataFrame] = None, required_limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Запрашивает 15m данные с биржи и сохраняет в кэш.
        Если есть существующий кэш, подгружает только новые данные.
        
        Args:
            symbol: Торговая пара
            existing_cache: Существующий кэш (если есть) для подгрузки только новых данных
            required_limit: Минимальное количество свечей, которое нужно иметь
        
        Returns:
            DataFrame с 15m данными или None при ошибке
        """
        try:
            from pathlib import Path
            ml_data_dir = Path("ml_data")
            ml_data_dir.mkdir(exist_ok=True)
            cache_file = ml_data_dir / f"{symbol}_15_cache.csv"
            
            # Определяем, сколько свечей нужно запросить
            if existing_cache is not None and not existing_cache.empty:
                if isinstance(existing_cache.index, pd.DatetimeIndex) and len(existing_cache) > 0:
                    # Если есть кэш, запрашиваем последние свечи для обновления
                    # Запрашиваем достаточно, чтобы гарантированно получить самую новую свечу
                    # Для 15m свечей запрашиваем последние 20 свечей (5 часов данных) - этого достаточно
                    # чтобы покрыть возможные пропуски и получить самую новую свечу
                    limit = 20  # Запрашиваем последние 20 свечей для обновления
                    logger.debug(f"[{symbol}] Updating cache: have {len(existing_cache)} candles, fetching last {limit} candles from exchange")
                else:
                    limit = required_limit
            else:
                # Если кэша нет, запрашиваем полное количество
                limit = required_limit
            
            # Запрашиваем 15m данные с биржи
            # Используем "15" для 15-минутных свечей (независимо от основного timeframe)
            logger.info(f"[{symbol}] Fetching 15m data from exchange (limit={limit})...")
            df_new = self.bybit.get_kline_df(symbol, "15", limit)
            
            if df_new.empty:
                logger.warning(f"[{symbol}] No 15m data received from exchange")
                return existing_cache  # Возвращаем существующий кэш, если новый запрос пуст
            
            # Проверяем наличие необходимых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df_new.columns for col in required_cols):
                logger.warning(f"[{symbol}] 15m data from exchange missing required columns")
                return existing_cache
            
            # Преобразуем timestamp в datetime
            if "timestamp" in df_new.columns:
                # Парсим timestamp: может быть строка, datetime или число (миллисекунды)
                if pd.api.types.is_numeric_dtype(df_new["timestamp"]):
                    # Если это число, интерпретируем как миллисекунды
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit='ms', errors='coerce')
                else:
                    # Если это строка или datetime, парсим как обычно
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors='coerce')
                # Удаляем строки с некорректными датами
                invalid_dates = df_new["timestamp"].isna()
                if invalid_dates.any():
                    logger.warning(f"[{symbol}] Found {invalid_dates.sum()} rows with invalid timestamps from exchange, removing them")
                    df_new = df_new[~invalid_dates].copy()
                
                if len(df_new) == 0:
                    logger.warning(f"[{symbol}] No valid data after timestamp parsing from exchange")
                    return existing_cache
                
                df_new = df_new.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                df_new = df_new.set_index("timestamp")
                
                # Проверяем, что индекс корректный и даты разумные
                if not isinstance(df_new.index, pd.DatetimeIndex):
                    logger.warning(f"[{symbol}] Failed to create DatetimeIndex from exchange data")
                    return existing_cache
                
                if len(df_new) > 0:
                    first_date = df_new.index[0]
                    last_date = df_new.index[-1]
                    if first_date.year < 2020 or last_date.year < 2020:
                        logger.warning(f"[{symbol}] Invalid dates from exchange: first={first_date}, last={last_date}")
                        return existing_cache
                    logger.debug(f"[{symbol}] Exchange data date range: {first_date} to {last_date}")
            elif not isinstance(df_new.index, pd.DatetimeIndex):
                logger.warning(f"[{symbol}] Could not set timestamp index for new 15m data")
                return existing_cache
            
            # Объединяем с существующим кэшем, если есть
            if existing_cache is not None and not existing_cache.empty:
                # Объединяем данные, удаляя дубликаты
                df_combined = pd.concat([existing_cache, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # Оставляем последние значения для дубликатов
                df_combined = df_combined.sort_index()
                
                # Берем последние required_limit свечей (или больше, если нужно)
                if len(df_combined) > required_limit:
                    df_combined = df_combined.tail(required_limit)
                
                df_final = df_combined
                logger.info(f"[{symbol}] Merged cache: {len(existing_cache)} old + {len(df_new)} new = {len(df_final)} total candles")
            else:
                df_final = df_new
                logger.info(f"[{symbol}] Created new cache with {len(df_final)} candles")
            
            # Сохраняем в кэш
            try:
                df_to_save = df_final.reset_index() if isinstance(df_final.index, pd.DatetimeIndex) else df_final.copy()
                
                # Убеждаемся, что timestamp в правильном формате для сохранения
                if "timestamp" in df_to_save.columns:
                    # Проверяем, что timestamp - это datetime, а не что-то другое
                    if not pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        # Если это не datetime, пытаемся преобразовать
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        # Удаляем некорректные даты
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                    
                    # Сохраняем timestamp в формате строки для читаемости
                    # Проверяем, что timestamp - это datetime перед форматированием
                    if pd.api.types.is_datetime64_any_dtype(df_to_save["timestamp"]):
                        df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Если это не datetime, пытаемся преобразовать сначала
                        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"], errors='coerce')
                        invalid = df_to_save["timestamp"].isna()
                        if invalid.any():
                            logger.warning(f"[{symbol}] Removing {invalid.sum()} rows with invalid timestamps before saving")
                            df_to_save = df_to_save[~invalid].copy()
                        if len(df_to_save) > 0:
                            df_to_save["timestamp"] = df_to_save["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"[{symbol}] ✅ Saved {len(df_final)} 15m candles to cache: {cache_file}")
                if len(df_to_save) > 0:
                    logger.debug(f"[{symbol}] Saved cache date range: {df_to_save['timestamp'].iloc[0]} to {df_to_save['timestamp'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to save 15m cache: {e}", exc_info=True)
            
            return df_final
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch and cache 15m data: {e}", exc_info=True)
            return existing_cache  # Возвращаем существующий кэш при ошибке

    async def _get_btc_signal(self) -> Optional[Dict]:
        """
        Получает сигнал BTCUSDT для проверки направления других пар.
        Использует кэш на 5 минут, чтобы не делать лишние запросы.
        
        Returns:
            Dict с ключами 'action' (Action) и 'confidence' (float) или None
        """
        import time
        
        # Проверяем кэш (актуален 5 минут)
        current_time = time.time()
        if (self._btc_signal_cache is not None and 
            self._btc_signal_cache_time is not None and 
            current_time - self._btc_signal_cache_time < 300):  # 5 минут
            return self._btc_signal_cache
        
        # Если BTCUSDT не в активных символах, возвращаем None
        if "BTCUSDT" not in self.state.active_symbols:
            return None
        
        try:
            # Получаем данные BTCUSDT
            btc_df = await asyncio.to_thread(
                self.bybit.get_kline_df,
                "BTCUSDT",
                self.settings.timeframe,
                200
            )
            
            if btc_df.empty or len(btc_df) < 2:
                return None
            
            # Инициализируем стратегию BTCUSDT если нужно
            if "BTCUSDT" not in self.strategies:
                model_path = self.state.symbol_models.get("BTCUSDT")
                if not model_path:
                    from pathlib import Path
                    models = list(Path("ml_models").glob("*_BTCUSDT_*.pkl"))
                    if models:
                        ob_models = [p for p in models if "_ob" in p.stem]
                        if ob_models:
                            models = ob_models + [p for p in models if p not in ob_models]
                        model_path = str(models[0])
                        self.state.symbol_models["BTCUSDT"] = model_path
                
                if model_path:
                    ms = self.settings.ml_strategy
                    self.strategies["BTCUSDT"] = MLStrategy(
                        model_path=model_path,
                        confidence_threshold=ms.confidence_threshold,
                        min_signal_strength=ms.min_signal_strength,
                        stability_filter=ms.stability_filter,
                        min_signals_per_day=ms.min_signals_per_day,
                        max_signals_per_day=ms.max_signals_per_day,
                        use_dynamic_ensemble_weights=getattr(ms, "use_dynamic_ensemble_weights", False),
                        adx_trend_threshold=getattr(ms, "adx_trend_threshold", 25.0),
                        adx_flat_threshold=getattr(ms, "adx_flat_threshold", 20.0),
                        trend_weights=getattr(ms, "trend_weights", None),
                        flat_weights=getattr(ms, "flat_weights", None),
                        use_fixed_sl_from_risk=getattr(ms, "use_fixed_sl_from_risk", False),
                    )
                else:
                    return None
            
            # Получаем позицию BTCUSDT
            try:
                btc_pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol="BTCUSDT")
                btc_has_pos = None
                if btc_pos_info and isinstance(btc_pos_info, dict) and btc_pos_info.get("retCode") == 0:
                    result = btc_pos_info.get("result")
                    if result and isinstance(result, dict):
                        list_data = result.get("list", [])
                        if list_data and len(list_data) > 0:
                            p = list_data[0]
                            if p and isinstance(p, dict):
                                btc_size = float(p.get("size", 0))
                                if btc_size > 0:
                                    btc_side = p.get("side")
                                    btc_has_pos = Bias.LONG if btc_side == "Buy" else Bias.SHORT
            except Exception as e:
                logger.debug(f"Error getting BTCUSDT position: {e}")
                btc_has_pos = None
            
            # Генерируем сигнал BTCUSDT
            btc_strategy = self.strategies["BTCUSDT"]
            btc_row = btc_df.iloc[-2] if len(btc_df) >= 2 else btc_df.iloc[-1]
            btc_current_price = btc_df.iloc[-1]['close']
            
            btc_signal = await asyncio.to_thread(
                btc_strategy.generate_signal,
                row=btc_row,
                df=btc_df.iloc[:-1] if len(btc_df) >= 2 else btc_df,
                has_position=btc_has_pos,
                current_price=btc_current_price,
                leverage=self.settings.get_leverage_for_symbol("BTCUSDT")
            )
            
            if btc_signal:
                # Сохраняем в кэш
                indicators_info = btc_signal.indicators_info if btc_signal.indicators_info and isinstance(btc_signal.indicators_info, dict) else {}
                btc_confidence = indicators_info.get('confidence', 0) if isinstance(indicators_info, dict) else 0
                
                self._btc_signal_cache = {
                    'action': btc_signal.action,
                    'confidence': btc_confidence
                }
                self._btc_signal_cache_time = current_time
                
                return self._btc_signal_cache
            
        except Exception as e:
            logger.debug(f"Error getting BTCUSDT signal: {e}")
        
        return None

    async def _close_position_market(self, symbol: str, side: Bias, size: float):
        """Закрывает позицию по рынку (reduce_only)."""
        if size <= 0:
            return
        close_side = "Sell" if side == Bias.LONG else "Buy"
        logger.info(f"[{symbol}] Closing position by market for reverse: {size} {close_side}")
        resp = await asyncio.to_thread(
            self.bybit.place_order,
            symbol=symbol,
            side=close_side,
            qty=size,
            order_type="Market",
            reduce_only=True,
        )
        if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
            await self.notifier.high(f"🔁 РЕВЕРС: позиция {symbol} закрыта и будет открыта в обратную сторону")
        else:
            logger.error(f"[{symbol}] Failed to close position for reverse: {resp}")
    
    async def update_trailing_stop(self, symbol: str, position_info: dict):
        """Активирует трейлинг стоп при достижении порога прибыли"""
        try:
            if not self.settings.risk.enable_trailing_stop:
                return
            
            if not position_info or not isinstance(position_info, dict):
                return
            
            if not position_info.get("size"):
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            trailing_stop = position_info.get("trailingStop")

            if not bool(getattr(self.settings.risk, "use_take_profit", True)):
                if symbol not in self._tp_cleared_for_symbol:
                    try:
                        existing_tp = position_info.get("takeProfit")
                        if existing_tp:
                            existing_tp_f = float(existing_tp)
                            if existing_tp_f > 0:
                                resp = await asyncio.to_thread(
                                    self.bybit.set_trading_stop,
                                    symbol=symbol,
                                    take_profit=0.0,
                                )
                                if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                                    self._tp_cleared_for_symbol.add(symbol)
                    except Exception:
                        pass
            
            if not entry_price or not mark_price:
                return
            
            leverage = self.settings.get_leverage_for_symbol(symbol)
            if leverage is None or leverage <= 0:
                leverage = DEFAULT_LEVERAGE
            
            activation_pct = get_trailing_activation_pct(self.settings, leverage)
            
            # Рассчитываем текущий PnL в процентах
            if side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price)
            else:  # Sell
                pnl_pct = ((entry_price - mark_price) / entry_price)
            
            try:
                trailing_stop_val = float(trailing_stop) if trailing_stop not in (None, "", 0, "0") else 0.0
            except Exception:
                trailing_stop_val = 0.0

            if pnl_pct >= activation_pct and trailing_stop_val <= 0:
                trailing_distance = mark_price * self.settings.risk.trailing_stop_distance_pct
                
                logger.info(f"Activating trailing stop for {symbol}: distance={trailing_distance:.6f} (PnL: {pnl_pct*100:.2f}%)")
                resp = await asyncio.to_thread(
                    self.bybit.set_trading_stop,
                    symbol=symbol,
                    trailing_stop=trailing_distance
                )
                
                if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                    await self.notifier.medium(
                        f"📊 ТРЕЙЛИНГ СТОП АКТИВИРОВАН\n{symbol} | distance={trailing_distance:.2f}\nТекущий PnL: +{pnl_pct*100:.2f}%"
                    )
        
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")
    
    async def check_partial_close(self, symbol: str, position_info: dict):
        """Проверяет и выполняет частичное закрытие позиции"""
        try:
            if not self.settings.risk.enable_partial_close:
                return
            
            if not position_info or not isinstance(position_info, dict):
                return
            
            if not position_info.get("size"):
                return
            
            size = float(position_info.get("size", 0))
            if size == 0:
                return
            
            side = position_info.get("side")
            entry_price = float(position_info.get("avgPrice", 0))
            mark_price = float(position_info.get("markPrice", entry_price))
            leverage_str = position_info.get("leverage", "10")
            try:
                leverage = int(float(leverage_str))
            except (ValueError, TypeError):
                leverage = self.settings.get_leverage_for_symbol(symbol) or DEFAULT_LEVERAGE
            
            take_profit = position_info.get("takeProfit")
            if not take_profit:
                local_pos = self.state.get_open_position(symbol)
                if local_pos and local_pos.take_profit:
                    take_profit = local_pos.take_profit
            
            if not entry_price or not mark_price or not take_profit:
                return
            
            take_profit_price = float(take_profit)
            
            # Рассчитываем прогресс к TP
            if side == "Buy":
                distance_to_tp = take_profit_price - entry_price
                current_progress = mark_price - entry_price
            else:  # Sell
                distance_to_tp = entry_price - take_profit_price
                current_progress = entry_price - mark_price
            
            if distance_to_tp <= 0:
                return
            
            progress_pct = current_progress / distance_to_tp
            
            # Корректируем прогресс с учетом режима и плеча
            adjusted_progress = get_partial_close_progress_pct(self.settings, progress_pct, leverage)
            
            # Проверяем уровни частичного закрытия
            for level_progress, close_pct in self.settings.risk.partial_close_levels:
                if adjusted_progress >= level_progress:
                    # Проверяем, не закрывали ли мы уже на этом уровне
                    # (это можно отслеживать через метаданные в state)
                    
                    # Рассчитываем количество для закрытия
                    close_qty = size * close_pct
                    
                    # Округляем
                    qty_step = self.bybit.get_qty_step(symbol)
                    close_qty = round(close_qty / qty_step) * qty_step
                    
                    if close_qty > 0:
                        mode = getattr(self.settings.risk, 'partial_close_mode', 'price')
                        if mode == "margin":
                            logger.info(f"Partial close {symbol}: {close_pct*100}% at {adjusted_progress*100:.1f}% margin-progress (raw: {progress_pct*100:.1f}%, leverage: {leverage}x)")
                        else:
                            logger.info(f"Partial close {symbol}: {close_pct*100}% at {progress_pct*100:.1f}% to TP")
                        
                        # Закрываем частично (reduce_only ордер)
                        close_side = "Sell" if side == "Buy" else "Buy"
                        resp = await asyncio.to_thread(
                            self.bybit.place_order,
                            symbol=symbol,
                            side=close_side,
                            qty=close_qty,
                            order_type="Market",
                            reduce_only=True
                        )
                        
                        if resp and isinstance(resp, dict) and resp.get("retCode") == 0:
                            # Обновляем количество в локальной позиции, чтобы корректно считать PnL при полном закрытии
                            try:
                                new_qty = None
                                with self.state.lock:
                                    local_pos = None
                                    for trade in reversed(self.state.trades):
                                        if trade.symbol == symbol and trade.status == "open":
                                            local_pos = trade
                                            break
                                    if local_pos:
                                        new_qty = local_pos.qty - close_qty
                                        if new_qty < 0:
                                            new_qty = 0
                                        local_pos.qty = new_qty
                                if new_qty is not None:
                                    self.state.save()
                                    logger.info(f"Updated local position size for {symbol} after partial close: {new_qty}")
                            except Exception as e_update:
                                logger.error(f"Error updating local position size: {e_update}")

                            await self.notifier.high(
                                f"💰 ЧАСТИЧНОЕ ЗАКРЫТИЕ\n{symbol} | {close_pct*100}%\nПрогресс к TP: {progress_pct*100:.1f}%"
                            )
                        
                        break  # Закрываем только на одном уровне за раз
        
        except Exception as e:
            logger.error(f"Error checking partial close for {symbol}: {e}")
    
    async def handle_position_closed(self, symbol: str, local_pos: TradeRecord):
        """Обрабатывает закрытие позиции, которая была открыта локально, но закрылась на бирже"""
        try:
            logger.info(f"Position {symbol} closed on exchange, updating state...")
            
            # Пытаемся получить информацию о закрытии из истории исполнений
            # Увеличиваем временной диапазон до 1 часа, чтобы найти закрытие
            import time
            from datetime import datetime, timedelta
            
            end_time = int(time.time() * 1000)
            start_time = int((time.time() - 3600) * 1000)  # 1 час назад (было 5 минут)
            
            exit_price = None
            pnl_usd = 0.0
            pnl_pct = 0.0
            
            # Метод 1: Пытаемся получить из закрытых позиций (closed PnL) - самый точный источник
            # Делаем несколько попыток с небольшой задержкой, так как данные могут появиться не сразу
            # НЕ передаем endTime, чтобы избежать проблем с рассинхроном времени (ПК отстает от сервера)
            total_fee_usd = 0.0
            
            for attempt in range(3):
                try:
                    if attempt > 0:
                        await asyncio.sleep(2.0) # Увеличиваем паузу до 2 сек
                    
                    closed_pnl = await asyncio.to_thread(
                        self.bybit.get_closed_pnl,
                        symbol=symbol,
                        limit=50  # Берем последние 50 записей (Bybit отдает от новых к старым)
                    )
                    
                    if closed_pnl and isinstance(closed_pnl, dict) and closed_pnl.get("retCode") == 0:
                        result = closed_pnl.get("result")
                        if result and isinstance(result, dict):
                            pnl_list = result.get("list", [])
                            if pnl_list and len(pnl_list) > 0:
                                # Ищем ВСЕ закрытые позиции для этого символа, которые относятся к нашей сделке
                                # Сортируем по времени создания (createdTime), чтобы взять самые свежие (API и так это делает, но для надежности)
                                try:
                                    pnl_list.sort(key=lambda x: int(x.get("createdTime", 0)), reverse=True)
                                except:
                                    pass
                                    
                                found_pnl = False
                                accumulated_pnl = 0.0
                                accumulated_fee = 0.0
                                last_exit_price = 0.0
                                last_exit_time = 0
                                
                                # Преобразуем entry_time в timestamp ms для фильтрации
                                # Добавляем буфер 1 час (3600000 мс) на случай, если часы ПК спешат вперед относительно сервера
                                try:
                                    entry_dt = datetime.fromisoformat(local_pos.entry_time)
                                    entry_ts = int(entry_dt.timestamp() * 1000)
                                    entry_ts_with_buffer = entry_ts - 3600000 
                                except:
                                    entry_ts_with_buffer = 0
                                
                                close_side = "Sell" if local_pos.side == "Buy" else "Buy"
                                expected_sides = {local_pos.side, close_side}
                                
                                for pnl_item in pnl_list:
                                    if pnl_item and isinstance(pnl_item, dict):
                                        # Логируем все ключи для отладки
                                        logger.debug(f"PnL item keys: {list(pnl_item.keys())}")
                                        
                                        pnl_symbol = pnl_item.get("symbol", "")
                                        pnl_side = str(pnl_item.get("side", "")).capitalize()
                                        created_time = int(pnl_item.get("createdTime", 0))
                                        
                                        # Проверяем, что это наша позиция (тот же символ, сторона и создана ПОСЛЕ открытия с учетом буфера)
                                        if pnl_symbol == symbol and pnl_side in expected_sides and created_time > entry_ts_with_buffer:
                                            # Накапливаем PnL и комиссии
                                            closed_pnl_val = float(pnl_item.get("closedPnl", 0))
                                            accumulated_pnl += closed_pnl_val
                                            
                                            # Извлекаем точные комиссии и funding из API
                                            open_fee = float(pnl_item.get("openFee", 0))
                                            close_fee = float(pnl_item.get("closeFee", 0))
                                            # Пробуем несколько возможных ключей для funding fee
                                            funding_fee = 0.0
                                            for key in ["sumFundingFee", "fundingFee", "totalFundingFee"]:
                                                if key in pnl_item:
                                                    funding_fee = float(pnl_item.get(key, 0))
                                                    break
                                            total_fee_from_api = open_fee + close_fee + funding_fee
                                            
                                            # Вычисляем комиссию для этой части (альтернативный расчет)
                                            qty_val = float(pnl_item.get("qty", 0))
                                            entry_price_val = float(pnl_item.get("avgEntryPrice", 0))
                                            exit_price_val = float(pnl_item.get("avgExitPrice", 0))
                                            
                                            if exit_price_val > 0:
                                                if last_exit_time < created_time:
                                                    last_exit_price = exit_price_val
                                                    last_exit_time = created_time
                                                
                                                # Gross PnL (Position P&L)
                                                if local_pos.side == "Buy":
                                                    gross_pnl = (exit_price_val - entry_price_val) * qty_val
                                                else:
                                                    gross_pnl = (entry_price_val - exit_price_val) * qty_val
                                                
                                                # Fee = Gross - Net (должно совпадать с total_fee_from_api)
                                                calculated_fee = gross_pnl - closed_pnl_val
                                                accumulated_fee += calculated_fee
                                                
                                                # Логируем детали для отладки
                                                logger.debug(f"PnL item: closedPnl={closed_pnl_val:.4f}, openFee={open_fee:.4f}, closeFee={close_fee:.4f}, funding={funding_fee:.4f}, total_fee_api={total_fee_from_api:.4f}, calculated_fee={calculated_fee:.4f}")
                                                
                                                found_pnl = True
                                        elif pnl_symbol == symbol and created_time < entry_ts_with_buffer:
                                            # Если дошли до записей старее буфера открытия, останавливаемся (так как список отсортирован)
                                            # Это предотвращает захват старых сделок
                                            break
                                
                                if found_pnl:
                                    exit_price = last_exit_price
                                    pnl_usd = accumulated_pnl
                                    total_fee_usd = accumulated_fee
                                    
                                    # Рассчитываем процент PnL от начальной маржи
                                    margin = (local_pos.entry_price * local_pos.qty) / local_pos.leverage
                                    if margin > 0:
                                        pnl_pct = (pnl_usd / margin) * 100
                                    
                                    # ВАЖНО: Мы уже получили чистый PnL (Net PnL) из поля closedPnl биржи.
                                    # В него уже включены все комиссии (open, close, funding).
                                    # accumulated_fee содержит сумму всех этих издержек.
                                    # Нам НЕ нужно вычитать total_fee_usd из pnl_usd повторно.
                                        
                                    logger.info(f"Found aggregated PnL data: exit_price={exit_price:.2f}, pnl_usd={pnl_usd:.6f}, pnl_pct={pnl_pct:.4f}%, total_fee={total_fee_usd:.4f}")
                                    break
                                
                                if exit_price is not None:
                                    break
                except Exception as e:
                    logger.warning(f"Error getting closed PnL for {symbol} (attempt {attempt+1}): {e}")
            
            # Метод 2: Если не нашли в closed PnL, пытаемся получить из истории исполнений
            try:
                executions = await asyncio.to_thread(
                    self.bybit.get_execution_list,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    limit=50  # Увеличиваем лимит
                )
                
                if executions and isinstance(executions, dict) and executions.get("retCode") == 0:
                    result = executions.get("result")
                    if result and isinstance(result, dict):
                        exec_list = result.get("list", [])
                        if exec_list and len(exec_list) > 0:
                            # Ищем закрывающий ордер (reduceOnly или противоположный side)
                            close_side = "Sell" if local_pos.side == "Buy" else "Buy"
                            for exec_item in exec_list:
                                if exec_item and isinstance(exec_item, dict):
                                    exec_side = exec_item.get("side", "")
                                    # Ищем исполнение противоположного направления или reduceOnly
                                    if exec_side == close_side or exec_item.get("reduceOnly", False):
                                        exit_price = float(exec_item.get("execPrice", 0))
                                        if exit_price > 0:
                                            logger.info(f"Found exit price from execution list: {exit_price}")
                                            break
            except Exception as e:
                logger.warning(f"Error getting execution list for {symbol}: {e}")
            
            # Метод 3: Если не нашли в closed PnL и execution list, пытаемся получить из текущей позиции
            if exit_price is None or exit_price == 0:
                try:
                    # Получаем информацию о текущей позиции (может быть закрыта недавно)
                    pos_info = await asyncio.to_thread(self.bybit.get_position_info, symbol=symbol)
                    if pos_info and isinstance(pos_info, dict) and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data and len(list_data) > 0:
                                position = list_data[0]
                                if position and isinstance(position, dict):
                                    # Если позиция закрыта (size == 0), используем markPrice
                                    size = float(position.get("size", 0))
                                    if size == 0:
                                        mark_price = float(position.get("markPrice", 0))
                                        if mark_price > 0:
                                            exit_price = mark_price
                                            logger.info(f"Using markPrice as exit price: {exit_price}")
                except Exception as e:
                    logger.warning(f"Error getting position info for closed position {symbol}: {e}")
            
            # Метод 4: Если все еще не нашли, используем текущую цену из свечей
            if exit_price is None or exit_price == 0:
                try:
                    df = await asyncio.to_thread(
                        self.bybit.get_kline_df,
                        symbol,
                        self.settings.timeframe,
                        1
                    )
                    if not df.empty:
                        exit_price = float(df['close'].iloc[-1])
                        logger.info(f"Using current price from candles as exit price: {exit_price}")
                except Exception as e:
                    logger.warning(f"Error getting current price for {symbol}: {e}")
            
            # Если все методы не сработали, используем entry_price (но это плохо)
            if exit_price is None or exit_price == 0:
                exit_price = local_pos.entry_price
                logger.warning(f"Could not determine exit price for {symbol}, using entry_price: {exit_price}")
            
            # Рассчитываем PnL
            # ВАЖНО: Мы уже получили pnl_usd и pnl_pct из агрегированных данных closedPnl (если они были найдены)
            # Если pnl_usd == 0 (данные не найдены), тогда считаем вручную
            
            logger.debug(f"PnL calculation start: pnl_usd={pnl_usd}, pnl_pct={pnl_pct}, exit_price={exit_price}")
            
            if pnl_usd == 0 and exit_price is not None:
                leverage = local_pos.leverage
                
                logger.debug(f"Manual PnL calculation for {symbol}: entry_price={local_pos.entry_price}, qty={local_pos.qty}, side={local_pos.side}, leverage={leverage}, exit_price={exit_price}")
                
                if local_pos.side == "Buy":
                    price_diff_pct = ((exit_price - local_pos.entry_price) / local_pos.entry_price)
                    pnl_pct = price_diff_pct * leverage * 100
                else:  # Sell
                    price_diff_pct = ((local_pos.entry_price - exit_price) / local_pos.entry_price)
                    pnl_pct = price_diff_pct * leverage * 100
                
                # PnL в USD = (процент PnL / 100) * маржа
                # Маржа = entry_price * qty / leverage
                margin = (local_pos.entry_price * local_pos.qty) / leverage
                pnl_usd = (pnl_pct / 100) * margin
                
                logger.debug(f"Manual PnL: price_diff_pct={price_diff_pct:.8f}, pnl_pct_raw={pnl_pct:.8f}%, margin={margin:.6f}, pnl_usd_before_fee={pnl_usd:.6f}")

                # Учитываем комиссию биржи (если считаем вручную)
                fee_usd = self._calculate_fees_usd(local_pos.entry_price, exit_price, local_pos.qty)
                if fee_usd > 0:
                    pnl_usd -= fee_usd
                    total_fee_usd = fee_usd # Записываем расчетную комиссию
                    if margin > 0:
                        pnl_pct = (pnl_usd / margin) * 100
                    logger.info(
                        f"Applied fees for {symbol}: fee_usd={fee_usd:.4f}, pnl_usd={pnl_usd:.6f}, pnl_pct={pnl_pct:.4f}%"
                    )
            
            logger.info(f"Calculated PnL for {symbol}: exit_price={exit_price:.2f}, pnl_pct={pnl_pct:.4f}%, pnl_usd={pnl_usd:.6f}")
            
            # Определяем причину закрытия
            exit_reason = "TP" if pnl_usd > 0 else "SL"
            # Можно добавить более детальную причину, если доступна информация о trailing stop и т.д.
            
            # Обновляем статус сделки (может установить кулдаун от убытков)
            # Передаем реальную комиссию (если она была рассчитана)
            self.state.update_trade_on_close(
                symbol,
                exit_price,
                pnl_usd,
                pnl_pct,
                exit_reason,
                commission=total_fee_usd,
                enable_loss_cooldown=bool(getattr(self.settings.risk, "enable_loss_cooldown", False)),
            )
            
            # Устанавливаем кулдаун до закрытия следующей свечи (15 минут)
            # Этот кулдаун не перезапишет более длительный кулдаун от убытков
            self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)

            if exit_reason == "TP" and self.settings.risk.tp_reentry_enabled:
                wait_candles = max(0, int(self.settings.risk.tp_reentry_wait_candles))
                if wait_candles > 0:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    next_close = self._next_candle_close_utc(self.settings.timeframe)
                    wait_until = next_close + pd.Timedelta(minutes=self._timeframe_minutes(self.settings.timeframe) * wait_candles)
                    self.state.set_tp_reentry_guard(
                        symbol=symbol,
                        side=local_pos.side,
                        exit_time_utc=now_utc.isoformat(),
                        exit_price=float(exit_price),
                        wait_until_utc=wait_until.isoformat(),
                        wait_candles=wait_candles,
                    )
                    logger.info(
                        f"[{symbol}] TP reentry guard enabled: side={local_pos.side}, wait_candles={wait_candles}, wait_until_utc={wait_until.isoformat()}"
                    )
            
            # Отправляем уведомление
            pnl_emoji = "✅" if pnl_usd > 0 else "❌"
            await self.notifier.high(
                f"{pnl_emoji} ПОЗИЦИЯ ЗАКРЫТА ({exit_reason})\n"
                f"{symbol} {local_pos.side}\n"
                f"Вход: ${local_pos.entry_price:.2f}\n"
                f"Выход: ${exit_price:.2f}\n"
                f"PnL: {pnl_usd:+.6f} USD ({pnl_pct:+.4f}%)"
            )
            
            logger.info(f"Position {symbol} closed: PnL={pnl_usd:.6f} USD ({pnl_pct:.4f}%)")
            
        except Exception as e:
            logger.error(f"Error handling closed position for {symbol}: {e}")
            # В случае ошибки пытаемся получить текущую цену и закрыть позицию
            try:
                # Пытаемся получить текущую цену из свечей
                df = await asyncio.to_thread(
                    self.bybit.get_kline_df,
                    symbol,
                    self.settings.timeframe,
                    1
                )
                if not df.empty:
                    exit_price = float(df['close'].iloc[-1])
                    # Рассчитываем PnL даже при ошибке
                    if local_pos.side == "Buy":
                        pnl_pct = ((exit_price - local_pos.entry_price) / local_pos.entry_price) * 100
                    else:
                        pnl_pct = ((local_pos.entry_price - exit_price) / local_pos.entry_price) * 100
                    margin = (local_pos.entry_price * local_pos.qty) / local_pos.leverage
                    pnl_usd = (pnl_pct / 100) * margin
                    fee_usd = self._calculate_fees_usd(local_pos.entry_price, exit_price, local_pos.qty)
                    if fee_usd > 0:
                        pnl_usd -= fee_usd
                        if margin > 0:
                            pnl_pct = (pnl_usd / margin) * 100
                    self.state.update_trade_on_close(
                        symbol,
                        exit_price,
                        pnl_usd,
                        pnl_pct,
                        "MANUAL_CLOSE",
                        commission=fee_usd,
                        enable_loss_cooldown=bool(getattr(self.settings.risk, "enable_loss_cooldown", False)),
                    )
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
                else:
                    # Если не удалось получить цену, используем entry_price с нулевым PnL
                    self.state.update_trade_on_close(
                        symbol,
                        local_pos.entry_price,
                        0.0,
                        0.0,
                        "ERROR_CLOSE",
                        enable_loss_cooldown=bool(getattr(self.settings.risk, "enable_loss_cooldown", False)),
                    )
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
            except Exception as e2:
                logger.error(f"Error in fallback close handling for {symbol}: {e2}")
                # Последняя попытка - закрываем с entry_price
                try:
                    self.state.update_trade_on_close(
                        symbol,
                        local_pos.entry_price,
                        0.0,
                        0.0,
                        "ERROR_CLOSE",
                        enable_loss_cooldown=bool(getattr(self.settings.risk, "enable_loss_cooldown", False)),
                    )
                    # Устанавливаем кулдаун до закрытия следующей свечи
                    self.state.set_cooldown_until_next_candle(symbol, self.settings.timeframe)
                except:
                    pass
    
    async def sync_positions_with_exchange(self):
        """Синхронизирует локальное состояние с позициями на бирже при старте"""
        logger.info("Syncing positions with exchange...")
        
        try:
            for symbol in self.state.active_symbols:
                try:
                    # Получаем позицию с биржи
                    pos_info = await asyncio.to_thread(
                        self.bybit.get_position_info,
                        symbol=symbol
                    )
                    
                    if pos_info and pos_info.get("retCode") == 0:
                        result = pos_info.get("result")
                        if result and isinstance(result, dict):
                            list_data = result.get("list", [])
                            if list_data and len(list_data) > 0:
                                position = list_data[0]
                                if position and isinstance(position, dict):
                                    size = float(position.get("size", 0))
                                    
                                    if size > 0:
                                        # Есть открытая позиция на бирже
                                        side = position.get("side")
                                        entry_price = float(position.get("avgPrice", 0))
                                        
                                        # Проверяем, есть ли она в локальном состоянии
                                        local_pos = self.state.get_open_position(symbol)
                                        
                                        if not local_pos:
                                            # Позиции нет в локальном состоянии, добавляем
                                            logger.info(f"Found open position on exchange for {symbol}, adding to state")
                                            
                                            trade = TradeRecord(
                                                symbol=symbol,
                                                side=side,
                                                entry_price=entry_price,
                                                qty=size,
                                                status="open",
                                                model_name=self.state.symbol_models.get(symbol, "")
                                            )
                                            self.state.add_trade(trade)
                                            
                                            await self.notifier.medium(
                                                f"🔄 СИНХРОНИЗАЦИЯ\nНайдена открытая позиция:\n{symbol} {side} | Размер: {size}"
                                            )
                                        else:
                                            # Позиция есть, обновляем данные если нужно
                                            if abs(local_pos.qty - size) > 0.0001 or abs(local_pos.entry_price - entry_price) > 0.01:
                                                logger.info(f"Updating position data for {symbol}")
                                                self.state.update_position(symbol, size, entry_price)
                                    else:
                                        # Позиции нет на бирже (size == 0), но может быть в локальном состоянии
                                        local_pos = self.state.get_open_position(symbol)
                                        if local_pos:
                                            # Закрываем локальную позицию
                                            logger.warning(f"Position {symbol} closed on exchange but open locally, closing in state")
                                            await self.handle_position_closed(symbol, local_pos)
                
                except Exception as e:
                    logger.error(f"Error syncing position for {symbol}: {e}")
            
            logger.info("Position sync completed")
        
        except Exception as e:
            logger.error(f"Error during position sync: {e}")

    async def _maintenance_loop(self):
        """Цикл периодического обслуживания: очистка логов, архивация и т.д."""
        logger.info("Starting Maintenance Loop...")
        # Первая очистка через 1 час после старта, затем раз в неделю
        await asyncio.sleep(3600)

        while True:
            try:
                now = datetime.now()
                # Очищаем логи раз в неделю (в воскресенье ночью)
                if now.weekday() == 6 and now.hour == 3:
                    logger.info("📅 Weekly maintenance: clearing logs...")
                    self._auto_clear_logs()
                    # Ждем час, чтобы не зациклилось в 3 часа ночи
                    await asyncio.sleep(3600)

                # Проверяем раз в час
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)

    def _auto_clear_logs(self):
        """Автоматическая очистка основных логов (удаление старых и очистка текущих)."""
        log_dir = Path("logs")
        if not log_dir.exists():
            return

        logs_to_clear = ["bot.log", "errors.log", "signals.log", "trades.log", "ai_entry_audit.jsonl"]

        for log_name in logs_to_clear:
            base_file = log_dir / log_name
            if not base_file.exists():
                continue

            try:
                # 1. Удаляем все ротированные файлы (log.1, log.2 и т.д.)
                for p in sorted(log_dir.glob(f"{log_name}.*")):
                    if p.is_file():
                        try:
                            p.unlink(missing_ok=True)
                            logger.info(f"Deleted rotated log file: {p.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {p.name}: {e}")

                # 2. Очищаем основной файл (truncate), сохраняя дескриптор открытым
                # Это самый безопасный способ для работающего процесса
                with open(base_file, "w", encoding="utf-8") as f:
                    f.write(f"--- Log auto-cleared and restarted at {datetime.now().isoformat()} ---\n")

                logger.info(f"Main log file cleared: {log_name}")
            except Exception as e:
                logger.error(f"Failed to perform auto-cleanup for {log_name}: {e}")
