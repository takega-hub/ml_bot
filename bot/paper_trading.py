"""
Paper Trading Module for online testing of experimental models.
Provides virtual trading simulation using real-time candle data without real orders.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd

from bot.strategy import Action, Signal, Bias
from bot.ml.strategy_ml import MLStrategy
from bot.ml.mtf_strategy import MultiTimeframeMLStrategy

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a virtual trade in paper trading."""
    id: str
    experiment_id: str
    symbol: str
    action: Action
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing: Optional[dict] = None
    status: str = "open"  # "open", "closed", "cancelled"
    reason: str = ""
    indicators_info: Optional[dict] = None


@dataclass
class PaperMetrics:
    """Metrics for paper trading session."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration_hours: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_confidence: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_mae_ratio: float = 0.0
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    signals_with_tp_sl_pct: float = 0.0
    avg_position_size_usd: float = 100.0
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


class PaperBroker:
    """Virtual broker for paper trading."""
    
    def __init__(self, initial_balance: float = 10000.0, commission: float = 0.0006, slippage_bps: float = 0.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.slippage_bps = slippage_bps
        self.position: Optional[PaperTrade] = None
        self.trades: List[PaperTrade] = []
        
    def calculate_quantity(self, entry_price: float, base_order_usd: float = 100.0) -> float:
        """Calculate position quantity based on base order size."""
        return base_order_usd / entry_price
    
    def apply_slippage(self, price: float, is_entry: bool = True) -> float:
        """Apply slippage to price."""
        if self.slippage_bps > 0:
            slippage_factor = self.slippage_bps / 10000
            if is_entry:
                return price * (1 + slippage_factor)
            else:
                return price * (1 - slippage_factor)
        return price
    
    def open_position(self, signal: Signal, current_price: float, candle_timestamp: datetime) -> Optional[PaperTrade]:
        """Open a virtual position based on signal."""
        if self.position is not None:
            logger.warning("Position already open, cannot open new position")
            return None
            
        # Apply slippage to entry price
        entry_price = self.apply_slippage(current_price, is_entry=True)
        
        # Calculate quantity
        quantity = self.calculate_quantity(entry_price)
        
        # Calculate commission
        commission_cost = entry_price * quantity * self.commission
        
        # Create trade
        trade_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trade = PaperTrade(
            id=trade_id,
            experiment_id="",  # Will be set by manager
            symbol="",  # Will be set by manager
            action=signal.action,
            entry_price=entry_price,
            entry_time=candle_timestamp,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing=signal.trailing,
            reason=signal.reason,
            indicators_info=signal.indicators_info
        )
        
        # Update balance (deduct commission)
        self.balance -= commission_cost
        
        self.position = trade
        self.trades.append(trade)
        
        logger.info(f"Opened {signal.action.value} position at {entry_price:.2f}, quantity: {quantity:.4f}")
        return trade
    
    def check_exit(self, current_price: float, high: float, low: float, candle_timestamp: datetime) -> Optional[PaperTrade]:
        """Check if position should be closed based on TP/SL/trailing."""
        if self.position is None:
            return None
            
        trade = self.position
        exit_price = None
        exit_reason = ""
        
        # Check stop loss
        if trade.stop_loss is not None:
            if trade.action == Action.LONG and low <= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif trade.action == Action.SHORT and high >= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
        
        # Check take profit
        if exit_price is None and trade.take_profit is not None:
            if trade.action == Action.LONG and high >= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"
            elif trade.action == Action.SHORT and low <= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"
        
        # Check trailing stop
        if exit_price is None and trade.trailing is not None:
            trailing_stop = trade.trailing.get('stop_loss')
            if trailing_stop is not None:
                if trade.action == Action.LONG and low <= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "trailing_stop"
                elif trade.action == Action.SHORT and high >= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "trailing_stop"
        
        # If no TP/SL hit, check if we should close at current price (for simulation)
        # In real paper trading, we might want to hold until TP/SL
        # For now, we'll only close on TP/SL
        
        if exit_price is not None:
            # Apply slippage to exit price
            exit_price = self.apply_slippage(exit_price, is_entry=False)
            
            # Calculate PnL
            if trade.action == Action.LONG:
                pnl = (exit_price - trade.entry_price) * trade.quantity
            else:  # SHORT
                pnl = (trade.entry_price - exit_price) * trade.quantity
            
            # Deduct commission
            commission_cost = exit_price * trade.quantity * self.commission
            pnl -= commission_cost
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = candle_timestamp
            trade.pnl = pnl
            trade.pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
            trade.status = "closed"
            trade.reason = exit_reason
            
            # Update balance
            self.balance += pnl
            
            # Clear position
            self.position = None
            
            logger.info(f"Closed position at {exit_price:.2f}, PnL: {pnl:.2f}, Reason: {exit_reason}")
            return trade
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from trades."""
        if not self.trades:
            return {}
            
        closed_trades = [t for t in self.trades if t.status == "closed"]
        if not closed_trades:
            return {}
            
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_pnl_pct = sum(t.pnl_pct for t in closed_trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t.pnl for t in closed_trades if t.pnl > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.pnl for t in closed_trades if t.pnl < 0) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(sum(t.pnl for t in closed_trades if t.pnl > 0) / sum(t.pnl for t in closed_trades if t.pnl < 0)) if losing_trades > 0 else float('inf')
        
        # Calculate drawdown and equity curve
        equity = self.initial_balance
        max_equity = equity
        max_drawdown = 0
        equity_curve = [equity]
        timestamps = [datetime.now()]  # Start with current time
        
        for trade in closed_trades:
            equity += trade.pnl
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            max_drawdown = max(max_drawdown, drawdown)
            equity_curve.append(equity)
            timestamps.append(trade.exit_time or datetime.now())
        
        max_drawdown_pct = (max_drawdown / max_equity) * 100 if max_equity > 0 else 0
        
        # Calculate trade durations
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in closed_trades if t.exit_time]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Best/worst trades
        best_trade = max(closed_trades, key=lambda t: t.pnl) if closed_trades else None
        worst_trade = min(closed_trades, key=lambda t: t.pnl) if closed_trades else None
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        streak_type = None
        
        for trade in closed_trades:
            if trade.pnl > 0:
                if streak_type == "win":
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = "win"
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                if streak_type == "loss":
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = "loss"
                consecutive_losses = max(consecutive_losses, current_streak)
        
        # Average confidence
        avg_confidence = sum(t.indicators_info.get('confidence', 0) for t in closed_trades if t.indicators_info) / len(closed_trades) if closed_trades else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "avg_trade_duration_hours": avg_duration,
            "best_trade_pnl": best_trade.pnl if best_trade else 0,
            "worst_trade_pnl": worst_trade.pnl if worst_trade else 0,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "largest_win": best_trade.pnl if best_trade else 0,
            "largest_loss": worst_trade.pnl if worst_trade else 0,
            "avg_confidence": avg_confidence,
            "total_signals": len(closed_trades),
            "long_signals": sum(1 for t in closed_trades if t.action == Action.LONG),
            "short_signals": sum(1 for t in closed_trades if t.action == Action.SHORT),
            "signals_with_tp_sl_pct": 100.0,  # All trades have TP/SL in this implementation
            "avg_position_size_usd": 100.0,  # Default base order size
            "equity_curve": equity_curve,
            "timestamps": [ts.isoformat() for ts in timestamps]
        }


class PaperSession:
    """Paper trading session for a specific experiment."""
    
    def __init__(self, experiment_id: str, symbol: str, strategy, broker: PaperBroker):
        self.experiment_id = experiment_id
        self.symbol = symbol
        self.strategy = strategy
        self.broker = broker
        self.is_active = False
        self.start_time = None
        self.end_time = None
        self.metrics = PaperMetrics()
        
    def start(self):
        """Start the paper trading session."""
        self.is_active = True
        self.start_time = datetime.now()
        logger.info(f"Started paper trading session for {self.experiment_id} on {self.symbol}")
        
    def stop(self):
        """Stop the paper trading session."""
        self.is_active = False
        self.end_time = datetime.now()
        logger.info(f"Stopped paper trading session for {self.experiment_id} on {self.symbol}")
        
    def process_bar(self, row: pd.Series, df: pd.DataFrame, current_price: float, high: float, low: float, candle_timestamp: datetime):
        """Process a new bar and generate signals."""
        if not self.is_active:
            return
            
        # Check for exit first
        exit_trade = self.broker.check_exit(current_price, high, low, candle_timestamp)
        if exit_trade:
            exit_trade.experiment_id = self.experiment_id
            exit_trade.symbol = self.symbol
            self._update_metrics()
            
        # Generate signal if no position
        if self.broker.position is None:
            try:
                # Generate signal using the strategy
                if hasattr(self.strategy, 'predict_combined'):
                    # MTF strategy
                    signal = self.strategy.generate_signal(
                        row=row,
                        df_15m=df,
                        df_1h=None,  # Will be handled by strategy
                        has_position=None,
                        current_price=current_price,
                        leverage=1.0,  # Paper trading uses 1x leverage
                        target_profit_pct_margin=0.01,
                        max_loss_pct_margin=0.01,
                        stop_loss_pct=0.02,
                        take_profit_pct=0.04,
                    )
                else:
                    # Single timeframe strategy
                    signal = self.strategy.generate_signal(
                        row=row,
                        df=df,
                        has_position=None,
                        current_price=current_price,
                        leverage=1.0,
                        target_profit_pct_margin=0.01,
                        max_loss_pct_margin=0.01,
                        stop_loss_pct=0.02,
                        take_profit_pct=0.04,
                    )
                    
                if signal and signal.action != Action.HOLD:
                    # Open position
                    trade = self.broker.open_position(signal, current_price, candle_timestamp)
                    if trade:
                        trade.experiment_id = self.experiment_id
                        trade.symbol = self.symbol
                        self._update_metrics()
                        
            except Exception as e:
                logger.error(f"Error generating signal for {self.experiment_id}: {e}")
                
    def _update_metrics(self):
        """Update metrics from broker trades."""
        metrics_dict = self.broker.get_metrics()
        if metrics_dict:
            for key, value in metrics_dict.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
                    
    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "is_active": self.is_active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_position": {
                "action": self.broker.position.action.value if self.broker.position else None,
                "entry_price": self.broker.position.entry_price if self.broker.position else None,
                "entry_time": self.broker.position.entry_time.isoformat() if self.broker.position else None,
            } if self.broker.position else None,
            "balance": self.broker.balance,
            "total_trades": len(self.broker.trades),
            "open_trades": len([t for t in self.broker.trades if t.status == "open"]),
            "closed_trades": len([t for t in self.broker.trades if t.status == "closed"]),
        }


class PaperTradingManager:
    """Manager for paper trading sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, PaperSession] = {}
        self.experiments_file = Path("experiments.json")
        
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data from experiments.json."""
        if not self.experiments_file.exists():
            logger.error(f"Experiments file not found: {self.experiments_file}")
            return None
            
        try:
            with open(self.experiments_file, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
                
            if experiment_id not in experiments:
                logger.error(f"Experiment {experiment_id} not found in experiments.json")
                return None
                
            return experiments[experiment_id]
        except Exception as e:
            logger.error(f"Error loading experiment {experiment_id}: {e}")
            return None
            
    def create_strategy(self, experiment_data: Dict[str, Any]) -> Union[MLStrategy, MultiTimeframeMLStrategy, None]:
        """Create strategy from experiment data."""
        results = experiment_data.get('results', {})
        models = results.get('models', {})
        
        model_15m = models.get('15m')
        model_1h = models.get('1h')
        
        # Check if model files exist
        if model_15m and not Path(model_15m).exists():
            logger.warning(f"15m model file not found: {model_15m}")
            model_15m = None
            
        if model_1h and not Path(model_1h).exists():
            logger.warning(f"1h model file not found: {model_1h}")
            model_1h = None
        
        # If models don't exist, try to find alternative models
        symbol = experiment_data.get('symbol')
        if symbol:
            if not model_15m:
                # Try to find any 15m model for this symbol
                models_dir = Path("ml_models")
                candidates_15m = list(models_dir.glob(f"*{symbol}*15*15m*.pkl"))
                if candidates_15m:
                    model_15m = str(candidates_15m[0])
                    logger.info(f"Found alternative 15m model: {model_15m}")
            
            if not model_1h:
                # Try to find any 1h model for this symbol
                models_dir = Path("ml_models")
                candidates_1h = list(models_dir.glob(f"*{symbol}*60*1h*.pkl"))
                if candidates_1h:
                    model_1h = str(candidates_1h[0])
                    logger.info(f"Found alternative 1h model: {model_1h}")
        
        if model_1h and model_15m:
            # MTF strategy
            logger.info(f"Creating MTF strategy with 1h: {model_1h}, 15m: {model_15m}")
            return MultiTimeframeMLStrategy(
                model_1h_path=model_1h,
                model_15m_path=model_15m,
                confidence_threshold_1h=0.50,
                confidence_threshold_15m=0.35,
                require_alignment=True,
                alignment_mode="strict",
            )
        elif model_15m:
            # Single timeframe strategy
            logger.info(f"Creating single timeframe strategy with model: {model_15m}")
            return MLStrategy(
                model_path=model_15m,
                confidence_threshold=0.35,
                min_signal_strength="умеренное",
            )
        else:
            logger.error(f"No valid models found for experiment {experiment_data.get('id')}")
            return None
            
    def start_session(self, experiment_id: str) -> Optional[PaperSession]:
        """Start a paper trading session for an experiment."""
        if experiment_id in self.sessions:
            logger.warning(f"Session for {experiment_id} already exists")
            return self.sessions[experiment_id]
            
        # Load experiment data
        experiment_data = self.load_experiment(experiment_id)
        if not experiment_data:
            return None
            
        # Create strategy
        strategy = self.create_strategy(experiment_data)
        if not strategy:
            return None
            
        # Create broker and session
        broker = PaperBroker()
        session = PaperSession(
            experiment_id=experiment_id,
            symbol=experiment_data.get('symbol', 'UNKNOWN'),
            strategy=strategy,
            broker=broker
        )
        
        # Store session
        self.sessions[experiment_id] = session
        session.start()
        
        logger.info(f"Started paper trading session for {experiment_id}")
        return session
        
    def stop_session(self, experiment_id: str) -> bool:
        """Stop a paper trading session."""
        if experiment_id not in self.sessions:
            logger.warning(f"Session for {experiment_id} not found")
            return False
            
        session = self.sessions[experiment_id]
        session.stop()
        del self.sessions[experiment_id]
        
        logger.info(f"Stopped paper trading session for {experiment_id}")
        return True
        
    def get_session_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a paper trading session."""
        if experiment_id not in self.sessions:
            return None
            
        return self.sessions[experiment_id].get_status()
        
    def get_all_sessions_status(self) -> List[Dict[str, Any]]:
        """Get status of all active sessions."""
        return [session.get_status() for session in self.sessions.values()]
        
    def on_bar(self, symbol: str, row: pd.Series, df: pd.DataFrame, current_price: float, high: float, low: float, candle_timestamp: datetime):
        """Process a new bar for all active sessions on this symbol."""
        for session in self.sessions.values():
            if session.symbol == symbol and session.is_active:
                try:
                    session.process_bar(row, df, current_price, high, low, candle_timestamp)
                except Exception as e:
                    logger.error(f"Error processing bar for session {session.experiment_id}: {e}")
                    
    def get_trades(self, experiment_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trades for a specific experiment."""
        if experiment_id not in self.sessions:
            return []
            
        session = self.sessions[experiment_id]
        trades = session.broker.trades[-limit:] if limit else session.broker.trades
        
        return [{
            "id": trade.id,
            "experiment_id": trade.experiment_id,
            "symbol": trade.symbol,
            "action": trade.action.value,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_price": trade.exit_price,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "status": trade.status,
            "reason": trade.reason,
        } for trade in trades]
        
    def get_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific experiment."""
        if experiment_id not in self.sessions:
            return None
            
        session = self.sessions[experiment_id]
        return {
            "experiment_id": experiment_id,
            "symbol": session.symbol,
            "metrics": session.metrics.__dict__,
            "balance": session.broker.balance,
            "initial_balance": session.broker.initial_balance,
        }
    
    def get_realtime_chart_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time chart data for a specific experiment."""
        if experiment_id not in self.sessions:
            return None
            
        session = self.sessions[experiment_id]
        broker = session.broker
        
        # Get current equity curve
        equity_curve = []
        timestamps = []
        
        # Start with initial balance
        equity_curve.append(broker.initial_balance)
        timestamps.append(datetime.now().isoformat())
        
        # Add closed trades
        for trade in broker.trades:
            if trade.status == "closed" and trade.exit_time:
                equity_curve.append(equity_curve[-1] + trade.pnl)
                timestamps.append(trade.exit_time.isoformat())
        
        # Add current position value if open
        if broker.position:
            current_value = equity_curve[-1]  # Start with last closed equity
            # Add unrealized PnL based on current price
            # This would need current market price, which we don't have here
            # For now, we'll just use the last equity value
        
        return {
            "experiment_id": experiment_id,
            "symbol": session.symbol,
            "equity_curve": equity_curve,
            "timestamps": timestamps,
            "current_balance": broker.balance,
            "initial_balance": broker.initial_balance,
            "total_trades": len(broker.trades),
            "open_trades": len([t for t in broker.trades if t.status == "open"]),
            "closed_trades": len([t for t in broker.trades if t.status == "closed"]),
            "is_active": session.is_active,
        }