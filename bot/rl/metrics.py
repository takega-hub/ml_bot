"""
Метрики для оценки производительности PPO стратегии.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict


def calculate_metrics(trades: List[Dict], initial_capital: float) -> Dict[str, float]:
    """
    Вычисляет метрики производительности.
    
    Args:
        trades: Список сделок с полями: entry_step, exit_step, entry_price, exit_price, 
                side, size, pnl, commission, reason
        initial_capital: Начальный капитал
    
    Returns:
        Словарь с метриками
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_rr": 0.0,
            "total_pnl": 0.0,
            "total_commission": 0.0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }
    
    # Анализ сделок
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] < 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    total_gross_profit = sum(t["pnl"] for t in winning_trades)
    total_gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
    
    avg_win = total_gross_profit / len(winning_trades) if winning_trades else 0.0
    avg_loss = total_gross_loss / len(losing_trades) if losing_trades else 0.0
    
    total_pnl = sum(t["pnl"] for t in trades)
    total_commission = sum(t.get("commission", 0) for t in trades)
    net_pnl = total_pnl - total_commission
    
    return_pct = (net_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    # Risk-Reward ratio (средний)
    if winning_trades and losing_trades:
        avg_win_abs = avg_win
        avg_loss_abs = avg_loss
        avg_rr = avg_win_abs / avg_loss_abs if avg_loss_abs > 0 else 0.0
    else:
        avg_rr = 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_rr": avg_rr,
        "total_pnl": total_pnl,
        "total_commission": total_commission,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "gross_profit": total_gross_profit,
        "gross_loss": total_gross_loss,
    }


def calculate_equity_metrics(equity_curve: List[float], initial_capital: float) -> Dict[str, float]:
    """
    Вычисляет метрики equity curve.
    
    Args:
        equity_curve: Список значений equity по шагам
        initial_capital: Начальный капитал
    
    Returns:
        Словарь с метриками
    """
    if not equity_curve:
        return {
            "final_equity": initial_capital,
            "max_equity": initial_capital,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }
    
    equity_array = np.array(equity_curve)
    
    final_equity = equity_array[-1]
    max_equity = equity_array.max()
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (running_max - equity_array) / running_max
    max_drawdown = drawdown.max()
    max_drawdown_pct = max_drawdown * 100
    
    # Returns для Sharpe/Sortino
    returns = np.diff(equity_array) / equity_array[:-1]
    
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 96)  # 15m bars
        # Sortino: только отрицательные returns
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = np.mean(returns) / (downside_std + 1e-8) * np.sqrt(252 * 96)
        else:
            sortino_ratio = sharpe_ratio
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    
    return {
        "final_equity": float(final_equity),
        "max_equity": float(max_equity),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
    }


def generate_trade_report(trades: List[Dict], equity_curve: List[float], initial_capital: float) -> str:
    """
    Генерирует текстовый отчет о производительности.
    
    Args:
        trades: Список сделок
        equity_curve: Equity curve
        initial_capital: Начальный капитал
    
    Returns:
        Текстовый отчет
    """
    trade_metrics = calculate_metrics(trades, initial_capital)
    equity_metrics = calculate_equity_metrics(equity_curve, initial_capital)
    
    report = []
    report.append("=" * 60)
    report.append("PPO Strategy Performance Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("Trade Metrics:")
    report.append(f"  Total Trades: {trade_metrics['total_trades']}")
    report.append(f"  Win Rate: {trade_metrics['win_rate']:.2%}")
    report.append(f"  Profit Factor: {trade_metrics['profit_factor']:.2f}")
    report.append(f"  Avg Win: ${trade_metrics['avg_win']:.2f}")
    report.append(f"  Avg Loss: ${trade_metrics['avg_loss']:.2f}")
    report.append(f"  Avg RR: {trade_metrics['avg_rr']:.2f}")
    report.append("")
    
    report.append("PnL:")
    report.append(f"  Gross Profit: ${trade_metrics['gross_profit']:.2f}")
    report.append(f"  Gross Loss: ${trade_metrics['gross_loss']:.2f}")
    report.append(f"  Total Commission: ${trade_metrics['total_commission']:.2f}")
    report.append(f"  Net PnL: ${trade_metrics['net_pnl']:.2f}")
    report.append(f"  Return: {trade_metrics['return_pct']:.2f}%")
    report.append("")
    
    report.append("Equity Metrics:")
    report.append(f"  Final Equity: ${equity_metrics['final_equity']:.2f}")
    report.append(f"  Max Equity: ${equity_metrics['max_equity']:.2f}")
    report.append(f"  Max Drawdown: {equity_metrics['max_drawdown_pct']:.2f}%")
    report.append(f"  Sharpe Ratio: {equity_metrics['sharpe_ratio']:.2f}")
    report.append(f"  Sortino Ratio: {equity_metrics['sortino_ratio']:.2f}")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)
