import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trade_history(data_dir):
    """Load all trade history files and concatenate them"""
    files = sorted(Path(data_dir).glob("trades_history_*.xlsx"))
    if not files:
        logger.warning("No trade history files found")
        return pd.DataFrame()
    
    frames = []
    for f in files:
        try:
            df = pd.read_excel(f)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    if not frames:
        return pd.DataFrame()
        
    return pd.concat(frames, ignore_index=True)

def calculate_kpis(df):
    """Calculate Key Performance Indicators"""
    if df.empty:
        return {}
        
    # Ensure required columns exist
    required_cols = ['PnL (USD)', 'PnL (%)', 'Вход', 'Выход', 'Размер', 'Символ', 'Сторона']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            return {}

    # Basic stats
    total_trades = len(df)
    
    # Filter out zero PnL trades (breakeven/errors) if needed, but usually we keep them
    # df = df[df['PnL (USD)'] != 0]
    
    winning_trades = df[df['PnL (USD)'] > 0]
    losing_trades = df[df['PnL (USD)'] <= 0]
    
    wins = len(winning_trades)
    losses = len(losing_trades)
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df['PnL (USD)'].sum()
    avg_pnl = df['PnL (USD)'].mean()
    
    gross_profit = winning_trades['PnL (USD)'].sum()
    gross_loss = abs(losing_trades['PnL (USD)'].sum())
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Drawdown calculation (on cumulative PnL)
    df = df.sort_values(by='Вход Время' if 'Вход Время' in df.columns else df.index)
    df['cumulative_pnl'] = df['PnL (USD)'].cumsum()
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    # Average Win/Loss
    avg_win = winning_trades['PnL (USD)'].mean() if wins > 0 else 0
    avg_loss = losing_trades['PnL (USD)'].mean() if losses > 0 else 0
    
    # Sharpe Ratio (simplified, assuming risk-free rate = 0)
    # Using std dev of returns
    returns_std = df['PnL (USD)'].std()
    sharpe_ratio = (avg_pnl / returns_std) if returns_std > 0 else 0
    # Annualize Sharpe (assuming ~5 trades/day -> 1825 trades/year)
    # This is a rough approximation
    sharpe_annualized = sharpe_ratio * np.sqrt(total_trades) 

    return {
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Total PnL": f"${total_pnl:.2f}",
        "Avg PnL": f"${avg_pnl:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown": f"${max_drawdown:.2f}",
        "Avg Win": f"${avg_win:.2f}",
        "Avg Loss": f"${avg_loss:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}"
    }

def main():
    project_root = Path(__file__).parent.parent
    trade_dir = project_root / "trade_history"
    
    logger.info("Loading trade history...")
    df = load_trade_history(trade_dir)
    
    if df.empty:
        logger.info("No trades found.")
        return

    logger.info(f"Loaded {len(df)} trades.")
    
    # Overall KPIs
    logger.info("-" * 30)
    logger.info("OVERALL PERFORMANCE")
    logger.info("-" * 30)
    kpis = calculate_kpis(df)
    for k, v in kpis.items():
        logger.info(f"{k}: {v}")
        
    # Per Symbol KPIs
    if 'Символ' in df.columns:
        logger.info("\n" + "-" * 30)
        logger.info("PERFORMANCE BY SYMBOL")
        logger.info("-" * 30)
        for symbol in df['Символ'].unique():
            symbol_df = df[df['Символ'] == symbol]
            logger.info(f"\nSymbol: {symbol}")
            symbol_kpis = calculate_kpis(symbol_df)
            for k, v in symbol_kpis.items():
                logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
