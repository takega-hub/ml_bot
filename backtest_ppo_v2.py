"""
Entry-Only PPO Backtest Script.
"""
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from bot.rl.data_preparation import prepare_ppo_data, split_data
from bot.rl.trading_env_v2 import TradingEnvV2
from bot.rl.ppo_agent import ActorCritic
from bot.rl.ppo_trainer import PPOTrainer
from bot.rl.risk_manager import RiskManager
from bot.rl.metrics import calculate_metrics, generate_trade_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_ppo_v2(
    checkpoint_path: str,
    csv_path: str,
    symbol: str = "BTCUSDT",
    initial_capital: float = 10000.0,
    train_days: int = 730,
    val_days: int = 14,
    oos_days: int = 30,
    commission_rate: float = 0.0006,
    slippage_bps: float = 3.0,
    min_bars_between_trades: int = 8,
    min_adx: float = 0.0,
    min_atr_pct: float = 0.0,
    rr_default: float = 3.0,
    rr_min: float = 2.5,
    rr_max: float = 3.5,
    reward_scale: float = 100.0,
    stress_test: bool = True,
    use_trend_filter: bool = True,
    allowed_side: str = "both",
):
    """Бэктест Entry-Only PPO."""
    # Подготовка данных
    logger.info("Preparing data...")
    df = prepare_ppo_data(csv_path, symbol=symbol, enable_mtf=True)
    
    # Временная среда для размеров
    temp_env = TradingEnvV2(df.iloc[:100], initial_capital=initial_capital, allowed_side=allowed_side)
    state_size = temp_env.get_state_size()
    action_size = temp_env.get_action_size()
    
    logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    # Загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic(state_size=state_size, action_size=action_size)
    trainer = PPOTrainer(agent=agent, device=device)
    
    try:
        trainer.load_checkpoint(checkpoint_path)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    
    # Разделение данных
    df_train, df_val, df_oos = split_data(
        df, val_days=val_days, oos_days=oos_days, train_days=train_days
    )
    
    logger.info(f"OOS data: {len(df_oos)} rows ({df_oos.index[0]} to {df_oos.index[-1]})")
    
    # Risk manager
    risk_manager = RiskManager(default_rr=rr_default, min_rr=rr_min, max_rr=rr_max)
    
    def risk_manager_callback(side, entry_price, row, df_history):
        sl, tp, rr, _ = risk_manager.calculate_tp_sl(
            "LONG" if side == "LONG" else "SHORT",
            entry_price, row, df_history
        )
        return sl, tp, rr
    
    # Базовый бэктест
    logger.info("Running base backtest...")
    results_base = run_backtest(
        agent, df_oos, initial_capital, commission_rate, slippage_bps,
        risk_manager_callback, min_bars_between_trades, min_adx, min_atr_pct, reward_scale,
        use_trend_filter, allowed_side
    )
    
    # Отчёт
    report = generate_trade_report(
        results_base["trades"],
        results_base["equity_curve"],
        initial_capital
    )
    
    print("\n" + "=" * 60)
    print("ENTRY-ONLY PPO BACKTEST RESULTS")
    print("=" * 60)
    print(report)
    
    # Статистика по exit reasons
    trades = results_base["trades"]
    if trades:
        tp_count = sum(1 for t in trades if t["reason"] == "take_profit")
        sl_count = sum(1 for t in trades if t["reason"] == "stop_loss")
        other_count = len(trades) - tp_count - sl_count
        
        print(f"\nExit Reasons:")
        print(f"  Take Profit: {tp_count} ({tp_count/len(trades)*100:.1f}%)")
        print(f"  Stop Loss: {sl_count} ({sl_count/len(trades)*100:.1f}%)")
        if other_count > 0:
            print(f"  Other: {other_count} ({other_count/len(trades)*100:.1f}%)")
        
        # Long vs Short
        long_trades = [t for t in trades if t["side"] == "LONG"]
        short_trades = [t for t in trades if t["side"] == "SHORT"]
        print(f"\nDirection Balance:")
        print(f"  LONG: {len(long_trades)} trades, PnL: ${sum(t['net_pnl'] for t in long_trades):.2f}")
        print(f"  SHORT: {len(short_trades)} trades, PnL: ${sum(t['net_pnl'] for t in short_trades):.2f}")
    
    # Сохранение
    output_dir = Path("backtest_reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"ppo_v2_backtest_{symbol}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Экспорт trades и equity
    if trades:
        trades_path = output_dir / f"ppo_v2_trades_{symbol}_{timestamp}.csv"
        pd.DataFrame(trades).to_csv(trades_path, index=False)
        logger.info(f"Trades saved to {trades_path}")
    
    equity_path = output_dir / f"ppo_v2_equity_{symbol}_{timestamp}.csv"
    pd.DataFrame({"equity": results_base["equity_curve"]}).to_csv(equity_path, index=False)
    
    logger.info(f"Report saved to {report_path}")
    
    # Стресс-тесты
    if stress_test:
        logger.info("Running stress tests...")
        
        results_stress1 = run_backtest(
            agent, df_oos, initial_capital, commission_rate * 2, slippage_bps,
            risk_manager_callback, min_bars_between_trades, min_adx, min_atr_pct, reward_scale,
            use_trend_filter, allowed_side
        )
        
        results_stress2 = run_backtest(
            agent, df_oos, initial_capital, commission_rate, slippage_bps * 2,
            risk_manager_callback, min_bars_between_trades, min_adx, min_atr_pct, reward_scale,
            use_trend_filter, allowed_side
        )
        
        results_stress3 = run_backtest(
            agent, df_oos, initial_capital, commission_rate * 2, slippage_bps * 2,
            risk_manager_callback, min_bars_between_trades, min_adx, min_atr_pct, reward_scale,
            use_trend_filter, allowed_side
        )
        
        base_metrics = calculate_metrics(results_base["trades"], initial_capital)
        stress1_metrics = calculate_metrics(results_stress1["trades"], initial_capital)
        stress2_metrics = calculate_metrics(results_stress2["trades"], initial_capital)
        stress3_metrics = calculate_metrics(results_stress3["trades"], initial_capital)
        
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Base: Net PnL=${base_metrics['net_pnl']:.2f}, Return={base_metrics['return_pct']:.2f}%")
        print(f"Comm x2: Net PnL=${stress1_metrics['net_pnl']:.2f}, Return={stress1_metrics['return_pct']:.2f}%")
        print(f"Slip x2: Net PnL=${stress2_metrics['net_pnl']:.2f}, Return={stress2_metrics['return_pct']:.2f}%")
        print(f"Both x2: Net PnL=${stress3_metrics['net_pnl']:.2f}, Return={stress3_metrics['return_pct']:.2f}%")
        
        stress_path = output_dir / f"ppo_v2_stress_{symbol}_{timestamp}.txt"
        with open(stress_path, "w") as f:
            f.write("STRESS TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Base: Net PnL=${base_metrics['net_pnl']:.2f}, Return={base_metrics['return_pct']:.2f}%\n")
            f.write(f"Comm x2: Net PnL=${stress1_metrics['net_pnl']:.2f}, Return={stress1_metrics['return_pct']:.2f}%\n")
            f.write(f"Slip x2: Net PnL=${stress2_metrics['net_pnl']:.2f}, Return={stress2_metrics['return_pct']:.2f}%\n")
            f.write(f"Both x2: Net PnL=${stress3_metrics['net_pnl']:.2f}, Return={stress3_metrics['return_pct']:.2f}%\n")


def run_backtest(
    agent, df, initial_capital, commission_rate, slippage_bps,
    risk_manager_callback, min_bars_between_trades, min_adx, min_atr_pct, reward_scale,
    use_trend_filter=True, allowed_side="both"
) -> dict:
    """Запуск бэктеста."""
    env = TradingEnvV2(
        df,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_bps=slippage_bps,
        min_bars_between_trades=min_bars_between_trades,
        min_adx=min_adx,
        min_atr_pct=min_atr_pct,
        reward_scale=reward_scale,
        use_trend_filter=use_trend_filter,
        allowed_side=allowed_side,
    )
    
    device = next(agent.parameters()).device
    agent.eval()
    
    obs = env.reset(start_step=0)
    equity_curve = [initial_capital]
    
    with torch.no_grad():
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.item()
            
            obs, reward, done, info = env.step(action, risk_manager_callback)
            equity_curve.append(info.get("equity", initial_capital))
            
            if done:
                break
    
    # Закрываем открытую позицию
    if env.position_side is not None:
        last_row = df.iloc[-1]
        env.close_open_position(last_row, "end_of_test")
        equity_curve.append(env.equity)
    
    return {"trades": env.trades, "equity_curve": equity_curve}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Entry-Only PPO")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--no-stress-test", action="store_true")
    parser.add_argument("--min-bars-between-trades", type=int, default=8)
    parser.add_argument("--min-adx", type=float, default=0.0)
    parser.add_argument("--min-atr-pct", type=float, default=0.0)
    parser.add_argument("--rr-default", type=float, default=3.0)
    parser.add_argument("--rr-min", type=float, default=2.5)
    parser.add_argument("--rr-max", type=float, default=3.5)
    parser.add_argument("--commission-rate", type=float, default=0.0006)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable trend filter")
    parser.add_argument("--allowed-side", type=str, default="both",
                        choices=["long", "short", "both"],
                        help="Specialize model: 'long', 'short', or 'both'")
    
    args = parser.parse_args()
    
    backtest_ppo_v2(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        symbol=args.symbol,
        train_days=args.train_days,
        val_days=args.val_days,
        oos_days=args.oos_days,
        stress_test=not args.no_stress_test,
        min_bars_between_trades=args.min_bars_between_trades,
        min_adx=args.min_adx,
        min_atr_pct=args.min_atr_pct,
        rr_default=args.rr_default,
        rr_min=args.rr_min,
        rr_max=args.rr_max,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        use_trend_filter=not args.no_trend_filter,
        allowed_side=args.allowed_side,
    )
