"""
Скрипт бэктеста PPO стратегии на OOS данных (14 дней).
"""
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Optional

from bot.rl.data_preparation import prepare_ppo_data, split_data
from bot.rl.trading_env import TradingEnv
from bot.rl.ppo_agent import ActorCritic
from bot.rl.ppo_trainer import PPOTrainer
from bot.rl.risk_manager import RiskManager
from bot.rl.metrics import calculate_metrics, calculate_equity_metrics, generate_trade_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PRESETS = {
    "fast": {
        "min_bars_between_trades": 8,
        "min_hold_bars": 4,
        "reward_churn_penalty": 0.02,
        "reward_dd_penalty": 0.05,
        "reward_trade_penalty": 0.005,
        "min_adx": 8.0,
        "min_atr_pct": 0.05,
        "rr_default": 2.2,
        "rr_min": 2.0,
        "rr_max": 3.0,
        "max_leverage": 1.0,
        "commission_rate": 0.0006,
        "slippage_bps": 3.0,
        "reward_trend_bonus": 0.01,
        "reward_tp_progress": 0.02,
        "reward_one_sided_penalty": 0.02,
        "action_space": "hold",
    },
    "moderate": {
        "min_bars_between_trades": 4,
        "min_hold_bars": 6,
        "reward_churn_penalty": 0.008,
        "reward_dd_penalty": 0.1,
        "reward_trade_penalty": 0.0015,
        "min_adx": 10.0,
        "min_atr_pct": 0.06,
        "rr_default": 3.5,
        "rr_min": 3.0,
        "rr_max": 4.0,
        "max_leverage": 1.0,
        "commission_rate": 0.0006,
        "slippage_bps": 3.0,
        "reward_trend_bonus": 0.02,
        "reward_tp_progress": 0.5,
        "reward_one_sided_penalty": 0.12,
        "action_space": "hold",
    },
    "strict": {
        "min_bars_between_trades": 32,
        "min_hold_bars": 16,
        "reward_churn_penalty": 0.05,
        "reward_dd_penalty": 0.2,
        "reward_trade_penalty": 0.02,
        "min_adx": 18.0,
        "min_atr_pct": 0.15,
        "rr_default": 2.2,
        "rr_min": 2.0,
        "rr_max": 3.0,
        "max_leverage": 1.0,
        "commission_rate": 0.001,
        "slippage_bps": 5.0,
        "reward_trend_bonus": 0.03,
        "reward_tp_progress": 0.05,
        "reward_one_sided_penalty": 0.04,
        "action_space": "hold",
    },
}


def backtest_ppo(
    checkpoint_path: str,
    csv_path: str,
    symbol: str = "BTCUSDT",
    initial_capital: float = 10000.0,
    oos_days: int = 14,
    val_days: int = 7,
    train_days: Optional[int] = None,
    commission_rate: float = 0.001,
    slippage_bps: float = 5.0,
    stress_test: bool = True,
    min_bars_between_trades: int = 8,
    reward_churn_penalty: float = 0.05,
    reward_dd_penalty: float = 0.2,
    min_adx: float = 18.0,
    min_atr_pct: float = 0.15,
    rr_default: float = 2.2,
    rr_min: float = 2.0,
    rr_max: float = 3.0,
    max_leverage: float = 1.0,
    min_hold_bars: int = 2,
    reward_trade_penalty: float = 0.0,
    reward_trend_bonus: float = 0.0,
    reward_tp_progress: float = 0.0,
    reward_one_sided_penalty: float = 0.0,
    action_space: str = "hold",
):
    """
    Бэктест PPO стратегии на OOS данных.
    
    Args:
        checkpoint_path: Путь к чекпоинту модели
        csv_path: Путь к CSV с данными
        symbol: Символ
        initial_capital: Начальный капитал
        oos_days: Количество дней для OOS теста
        commission_rate: Комиссия
        slippage_bps: Слиппедж в bps
        stress_test: Выполнить стресс-тесты
    """
    # Подготовка данных (нужно для определения state_size)
    logger.info("Preparing data...")
    df = prepare_ppo_data(csv_path, symbol=symbol, enable_mtf=True)
    
    # Создаем временную среду для определения размеров
    temp_env = TradingEnv(
        df.iloc[:100],
        initial_capital=initial_capital,
        action_space=action_space,
    )
    state_size = temp_env.get_state_size()
    action_size = temp_env.get_action_size()
    
    logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    # Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = ActorCritic(state_size=state_size, action_size=action_size)
    trainer = PPOTrainer(agent=agent, device=device)
    
    try:
        trainer.load_checkpoint(checkpoint_path)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    
    # Разделение на сплиты
    df_train, df_val, df_oos = split_data(
        df,
        val_days=val_days,
        oos_days=oos_days,
        train_days=train_days,
    )
    
    logger.info(f"OOS data: {len(df_oos)} rows ({df_oos.index[0]} to {df_oos.index[-1]})")
    
    # Создаем risk manager
    risk_manager = RiskManager(default_rr=rr_default, min_rr=rr_min, max_rr=rr_max)
    
    def risk_manager_callback(side, entry_price, row, df_history):
        """Callback для risk manager."""
        side_str = "LONG" if side == "LONG" else "SHORT"
        sl, tp, rr, _ = risk_manager.calculate_tp_sl(
            side_str, entry_price, row, df_history
        )
        return sl, tp, rr
    
    # Базовый бэктест
    logger.info("Running base backtest...")
    results_base = run_backtest(
        agent,
        df_oos,
        initial_capital,
        commission_rate,
        slippage_bps,
        risk_manager_callback,
        min_bars_between_trades=min_bars_between_trades,
        min_hold_bars=min_hold_bars,
        reward_trade_penalty=reward_trade_penalty,
        reward_churn_penalty=reward_churn_penalty,
        reward_dd_penalty=reward_dd_penalty,
        min_adx=min_adx,
        min_atr_pct=min_atr_pct,
        max_leverage=max_leverage,
        reward_trend_bonus=reward_trend_bonus,
        reward_tp_progress=reward_tp_progress,
        reward_one_sided_penalty=reward_one_sided_penalty,
        action_space=action_space,
    )
    
    # Генерируем отчет
    report = generate_trade_report(
        results_base["trades"],
        results_base["equity_curve"],
        initial_capital
    )
    
    print("\n" + "=" * 60)
    print("BASE BACKTEST RESULTS")
    print("=" * 60)
    print(report)
    
    # Сохраняем результаты
    output_dir = Path("backtest_reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"ppo_backtest_{symbol}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")

    # Экспорт trades и equity curve в CSV
    trades_path = output_dir / f"ppo_trades_{symbol}_{timestamp}.csv"
    equity_path = output_dir / f"ppo_equity_{symbol}_{timestamp}.csv"

    if results_base["trades"]:
        pd.DataFrame(results_base["trades"]).to_csv(trades_path, index=False)
    pd.DataFrame({"equity": results_base["equity_curve"]}).to_csv(equity_path, index=False)

    logger.info(f"Trades saved to {trades_path}")
    logger.info(f"Equity curve saved to {equity_path}")
    
    # Стресс-тесты
    if stress_test:
        logger.info("Running stress tests...")
        
        # Тест 1: Комиссия x2
        logger.info("Stress test 1: Commission x2")
        results_stress1 = run_backtest(
            agent,
            df_oos,
            initial_capital,
            commission_rate * 2,
            slippage_bps,
            risk_manager_callback,
            min_bars_between_trades=min_bars_between_trades,
            min_hold_bars=min_hold_bars,
            reward_trade_penalty=reward_trade_penalty,
            reward_churn_penalty=reward_churn_penalty,
            reward_dd_penalty=reward_dd_penalty,
            min_adx=min_adx,
            min_atr_pct=min_atr_pct,
            max_leverage=max_leverage,
            reward_trend_bonus=reward_trend_bonus,
            reward_tp_progress=reward_tp_progress,
            reward_one_sided_penalty=reward_one_sided_penalty,
            action_space=action_space,
        )
        
        # Тест 2: Слиппедж x2
        logger.info("Stress test 2: Slippage x2")
        results_stress2 = run_backtest(
            agent,
            df_oos,
            initial_capital,
            commission_rate,
            slippage_bps * 2,
            risk_manager_callback,
            min_bars_between_trades=min_bars_between_trades,
            min_hold_bars=min_hold_bars,
            reward_trade_penalty=reward_trade_penalty,
            reward_churn_penalty=reward_churn_penalty,
            reward_dd_penalty=reward_dd_penalty,
            min_adx=min_adx,
            min_atr_pct=min_atr_pct,
            max_leverage=max_leverage,
            reward_trend_bonus=reward_trend_bonus,
            reward_tp_progress=reward_tp_progress,
            reward_one_sided_penalty=reward_one_sided_penalty,
            action_space=action_space,
        )
        
        # Тест 3: Комиссия + Слиппедж x2
        logger.info("Stress test 3: Commission + Slippage x2")
        results_stress3 = run_backtest(
            agent,
            df_oos,
            initial_capital,
            commission_rate * 2,
            slippage_bps * 2,
            risk_manager_callback,
            min_bars_between_trades=min_bars_between_trades,
            min_hold_bars=min_hold_bars,
            reward_trade_penalty=reward_trade_penalty,
            reward_churn_penalty=reward_churn_penalty,
            reward_dd_penalty=reward_dd_penalty,
            min_adx=min_adx,
            min_atr_pct=min_atr_pct,
            max_leverage=max_leverage,
            reward_trend_bonus=reward_trend_bonus,
            reward_tp_progress=reward_tp_progress,
            reward_one_sided_penalty=reward_one_sided_penalty,
            action_space=action_space,
        )
        
        # Сравнение результатов
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        
        base_metrics = calculate_metrics(results_base["trades"], initial_capital)
        stress1_metrics = calculate_metrics(results_stress1["trades"], initial_capital)
        stress2_metrics = calculate_metrics(results_stress2["trades"], initial_capital)
        stress3_metrics = calculate_metrics(results_stress3["trades"], initial_capital)
        
        print(f"\nBase (Commission {commission_rate:.3f}, Slippage {slippage_bps}bps):")
        print(f"  Net PnL: ${base_metrics['net_pnl']:.2f} ({base_metrics['return_pct']:.2f}%)")
        print(f"  Win Rate: {base_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {base_metrics['profit_factor']:.2f}")
        
        print(f"\nStress 1 (Commission {commission_rate*2:.3f}, Slippage {slippage_bps}bps):")
        print(f"  Net PnL: ${stress1_metrics['net_pnl']:.2f} ({stress1_metrics['return_pct']:.2f}%)")
        print(f"  Win Rate: {stress1_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {stress1_metrics['profit_factor']:.2f}")
        
        print(f"\nStress 2 (Commission {commission_rate:.3f}, Slippage {slippage_bps*2}bps):")
        print(f"  Net PnL: ${stress2_metrics['net_pnl']:.2f} ({stress2_metrics['return_pct']:.2f}%)")
        print(f"  Win Rate: {stress2_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {stress2_metrics['profit_factor']:.2f}")
        
        print(f"\nStress 3 (Commission {commission_rate*2:.3f}, Slippage {slippage_bps*2}bps):")
        print(f"  Net PnL: ${stress3_metrics['net_pnl']:.2f} ({stress3_metrics['return_pct']:.2f}%)")
        print(f"  Win Rate: {stress3_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {stress3_metrics['profit_factor']:.2f}")
        
        # Сохраняем стресс-тесты
        stress_report_path = output_dir / f"ppo_stress_test_{symbol}_{timestamp}.txt"
        with open(stress_report_path, "w") as f:
            f.write("STRESS TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Base: Net PnL=${base_metrics['net_pnl']:.2f}, Return={base_metrics['return_pct']:.2f}%\n")
            f.write(f"Stress 1 (Comm x2): Net PnL=${stress1_metrics['net_pnl']:.2f}, Return={stress1_metrics['return_pct']:.2f}%\n")
            f.write(f"Stress 2 (Slippage x2): Net PnL=${stress2_metrics['net_pnl']:.2f}, Return={stress2_metrics['return_pct']:.2f}%\n")
            f.write(f"Stress 3 (Both x2): Net PnL=${stress3_metrics['net_pnl']:.2f}, Return={stress3_metrics['return_pct']:.2f}%\n")
        
        logger.info(f"Stress test report saved to {stress_report_path}")


def run_backtest(
    agent: ActorCritic,
    df: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    slippage_bps: float,
    risk_manager_callback,
    min_bars_between_trades: int = 8,
    min_hold_bars: int = 2,
    reward_trade_penalty: float = 0.0,
    reward_churn_penalty: float = 0.05,
    reward_dd_penalty: float = 0.2,
    min_adx: float = 18.0,
    min_atr_pct: float = 0.15,
    max_leverage: float = 1.0,
    reward_trend_bonus: float = 0.0,
    reward_tp_progress: float = 0.0,
    reward_one_sided_penalty: float = 0.0,
    action_space: str = "hold",
) -> dict:
    """
    Запускает бэктест на данных.
    
    Returns:
        Словарь с результатами: trades, equity_curve
    """
    env = TradingEnv(
        df,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_bps=slippage_bps,
        max_leverage=max_leverage,
        reward_churn_penalty=reward_churn_penalty,
        reward_dd_penalty=reward_dd_penalty,
        min_bars_between_trades=min_bars_between_trades,
        min_hold_bars=min_hold_bars,
        reward_trade_penalty=reward_trade_penalty,
        min_adx=min_adx,
        min_atr_pct=min_atr_pct,
        reward_trend_bonus=reward_trend_bonus,
        reward_tp_progress=reward_tp_progress,
        reward_one_sided_penalty=reward_one_sided_penalty,
        action_space=action_space,
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

    # Закрываем открытую позицию в конце теста, чтобы учесть нереализованный PnL
    if env.position_side is not None:
        last_row = df.iloc[min(env.current_step, len(df) - 1)]
        env.close_open_position(last_row, reason="end_of_test")
        equity_curve.append(env.equity)

    return {
        "trades": env.trades,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest PPO trading strategy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--oos-days", type=int, default=14, help="OOS days")
    parser.add_argument("--val-days", type=int, default=7, help="Validation days")
    parser.add_argument("--train-days", type=int, default=730, help="Train window in days")
    parser.add_argument("--no-stress-test", action="store_true", help="Skip stress tests")
    parser.add_argument("--preset", type=str, choices=["fast", "moderate", "strict"], help="Parameter preset")
    parser.add_argument("--min-bars-between-trades", type=int, default=8, help="Min bars between trades")
    parser.add_argument("--reward-churn-penalty", type=float, default=0.05, help="Churn penalty")
    parser.add_argument("--reward-dd-penalty", type=float, default=0.2, help="Drawdown penalty")
    parser.add_argument("--min-adx", type=float, default=18.0, help="Min ADX filter for entries")
    parser.add_argument("--min-atr-pct", type=float, default=0.15, help="Min ATR%% filter for entries")
    parser.add_argument("--rr-default", type=float, default=2.2, help="Default RR")
    parser.add_argument("--rr-min", type=float, default=2.0, help="Min RR")
    parser.add_argument("--rr-max", type=float, default=3.0, help="Max RR")
    parser.add_argument("--max-leverage", type=float, default=1.0, help="Max leverage cap")
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Min hold bars")
    parser.add_argument("--reward-trade-penalty", type=float, default=0.0, help="Fixed trade penalty (capital fraction)")
    parser.add_argument("--reward-trend-bonus", type=float, default=0.0, help="Trend bonus multiplier")
    parser.add_argument("--reward-tp-progress", type=float, default=0.0, help="TP progress reward multiplier")
    parser.add_argument("--reward-one-sided-penalty", type=float, default=0.0, help="Penalty for one-sided trading")
    parser.add_argument("--action-space", type=str, choices=["hold", "no_hold", "no_close"], default="hold", help="Action space")
    
    args = parser.parse_args()
    
    preset = PRESETS.get(args.preset) if args.preset else None
    
    # Дефолтные значения из parser (для сравнения)
    defaults = {
        "min_bars_between_trades": 8,
        "min_hold_bars": 2,
        "reward_trade_penalty": 0.0,
        "reward_churn_penalty": 0.05,
        "reward_dd_penalty": 0.2,
        "min_adx": 18.0,
        "min_atr_pct": 0.15,
        "rr_default": 2.2,
        "rr_min": 2.0,
        "rr_max": 3.0,
        "reward_trend_bonus": 0.0,
        "action_space": "hold",
    }
    
    # Функция для получения параметра: явно указанный > preset > default
    def get_param(arg_name, preset_key=None):
        arg_value = getattr(args, arg_name)
        # Если параметр указан явно (не равен default), используем его
        if arg_name in defaults and arg_value != defaults[arg_name]:
            return arg_value
        # Иначе используем preset, если он есть
        if preset and preset_key and preset_key in preset:
            return preset[preset_key]
        # Иначе используем значение из args (которое может быть default)
        return arg_value
    
    backtest_ppo(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        symbol=args.symbol,
        oos_days=args.oos_days,
        val_days=args.val_days,
        train_days=args.train_days,
        stress_test=not args.no_stress_test,
        min_bars_between_trades=get_param("min_bars_between_trades", "min_bars_between_trades"),
        min_hold_bars=get_param("min_hold_bars", "min_hold_bars"),
        reward_trade_penalty=get_param("reward_trade_penalty", "reward_trade_penalty"),
        reward_churn_penalty=get_param("reward_churn_penalty", "reward_churn_penalty"),
        reward_dd_penalty=get_param("reward_dd_penalty", "reward_dd_penalty"),
        min_adx=get_param("min_adx", "min_adx"),
        min_atr_pct=get_param("min_atr_pct", "min_atr_pct"),
        rr_default=get_param("rr_default", "rr_default"),
        rr_min=get_param("rr_min", "rr_min"),
        rr_max=get_param("rr_max", "rr_max"),
        max_leverage=preset["max_leverage"] if preset else args.max_leverage,
        commission_rate=preset["commission_rate"] if preset else 0.001,
        slippage_bps=preset["slippage_bps"] if preset else 5.0,
        reward_trend_bonus=get_param("reward_trend_bonus", "reward_trend_bonus"),
        reward_tp_progress=preset["reward_tp_progress"] if preset and "reward_tp_progress" in preset else args.reward_tp_progress,
        reward_one_sided_penalty=preset["reward_one_sided_penalty"] if preset and "reward_one_sided_penalty" in preset else args.reward_one_sided_penalty,
        action_space=get_param("action_space", "action_space"),
    )
