"""
Скрипт обучения PPO торговой стратегии.
"""
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from bot.rl.data_preparation import prepare_ppo_data, split_data
from bot.rl.trading_env import TradingEnv, Action
from bot.rl.ppo_agent import ActorCritic
from bot.rl.ppo_trainer import PPOTrainer, RolloutBuffer
from bot.rl.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_ppo(
    csv_path: str,
    symbol: str = "BTCUSDT",
    output_dir: str = "ppo_models",
    initial_capital: float = 10000.0,
    rollout_steps: int = 2048,
    train_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    val_days: int = 7,
    oos_days: int = 14,
    train_days: int = 730,
    num_iterations: int = 100,
    save_freq: int = 10,
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
    commission_rate: float = 0.001,
    slippage_bps: float = 5.0,
    reward_trend_bonus: float = 0.0,
    reward_tp_progress: float = 0.0,
    reward_one_sided_penalty: float = 0.0,
    action_space: str = "hold",
    entropy_coef: float = 0.01,
):
    """
    Обучает PPO агента.
    
    Args:
        csv_path: Путь к CSV с данными
        symbol: Символ
        output_dir: Директория для сохранения моделей
        initial_capital: Начальный капитал
        rollout_steps: Количество шагов в rollout
        train_epochs: Количество эпох обучения на rollout
        batch_size: Размер батча
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_eps: PPO clip epsilon
        val_days: Дней для валидации
        oos_days: Дней для OOS теста
        num_iterations: Количество итераций обучения
        save_freq: Частота сохранения чекпоинтов
    """
    # Создаем директорию для моделей
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Подготовка данных
    logger.info("Preparing data...")
    df = prepare_ppo_data(csv_path, symbol=symbol, enable_mtf=True)
    
    # Разделение на сплиты
    df_train, df_val, df_oos = split_data(
        df,
        val_days=val_days,
        oos_days=oos_days,
        train_days=train_days,
    )
    
    logger.info(f"Train: {len(df_train)} rows, Val: {len(df_val)} rows, OOS: {len(df_oos)} rows")
    
    # Создаем среду для обучения
    env = TradingEnv(
        df_train,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_bps=slippage_bps,
        risk_per_trade_pct=1.0,
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
    
    # Создаем risk manager
    risk_manager = RiskManager(default_rr=rr_default, min_rr=rr_min, max_rr=rr_max)
    
    def risk_manager_callback(side, entry_price, row, df_history):
        """Callback для risk manager."""
        side_str = "LONG" if side == "LONG" else "SHORT"
        sl, tp, rr, _ = risk_manager.calculate_tp_sl(
            side_str, entry_price, row, df_history
        )
        return sl, tp, rr
    
    # Создаем агента
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    agent = ActorCritic(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=[256, 128, 64],
        use_layer_norm=True,
    )
    
    # Создаем trainer
    trainer = PPOTrainer(
        agent=agent,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
    )
    
    # Обучение
    logger.info("Starting training...")
    
    best_val_return = -np.inf
    buffer = RolloutBuffer(capacity=rollout_steps)
    
    for iteration in tqdm(range(num_iterations), desc="Training iterations"):
        # Rollout
        obs = env.reset()
        buffer.reset()
        
        episode_reward = 0.0
        episode_length = 0
        
        for step in tqdm(range(rollout_steps), desc="Rollout", leave=False):
            # Получаем действие от агента
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(obs_tensor)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            # Выполняем шаг в среде
            next_obs, reward, done, info = env.step(action, risk_manager_callback)
            
            # Сохраняем в буфер
            buffer.add(obs, action, log_prob, reward, value, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Вычисляем GAE после завершения rollout
        if len(buffer) > 0:
            # Получаем value для последнего состояния (0 если done)
            if done:
                next_value = 0.0
            else:
                last_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    next_value = agent.get_value(last_obs_tensor).item()
            
            buffer.compute_gae(next_value, gamma, gae_lambda)
            
            # Обучение
            metrics = trainer.train_step(buffer, epochs=train_epochs, batch_size=batch_size)
            
            # Логирование
            if (iteration + 1) % 1 == 0:
                logger.info(
                    f"Iteration {iteration+1}/{num_iterations}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Policy Loss={metrics.get('policy_loss', 0):.4f}, "
                    f"Value Loss={metrics.get('value_loss', 0):.4f}, "
                    f"Equity={info.get('equity', 0):.2f}"
                )
            
            # Валидация
            if (iteration + 1) % 5 == 0:
                val_return = validate_agent(
                    agent,
                    df_val,
                    initial_capital,
                    risk_manager_callback,
                    min_bars_between_trades=min_bars_between_trades,
                    min_hold_bars=min_hold_bars,
                    reward_trade_penalty=reward_trade_penalty,
                    reward_churn_penalty=reward_churn_penalty,
                    reward_dd_penalty=reward_dd_penalty,
                    min_adx=min_adx,
                    min_atr_pct=min_atr_pct,
                    max_leverage=max_leverage,
                    commission_rate=commission_rate,
                    slippage_bps=slippage_bps,
                    reward_trend_bonus=reward_trend_bonus,
                    reward_tp_progress=reward_tp_progress,
                    reward_one_sided_penalty=reward_one_sided_penalty,
                    action_space=action_space,
                )
                logger.info(f"Validation Return: {val_return:.2f}%")
                
                if val_return > best_val_return:
                    best_val_return = val_return
                    checkpoint_path = output_path / f"ppo_{symbol}_best.pth"
                    trainer.save_checkpoint(
                        str(checkpoint_path),
                        metadata={
                            "iteration": iteration + 1,
                            "val_return": val_return,
                            "symbol": symbol,
                        }
                    )
                    logger.info(f"Saved best model (val_return={val_return:.2f}%)")
            
            # Сохранение чекпоинта
            if (iteration + 1) % save_freq == 0:
                checkpoint_path = output_path / f"ppo_{symbol}_iter_{iteration+1}.pth"
                trainer.save_checkpoint(
                    str(checkpoint_path),
                    metadata={
                        "iteration": iteration + 1,
                        "symbol": symbol,
                    }
                )
    
    # Всегда сохраняем финальный чекпоинт
    final_checkpoint = output_path / f"ppo_{symbol}_final.pth"
    trainer.save_checkpoint(
        str(final_checkpoint),
        metadata={
            "iteration": num_iterations,
            "symbol": symbol,
            "val_return": best_val_return,
        }
    )
    logger.info(f"Final checkpoint saved to {final_checkpoint}")

    logger.info("Training completed!")
    logger.info(f"Best validation return: {best_val_return:.2f}%")


def validate_agent(
    agent,
    df_val,
    initial_capital,
    risk_manager_callback,
    min_bars_between_trades: int = 8,
    min_hold_bars: int = 2,
    reward_trade_penalty: float = 0.0,
    reward_churn_penalty: float = 0.05,
    reward_dd_penalty: float = 0.2,
    min_adx: float = 18.0,
    min_atr_pct: float = 0.15,
    max_leverage: float = 1.0,
    commission_rate: float = 0.001,
    slippage_bps: float = 5.0,
    reward_trend_bonus: float = 0.0,
    reward_tp_progress: float = 0.0,
    reward_one_sided_penalty: float = 0.0,
    action_space: str = "hold",
):
    """Валидация агента на validation set."""
    env = TradingEnv(
        df_val,
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
    
    obs = env.reset(start_step=0)
    total_reward = 0.0
    
    device = next(agent.parameters()).device
    agent.eval()
    with torch.no_grad():
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.item()
            
            obs, reward, done, info = env.step(action, risk_manager_callback)
            total_reward += reward
            
            if done:
                break
    
    final_equity = info.get("equity", initial_capital)
    return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    
    return return_pct


PRESETS = {
    "fast": {
        "min_bars_between_trades": 8,
        "min_hold_bars": 4,
        "reward_churn_penalty": 0.02,
        "reward_dd_penalty": 0.1,
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
        "reward_dd_penalty": 0.2,
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
        "reward_dd_penalty": 0.4,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO trading strategy")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with 15m data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--output", type=str, default="ppo_models", help="Output directory")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train-days", type=int, default=730, help="Train window in days")
    parser.add_argument("--val-days", type=int, default=7, help="Validation window in days")
    parser.add_argument("--oos-days", type=int, default=14, help="OOS window in days")
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
    parser.add_argument("--commission-rate", type=float, default=0.001, help="Commission rate per side")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--reward-trend-bonus", type=float, default=0.0, help="Trend bonus multiplier")
    parser.add_argument("--reward-tp-progress", type=float, default=0.0, help="TP progress reward multiplier")
    parser.add_argument("--reward-one-sided-penalty", type=float, default=0.0, help="Penalty for one-sided trading")
    parser.add_argument("--action-space", type=str, choices=["hold", "no_hold", "no_close"], default="hold", help="Action space")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    
    args = parser.parse_args()
    
    preset = PRESETS.get(args.preset) if args.preset else None
    train_ppo(
        csv_path=args.csv,
        symbol=args.symbol,
        output_dir=args.output,
        num_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_days=args.train_days,
        val_days=args.val_days,
        oos_days=args.oos_days,
        min_bars_between_trades=preset["min_bars_between_trades"] if preset else args.min_bars_between_trades,
        reward_churn_penalty=preset["reward_churn_penalty"] if preset else args.reward_churn_penalty,
        reward_dd_penalty=preset["reward_dd_penalty"] if preset else args.reward_dd_penalty,
        min_adx=preset["min_adx"] if preset else args.min_adx,
        min_atr_pct=preset["min_atr_pct"] if preset else args.min_atr_pct,
        rr_default=preset["rr_default"] if preset else args.rr_default,
        rr_min=preset["rr_min"] if preset else args.rr_min,
        rr_max=preset["rr_max"] if preset else args.rr_max,
        max_leverage=preset["max_leverage"] if preset else args.max_leverage,
        min_hold_bars=preset["min_hold_bars"] if preset else args.min_hold_bars,
        reward_trade_penalty=preset["reward_trade_penalty"] if preset else args.reward_trade_penalty,
        commission_rate=preset["commission_rate"] if preset else args.commission_rate,
        slippage_bps=preset["slippage_bps"] if preset else args.slippage_bps,
        reward_trend_bonus=preset["reward_trend_bonus"] if preset else args.reward_trend_bonus,
        reward_tp_progress=preset["reward_tp_progress"] if preset else args.reward_tp_progress,
        reward_one_sided_penalty=preset["reward_one_sided_penalty"] if preset else args.reward_one_sided_penalty,
        action_space=preset["action_space"] if preset else args.action_space,
        entropy_coef=args.entropy_coef,
    )
