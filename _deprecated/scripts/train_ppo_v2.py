"""
Entry-Only PPO Training Script.
Упрощенный подход: агент только входит, выходы по TP/SL.
"""
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from bot.rl.data_preparation import prepare_ppo_data, split_data
from bot.rl.trading_env_v2 import TradingEnvV2, Action
from bot.rl.ppo_agent import ActorCritic
from bot.rl.ppo_trainer import PPOTrainer, RolloutBuffer
from bot.rl.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_ppo_v2(
    csv_path: str,
    symbol: str = "BTCUSDT",
    output_dir: str = "ppo_models",
    initial_capital: float = 10000.0,
    rollout_steps: int = 4096,
    train_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.5,  # Максимальный exploration для бинарного выбора
    train_days: int = 730,
    val_days: int = 14,
    oos_days: int = 30,
    num_iterations: int = 200,
    # Trading params
    commission_rate: float = 0.0006,
    slippage_bps: float = 3.0,
    min_bars_between_trades: int = 8,  # 2 часа между сделками (базовое)
    min_adx: float = 0.0,  # Без фильтра ADX (базовое)
    min_atr_pct: float = 0.0,  # Без фильтра ATR (базовое)
    rr_default: float = 3.0,  # Классический RR
    rr_min: float = 2.5,
    rr_max: float = 3.5,
    reward_scale: float = 100.0,
    use_trend_filter: bool = True,  # Фильтр тренда
    allowed_side: str = "both",  # "long", "short", или "both"
):
    """Обучает Entry-Only PPO агента."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Подготовка данных
    logger.info("Preparing data...")
    df = prepare_ppo_data(csv_path, symbol=symbol, enable_mtf=True)
    
    df_train, df_val, df_oos = split_data(
        df, val_days=val_days, oos_days=oos_days, train_days=train_days
    )
    
    logger.info(f"Train: {len(df_train)} rows, Val: {len(df_val)} rows, OOS: {len(df_oos)} rows")
    
    # Среда
    env = TradingEnvV2(
        df_train,
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
    
    # Risk manager
    risk_manager = RiskManager(default_rr=rr_default, min_rr=rr_min, max_rr=rr_max)
    
    def risk_manager_callback(side, entry_price, row, df_history):
        sl, tp, rr, _ = risk_manager.calculate_tp_sl(
            "LONG" if side == "LONG" else "SHORT",
            entry_price, row, df_history
        )
        return sl, tp, rr
    
    # Агент
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    agent = ActorCritic(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=[256, 128, 64],
        use_layer_norm=True,
    )
    
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
    
    for iteration in tqdm(range(num_iterations), desc="Training"):
        obs = env.reset()
        buffer.reset()
        
        total_reward = 0.0
        trades_this_episode = 0
        opened_count = 0  # DEBUG: открытые позиции
        action_counts = {0: 0, 1: 0}  # DEBUG: подсчёт действий
        rejected_count = 0  # DEBUG: отклонённые сделки
        
        for step in range(rollout_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(obs_tensor)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            next_obs, reward, done, info = env.step(action, risk_manager_callback)
            
            # DEBUG: подсчёт действий
            action_counts[action] = action_counts.get(action, 0) + 1
            if "action_rejected" in info:
                rejected_count += 1
            if info.get("action") in ["opened_long", "opened_short"]:
                opened_count += 1
            
            buffer.add(obs, action, log_prob, reward, value, done)
            
            obs = next_obs
            total_reward += reward
            if "trade_closed" in info:
                trades_this_episode += 1
            
            if done:
                obs = env.reset()
        
        # GAE
        if len(buffer) > 0:
            if done:
                next_value = 0.0
            else:
                last_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    next_value = agent.get_value(last_obs_tensor).item()
            
            buffer.compute_gae(next_value, gamma, gae_lambda)
            metrics = trainer.train_step(buffer, epochs=train_epochs, batch_size=batch_size)
            
            if (iteration + 1) % 5 == 0:
                logger.info(
                    f"Iter {iteration+1}: Reward={total_reward:.2f}, "
                    f"Opened={opened_count}, Closed={trades_this_episode}, "
                    f"Actions={action_counts}"
                )
            
            # Валидация
            if (iteration + 1) % 10 == 0:
                val_return, val_trades = validate_agent(
                    agent, df_val, initial_capital, risk_manager_callback,
                    commission_rate, slippage_bps, min_bars_between_trades,
                    min_adx, min_atr_pct, reward_scale, use_trend_filter,
                    allowed_side
                )
                logger.info(f"Validation: Return={val_return:.2f}%, Trades={val_trades}")
                
                if val_return > best_val_return:
                    best_val_return = val_return
                    # Имя файла учитывает специализацию
                    side_suffix = f"_{allowed_side}" if allowed_side != "both" else ""
                    checkpoint_path = output_path / f"ppo_v2_{symbol}{side_suffix}_best.pth"
                    trainer.save_checkpoint(
                        str(checkpoint_path),
                        metadata={"iteration": iteration + 1, "val_return": val_return, "allowed_side": allowed_side}
                    )
                    logger.info(f"Saved best model (val_return={val_return:.2f}%)")
    
    # Финальный чекпоинт
    side_suffix = f"_{allowed_side}" if allowed_side != "both" else ""
    final_checkpoint = output_path / f"ppo_v2_{symbol}{side_suffix}_final.pth"
    trainer.save_checkpoint(str(final_checkpoint), metadata={"iteration": num_iterations})
    
    logger.info("Training completed!")
    logger.info(f"Best validation return: {best_val_return:.2f}%")


def validate_agent(
    agent, df_val, initial_capital, risk_manager_callback,
    commission_rate, slippage_bps, min_bars_between_trades,
    min_adx, min_atr_pct, reward_scale, use_trend_filter=True,
    allowed_side="both"
):
    """Валидация агента."""
    env = TradingEnvV2(
        df_val,
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
    
    obs = env.reset(start_step=0)
    device = next(agent.parameters()).device
    agent.eval()
    
    with torch.no_grad():
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.item()
            
            obs, reward, done, info = env.step(action, risk_manager_callback)
            
            if done:
                break
    
    # Закрываем открытую позицию
    if env.position_side is not None:
        last_row = df_val.iloc[-1]
        env.close_open_position(last_row, "end_of_test")
    
    final_equity = env.equity
    return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    
    return return_pct, len(env.trades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entry-Only PPO")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--output", type=str, default="ppo_models")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.5)
    parser.add_argument("--min-bars-between-trades", type=int, default=8)
    parser.add_argument("--min-adx", type=float, default=0.0)
    parser.add_argument("--min-atr-pct", type=float, default=0.0)
    parser.add_argument("--rr-default", type=float, default=3.0)
    parser.add_argument("--rr-min", type=float, default=2.5)
    parser.add_argument("--rr-max", type=float, default=3.5)
    parser.add_argument("--commission-rate", type=float, default=0.0006)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable trend filter")
    parser.add_argument("--allowed-side", type=str, default="both", 
                        choices=["long", "short", "both"],
                        help="Specialize model: 'long', 'short', or 'both'")
    
    args = parser.parse_args()
    
    train_ppo_v2(
        csv_path=args.csv,
        symbol=args.symbol,
        output_dir=args.output,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        train_days=args.train_days,
        val_days=args.val_days,
        oos_days=args.oos_days,
        min_bars_between_trades=args.min_bars_between_trades,
        min_adx=args.min_adx,
        min_atr_pct=args.min_atr_pct,
        rr_default=args.rr_default,
        rr_min=args.rr_min,
        rr_max=args.rr_max,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        reward_scale=args.reward_scale,
        use_trend_filter=not args.no_trend_filter,
        allowed_side=args.allowed_side,
    )
