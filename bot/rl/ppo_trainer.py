"""
PPO Trainer: обучение PPO агента с GAE и clipped objective.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

from bot.rl.ppo_agent import ActorCritic

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Буфер для хранения rollout данных.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Очищает буфер."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Добавляет шаг в буфер."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Вычисляет GAE (Generalized Advantage Estimation).
        
        Args:
            next_value: Value для следующего состояния (после последнего шага)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = next_value
        
        # Идем в обратном порядке
        for step in reversed(range(len(self.rewards))):
            if self.dones[step]:
                delta = self.rewards[step] - self.values[step]
                gae = delta
            else:
                delta = self.rewards[step] + gamma * next_value - self.values[step]
                gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            next_value = self.values[step]
        
        self.advantages = np.array(advantages, dtype=np.float32)
        self.returns = np.array(returns, dtype=np.float32)
    
    def get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Получает батч данных по индексам."""
        return {
            "observations": torch.FloatTensor(np.array([self.observations[i] for i in indices])),
            "actions": torch.LongTensor(np.array([self.actions[i] for i in indices])),
            "old_log_probs": torch.FloatTensor(np.array([self.log_probs[i] for i in indices])),
            "advantages": torch.FloatTensor(self.advantages[indices]),
            "returns": torch.FloatTensor(self.returns[indices]),
        }
    
    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO Trainer для обучения агента.
    """
    
    def __init__(
        self,
        agent: ActorCritic,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            agent: ActorCritic агент
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_eps: PPO clip epsilon
            value_coef: Коэффициент value loss
            entropy_coef: Коэффициент entropy bonus
            max_grad_norm: Gradient clipping
            device: Устройство (cuda/cpu)
        """
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.agent.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        
        # Метрики
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
    
    def train_step(
        self,
        buffer: RolloutBuffer,
        epochs: int = 5,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Выполняет один шаг обучения PPO.
        
        Args:
            buffer: RolloutBuffer с данными
            epochs: Количество эпох обновления
            batch_size: Размер батча
        
        Returns:
            Словарь с метриками обучения
        """
        if len(buffer) == 0:
            return {}
        
        # Нормализуем advantages
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages
        
        # Статистика для логирования
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        
        # Обучение на нескольких эпохах
        indices = np.arange(len(buffer))
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch = buffer.get_batch(batch_indices)
                
                # Переносим на устройство
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                log_probs, entropy, values = self.agent.evaluate_actions(
                    batch["observations"],
                    batch["actions"],
                )
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                advantages_batch = batch["advantages"]
                
                # Clipped policy loss
                policy_loss_1 = ratio * advantages_batch
                policy_loss_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantages_batch
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])
                
                # Entropy bonus
                entropy_bonus = -self.entropy_coef * entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + entropy_bonus
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Метрики
                with torch.no_grad():
                    approx_kl = (batch["old_log_probs"] - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
        
        # Средние метрики
        n_updates = epochs * (len(buffer) // batch_size + (1 if len(buffer) % batch_size > 0 else 0))
        
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "clip_fraction": total_clip_fraction / n_updates,
        }
        
        # Сохраняем метрики
        for key, value in metrics.items():
            self.training_stats[key].append(value)
        
        return metrics
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None):
        """Сохраняет чекпоинт модели."""
        checkpoint = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Загружает чекпоинт модели."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", {})
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint.get("metadata", {})
