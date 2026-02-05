"""
PPO Agent: Actor-Critic сеть для торговой стратегии.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic сеть для PPO.
    
    Архитектура:
    - Общий encoder (MLP)
    - Actor head: policy (categorical distribution)
    - Critic head: value function
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list = [256, 128, 64],
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        """
        Args:
            state_size: Размер state (наблюдения)
            action_size: Размер action space
            hidden_sizes: Размеры скрытых слоев
            activation: Функция активации ("relu", "tanh", "elu")
            use_layer_norm: Использовать LayerNorm
        """
        super(ActorCritic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_layer_norm = use_layer_norm
        
        # Выбираем активацию
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Общий encoder
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.activation)
            input_size = hidden_size
        
        self.encoder = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(input_size, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(input_size, 1)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_size)
        
        Returns:
            (action_logits, value)
        """
        features = self.encoder(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Получает действие и value для состояния.
        
        Args:
            state: (batch_size, state_size)
            action: Опционально уже выбранное действие (для вычисления log_prob)
        
        Returns:
            (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        
        # Categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Получает только value для состояния."""
        _, value = self.forward(state)
        return value.squeeze(-1)
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Оценивает действия (для PPO update).
        
        Args:
            state: (batch_size, state_size)
            action: (batch_size,)
        
        Returns:
            (log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)
