"""
AlphaZero-style agent combining Transformer + Policy/Value heads with PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

try:
    from .transformer_encoder import TransformerEncoder
    from .policy_network import PolicyNetwork, BetaPolicyNetwork
    from .value_network import ValueNetwork
except ImportError:
    from transformer_encoder import TransformerEncoder
    from policy_network import PolicyNetwork, BetaPolicyNetwork
    from value_network import ValueNetwork


@dataclass
class AgentConfig:
    """Agent configuration"""
    # Input dimensions
    input_dim: int = 258
    seq_len: int = 60
    n_assets: int = 4
    
    # Transformer
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    
    # Policy/Value heads
    hidden_dims: List[int] = None
    dropout: float = 0.1
    
    # Policy type
    use_beta_policy: bool = False
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class AlphaZeroAgent(nn.Module):
    """
    AlphaZero-style trading agent
    
    Architecture:
    1. Transformer encoder processes temporal market data
    2. Policy head outputs position allocations
    3. Value head estimates expected return
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__()
        
        self.config = config or AgentConfig()
        
        # Transformer encoder for state representation
        self.encoder = TransformerEncoder(
            input_dim=self.config.input_dim,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            max_seq_len=self.config.seq_len + 10,  # Buffer
        )
        
        # Policy network
        if self.config.use_beta_policy:
            self.policy = BetaPolicyNetwork(
                input_dim=self.config.d_model,
                n_assets=self.config.n_assets,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
            )
        else:
            self.policy = PolicyNetwork(
                input_dim=self.config.d_model,
                n_assets=self.config.n_assets,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
            )
        
        # Value network
        self.value = ValueNetwork(
            input_dim=self.config.d_model,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation tensor of shape (batch, seq_len, input_dim)
            deterministic: If True, use mean action
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        # Encode state
        state = self.encoder(obs)  # (batch, d_model)
        
        # Get action from policy
        action, log_prob, entropy = self.policy(state, deterministic=deterministic)
        
        # Get value estimate
        value = self.value(state)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations"""
        state = self.encoder(obs)
        return self.value(state)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        state = self.encoder(obs)
        log_prob, entropy = self.policy.evaluate_actions(state, actions)
        value = self.value(state)
        return log_prob, entropy, value


class PPOTrainer:
    """
    PPO trainer for AlphaZero agent
    """
    
    def __init__(
        self,
        agent: AlphaZeroAgent,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize PPO trainer
        
        Args:
            agent: AlphaZero agent to train
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of PPO epochs per update
            batch_size: Mini-batch size
            device: Device to use
        """
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
        )
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: Rewards tensor (n_steps,)
            values: Value estimates (n_steps,)
            dones: Done flags (n_steps,)
            next_value: Value estimate for final state
            
        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = torch.zeros(n_steps, device=self.device)
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform PPO update
        
        Args:
            obs: Observations (n_samples, seq_len, input_dim)
            actions: Actions (n_samples, action_dim)
            old_log_probs: Old log probabilities (n_samples,)
            advantages: Advantages (n_samples,)
            returns: Returns (n_samples,)
            
        Returns:
            Dictionary of training metrics
        """
        n_samples = len(obs)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            # Generate random indices
            indices = torch.randperm(n_samples, device=self.device)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.agent.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.clip_epsilon, 
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    total_approx_kl += approx_kl.item()
                    n_updates += 1
        
        # Step scheduler
        self.scheduler.step()
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "lr": self.optimizer.param_groups[0]["lr"],
        }


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all data as tensors"""
        return (
            torch.stack(self.observations).to(self.device),
            torch.stack(self.actions).to(self.device),
            torch.tensor(self.rewards, dtype=torch.float32, device=self.device),
            torch.stack(self.values).to(self.device),
            torch.stack(self.log_probs).to(self.device),
            torch.tensor(self.dones, dtype=torch.float32, device=self.device),
        )
    
    def __len__(self):
        return len(self.observations)


if __name__ == "__main__":
    # Test agent
    config = AgentConfig(
        input_dim=258,
        seq_len=60,
        n_assets=4,
        d_model=128,
        n_heads=8,
        n_layers=4,
    )
    
    agent = AlphaZeroAgent(config)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, config.seq_len, config.input_dim)
    
    action, log_prob, entropy, value = agent(obs)
    print(f"\nForward pass:")
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.4f}, {action.max():.4f}]")
    print(f"  Log prob: {log_prob.mean():.4f}")
    print(f"  Entropy: {entropy.mean():.4f}")
    print(f"  Value: {value.mean():.4f}")
    
    # Test deterministic action
    action_det, _, _, _ = agent(obs, deterministic=True)
    print(f"\nDeterministic action range: [{action_det.min():.4f}, {action_det.max():.4f}]")
    
    # Test evaluate actions
    log_prob_eval, entropy_eval, value_eval = agent.evaluate_actions(obs, action)
    print(f"\nEvaluate actions:")
    print(f"  Log prob: {log_prob_eval.mean():.4f}")
    print(f"  Entropy: {entropy_eval.mean():.4f}")
    print(f"  Value: {value_eval.mean():.4f}")
    
    # Test PPO trainer
    print("\n--- Testing PPO Trainer ---")
    trainer = PPOTrainer(agent, lr=1e-4, device="cpu")
    
    # Create dummy rollout data
    n_steps = 100
    obs_data = torch.randn(n_steps, config.seq_len, config.input_dim)
    actions_data = torch.rand(n_steps, 5)
    old_log_probs = torch.randn(n_steps)
    advantages = torch.randn(n_steps)
    returns = torch.randn(n_steps)
    
    # Update
    metrics = trainer.update(obs_data, actions_data, old_log_probs, advantages, returns)
    print(f"Training metrics: {metrics}")

