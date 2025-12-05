"""
Policy network for position allocation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs position allocations
    
    Outputs a distribution over actions (position weights)
    """
    
    def __init__(
        self,
        input_dim: int,
        n_assets: int = 4,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize policy network
        
        Args:
            input_dim: Dimension of input (from transformer encoder)
            n_assets: Number of assets to allocate
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_assets = n_assets
        self.action_dim = n_assets + 1  # weights + leverage factor
        
        hidden_dims = hidden_dims or [256, 128]
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, self.action_dim)
        self.log_std_head = nn.Linear(prev_dim, self.action_dim)
        
        # Bounds for log_std
        self.log_std_min = -20
        self.log_std_max = 2
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output layers with small weights"""
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            deterministic: If True, return mean action without sampling
            
        Returns:
            Tuple of (action, log_prob, entropy)
            - action: (batch, action_dim) values in [0, 1]
            - log_prob: (batch,) log probability of action
            - entropy: (batch,) entropy of distribution
        """
        # MLP forward
        h = self.mlp(x)
        
        # Get mean and log_std
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action_raw = mean
        else:
            action_raw = dist.rsample()  # Reparameterization trick
        
        # Apply sigmoid to get values in [0, 1]
        action = torch.sigmoid(action_raw)
        
        # Calculate log probability with correction for sigmoid transform
        log_prob = dist.log_prob(action_raw)
        # Jacobian correction for sigmoid
        log_prob = log_prob - torch.log(action * (1 - action) + 1e-8)
        log_prob = log_prob.sum(dim=-1)
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and evaluate log probability
        
        Args:
            x: Input tensor
            action: Optional action to evaluate (if None, sample new action)
            
        Returns:
            Tuple of (action, log_prob, entropy)
        """
        h = self.mlp(x)
        
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action_raw = dist.rsample()
            action = torch.sigmoid(action_raw)
        else:
            # Inverse sigmoid to get raw action
            action = torch.clamp(action, 1e-6, 1 - 1e-6)
            action_raw = torch.log(action / (1 - action))
        
        log_prob = dist.log_prob(action_raw)
        log_prob = log_prob - torch.log(action * (1 - action) + 1e-8)
        log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions
        
        Args:
            x: Input tensor
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        h = self.mlp(x)
        
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        # Inverse sigmoid
        actions = torch.clamp(actions, 1e-6, 1 - 1e-6)
        actions_raw = torch.log(actions / (1 - actions))
        
        log_prob = dist.log_prob(actions_raw)
        log_prob = log_prob - torch.log(actions * (1 - actions) + 1e-8)
        log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class BetaPolicyNetwork(nn.Module):
    """
    Policy network using Beta distribution (naturally bounded [0, 1])
    """
    
    def __init__(
        self,
        input_dim: int,
        n_assets: int = 4,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_assets = n_assets
        self.action_dim = n_assets + 1
        
        hidden_dims = hidden_dims or [256, 128]
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output alpha and beta parameters for Beta distribution
        self.alpha_head = nn.Linear(prev_dim, self.action_dim)
        self.beta_head = nn.Linear(prev_dim, self.action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.alpha_head.weight, gain=0.01)
        nn.init.constant_(self.alpha_head.bias, 1)  # Initialize to uniform
        nn.init.orthogonal_(self.beta_head.weight, gain=0.01)
        nn.init.constant_(self.beta_head.bias, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        
        # Get alpha and beta (must be positive)
        alpha = F.softplus(self.alpha_head(h)) + 1  # Minimum 1 for stability
        beta = F.softplus(self.beta_head(h)) + 1
        
        # Beta distribution
        dist = torch.distributions.Beta(alpha, beta)
        
        if deterministic:
            # Mean of Beta distribution
            action = alpha / (alpha + beta)
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        
        alpha = F.softplus(self.alpha_head(h)) + 1
        beta = F.softplus(self.beta_head(h)) + 1
        
        dist = torch.distributions.Beta(alpha, beta)
        
        log_prob = dist.log_prob(actions.clamp(1e-6, 1 - 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


if __name__ == "__main__":
    # Test policy network
    batch_size = 32
    input_dim = 128
    n_assets = 4
    
    print("Testing Gaussian Policy Network")
    print("=" * 50)
    
    policy = PolicyNetwork(input_dim, n_assets)
    print(f"Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    x = torch.randn(batch_size, input_dim)
    
    # Test forward
    action, log_prob, entropy = policy(x)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    # Test deterministic
    action_det, _, _ = policy(x, deterministic=True)
    print(f"Deterministic action range: [{action_det.min():.4f}, {action_det.max():.4f}]")
    
    # Test evaluate
    log_prob_eval, entropy_eval = policy.evaluate_actions(x, action)
    print(f"Evaluate log_prob: {log_prob_eval.mean():.4f}")
    
    print("\nTesting Beta Policy Network")
    print("=" * 50)
    
    beta_policy = BetaPolicyNetwork(input_dim, n_assets)
    print(f"Parameters: {sum(p.numel() for p in beta_policy.parameters()):,}")
    
    action, log_prob, entropy = beta_policy(x)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
    print(f"Log prob: {log_prob.mean():.4f}")
    print(f"Entropy: {entropy.mean():.4f}")

