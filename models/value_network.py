"""
Value network for estimating expected returns
"""
import torch
import torch.nn as nn
from typing import List


class ValueNetwork(nn.Module):
    """
    Value network that estimates expected return (V(s))
    
    Takes the state representation and outputs a scalar value
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize value network
        
        Args:
            input_dim: Dimension of input (from transformer encoder)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
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
        
        # Output head
        self.value_head = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output layer with small weights"""
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Value estimate of shape (batch,)
        """
        h = self.mlp(x)
        value = self.value_head(h)
        return value.squeeze(-1)


class DuelingValueNetwork(nn.Module):
    """
    Dueling value network with separate value and advantage streams
    
    V(s) = Value(s) + Advantage(s, a) - mean(Advantage)
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 5,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        hidden_dims = hidden_dims or [256, 128]
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], n_actions),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            State value of shape (batch,)
        """
        h = self.shared(x)
        
        value = self.value_stream(h)  # (batch, 1)
        advantage = self.advantage_stream(h)  # (batch, n_actions)
        
        # Combine using dueling formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        # Return state value (max Q)
        return q_values.max(dim=-1)[0]
    
    def forward_q(self, x: torch.Tensor) -> torch.Tensor:
        """Return full Q-values for all actions"""
        h = self.shared(x)
        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DistributionalValueNetwork(nn.Module):
    """
    Distributional value network (C51-style)
    
    Outputs a distribution over possible returns
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Support (possible return values)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms)
        )
        
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
        
        # Output distribution over atoms
        self.dist_head = nn.Linear(prev_dim, n_atoms)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.dist_head.weight, gain=0.01)
        nn.init.constant_(self.dist_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns expected value
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Expected value of shape (batch,)
        """
        h = self.mlp(x)
        logits = self.dist_head(h)  # (batch, n_atoms)
        probs = torch.softmax(logits, dim=-1)
        
        # Expected value
        expected_value = (probs * self.support).sum(dim=-1)
        return expected_value
    
    def forward_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return full probability distribution"""
        h = self.mlp(x)
        logits = self.dist_head(h)
        return torch.softmax(logits, dim=-1)
    
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits"""
        h = self.mlp(x)
        return self.dist_head(h)


if __name__ == "__main__":
    # Test value networks
    batch_size = 32
    input_dim = 128
    
    print("Testing Value Network")
    print("=" * 50)
    
    value_net = ValueNetwork(input_dim)
    print(f"Parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    
    x = torch.randn(batch_size, input_dim)
    value = value_net(x)
    print(f"Value shape: {value.shape}")
    print(f"Value range: [{value.min():.4f}, {value.max():.4f}]")
    
    print("\nTesting Dueling Value Network")
    print("=" * 50)
    
    dueling_net = DuelingValueNetwork(input_dim, n_actions=5)
    print(f"Parameters: {sum(p.numel() for p in dueling_net.parameters()):,}")
    
    value = dueling_net(x)
    print(f"Value shape: {value.shape}")
    q_values = dueling_net.forward_q(x)
    print(f"Q-values shape: {q_values.shape}")
    
    print("\nTesting Distributional Value Network")
    print("=" * 50)
    
    dist_net = DistributionalValueNetwork(input_dim)
    print(f"Parameters: {sum(p.numel() for p in dist_net.parameters()):,}")
    
    value = dist_net(x)
    print(f"Value shape: {value.shape}")
    dist = dist_net.forward_dist(x)
    print(f"Distribution shape: {dist.shape}")
    print(f"Distribution sum: {dist.sum(dim=-1).mean():.4f}")  # Should be ~1

