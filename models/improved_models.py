"""
Improved model architectures for trading
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.chomp1 = Chomp1d(padding)
        self.chomp2 = Chomp1d(padding)
        
        self.relu1 = nn.GELU()
        self.relu2 = nn.GELU()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = out.transpose(1, 2)  # (batch, seq, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (batch, channels, seq)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class Chomp1d(nn.Module):
    """Remove padding from the right side"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network - efficient for long sequences
    Better than LSTM for parallel training, similar to transformer
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 128, 256, 128]
        layers = []
        
        num_levels = len(hidden_dims)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # Take last timestep
        return out


class GatedTransformerBlock(nn.Module):
    """Transformer block with gating mechanism - better for noisy financial data"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Gating mechanism
        self.gate1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.gate2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with gating
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        gate1 = self.gate1(torch.cat([x, attn_out], dim=-1))
        x = self.norm1(x + gate1 * attn_out)
        
        # Feed-forward with gating
        ff_out = self.ff(x)
        gate2 = self.gate2(torch.cat([x, ff_out], dim=-1))
        x = self.norm2(x + gate2 * ff_out)
        
        return x


class GatedTransformerEncoder(nn.Module):
    """
    Gated Transformer - more robust for noisy financial time series
    The gating helps filter out noise and focus on relevant patterns
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gated transformer blocks
        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Return CLS token representation
        return self.norm(x[:, 0])


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for trading
    
    Conditions actions on:
    - Past states (market features)
    - Past actions (positions)
    - Past returns (realized + unrealized)
    - Target return (what we want to achieve)
    
    This approach is more stable than PPO and can target specific return/risk profiles
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 20,  # Context length
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings for each modality
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_seq_len * 3, d_model)
        
        # Transformer
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len * 3, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
            nn.Sigmoid(),  # Actions in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        states: torch.Tensor,      # (batch, seq, state_dim)
        actions: torch.Tensor,      # (batch, seq, action_dim)
        returns_to_go: torch.Tensor,  # (batch, seq, 1)
        timesteps: torch.Tensor,    # (batch, seq)
    ) -> torch.Tensor:
        """
        Forward pass - predicts next action given context
        
        Returns:
            Predicted action (batch, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed each modality
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        return_embeddings = self.return_embed(returns_to_go)
        
        # Add timestep embeddings
        time_embeddings = self.timestep_embed(timesteps)
        
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings
        
        # Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        # But for prediction we want: [R_t, s_t] -> predict a_t
        stacked = torch.stack([return_embeddings, state_embeddings, action_embeddings], dim=2)
        stacked = stacked.reshape(batch_size, seq_len * 3, self.d_model)
        
        # Add positional encoding
        stacked = stacked + self.pos_encoding[:, :stacked.size(1), :]
        
        # Create causal mask
        seq_len_total = stacked.size(1)
        mask = torch.triu(torch.ones(seq_len_total, seq_len_total, device=stacked.device), diagonal=1).bool()
        
        # Transform
        output = self.transformer(stacked, mask=mask)
        output = self.norm(output)
        
        # Get state positions (indices 1, 4, 7, ...) for action prediction
        # We predict action from state representation
        state_outputs = output[:, 1::3, :]  # Every 3rd starting from index 1
        
        # Predict action from last state
        action_pred = self.action_head(state_outputs[:, -1, :])
        
        return action_pred
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get action for deployment"""
        return self.forward(states, actions, returns_to_go, timesteps)


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder - simpler but effective baseline
    Often works well for shorter sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden states from both directions
        h_forward = h_n[-2]  # Last layer forward
        h_backward = h_n[-1]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        
        out = self.output_proj(h_combined)
        return self.norm(out)


class HybridEncoder(nn.Module):
    """
    Hybrid CNN + Transformer encoder
    
    - CNN captures local patterns (price movements, technical signals)
    - Transformer captures global dependencies (regime changes, trends)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # CNN for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        
        # Transformer for global patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.output_dim = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN: local features
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer: global patterns
        x = self.transformer(x)
        
        return self.norm(x[:, 0])


def get_best_encoder(
    input_dim: int,
    model_type: str = "gated_transformer",
    **kwargs
) -> nn.Module:
    """
    Factory function to get the best encoder for trading
    
    Args:
        input_dim: Input feature dimension
        model_type: One of:
            - "gated_transformer" (recommended for most cases)
            - "tcn" (faster training, good for longer sequences)
            - "lstm" (simpler baseline)
            - "hybrid" (best for capturing both local and global patterns)
            - "decision_transformer" (for goal-conditioned trading)
    """
    if model_type == "gated_transformer":
        return GatedTransformerEncoder(
            input_dim=input_dim,
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            n_layers=kwargs.get("n_layers", 6),
            d_ff=kwargs.get("d_ff", 1024),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    elif model_type == "tcn":
        return TemporalConvNet(
            input_dim=input_dim,
            hidden_dims=kwargs.get("hidden_dims", [128, 256, 256, 128]),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    elif model_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 256),
            n_layers=kwargs.get("n_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    elif model_type == "hybrid":
        return HybridEncoder(
            input_dim=input_dim,
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            n_layers=kwargs.get("n_layers", 4),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test all encoders
    batch_size = 32
    seq_len = 60
    input_dim = 258
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("=" * 60)
    print("Model Comparison for Trading")
    print("=" * 60)
    
    models = {
        "Gated Transformer": get_best_encoder(input_dim, "gated_transformer", d_model=256, n_layers=6),
        "TCN": get_best_encoder(input_dim, "tcn"),
        "LSTM": get_best_encoder(input_dim, "lstm"),
        "Hybrid CNN+Transformer": get_best_encoder(input_dim, "hybrid"),
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Time forward pass
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                out = model(x)
        elapsed = (time.time() - start) / 10 * 1000
        
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
        print(f"  Forward time: {elapsed:.2f}ms")
    
    # Test Decision Transformer
    print("\n" + "=" * 60)
    print("Decision Transformer (Goal-Conditioned)")
    print("=" * 60)
    
    dt = DecisionTransformer(
        state_dim=input_dim,
        action_dim=5,
        d_model=256,
        n_layers=4,
        max_seq_len=20,
    )
    
    context_len = 10
    states = torch.randn(batch_size, context_len, input_dim)
    actions = torch.rand(batch_size, context_len, 5)
    returns_to_go = torch.randn(batch_size, context_len, 1)
    timesteps = torch.arange(context_len).unsqueeze(0).expand(batch_size, -1)
    
    action_pred = dt(states, actions, returns_to_go, timesteps)
    
    print(f"  Parameters: {sum(p.numel() for p in dt.parameters()):,}")
    print(f"  Action prediction shape: {action_pred.shape}")
    print(f"  Action range: [{action_pred.min():.4f}, {action_pred.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("""
    For your use case (leveraged index trading with risk management):
    
    1. BEST CHOICE: Gated Transformer (256 dim, 6 layers)
       - Robust to noisy financial data
       - Good for regime detection
       - ~2.5M parameters
    
    2. FAST ALTERNATIVE: Hybrid CNN+Transformer
       - Captures both local patterns (price action) and global trends
       - Faster training than pure transformer
       - ~1.5M parameters
    
    3. FOR GOAL-CONDITIONED: Decision Transformer
       - Can target specific Sharpe ratio / return
       - More stable than PPO
       - Better for live trading
    """)

