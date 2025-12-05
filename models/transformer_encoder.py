"""
Transformer encoder for temporal state representation
"""
import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing temporal market data
    
    Takes a sequence of market features and outputs a rich state representation
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        Initialize transformer encoder
        
        Args:
            input_dim: Dimension of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        
        # Learned CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            State representation of shape (batch, d_model)
        """
        batch_size = x.size(0)
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transform
        x = self.transformer(x, mask=mask)
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Apply output normalization
        output = self.output_norm(cls_output)
        
        return output
    
    def forward_with_attention(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """
        Forward pass with attention weights for interpretability
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Manual forward through transformer layers to get attention
        attention_weights = []
        for layer in self.transformer.layers:
            # Self-attention with attention weights
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weights)
            
            # Rest of transformer layer
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))
        
        # Final norm
        x = self.transformer.norm(x)
        
        # CLS output
        cls_output = x[:, 0, :]
        output = self.output_norm(cls_output)
        
        return output, attention_weights


class TransformerEncoderWithPooling(nn.Module):
    """
    Transformer encoder with different pooling strategies
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        pooling: str = "mean",  # "cls", "mean", "last", "attention"
    ):
        super().__init__()
        
        self.pooling = pooling
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # CLS token (only for cls pooling)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention pooling
        if pooling == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )
        
        self.output_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Add CLS token if needed
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transform
        x = self.transformer(x)
        
        # Pool
        if self.pooling == "cls":
            output = x[:, 0, :]
        elif self.pooling == "mean":
            output = x.mean(dim=1)
        elif self.pooling == "last":
            output = x[:, -1, :]
        elif self.pooling == "attention":
            # Attention-weighted pooling
            attn_weights = self.attention_pool(x)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            output = (x * attn_weights).sum(dim=1)
        else:
            output = x.mean(dim=1)
        
        return self.output_norm(output)


if __name__ == "__main__":
    # Test transformer encoder
    batch_size = 32
    seq_len = 60
    input_dim = 258
    d_model = 128
    
    # Create model
    model = TransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention
    output, attn_weights = model.forward_with_attention(x)
    print(f"Attention weights: {len(attn_weights)} layers, shape: {attn_weights[0].shape}")
    
    # Test pooling variants
    print("\n--- Testing pooling variants ---")
    for pooling in ["cls", "mean", "last", "attention"]:
        model_pool = TransformerEncoderWithPooling(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=8,
            n_layers=4,
            pooling=pooling,
        )
        output = model_pool(x)
        print(f"Pooling={pooling}: output shape={output.shape}")

