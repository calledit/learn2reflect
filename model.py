import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ff.net.2.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: [B, T] → (logits: [B, T, vocab_size], hidden_states: list of n_layers [B, T, d_model])"""
        B, T = x.shape
        assert T <= self.cfg.context_length, f"Input length {T} exceeds context_length {self.cfg.context_length}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        hidden_states = []
        for block in self.blocks:
            h = block(h)
            hidden_states.append(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, hidden_states


# ──────────────────────────────────────────────── reflection transformer


class CausalCrossAttention(nn.Module):
    def __init__(self, d_ref: int, d_gen: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_ref % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_ref // n_heads
        self.dropout  = dropout
        self.q_proj   = nn.Linear(d_ref, d_ref, bias=False)
        self.kv_proj  = nn.Linear(d_gen, 2 * d_ref, bias=False)
        self.out_proj  = nn.Linear(d_ref, d_ref, bias=False)

    def forward(self, x: torch.Tensor, h_gen: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q    = self.q_proj(x)
        k, v = self.kv_proj(h_gen).chunk(2, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(y)


class ReflectionBlock(nn.Module):
    def __init__(self, d_ref: int, d_gen: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_ref)
        self.self_attn  = CausalSelfAttention(d_ref, n_heads, dropout)
        self.norm2      = nn.LayerNorm(d_ref)
        self.cross_attn = CausalCrossAttention(d_ref, d_gen, n_heads, dropout)
        self.norm3      = nn.LayerNorm(d_ref)
        self.ff         = FeedForward(d_ref, dropout)

    def forward(self, x: torch.Tensor, h_gen: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), h_gen)
        x = x + self.ff(self.norm3(x))
        return x


class ReflectionTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_ref = cfg.reflection_d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d_ref)
        self.pos_emb = nn.Embedding(cfg.context_length, d_ref)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([
            ReflectionBlock(d_ref, cfg.d_model, cfg.reflection_n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(d_ref)
        self.head = nn.Sequential(
            nn.Linear(d_ref, d_ref // 2, bias=True),
            nn.GELU(),
            nn.Linear(d_ref // 2, 1, bias=True),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, T]
        hidden_states: list of n_layers [B, T, d_gen] from Generator
        returns: [B, T] per-token loss predictions
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block, h_gen in zip(self.blocks, hidden_states):
            h = block(h, h_gen)
        h = self.norm(h)
        return self.head(h).squeeze(-1)
