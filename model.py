import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ──────────────────────────────────────────────────────────── generator


class GeneratorAttention(nn.Module):
    """Causal self-attention without output projection — returns per-head outputs."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

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
        return y.transpose(1, 2).contiguous()  # [B, T, n_heads, head_dim]


class FunctionGroup(nn.Module):
    """Per-attention-head MLP: head_dim → hidden1 → hidden2 → hidden1 → d_model residual."""
    def __init__(self, head_dim: int, d_model: int, hidden1: int, hidden2: int):
        super().__init__()
        self.norm = nn.LayerNorm(head_dim)
        self.net = nn.Sequential(
            nn.Linear(head_dim, hidden1, bias=False), nn.GELU(),
            nn.Linear(hidden1,  hidden2, bias=False), nn.GELU(),
            nn.Linear(hidden2,  hidden1, bias=False), nn.GELU(),
            nn.Linear(hidden1,  d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, head_dim] → [B, T, d_model]
        return self.net(self.norm(x))


class FunctionGroupBlock(nn.Module):
    """Transformer block: each attention head feeds its own MLP function group.
    Replaces both W_O and the FFN."""
    def __init__(self, d_model: int, n_heads: int, dropout: float, hidden1: int, hidden2: int):
        super().__init__()
        head_dim = d_model // n_heads
        self.norm1     = nn.LayerNorm(d_model)
        self.attn      = GeneratorAttention(d_model, n_heads, dropout)
        self.fn_groups = nn.ModuleList([
            FunctionGroup(head_dim, d_model, hidden1, hidden2) for _ in range(n_heads)
        ])
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = self.attn(self.norm1(x))                           # [B, T, n_heads, head_dim]
        residual = sum(fg(heads[:, :, i, :]) for i, fg in enumerate(self.fn_groups))
        return x + self.norm2(residual)

    def get_cache(self, x: torch.Tensor) -> tuple:
        """Full forward returning intermediates for the activation caching pass."""
        heads   = self.attn(self.norm1(x))
        fn_outs = [fg(heads[:, :, i, :]) for i, fg in enumerate(self.fn_groups)]
        return x + self.norm2(sum(fn_outs)), heads, fn_outs

    def forward_cached(self, x: torch.Tensor, selected_head: int,
                       cached_head_out: torch.Tensor, cached_sibling_sum: torch.Tensor) -> torch.Tensor:
        """Isolated-training forward: selected group in graph, siblings pre-summed and detached."""
        residual = self.fn_groups[selected_head](cached_head_out) + cached_sibling_sum
        return x + self.norm2(residual)


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([
            FunctionGroupBlock(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.fn_hidden1, cfg.fn_hidden2)
            for _ in range(cfg.n_layers)
        ])
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Function group output projections start silent so groups earn their contribution
        for block in self.blocks:
            for fg in block.fn_groups:
                nn.init.zeros_(fg.net[-1].weight)

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
        assert T <= self.cfg.context_length
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        hidden_states = []
        for block in self.blocks:
            h = block(h)
            hidden_states.append(h)
        h = self.norm(h)
        return self.lm_head(h), hidden_states

    def forward_from_cache(
        self,
        layer_idx: int,
        h_cached: torch.Tensor,
        selected_head: int,
        cached_head_out: torch.Tensor,
        cached_sibling_sum: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Isolated-training forward starting from cached input at layer_idx.
        Only the selected function group is in the computation graph."""
        h = self.blocks[layer_idx].forward_cached(
            h_cached, selected_head, cached_head_out, cached_sibling_sum
        )
        for i in range(layer_idx + 1, len(self.blocks)):
            h = self.blocks[i](h)
        logits = self.lm_head(self.norm(h))
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), y.reshape(-1))


# ──────────────────────────────────────────────── reflection transformer


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
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


class CausalCrossAttention(nn.Module):
    def __init__(self, d_ref: int, d_gen: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_ref % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_ref // n_heads
        self.dropout  = dropout
        self.q_proj   = nn.Linear(d_ref, d_ref, bias=False)
        self.kv_proj  = nn.Linear(d_gen, 2 * d_ref, bias=False)
        self.out_proj = nn.Linear(d_ref, d_ref, bias=False)

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
        self.ff         = nn.Sequential(
            nn.Linear(d_ref, 4 * d_ref, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_ref, d_ref, bias=False),
        )

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
        self.loss_head = nn.Sequential(
            nn.Linear(d_ref, d_ref // 2, bias=True),
            nn.GELU(),
            nn.Linear(d_ref // 2, 1, bias=True),
        )
        self.selection_head = nn.Sequential(
            nn.Linear(d_ref, d_ref, bias=True),
            nn.GELU(),
            nn.Linear(d_ref, cfg.n_layers * cfg.n_heads, bias=True),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor],
                return_selection: bool = False) -> torch.Tensor | tuple:
        """
        x: [B, T]
        hidden_states: list of n_layers [B, T, d_gen] from Generator
        return_selection: also return selection logits [n_layers * n_heads]
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block, h_gen in zip(self.blocks, hidden_states):
            h = block(h, h_gen)
        h = self.norm(h)
        loss_pred = self.loss_head(h).squeeze(-1)           # [B, T]
        if return_selection:
            pooled          = h.mean(dim=1).mean(dim=0)    # [d_ref]
            selection_logits = self.selection_head(pooled) # [n_layers * n_heads]
            return loss_pred, selection_logits
        return loss_pred
