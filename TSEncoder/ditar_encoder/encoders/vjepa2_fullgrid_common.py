from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

from ditar_encoder.encoders.text_normalization import normalize_instruction


















class FrozenT5TextConditioner(nn.Module):
    def __init__(
        self,
        t5_model_id: str = 't5-base',
        t5_model_path: Optional[str] = None,
        t5_max_length: int = 512,
        text_cache_size: int = 0,
    ) -> None:
        super().__init__()
        # Path precedence: explicit arg -> T5_BASE_DIR env -> HF hub id (default).
        t5_source = t5_model_path or os.environ.get('T5_BASE_DIR') or t5_model_id
        if t5_source != t5_model_id and not os.path.isdir(t5_source):
            t5_source = t5_model_id

        self.encoder = T5EncoderModel.from_pretrained(t5_source)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_source)
        self.hidden_size = int(getattr(self.encoder.config, 'd_model', 768))
        self.max_length = int(t5_max_length)
        self.text_cache_size = max(int(text_cache_size), 0)
        self._cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _normalize_instruction(self, instruction: Optional[str]) -> str:
        # Single source of truth shared with the training data pipeline (audit A6).
        return normalize_instruction(instruction)

    def _cache_item(self, text: str, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        if self.text_cache_size <= 0:
            return
        self._cache[text] = (
            tokens.detach().to(device='cpu', dtype=torch.float16),
            mask.detach().to(device='cpu', dtype=torch.bool),
        )
        self._cache.move_to_end(text)
        while len(self._cache) > self.text_cache_size:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _encode_uncached(self, instructions: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(instructions),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        device = self.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.encoder(**encoded)
        tokens = outputs.last_hidden_state.float()
        mask = encoded['attention_mask'].to(device=device, dtype=torch.bool)
        return tokens, mask

    @torch.no_grad()
    def forward(self, instructions: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = [self._normalize_instruction(instruction) for instruction in instructions]
        if self.text_cache_size <= 0:
            return self._encode_uncached(normalized)

        batch_items: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for text in dict.fromkeys(normalized):
            if text in self._cache:
                batch_items[text] = self._cache[text]
                self._cache.move_to_end(text)

        missing = [text for text in dict.fromkeys(normalized) if text not in batch_items]
        if missing:
            missing_tokens, missing_mask = self._encode_uncached(missing)
            for text, tokens, mask in zip(missing, missing_tokens, missing_mask):
                valid_len = int(mask.sum().item())
                item = (
                    tokens[:valid_len].detach().to(device='cpu', dtype=torch.float16),
                    mask[:valid_len].detach().to(device='cpu', dtype=torch.bool),
                )
                batch_items[text] = item
                self._cache_item(text, tokens[:valid_len], mask[:valid_len])

        max_len = max(batch_items[text][0].shape[0] for text in normalized)
        device = self.device
        tokens = torch.zeros(len(normalized), max_len, self.hidden_size, device=device, dtype=torch.float32)
        mask = torch.zeros(len(normalized), max_len, device=device, dtype=torch.bool)
        for batch_idx, text in enumerate(normalized):
            cached_tokens, cached_mask = batch_items[text]
            length = int(cached_mask.shape[0])
            tokens[batch_idx, :length] = cached_tokens.to(device=device, dtype=torch.float32)
            mask[batch_idx, :length] = cached_mask.to(device=device, dtype=torch.bool)
        return tokens, mask

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device








class FullGridTokenBottleneck(nn.Module):
    def __init__(
        self,
        model_dim: int,
        bottleneck_dim: int = 64,
        bottleneck_hidden_dim: int = 256,
        bottleneck_regularizer: str = 'kl',
        post_norm_after_decode: bool = False,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.bottleneck_hidden_dim = int(bottleneck_hidden_dim)
        self.post_norm_after_decode = bool(post_norm_after_decode)
        self.bottleneck_regularizer = str(bottleneck_regularizer or 'none').strip().lower()
        if self.bottleneck_regularizer not in {'none', 'kl'}:
            raise ValueError(
                f'Unsupported bottleneck_regularizer={bottleneck_regularizer!r}. Expected "none" or "kl".'
            )

        if self.bottleneck_dim > 0:
            self.pre = nn.Sequential(
                nn.LayerNorm(self.model_dim),
                nn.Linear(self.model_dim, self.bottleneck_hidden_dim),
                nn.GELU(),
            )
            if self.bottleneck_regularizer == 'kl':
                self.to_mu = nn.Linear(self.bottleneck_hidden_dim, self.bottleneck_dim)
                self.to_logvar = nn.Linear(self.bottleneck_hidden_dim, self.bottleneck_dim)
                self.to_latent = None
            else:
                self.to_mu = None
                self.to_logvar = None
                self.to_latent = nn.Linear(self.bottleneck_hidden_dim, self.bottleneck_dim)
            self.to_model = nn.Linear(self.bottleneck_dim, self.model_dim)
            self.post_decode_norm = (
                nn.LayerNorm(self.model_dim, elementwise_affine=False)
                if self.post_norm_after_decode
                else nn.Identity()
            )
        else:
            self.pre = None
            self.to_mu = None
            self.to_logvar = None
            self.to_latent = None
            self.to_model = None
            self.post_decode_norm = nn.Identity()

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = tokens.new_zeros(())
        if self.bottleneck_dim <= 0:
            return {
                'latent': tokens,
                'decoded': tokens,
                'kl_loss': zero,
                'mu': None,
                'logvar': None,
                'per_dim_kl': None,
            }

        hidden = self.pre(tokens)
        mu = None
        logvar = None
        per_dim_kl = None
        if self.bottleneck_regularizer == 'kl':
            mu = self.to_mu(hidden)
            logvar = self.to_logvar(hidden).clamp(min=-30.0, max=20.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                latent = mu + std * torch.randn_like(std)
            else:
                latent = mu
            per_dim_kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
            kl_loss = per_dim_kl.mean()
        else:
            latent = self.to_latent(hidden)
            kl_loss = zero

        decoded = self.post_decode_norm(self.to_model(latent))
        return {
            'latent': latent,
            'decoded': decoded,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar,
            'per_dim_kl': per_dim_kl,
        }






def adaln_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.norm_msa = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(6, dim=-1)
        attn_input = adaln_modulate(self.norm_msa(tokens), shift_msa, scale_msa)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + gate_msa.unsqueeze(1) * attn_output

        mlp_input = adaln_modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        if key_padding_mask is not None:
            tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return tokens








FLAT_SELF_ATTN_ADALN_ZERO = 'flat_self_attn_adaln_zero'
FACTORIZED_QUERY_CROSS_V1 = 'factorized_query_cross_v1'
