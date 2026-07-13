from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from einops import rearrange

from ditar_encoder.encoders import vjepa2_fullgrid_common as common
from ditar_encoder.lam.defaults import DEFAULT_BOTTLENECK_DIM, DEFAULT_BOTTLENECK_HIDDEN_DIM, DEFAULT_CONV_CHANNELS, DEFAULT_DEC_BLOCKS, DEFAULT_ENC_BLOCKS, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS, _group_norm_groups


TARGET_MODE_ABSOLUTE = 'absolute'
SPATIAL_GRID_ENCODER = 'spatial_grid_v1'
QUERYSET_CROSS_ENCODER = 'queryset_cross_v1'
DECODER_ARCH_ANCHOR_FULL_ATTN = 'anchor_full_attn_v1'
DECODER_ARCH_SOURCE_ADDITIVE = 'source_additive_v1'
DECODER_ARCH_SOURCE_QUERY_SELF_ATTN = 'source_query_self_attn_v1'
KL_SCOPE_TARGET = 'target'
KL_SCOPE_ALL = 'all'


class SpatialOnlyLatentReducer3D(nn.Module):
    """WAN-style compact reducer that preserves the teacher temporal axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        norm_groups = _group_norm_groups(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_encoder_arch(encoder_arch: str) -> str:
    normalized = str(encoder_arch or SPATIAL_GRID_ENCODER).strip().lower()
    if normalized not in {SPATIAL_GRID_ENCODER, QUERYSET_CROSS_ENCODER}:
        raise ValueError(
            f'Unsupported encoder_arch={encoder_arch!r}. Expected '
            f'{SPATIAL_GRID_ENCODER!r} or {QUERYSET_CROSS_ENCODER!r}.'
        )
    return normalized


def normalize_decoder_arch(decoder_arch: str) -> str:
    normalized = str(decoder_arch or DECODER_ARCH_ANCHOR_FULL_ATTN).strip().lower()
    valid = {
        DECODER_ARCH_ANCHOR_FULL_ATTN,
        DECODER_ARCH_SOURCE_ADDITIVE,
        DECODER_ARCH_SOURCE_QUERY_SELF_ATTN,
    }
    if normalized not in valid:
        raise ValueError(
            f'Unsupported decoder_arch={decoder_arch!r}. Expected one of '
            f'{sorted(valid)!r}.'
        )
    return normalized


def normalize_kl_scope(kl_scope: str) -> str:
    normalized = str(kl_scope or KL_SCOPE_TARGET).strip().lower()
    if normalized not in {KL_SCOPE_TARGET, KL_SCOPE_ALL}:
        raise ValueError(f'Unsupported kl_scope={kl_scope!r}. Expected "target" or "all".')
    return normalized


def aggregate_token_kl(
    per_dim_kl: torch.Tensor,
    target_step_indices: torch.Tensor,
    kl_scope: str,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if per_dim_kl.ndim != 4:
        raise ValueError(f'Expected per_dim_kl [B, T, Q, D], got {tuple(per_dim_kl.shape)}.')
    target_step_indices = target_step_indices.to(device=per_dim_kl.device, dtype=torch.long)
    if target_step_indices.ndim != 1 or target_step_indices.shape[0] != per_dim_kl.shape[0]:
        raise ValueError(
            'Expected target_step_indices [B], '
            f'got {tuple(target_step_indices.shape)} for batch_size={per_dim_kl.shape[0]}.'
        )

    per_token_kl = per_dim_kl.sum(dim=-1)
    target_token_kl = gather_step_tokens(per_token_kl.unsqueeze(-1), target_step_indices).squeeze(-1)
    kl_target_loss = target_token_kl.mean()
    kl_all_loss = per_token_kl.mean()
    normalized_scope = normalize_kl_scope(kl_scope)
    kl_loss = kl_target_loss if normalized_scope == KL_SCOPE_TARGET else kl_all_loss
    return kl_loss, {
        'kl_target_loss': kl_target_loss,
        'kl_all_loss': kl_all_loss,
    }


class QuerySetCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.visual_cross_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross_query = nn.LayerNorm(model_dim)
        self.norm_cross_memory = nn.LayerNorm(model_dim)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def _memory_with_optional_text(
        self,
        visual_memory: torch.Tensor,
        text_memory: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_steps, num_visual_tokens = visual_memory.shape[:3]
        if text_memory is None:
            return visual_memory, None
        if text_mask is None:
            text_mask = torch.ones(text_memory.shape[:2], dtype=torch.bool, device=text_memory.device)
        text_memory = text_memory.to(device=visual_memory.device, dtype=visual_memory.dtype)
        repeated_text = text_memory.unsqueeze(1).expand(-1, num_steps, -1, -1)
        memory = torch.cat([visual_memory, repeated_text], dim=2)

        visual_mask = torch.zeros(
            batch_size,
            num_steps,
            num_visual_tokens,
            dtype=torch.bool,
            device=visual_memory.device,
        )
        repeated_text_mask = ~text_mask.to(device=visual_memory.device, dtype=torch.bool).unsqueeze(1).expand(
            -1,
            num_steps,
            -1,
        )
        memory_padding_mask = torch.cat([visual_mask, repeated_text_mask], dim=2)
        return memory, rearrange(memory_padding_mask, 'b t p -> (b t) p')

    def forward(
        self,
        query_tokens: torch.Tensor,
        visual_memory: torch.Tensor,
        text_memory: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_steps, query_count, model_dim = query_tokens.shape
        memory, memory_padding_mask = self._memory_with_optional_text(visual_memory, text_memory, text_mask)

        query_flat = rearrange(self.norm_cross_query(query_tokens), 'b t q d -> (b t) q d')
        memory_flat = rearrange(self.norm_cross_memory(memory), 'b t p d -> (b t) p d')
        cross_output, _ = self.visual_cross_attn(
            query_flat,
            memory_flat,
            memory_flat,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + rearrange(cross_output, '(b t) q d -> b t q d', b=batch_size, t=num_steps)

        self_query = rearrange(self.norm_self(query_tokens), 'b t q d -> b (t q) d')
        self_output, _ = self.query_self_attn(
            self_query,
            self_query,
            self_query,
            need_weights=False,
        )
        query_tokens = query_tokens + rearrange(
            self_output,
            'b (t q) d -> b t q d',
            t=num_steps,
            q=query_count,
        )
        query_tokens = query_tokens + self.ffn(self.norm_ffn(query_tokens))
        return query_tokens


class QuerySetVideoEncoder(nn.Module):
    QUERY_STEP_EMBED_ADDITIVE = 'additive'
    QUERY_STEP_EMBED_DIRECT = 'direct'
    _VALID_QUERY_STEP_EMBED_MODES = {QUERY_STEP_EMBED_ADDITIVE, QUERY_STEP_EMBED_DIRECT}

    def __init__(
        self,
        context_dim: int,
        model_dim: int,
        num_steps: int,
        query_num_tokens: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        text_dim: Optional[int] = None,
        max_text_tokens: int = 128,
        project_context: bool = True,
        query_step_embed_mode: str = QUERY_STEP_EMBED_ADDITIVE,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.model_dim = int(model_dim)
        self.num_steps = int(num_steps)
        self.query_num_tokens = int(query_num_tokens)
        self.max_text_tokens = int(max_text_tokens)
        if self.query_num_tokens <= 0:
            raise ValueError(f'query_num_tokens must be positive, got {self.query_num_tokens}.')
        normalized_mode = str(query_step_embed_mode or self.QUERY_STEP_EMBED_ADDITIVE).strip().lower()
        if normalized_mode not in self._VALID_QUERY_STEP_EMBED_MODES:
            raise ValueError(
                f'query_step_embed_mode must be one of {sorted(self._VALID_QUERY_STEP_EMBED_MODES)}, '
                f'got {query_step_embed_mode!r}.'
            )
        self.query_step_embed_mode = normalized_mode

        if project_context or self.context_dim != self.model_dim:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
            )
        else:
            self.context_proj = nn.Identity()

        if self.query_step_embed_mode == self.QUERY_STEP_EMBED_DIRECT:
            self.query_step_embed = nn.Parameter(
                torch.zeros(1, self.num_steps, self.query_num_tokens, self.model_dim)
            )
        else:
            self.query_embed = nn.Parameter(torch.zeros(1, 1, self.query_num_tokens, self.model_dim))
            self.step_embed = nn.Parameter(torch.zeros(1, self.num_steps, 1, self.model_dim))
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.query_proj = nn.Linear(self.model_dim, self.model_dim)

        self.text_proj = None
        self.text_position_embed = None
        self.text_type_embed = None
        if text_dim is not None:
            self.text_proj = nn.Sequential(nn.LayerNorm(int(text_dim)), nn.Linear(int(text_dim), self.model_dim))
            self.text_position_embed = nn.Parameter(torch.zeros(1, self.max_text_tokens, self.model_dim))
            self.text_type_embed = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        self.blocks = nn.ModuleList(
            [
                QuerySetCrossAttentionBlock(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(self.model_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.query_step_embed_mode == self.QUERY_STEP_EMBED_DIRECT:
            nn.init.normal_(self.query_step_embed, std=0.02)
        else:
            nn.init.normal_(self.query_embed, std=0.02)
            nn.init.normal_(self.step_embed, std=0.02)
        if self.text_position_embed is not None:
            nn.init.normal_(self.text_position_embed, std=0.02)
        if self.text_type_embed is not None:
            nn.init.normal_(self.text_type_embed, std=0.02)

    def _query_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.query_step_embed_mode == self.QUERY_STEP_EMBED_DIRECT:
            query_tokens = self.query_step_embed
        else:
            query_tokens = self.query_embed + self.step_embed
        query_tokens = self.query_proj(self.query_norm(query_tokens))
        return query_tokens.expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype)

    def _project_text(
        self,
        text_tokens: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if text_tokens is None:
            return None, None
        if self.text_proj is None or self.text_position_embed is None or self.text_type_embed is None:
            raise ValueError('This QuerySetVideoEncoder was created without text conditioning.')
        if text_tokens.ndim != 3:
            raise ValueError(f'Expected text_tokens [B, L, D], got {tuple(text_tokens.shape)}.')
        if text_tokens.shape[1] > self.max_text_tokens:
            raise ValueError(
                f'Text token count {text_tokens.shape[1]} exceeds max_text_tokens={self.max_text_tokens}.'
            )
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        projected = self.text_proj(text_tokens.to(device=device, dtype=self.text_proj[1].weight.dtype))
        projected = projected.to(device=device, dtype=dtype)
        projected = projected + self.text_position_embed[:, : projected.shape[1]].to(device=device, dtype=dtype)
        projected = projected + self.text_type_embed.to(device=device, dtype=dtype)
        return projected, text_mask.to(device=device, dtype=torch.bool)

    def forward(
        self,
        context_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_tokens.ndim != 4:
            raise ValueError(f'Expected context_tokens [B, T, P, C], got {tuple(context_tokens.shape)}.')
        if context_tokens.shape[1] != self.num_steps:
            raise ValueError(f'Expected {self.num_steps} context steps, got {context_tokens.shape[1]}.')
        context = self.context_proj(context_tokens)
        query_tokens = self._query_tokens(
            batch_size=context.shape[0],
            device=context.device,
            dtype=context.dtype,
        )
        text_memory, text_mask = self._project_text(text_tokens, text_mask, context.device, context.dtype)
        for block in self.blocks:
            query_tokens = block(
                query_tokens=query_tokens,
                visual_memory=context,
                text_memory=text_memory,
                text_mask=text_mask,
            )
        return self.out_norm(query_tokens)


class AnchorStepFullAttentionDecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.query_self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.memory_cross_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_cross_query = nn.LayerNorm(model_dim)
        self.norm_cross_memory = nn.LayerNorm(model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_tokens = query_tokens + self.query_self_attn(
            self.norm_self(query_tokens),
            self.norm_self(query_tokens),
            self.norm_self(query_tokens),
            need_weights=False,
        )[0]
        query_tokens = query_tokens + self.memory_cross_attn(
            self.norm_cross_query(query_tokens),
            self.norm_cross_memory(memory_tokens),
            self.norm_cross_memory(memory_tokens),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        query_tokens = query_tokens + self.ffn(self.norm_ffn(query_tokens))
        return query_tokens


class AnchorStepFullAttentionDecoder(nn.Module):
    SOURCE_TOKEN_TYPE = 0
    TI_TOKEN_TYPE = 1
    TC_TOKEN_TYPE = 2
    TEXT_TOKEN_TYPE = 3

    def __init__(
        self,
        model_dim: int,
        out_dim: int,
        source_dim: int,
        ti_dim: int,
        tc_dim: int,
        text_dim: int,
        num_steps: int,
        source_num_patches: int,
        reduced_num_patches: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        max_text_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.out_dim = int(out_dim)
        self.num_steps = int(num_steps)
        self.source_num_patches = int(source_num_patches)
        self.reduced_num_patches = int(reduced_num_patches)
        self.max_text_tokens = int(max_text_tokens)
        if self.num_steps < 2:
            raise ValueError(f'Predictive decoder expects at least two teacher steps, got {self.num_steps}.')

        self.query_base = nn.Parameter(torch.zeros(1, self.source_num_patches, self.model_dim))
        self.query_patch_embed = nn.Parameter(torch.zeros(1, self.source_num_patches, self.model_dim))
        self.source_patch_embed = nn.Parameter(torch.zeros(1, self.source_num_patches, self.model_dim))
        self.reduced_patch_embed = nn.Parameter(torch.zeros(1, self.reduced_num_patches, self.model_dim))
        self.text_position_embed = nn.Parameter(torch.zeros(1, self.max_text_tokens, self.model_dim))
        self.target_step_embed = nn.Embedding(self.num_steps, self.model_dim)
        self.token_type_embed = nn.Embedding(4, self.model_dim)

        self.source_proj = nn.Sequential(nn.LayerNorm(source_dim), nn.Linear(source_dim, self.model_dim))
        self.ti_proj = nn.Sequential(nn.LayerNorm(ti_dim), nn.Linear(ti_dim, self.model_dim))
        self.tc_proj = nn.Sequential(nn.LayerNorm(tc_dim), nn.Linear(tc_dim, self.model_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, self.model_dim))
        self.blocks = nn.ModuleList(
            [
                AnchorStepFullAttentionDecoderBlock(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(dec_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.out_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in (
            self.query_base,
            self.query_patch_embed,
            self.source_patch_embed,
            self.reduced_patch_embed,
            self.text_position_embed,
        ):
            nn.init.normal_(parameter, std=0.02)
        nn.init.normal_(self.target_step_embed.weight, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _validate_target_steps(self, target_step_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        target_step_indices = target_step_indices.to(dtype=torch.long)
        if target_step_indices.ndim != 1 or target_step_indices.shape[0] != batch_size:
            raise ValueError(
                'Expected target_step_indices to have shape [B], '
                f'got {tuple(target_step_indices.shape)} for batch_size={batch_size}.'
            )
        if torch.any(target_step_indices < 1) or torch.any(target_step_indices >= self.num_steps):
            raise ValueError(f'target_step_indices must be in [1, {self.num_steps - 1}].')
        return target_step_indices

    def _typed(self, tokens: torch.Tensor, token_type: int) -> torch.Tensor:
        type_embed = self.token_type_embed.weight[token_type].to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
        return tokens + type_embed

    def _query_tokens(
        self,
        batch_size: int,
        target_step_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        step_embed = self.target_step_embed(target_step_indices.to(device=device)).to(dtype=dtype).unsqueeze(1)
        query_tokens = self.query_base.to(device=device, dtype=dtype) + self.query_patch_embed.to(device=device, dtype=dtype)
        query_tokens = query_tokens.expand(batch_size, -1, -1) + step_embed
        return self._typed(query_tokens, self.SOURCE_TOKEN_TYPE)

    def _source_memory(self, source_tokens: torch.Tensor, target_step_indices: torch.Tensor) -> torch.Tensor:
        projected = self.source_proj(source_tokens)
        step_embed = self.target_step_embed(target_step_indices.to(device=projected.device)).to(
            dtype=projected.dtype
        ).unsqueeze(1)
        projected = projected + self.source_patch_embed.to(device=projected.device, dtype=projected.dtype)
        return self._typed(projected + step_embed, self.SOURCE_TOKEN_TYPE)

    def _reduced_memory(self, tokens: torch.Tensor, projection: nn.Module, token_type: int) -> torch.Tensor:
        projected = projection(tokens)
        projected = projected + self.reduced_patch_embed.to(device=projected.device, dtype=projected.dtype)
        return self._typed(projected, token_type)

    def _text_memory(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if text_tokens.shape[1] > self.max_text_tokens:
            raise ValueError(
                f'Text token count {text_tokens.shape[1]} exceeds decoder max_text_tokens={self.max_text_tokens}.'
            )
        projected = self.text_proj(text_tokens)
        position = self.text_position_embed[:, : projected.shape[1]].to(device=projected.device, dtype=projected.dtype)
        return self._typed(projected + position, self.TEXT_TOKEN_TYPE)

    def forward(
        self,
        source_tokens: torch.Tensor,
        ti_tokens_t: torch.Tensor,
        target_step_indices: torch.Tensor,
        tc_tokens_t: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if source_tokens.ndim != 3:
            raise ValueError(f'Expected source_tokens [B, P, C], got {tuple(source_tokens.shape)}.')
        if ti_tokens_t.ndim != 3:
            raise ValueError(f'Expected ti_tokens_t [B, P_reduced, C], got {tuple(ti_tokens_t.shape)}.')
        if source_tokens.shape[1] != self.source_num_patches:
            raise ValueError(
                f'Expected {self.source_num_patches} source patches, got {source_tokens.shape[1]}.'
            )
        if ti_tokens_t.shape[1] != self.reduced_num_patches:
            raise ValueError(
                f'Expected {self.reduced_num_patches} TI patches, got {ti_tokens_t.shape[1]}.'
            )
        batch_size = source_tokens.shape[0]
        target_step_indices = self._validate_target_steps(target_step_indices, batch_size).to(source_tokens.device)

        query_tokens = self._query_tokens(
            batch_size=batch_size,
            target_step_indices=target_step_indices,
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        memory_groups = [
            self._source_memory(source_tokens, target_step_indices),
            self._reduced_memory(ti_tokens_t, self.ti_proj, self.TI_TOKEN_TYPE),
        ]
        padding_masks = [
            torch.zeros(batch_size, self.source_num_patches, dtype=torch.bool, device=source_tokens.device),
            torch.zeros(batch_size, self.reduced_num_patches, dtype=torch.bool, device=source_tokens.device),
        ]

        if tc_tokens_t is not None:
            if tc_tokens_t.ndim != 3 or tc_tokens_t.shape[:2] != ti_tokens_t.shape[:2]:
                raise ValueError(
                    'Expected tc_tokens_t to match TI batch/patch shape, '
                    f'got {tuple(tc_tokens_t.shape)} vs {tuple(ti_tokens_t.shape)}.'
                )
            memory_groups.append(self._reduced_memory(tc_tokens_t, self.tc_proj, self.TC_TOKEN_TYPE))
            padding_masks.append(
                torch.zeros(batch_size, self.reduced_num_patches, dtype=torch.bool, device=source_tokens.device)
            )

        if text_tokens is not None:
            if text_tokens.ndim != 3 or text_tokens.shape[0] != batch_size:
                raise ValueError(f'Expected text_tokens [B, L, C], got {tuple(text_tokens.shape)}.')
            text_memory = self._text_memory(text_tokens.to(device=source_tokens.device))
            memory_groups.append(text_memory)
            if text_mask is None:
                text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=source_tokens.device)
            padding_masks.append(~text_mask.to(device=source_tokens.device, dtype=torch.bool))

        memory = torch.cat(memory_groups, dim=1)
        memory_padding_mask = torch.cat(padding_masks, dim=1)
        for block in self.blocks:
            query_tokens = block(query_tokens, memory, memory_padding_mask)
        return self.out_proj(self.out_norm(query_tokens))


class SourceAdditiveConditionAdapter(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        model_dim: int,
        num_heads: int,
        max_condition_tokens: int,
        dropout: float = 0.0,
        exact_condition_tokens: Optional[int] = None,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.model_dim = int(model_dim)
        self.max_condition_tokens = int(max_condition_tokens)
        self.exact_condition_tokens = None if exact_condition_tokens is None else int(exact_condition_tokens)
        if self.max_condition_tokens <= 0:
            raise ValueError(f'max_condition_tokens must be positive, got {self.max_condition_tokens}.')
        hidden_dim = int(self.model_dim * mlp_ratio)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, self.model_dim),
        )
        self.position_embed = nn.Parameter(torch.zeros(1, self.max_condition_tokens, self.model_dim))
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.condition_norm = nn.LayerNorm(self.model_dim)
        self.cross_attn = nn.MultiheadAttention(self.model_dim, num_heads, dropout=dropout, batch_first=True)
        self.delta_norm = nn.LayerNorm(self.model_dim)
        self.delta_ffn = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.model_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embed, std=0.02)

    @property
    def _projection_dtype(self) -> torch.dtype:
        return self.condition_proj[1].weight.dtype

    def _project_condition(
        self,
        condition_tokens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if condition_tokens.ndim != 3:
            raise ValueError(f'Expected condition tokens [B, P, C], got {tuple(condition_tokens.shape)}.')
        token_count = int(condition_tokens.shape[1])
        if self.exact_condition_tokens is not None and token_count != self.exact_condition_tokens:
            raise ValueError(f'Expected {self.exact_condition_tokens} condition tokens, got {token_count}.')
        if token_count > self.max_condition_tokens:
            raise ValueError(
                f'Condition token count {token_count} exceeds max_condition_tokens={self.max_condition_tokens}.'
            )
        projected = self.condition_proj(condition_tokens.to(device=device, dtype=self._projection_dtype))
        position = self.position_embed[:, :token_count].to(device=device, dtype=projected.dtype)
        return projected + position

    @staticmethod
    def _safe_padding_mask(
        condition: torch.Tensor,
        condition_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if condition_mask is None:
            return None
        if condition_mask.shape != condition.shape[:2]:
            raise ValueError(
                f'Expected condition_mask shape {tuple(condition.shape[:2])}, got {tuple(condition_mask.shape)}.'
            )
        valid_mask = condition_mask.to(device=condition.device, dtype=torch.bool)
        if condition.shape[1] == 0:
            return None
        key_padding_mask = ~valid_mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
            condition[all_masked, 0] = 0
        return key_padding_mask

    def forward(
        self,
        source_base: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self._project_condition(condition_tokens, source_base.device)
        if condition.shape[1] == 0:
            return source_base.new_zeros(source_base.shape)
        key_padding_mask = self._safe_padding_mask(condition, condition_mask)
        condition = self.condition_norm(condition)
        delta, _ = self.cross_attn(
            self.query_norm(source_base),
            condition,
            condition,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        delta = delta + self.delta_ffn(self.delta_norm(delta))
        return delta.to(dtype=source_base.dtype)


class SourceAdditiveDecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm_self(tokens)
        tokens = tokens + self.self_attn(normed, normed, normed, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens


class SourceAdditiveActionDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        out_dim: int,
        source_dim: int,
        ti_dim: int,
        tc_dim: int,
        text_dim: int,
        num_steps: int,
        source_num_patches: int,
        reduced_num_patches: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        max_text_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.out_dim = int(out_dim)
        self.source_dim = int(source_dim)
        self.num_steps = int(num_steps)
        self.source_num_patches = int(source_num_patches)
        self.reduced_num_patches = int(reduced_num_patches)
        self.max_text_tokens = int(max_text_tokens)
        if self.source_dim != self.out_dim:
            raise ValueError(
                'Source-additive decoder predicts source residuals and expects source_dim == out_dim, '
                f'got source_dim={self.source_dim}, out_dim={self.out_dim}.'
            )
        if self.num_steps < 2:
            raise ValueError(f'Predictive decoder expects at least two teacher steps, got {self.num_steps}.')

        self.source_proj = nn.Sequential(nn.LayerNorm(self.source_dim), nn.Linear(self.source_dim, self.model_dim))
        self.source_patch_embed = nn.Parameter(torch.zeros(1, self.source_num_patches, self.model_dim))
        self.target_step_embed = nn.Embedding(self.num_steps, self.model_dim)
        self.ti_adapter = SourceAdditiveConditionAdapter(
            condition_dim=ti_dim,
            model_dim=self.model_dim,
            num_heads=num_heads,
            max_condition_tokens=self.reduced_num_patches,
            exact_condition_tokens=self.reduced_num_patches,
            dropout=dropout,
        )
        self.tc_adapter = SourceAdditiveConditionAdapter(
            condition_dim=tc_dim,
            model_dim=self.model_dim,
            num_heads=num_heads,
            max_condition_tokens=self.reduced_num_patches,
            exact_condition_tokens=self.reduced_num_patches,
            dropout=dropout,
        )
        self.text_adapter = SourceAdditiveConditionAdapter(
            condition_dim=text_dim,
            model_dim=self.model_dim,
            num_heads=num_heads,
            max_condition_tokens=self.max_text_tokens,
            exact_condition_tokens=None,
            dropout=dropout,
        )
        self.ti_gate = nn.Parameter(torch.ones(()))
        self.tc_gate = nn.Parameter(torch.ones(()))
        self.text_gate = nn.Parameter(torch.ones(()))
        self.fuse_norm = nn.LayerNorm(self.model_dim)
        self.blocks = nn.ModuleList(
            [
                SourceAdditiveDecoderBlock(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(dec_blocks)
            ]
        )
        self.residual_proj = nn.Linear(self.model_dim, self.out_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.source_patch_embed, std=0.02)
        nn.init.normal_(self.target_step_embed.weight, std=0.02)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    @property
    def _projection_dtype(self) -> torch.dtype:
        return self.source_proj[1].weight.dtype

    def _validate_target_steps(self, target_step_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        target_step_indices = target_step_indices.to(dtype=torch.long)
        if target_step_indices.ndim != 1 or target_step_indices.shape[0] != batch_size:
            raise ValueError(
                'Expected target_step_indices to have shape [B], '
                f'got {tuple(target_step_indices.shape)} for batch_size={batch_size}.'
            )
        if torch.any(target_step_indices < 1) or torch.any(target_step_indices >= self.num_steps):
            raise ValueError(f'target_step_indices must be in [1, {self.num_steps - 1}].')
        return target_step_indices

    def _source_base(self, source_tokens: torch.Tensor, target_step_indices: torch.Tensor) -> torch.Tensor:
        projected = self.source_proj(source_tokens.to(dtype=self._projection_dtype))
        step_embed = self.target_step_embed(target_step_indices.to(device=projected.device)).to(dtype=projected.dtype)
        projected = projected + self.source_patch_embed.to(device=projected.device, dtype=projected.dtype)
        return projected + step_embed.unsqueeze(1)

    def forward(
        self,
        source_tokens: torch.Tensor,
        ti_tokens_t: torch.Tensor,
        target_step_indices: torch.Tensor,
        tc_tokens_t: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if source_tokens.ndim != 3:
            raise ValueError(f'Expected source_tokens [B, P, C], got {tuple(source_tokens.shape)}.')
        if ti_tokens_t.ndim != 3:
            raise ValueError(f'Expected ti_tokens_t [B, P_reduced, C], got {tuple(ti_tokens_t.shape)}.')
        if source_tokens.shape[1] != self.source_num_patches:
            raise ValueError(
                f'Expected {self.source_num_patches} source patches, got {source_tokens.shape[1]}.'
            )
        if ti_tokens_t.shape[1] != self.reduced_num_patches:
            raise ValueError(
                f'Expected {self.reduced_num_patches} TI patches, got {ti_tokens_t.shape[1]}.'
            )
        batch_size = source_tokens.shape[0]
        target_step_indices = self._validate_target_steps(target_step_indices, batch_size).to(source_tokens.device)
        source_base = self._source_base(source_tokens, target_step_indices)

        ti_delta = self.ti_adapter(source_base, ti_tokens_t.to(device=source_tokens.device))
        tc_delta = None
        if tc_tokens_t is not None:
            if tc_tokens_t.ndim != 3 or tc_tokens_t.shape[:2] != ti_tokens_t.shape[:2]:
                raise ValueError(
                    'Expected tc_tokens_t to match TI batch/patch shape, '
                    f'got {tuple(tc_tokens_t.shape)} vs {tuple(ti_tokens_t.shape)}.'
                )
            tc_delta = self.tc_adapter(source_base, tc_tokens_t.to(device=source_tokens.device))

        text_delta = None
        if text_tokens is not None:
            if text_tokens.ndim != 3 or text_tokens.shape[0] != batch_size:
                raise ValueError(f'Expected text_tokens [B, L, C], got {tuple(text_tokens.shape)}.')
            if text_mask is None:
                text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
            text_delta = self.text_adapter(
                source_base,
                text_tokens.to(device=source_tokens.device),
                condition_mask=text_mask.to(device=source_tokens.device, dtype=torch.bool),
            )

        fused = source_base + self.ti_gate.to(dtype=source_base.dtype) * ti_delta
        if text_delta is not None:
            fused = fused + self.text_gate.to(dtype=source_base.dtype) * text_delta
        if tc_delta is not None:
            fused = fused + self.tc_gate.to(dtype=source_base.dtype) * tc_delta
        decoded = self.fuse_norm(fused)
        for block in self.blocks:
            decoded = block(decoded)
        residual = self.residual_proj(decoded).to(dtype=source_tokens.dtype)
        return {
            'pred_latent_t': source_tokens + residual,
            'ti_delta': ti_delta,
            'tc_delta': tc_delta,
            'text_delta': text_delta,
        }


class SourceQuerySelfAttentionDecoder(nn.Module):
    """UniVLA / DreamDojo-style MAE decoder.

    Source V-JEPA tokens carry the per-position content of the prediction queries; TI / TC /
    text tokens are appended as additional sequence positions. ``dec_blocks`` of pure
    ``AdaLNZeroSelfAttentionBlock`` self-attention propagate information across the whole
    sequence. The future-step prediction is read from the source-aligned slots and added
    back to ``source_tokens`` as a zero-init residual, so the model predicts source-copy at
    initialization (matches UniVLA's MAE prior, which lets a 20-bit latent reach ~0.88
    DINO cosine).
    """

    SOURCE_TOKEN_TYPE = 0
    TI_TOKEN_TYPE = 1
    TC_TOKEN_TYPE = 2
    TEXT_TOKEN_TYPE = 3

    def __init__(
        self,
        model_dim: int,
        out_dim: int,
        source_dim: int,
        ti_dim: int,
        tc_dim: int,
        text_dim: int,
        num_steps: int,
        source_num_patches: int,
        reduced_num_patches: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        max_text_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.out_dim = int(out_dim)
        self.source_dim = int(source_dim)
        self.num_steps = int(num_steps)
        self.source_num_patches = int(source_num_patches)
        self.reduced_num_patches = int(reduced_num_patches)
        self.max_text_tokens = int(max_text_tokens)
        if self.source_dim != self.out_dim:
            raise ValueError(
                'Source-query self-attn decoder predicts a residual on the source feature space '
                f'and expects source_dim == out_dim, got source_dim={self.source_dim}, out_dim={self.out_dim}.'
            )
        if self.num_steps < 2:
            raise ValueError(f'Predictive decoder expects at least two teacher steps, got {self.num_steps}.')

        self.source_proj = nn.Sequential(nn.LayerNorm(self.source_dim), nn.Linear(self.source_dim, self.model_dim))
        self.ti_proj = nn.Sequential(nn.LayerNorm(ti_dim), nn.Linear(ti_dim, self.model_dim))
        self.tc_proj = nn.Sequential(nn.LayerNorm(tc_dim), nn.Linear(tc_dim, self.model_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, self.model_dim))

        self.source_patch_embed = nn.Parameter(torch.zeros(1, self.source_num_patches, self.model_dim))
        self.ti_patch_embed = nn.Parameter(torch.zeros(1, self.reduced_num_patches, self.model_dim))
        self.tc_patch_embed = nn.Parameter(torch.zeros(1, self.reduced_num_patches, self.model_dim))
        self.text_position_embed = nn.Parameter(torch.zeros(1, self.max_text_tokens, self.model_dim))
        self.target_step_embed = nn.Embedding(self.num_steps, self.model_dim)
        self.token_type_embed = nn.Embedding(4, self.model_dim)

        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                common.AdaLNZeroSelfAttentionBlock(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(dec_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(self.model_dim, elementwise_affine=False)
        self.residual_proj = nn.Linear(self.model_dim, self.out_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in (
            self.source_patch_embed,
            self.ti_patch_embed,
            self.tc_patch_embed,
            self.text_position_embed,
        ):
            nn.init.normal_(parameter, std=0.02)
        nn.init.normal_(self.target_step_embed.weight, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    @property
    def _projection_dtype(self) -> torch.dtype:
        return self.source_proj[1].weight.dtype

    def _validate_target_steps(self, target_step_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        target_step_indices = target_step_indices.to(dtype=torch.long)
        if target_step_indices.ndim != 1 or target_step_indices.shape[0] != batch_size:
            raise ValueError(
                'Expected target_step_indices to have shape [B], '
                f'got {tuple(target_step_indices.shape)} for batch_size={batch_size}.'
            )
        if torch.any(target_step_indices < 1) or torch.any(target_step_indices >= self.num_steps):
            raise ValueError(f'target_step_indices must be in [1, {self.num_steps - 1}].')
        return target_step_indices

    def _typed(self, tokens: torch.Tensor, token_type: int) -> torch.Tensor:
        type_embed = self.token_type_embed.weight[token_type].to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
        return tokens + type_embed

    def _source_sequence(
        self,
        source_tokens: torch.Tensor,
        target_step_indices: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.source_proj(source_tokens.to(dtype=self._projection_dtype))
        step_embed = self.target_step_embed(target_step_indices.to(device=projected.device)).to(dtype=projected.dtype)
        projected = projected + self.source_patch_embed.to(device=projected.device, dtype=projected.dtype)
        projected = projected + step_embed.unsqueeze(1)
        return self._typed(projected, self.SOURCE_TOKEN_TYPE)

    def _reduced_sequence(
        self,
        tokens: torch.Tensor,
        projection: nn.Module,
        patch_embed: torch.Tensor,
        token_type: int,
    ) -> torch.Tensor:
        projected = projection(tokens.to(dtype=self._projection_dtype))
        projected = projected + patch_embed.to(device=projected.device, dtype=projected.dtype)
        return self._typed(projected, token_type)

    def _text_sequence(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if text_tokens.shape[1] > self.max_text_tokens:
            raise ValueError(
                f'Text token count {text_tokens.shape[1]} exceeds decoder max_text_tokens={self.max_text_tokens}.'
            )
        projected = self.text_proj(text_tokens.to(dtype=self._projection_dtype))
        position = self.text_position_embed[:, : projected.shape[1]].to(device=projected.device, dtype=projected.dtype)
        return self._typed(projected + position, self.TEXT_TOKEN_TYPE)

    def _conditioning(
        self,
        ti_tokens_t: torch.Tensor,
        target_step_indices: torch.Tensor,
        tc_tokens_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Text participates only in the self-attention sequence, not in the AdaLN conditioning,
        # so we keep the conditioning summary action-shaped (model_dim) without a separate text pool.
        target_dtype = self.condition_proj[1].weight.dtype
        pooled = ti_tokens_t.to(dtype=target_dtype).mean(dim=1)
        if tc_tokens_t is not None:
            pooled = pooled + tc_tokens_t.to(dtype=target_dtype).mean(dim=1)
        step_embed = self.target_step_embed(target_step_indices.to(device=pooled.device)).to(dtype=target_dtype)
        return self.condition_proj(pooled + step_embed)

    def forward(
        self,
        source_tokens: torch.Tensor,
        ti_tokens_t: torch.Tensor,
        target_step_indices: torch.Tensor,
        tc_tokens_t: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if source_tokens.ndim != 3:
            raise ValueError(f'Expected source_tokens [B, P, C], got {tuple(source_tokens.shape)}.')
        if ti_tokens_t.ndim != 3:
            raise ValueError(f'Expected ti_tokens_t [B, P_reduced, C], got {tuple(ti_tokens_t.shape)}.')
        if source_tokens.shape[1] != self.source_num_patches:
            raise ValueError(
                f'Expected {self.source_num_patches} source patches, got {source_tokens.shape[1]}.'
            )
        if ti_tokens_t.shape[1] != self.reduced_num_patches:
            raise ValueError(
                f'Expected {self.reduced_num_patches} TI patches, got {ti_tokens_t.shape[1]}.'
            )
        batch_size = source_tokens.shape[0]
        target_step_indices = self._validate_target_steps(target_step_indices, batch_size).to(source_tokens.device)

        source_sequence = self._source_sequence(source_tokens, target_step_indices)
        ti_sequence = self._reduced_sequence(
            ti_tokens_t.to(device=source_tokens.device),
            self.ti_proj,
            self.ti_patch_embed,
            self.TI_TOKEN_TYPE,
        )

        token_groups = [source_sequence, ti_sequence]
        key_padding_groups = [
            torch.zeros(batch_size, self.source_num_patches, dtype=torch.bool, device=source_tokens.device),
            torch.zeros(batch_size, self.reduced_num_patches, dtype=torch.bool, device=source_tokens.device),
        ]

        if tc_tokens_t is not None:
            if tc_tokens_t.ndim != 3 or tc_tokens_t.shape[:2] != ti_tokens_t.shape[:2]:
                raise ValueError(
                    'Expected tc_tokens_t to match TI batch/patch shape, '
                    f'got {tuple(tc_tokens_t.shape)} vs {tuple(ti_tokens_t.shape)}.'
                )
            tc_sequence = self._reduced_sequence(
                tc_tokens_t.to(device=source_tokens.device),
                self.tc_proj,
                self.tc_patch_embed,
                self.TC_TOKEN_TYPE,
            )
            token_groups.append(tc_sequence)
            key_padding_groups.append(
                torch.zeros(batch_size, self.reduced_num_patches, dtype=torch.bool, device=source_tokens.device)
            )

        if text_tokens is not None:
            if text_tokens.ndim != 3 or text_tokens.shape[0] != batch_size:
                raise ValueError(f'Expected text_tokens [B, L, C], got {tuple(text_tokens.shape)}.')
            text_sequence = self._text_sequence(text_tokens.to(device=source_tokens.device))
            if text_mask is None:
                text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=source_tokens.device)
            token_groups.append(text_sequence)
            key_padding_groups.append(~text_mask.to(device=source_tokens.device, dtype=torch.bool))

        sequence = torch.cat(token_groups, dim=1)
        key_padding_mask = torch.cat(key_padding_groups, dim=1)
        conditioning = self._conditioning(
            ti_tokens_t=ti_tokens_t,
            target_step_indices=target_step_indices,
            tc_tokens_t=tc_tokens_t,
        )

        for block in self.blocks:
            sequence = block(sequence, conditioning, key_padding_mask=key_padding_mask)

        source_decoded = sequence[:, : self.source_num_patches]
        residual = self.residual_proj(self.out_norm(source_decoded)).to(dtype=source_tokens.dtype)
        return {
            'pred_latent_t': source_tokens + residual,
            'residual': residual,
        }


def build_predictive_decoder(
    decoder_arch: str,
    model_dim: int,
    out_dim: int,
    source_dim: int,
    ti_dim: int,
    tc_dim: int,
    text_dim: int,
    num_steps: int,
    source_num_patches: int,
    reduced_num_patches: int,
    dec_blocks: int,
    num_heads: int,
    dropout: float,
    max_text_tokens: int,
) -> nn.Module:
    normalized = normalize_decoder_arch(decoder_arch)
    decoder_cls: type[nn.Module]
    if normalized == DECODER_ARCH_ANCHOR_FULL_ATTN:
        decoder_cls = AnchorStepFullAttentionDecoder
    elif normalized == DECODER_ARCH_SOURCE_ADDITIVE:
        decoder_cls = SourceAdditiveActionDecoder
    elif normalized == DECODER_ARCH_SOURCE_QUERY_SELF_ATTN:
        decoder_cls = SourceQuerySelfAttentionDecoder
    else:
        raise RuntimeError(f'Unsupported decoder_arch={decoder_arch!r}.')
    return decoder_cls(
        model_dim=model_dim,
        out_dim=out_dim,
        source_dim=source_dim,
        ti_dim=ti_dim,
        tc_dim=tc_dim,
        text_dim=text_dim,
        num_steps=num_steps,
        source_num_patches=source_num_patches,
        reduced_num_patches=reduced_num_patches,
        dec_blocks=dec_blocks,
        num_heads=num_heads,
        dropout=dropout,
        max_text_tokens=max_text_tokens,
    )


def gather_step_tokens(tokens: torch.Tensor, step_indices: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f'Expected tokens [B, T, P, C], got {tuple(tokens.shape)}.')
    step_indices = step_indices.to(device=tokens.device, dtype=torch.long)
    gather_shape = (tokens.shape[0], 1, tokens.shape[2], tokens.shape[3])
    gathered = tokens.gather(dim=1, index=step_indices.view(-1, 1, 1, 1).expand(gather_shape))
    return gathered.squeeze(1)


class _PredictiveCompactBase(nn.Module):
    def _tokens_to_volume(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        return rearrange(tokens, 'b t (h w) d -> b d t h w', h=grid_size, w=grid_size)

    @staticmethod
    def _volume_to_tokens(volume: torch.Tensor) -> torch.Tensor:
        return rearrange(volume, 'b d t h w -> b t (h w) d')

    def _reduce_teacher_tokens(self, teacher_latents: torch.Tensor) -> torch.Tensor:
        teacher_volume = self._tokens_to_volume(teacher_latents, grid_size=self.full_grid_size)
        projected = self.input_proj(teacher_volume)
        reduced_volume = self.reducer(projected)
        return self._volume_to_tokens(reduced_volume)

    def _encode_teacher_context(self, teacher_latents: torch.Tensor) -> torch.Tensor:
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            return self._reduce_teacher_tokens(teacher_latents)
        if self.encoder_arch == QUERYSET_CROSS_ENCODER:
            return self.input_proj(teacher_latents)
        raise RuntimeError(f'Unsupported encoder_arch={self.encoder_arch!r}.')

    def _aggregate_bottleneck_kl(
        self,
        bottleneck_outputs: Dict[str, torch.Tensor],
        target_step_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        per_dim_kl = bottleneck_outputs.get('per_dim_kl')
        if per_dim_kl is None:
            zero = target_step_indices.new_zeros((), dtype=torch.float32)
            return zero, {
                'kl_target_loss': zero,
                'kl_all_loss': zero,
            }
        return aggregate_token_kl(per_dim_kl, target_step_indices, self.kl_scope)

    @staticmethod
    def _unpack_decoder_outputs(
        decoder_outputs: torch.Tensor | Mapping[str, Optional[torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(decoder_outputs, Mapping):
            return decoder_outputs, {}
        pred_latent_t = decoder_outputs.get('pred_latent_t')
        if pred_latent_t is None:
            raise KeyError('Decoder output mapping is missing "pred_latent_t".')
        aux_outputs: Dict[str, torch.Tensor] = {}
        for source_key, output_key in (
            ('ti_delta', 'decoder_ti_delta'),
            ('tc_delta', 'decoder_tc_delta'),
            ('text_delta', 'decoder_text_delta'),
            ('residual', 'decoder_residual'),
        ):
            value = decoder_outputs.get(source_key)
            if value is not None:
                aux_outputs[output_key] = value
        return pred_latent_t, aux_outputs

    def _target_step_indices(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_latents: torch.Tensor,
    ) -> torch.Tensor:
        if 'target_step_indices' in batch:
            return batch['target_step_indices'].to(device=teacher_latents.device, dtype=torch.long)

        batch_size = int(teacher_latents.shape[0])
        valid_steps = torch.ones(
            batch_size,
            self.full_num_steps,
            dtype=torch.bool,
            device=teacher_latents.device,
        )
        valid_steps[:, 0] = False
        if 'pad_mask' in batch:
            pad_mask = batch['pad_mask'].to(device=teacher_latents.device, dtype=torch.bool)
            tubelet = int(self.teacher.tubelet_size)
            if pad_mask.shape[1] % tubelet != 0:
                raise ValueError(
                    f'Expected pad_mask length divisible by tubelet_size={tubelet}, got {pad_mask.shape[1]}.'
                )
            valid_steps = valid_steps & pad_mask.view(batch_size, -1, tubelet).all(dim=-1)
            valid_steps[:, 0] = False

        fallback = torch.randint(
            low=1,
            high=self.full_num_steps,
            size=(batch_size,),
            device=teacher_latents.device,
        )
        sampled = []
        for batch_idx in range(batch_size):
            candidate_steps = torch.nonzero(valid_steps[batch_idx], as_tuple=False).flatten()
            if candidate_steps.numel() == 0:
                sampled.append(fallback[batch_idx])
            else:
                sampled.append(candidate_steps[torch.randint(candidate_steps.numel(), (), device=teacher_latents.device)])
        return torch.stack(sampled).to(dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
