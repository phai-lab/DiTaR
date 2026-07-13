"""WAN VAE queryset predictive Stage-1 LAM (T5-base text conditioning).

A frozen WAN-2.2 VAE teacher feeds a query-set cross-attention encoder + KL
bottleneck + source-query self-attention decoder; instructions are conditioned
through a frozen T5-base encoder (``FrozenT5TextConditioner``). This is the
task-irrelevant (TI) Stage-1 latent-action model.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ditar_encoder.encoders import vjepa2_fullgrid_common as common
from ditar_encoder.encoders import wan_forecast_common
from ditar_encoder.lam.vjepa2_wanconv_compact_predictive_lam import DECODER_ARCH_SOURCE_QUERY_SELF_ATTN, QUERYSET_CROSS_ENCODER, QuerySetVideoEncoder, _PredictiveCompactBase, build_predictive_decoder, gather_step_tokens, normalize_kl_scope
from ditar_encoder.lam.defaults import DEFAULT_BOTTLENECK_DIM, DEFAULT_BOTTLENECK_HIDDEN_DIM, DEFAULT_DEC_BLOCKS, DEFAULT_ENC_BLOCKS, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS
from ditar_encoder.lam.wan_vae_queryset_predictive_lam import _validate_queryset_only, _validate_self_attn_decoder, _wan_vae_target_step_indices


class WanVaeQuerysetPredictiveT5Stage1LatentActionModel(
    _PredictiveCompactBase
):
    """WAN VAE queryset Stage-1 LAM with frozen T5-base text conditioning."""

    _CHECKPOINT_EXCLUDED_PREFIXES = ('lam.teacher.', 'lam.t5_conditioner.')

    def __init__(
        self,
        bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM,
        bottleneck_hidden_dim: int = DEFAULT_BOTTLENECK_HIDDEN_DIM,
        model_dim: int = DEFAULT_MODEL_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        enc_blocks: int = DEFAULT_ENC_BLOCKS,
        dec_blocks: int = DEFAULT_DEC_BLOCKS,
        bottleneck_regularizer: str = 'kl',
        bottleneck_post_norm: bool = True,
        lam_dropout: float = 0.1,
        encoder_arch: str = QUERYSET_CROSS_ENCODER,
        decoder_arch: str = DECODER_ARCH_SOURCE_QUERY_SELF_ATTN,
        query_num_tokens: int = 64,
        query_blocks: int = 8,
        kl_scope: str = 'target',
        wan_vae_model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        wan_vae_model_path: Optional[str] = None,
        wan_vae_subfolder: str = 'vae',
        wan_vae_dtype: str = 'float32',
        wan_vae_sample_mode: str = 'mode',
        wan_vae_normalize_latents: bool = True,
        teacher_input_resolution: int = 256,
        num_frames: int = 17,
        t5_model_id: str = 't5-base',
        t5_model_path: Optional[str] = None,
        t5_max_length: int = 64,
        text_cache_size: int = 0,
        max_decoder_text_tokens: int = 256,
    ) -> None:
        torch.nn.Module.__init__(self)
        _validate_queryset_only(encoder_arch)
        _validate_self_attn_decoder(decoder_arch)

        self.teacher = wan_forecast_common.FrozenWanLatentTeacher(
            wan_vae_model_id=wan_vae_model_id,
            wan_vae_model_path=wan_vae_model_path,
            wan_vae_subfolder=wan_vae_subfolder,
            wan_vae_dtype=wan_vae_dtype,
            wan_latent_sample_mode=wan_vae_sample_mode,
            wan_normalize_latents=wan_vae_normalize_latents,
            teacher_num_frames=num_frames,
            teacher_input_resolution=teacher_input_resolution,
        )
        self.t5_conditioner = common.FrozenT5TextConditioner(
            t5_model_id=t5_model_id,
            t5_model_path=t5_model_path,
            t5_max_length=t5_max_length,
            text_cache_size=text_cache_size,
        )
        self.t5_conditioner.requires_grad_(False)
        self.t5_conditioner.eval()

        self.bottleneck_dim = int(bottleneck_dim)
        self.bottleneck_hidden_dim = int(bottleneck_hidden_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.conv_channels = int(model_dim)
        self.enc_blocks = int(enc_blocks)
        self.dec_blocks = int(dec_blocks)
        self.lam_dropout = float(lam_dropout)
        self.encoder_arch = QUERYSET_CROSS_ENCODER
        self.decoder_arch = DECODER_ARCH_SOURCE_QUERY_SELF_ATTN
        self.query_num_tokens = int(query_num_tokens)
        self.query_blocks = int(query_blocks)
        self.kl_scope = normalize_kl_scope(kl_scope)

        self.full_num_steps = int(self.teacher.teacher_num_steps)
        self.full_grid_size = int(self.teacher.teacher_grid_size)
        self.reduced_num_steps = self.full_num_steps
        self.reduced_grid_size = 0
        self.reduced_num_patches = self.query_num_tokens

        self.input_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(self.teacher.teacher_feature_dim),
            torch.nn.Linear(self.teacher.teacher_feature_dim, self.model_dim),
            torch.nn.LayerNorm(self.model_dim),
        )
        self.reducer = torch.nn.Identity()
        self.ti_encoder = QuerySetVideoEncoder(
            context_dim=self.model_dim,
            model_dim=self.model_dim,
            num_steps=self.reduced_num_steps,
            query_num_tokens=self.reduced_num_patches,
            num_blocks=self.query_blocks,
            num_heads=self.num_heads,
            dropout=self.lam_dropout,
            text_dim=self.t5_conditioner.hidden_size,
            max_text_tokens=max_decoder_text_tokens,
            project_context=False,
        )
        self.ti_bottleneck = common.FullGridTokenBottleneck(
            model_dim=self.model_dim,
            bottleneck_dim=self.bottleneck_dim,
            bottleneck_hidden_dim=self.bottleneck_hidden_dim,
            bottleneck_regularizer=bottleneck_regularizer,
            post_norm_after_decode=bottleneck_post_norm,
        )
        self.decoder = build_predictive_decoder(
            decoder_arch=self.decoder_arch,
            model_dim=self.model_dim,
            out_dim=self.teacher.teacher_feature_dim,
            source_dim=self.teacher.teacher_feature_dim,
            ti_dim=self.model_dim,
            tc_dim=self.model_dim,
            text_dim=self.t5_conditioner.hidden_size,
            num_steps=self.full_num_steps,
            source_num_patches=self.teacher.teacher_num_patches,
            reduced_num_patches=self.reduced_num_patches,
            dec_blocks=self.dec_blocks,
            num_heads=self.num_heads,
            dropout=self.lam_dropout,
            max_text_tokens=max_decoder_text_tokens,
        )
        self.decoder.tc_proj.requires_grad_(False)
        self.decoder.tc_patch_embed.requires_grad_(False)

        self.ti_token_dropout_p = 0.0
        self.source_token_dropout_p = 0.0

    def _target_step_indices(self, batch, teacher_latents):
        return _wan_vae_target_step_indices(self, batch, teacher_latents)

    def _resolve_text_condition(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        if 't5_text_token_bank' in batch:
            text_tokens = batch['t5_text_token_bank'].to(device=self.device, dtype=torch.float32)
            if 't5_text_token_mask' in batch:
                text_mask = batch['t5_text_token_mask'].to(device=self.device, dtype=torch.bool)
            else:
                text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=self.device)
            return text_tokens, text_mask
        if 'task_instruction' not in batch:
            raise KeyError('WAN VAE queryset T5 stage-1 requires batch["task_instruction"].')
        with torch.no_grad():
            tokens, mask = self.t5_conditioner(batch['task_instruction'])
        return tokens.to(device=self.device, dtype=torch.float32), mask.to(device=self.device, dtype=torch.bool)

    def _maybe_drop_source_tokens(self, source_tokens: torch.Tensor) -> torch.Tensor:
        # Token-level Bernoulli dropout on the source patch axis (256 tokens
        # for a 16x16 grid at WAN VAE z_dim=48). Mask shape [B, P_source, 1]
        # so each source token is dropped as a unit; survivors rescaled by
        # 1 / (1 - p) (inverted-dropout). Only at training time.
        if not self.training or self.source_token_dropout_p <= 0.0:
            return source_tokens
        keep_prob = 1.0 - self.source_token_dropout_p
        mask = torch.empty(
            source_tokens.shape[0], source_tokens.shape[1], 1,
            device=source_tokens.device, dtype=source_tokens.dtype,
        ).bernoulli_(keep_prob)
        return source_tokens * mask / keep_prob

    def _maybe_drop_ti_tokens(self, ti_tokens_t: torch.Tensor) -> torch.Tensor:
        # Token-level Bernoulli dropout on the 64 TI query tokens, applied only
        # at training time. Mask shape [B, P_reduced, 1] so each token is dropped
        # as a unit; surviving tokens are rescaled by 1 / (1 - p) to preserve the
        # expected magnitude (standard inverted-dropout convention).
        if not self.training or self.ti_token_dropout_p <= 0.0:
            return ti_tokens_t
        keep_prob = 1.0 - self.ti_token_dropout_p
        mask = torch.empty(
            ti_tokens_t.shape[0], ti_tokens_t.shape[1], 1,
            device=ti_tokens_t.device, dtype=ti_tokens_t.dtype,
        ).bernoulli_(keep_prob)
        return ti_tokens_t * mask / keep_prob

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'videos' not in batch:
            raise KeyError('Predictive compact stage-1 path requires batch["videos"].')

        with torch.no_grad():
            teacher_latents = self.teacher.encode(batch['videos'])
        text_tokens, text_mask = self._resolve_text_condition(batch)
        reduced_teacher_tokens = self._encode_teacher_context(teacher_latents)
        ti_encoded = self.ti_encoder(
            reduced_teacher_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        ti_outputs = self.ti_bottleneck(ti_encoded)
        ti_tokens = ti_outputs['decoded']
        target_step_indices = self._target_step_indices(batch, teacher_latents)
        kl_loss, kl_logs = self._aggregate_bottleneck_kl(ti_outputs, target_step_indices)

        source_tokens = teacher_latents[:, 0]
        source_tokens_for_decoder = self._maybe_drop_source_tokens(source_tokens)
        ti_tokens_t = gather_step_tokens(ti_tokens, target_step_indices)
        ti_tokens_t_for_decoder = self._maybe_drop_ti_tokens(ti_tokens_t)
        target_latent_t = gather_step_tokens(teacher_latents, target_step_indices)
        decoder_kwargs = {
            'source_tokens': source_tokens_for_decoder,
            'ti_tokens_t': ti_tokens_t_for_decoder,
            'target_step_indices': target_step_indices,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
        }
        pred_latent_t, decoder_aux_outputs = self._unpack_decoder_outputs(self.decoder(**decoder_kwargs))
        return {
            'pred_latent_t': pred_latent_t,
            'target_latent_t': target_latent_t,
            'target_step_indices': target_step_indices,
            'source_latent_l0': source_tokens,
            'teacher_latents': teacher_latents,
            'reduced_teacher_tokens': reduced_teacher_tokens,
            'ti_tokens': ti_tokens,
            'ti_tokens_t': ti_tokens_t,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'kl_loss': kl_loss,
            'kl_target_loss': kl_logs['kl_target_loss'],
            'kl_all_loss': kl_logs['kl_all_loss'],
            'bottleneck_latent': ti_outputs['latent'],
            'bottleneck_mu': ti_outputs['mu'],
            'bottleneck_logvar': ti_outputs['logvar'],
            'bottleneck_per_dim_kl': ti_outputs['per_dim_kl'],
            **decoder_aux_outputs,
        }
