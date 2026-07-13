"""WAN VAE queryset predictive LAM (parallel to vjepa2_wanconv_compact_predictive_lam).

Same architectural family as the V-JEPA2 queryset predictive path:
- queryset_cross_v1 encoder for TI (and shared trunk for stage-2 TI/TC),
- source_query_self_attn_v1 self-attention MAE-style decoder,
- KL bottleneck per query token,

but the reconstruction target is **WAN 2.2 VAE latents** instead of V-JEPA2
features. WAN VAE is causal in time (`scale_factor_temporal=4`,
`scale_factor_spatial=16`, `z_dim=48`), so we follow the WAN 4k+1 rule:
``num_frames = 4*k + 1`` → ``T_lat = k + 1`` latent steps. With ``k=4`` we feed
17 RGB frames and predict at one of 4 future latent steps.

Source-copy cosine on DROID with WAN VAE sits at ~0.76 (probe `99299` in
``docs/tmp/2026-04-27_per_step_source_copy_probe.md``), versus 0.44 for
V-JEPA2 — a much friendlier residual for the latent action bottleneck to model.

This module reuses the shared building blocks (``QuerySetVideoEncoder``,
``SourceQuerySelfAttentionDecoder``, ``FullGridTokenBottleneck``) and the
``_PredictiveCompactBase`` mixin. Only ``queryset_cross_v1`` is supported here
(WAN VAE's z_dim=48 makes a Conv3d reducer unnecessary); the V-JEPA2-specific
``spatial_grid_v1`` encoder path is not used.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from ditar_encoder.encoders import vjepa2_fullgrid_common as common
from ditar_encoder.encoders import wan_forecast_common
from ditar_encoder.lam.vjepa2_wanconv_compact_predictive_lam import DECODER_ARCH_SOURCE_QUERY_SELF_ATTN, DECODER_ARCH_SOURCE_ADDITIVE, QUERYSET_CROSS_ENCODER, SPATIAL_GRID_ENCODER, QuerySetVideoEncoder, _PredictiveCompactBase, build_predictive_decoder, gather_step_tokens, normalize_decoder_arch, normalize_encoder_arch, normalize_kl_scope
from ditar_encoder.lam.defaults import DEFAULT_BOTTLENECK_DIM, DEFAULT_BOTTLENECK_HIDDEN_DIM, DEFAULT_DEC_BLOCKS, DEFAULT_ENC_BLOCKS, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS


def _wan_vae_target_step_indices(model, batch, teacher_latents):
    """Sample target latent step for the WAN 4k+1 layout.

    WAN VAE produces ``T_lat = (T_in - 1) // 4 + 1`` latent steps from ``T_in``
    RGB frames. Latent step 0 covers RGB frame 0 only; latent step ``t >= 1``
    covers RGB frames ``[(t-1)*4 + 1 .. t*4]`` (causal). Therefore the pad mask
    over RGB frames is **not** evenly groupable by ``tubelet_size``. We override
    the V-JEPA2 path's grouping logic and walk the causal windows directly.
    """
    if 'target_step_indices' in batch:
        return batch['target_step_indices'].to(device=teacher_latents.device, dtype=torch.long)

    batch_size = int(teacher_latents.shape[0])
    full_num_steps = int(model.full_num_steps)
    valid_steps = torch.ones(batch_size, full_num_steps, dtype=torch.bool, device=teacher_latents.device)
    valid_steps[:, 0] = False  # source step is excluded as a target

    if 'pad_mask' in batch:
        pad_mask = batch['pad_mask'].to(device=teacher_latents.device, dtype=torch.bool)
        scale = int(model.teacher.scale_factor_temporal)
        for t in range(1, full_num_steps):
            window_start = (t - 1) * scale + 1
            window_end = t * scale + 1
            window = pad_mask[:, window_start:window_end]
            valid_steps[:, t] = valid_steps[:, t] & window.all(dim=-1)
        valid_steps[:, 0] = False

    fallback = torch.randint(low=1, high=full_num_steps, size=(batch_size,), device=teacher_latents.device)
    sampled = []
    for batch_idx in range(batch_size):
        candidates = torch.nonzero(valid_steps[batch_idx], as_tuple=False).flatten()
        if candidates.numel() == 0:
            sampled.append(fallback[batch_idx])
        else:
            sampled.append(candidates[torch.randint(candidates.numel(), (), device=teacher_latents.device)])
    return torch.stack(sampled).to(dtype=torch.long)


def _validate_queryset_only(encoder_arch: str) -> str:
    """The WAN VAE path supports only the queryset encoder; spatial-grid is omitted."""
    normalized = normalize_encoder_arch(encoder_arch)
    if normalized != QUERYSET_CROSS_ENCODER:
        raise ValueError(
            f'WAN VAE queryset predictive LAM supports only encoder_arch={QUERYSET_CROSS_ENCODER!r}, '
            f'got {encoder_arch!r}.'
        )
    return normalized


def _validate_self_attn_decoder(decoder_arch: str) -> str:
    """The WAN VAE path is built for the MAE source-query self-attn decoder."""
    normalized = normalize_decoder_arch(decoder_arch)
    if normalized != DECODER_ARCH_SOURCE_QUERY_SELF_ATTN:
        raise ValueError(
            f'WAN VAE queryset predictive LAM supports only '
            f'decoder_arch={DECODER_ARCH_SOURCE_QUERY_SELF_ATTN!r}, got {decoder_arch!r}.'
        )
    return normalized


class WanVaeQuerysetPredictiveStage2LatentActionModel(
    _PredictiveCompactBase
):
    """Stage-2 LAM with WAN VAE target.

    Mirrors the V-JEPA2 stage-2 queryset path: a single ``shared_query_encoder``
    emits ``query_num_tokens * 2`` query tokens that split into ``[TI, TC]``
    halves. Stage-1 ``ti_bottleneck`` weights are loaded and frozen via
    ``initialize_from_stage_one_state_dict`` (inherited from the parent). The
    only constructor change vs. the parent is the WAN teacher.
    """

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
        max_decoder_text_tokens: int = 256,
        separate_tc_encoder: bool = False,
        init_tc_from_ti: bool = False,
        tc_bootstrap_freeze: bool = False,
        ti_decoder_dropout_p: float = 0.0,
        freeze_ti_branch: bool = True,
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
        self.t5_conditioner = None  # text conditioner is unused in queryset stage-2
        self.separate_tc_encoder = bool(separate_tc_encoder)
        self.init_tc_from_ti = bool(init_tc_from_ti)
        self.tc_bootstrap_freeze = bool(tc_bootstrap_freeze)
        self.freeze_ti_branch_enabled = bool(freeze_ti_branch)
        if self.tc_bootstrap_freeze and not self.separate_tc_encoder:
            raise ValueError('tc_bootstrap_freeze requires separate_tc_encoder=true.')
        ti_dropout = float(ti_decoder_dropout_p)
        if not (0.0 <= ti_dropout <= 1.0):
            raise ValueError(f'ti_decoder_dropout_p must be in [0, 1], got {ti_dropout}.')
        self.ti_decoder_dropout_p = ti_dropout

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
        if self.separate_tc_encoder:
            self.shared_query_encoder = None
            self.ti_encoder = QuerySetVideoEncoder(
                context_dim=self.model_dim,
                model_dim=self.model_dim,
                num_steps=self.reduced_num_steps,
                query_num_tokens=self.reduced_num_patches,
                num_blocks=self.query_blocks,
                num_heads=self.num_heads,
                dropout=self.lam_dropout,
                project_context=False,
            )
            self.tc_encoder = QuerySetVideoEncoder(
                context_dim=self.model_dim,
                model_dim=self.model_dim,
                num_steps=self.reduced_num_steps,
                query_num_tokens=self.reduced_num_patches,
                num_blocks=self.query_blocks,
                num_heads=self.num_heads,
                dropout=self.lam_dropout,
                project_context=False,
            )
        else:
            self.shared_query_encoder = QuerySetVideoEncoder(
                context_dim=self.model_dim,
                model_dim=self.model_dim,
                num_steps=self.reduced_num_steps,
                query_num_tokens=self.reduced_num_patches * 2,
                num_blocks=self.query_blocks,
                num_heads=self.num_heads,
                dropout=self.lam_dropout,
                project_context=False,
            )
            self.ti_encoder = None
            self.tc_encoder = None
        self.ti_bottleneck = common.FullGridTokenBottleneck(
            model_dim=self.model_dim,
            bottleneck_dim=self.bottleneck_dim,
            bottleneck_hidden_dim=self.bottleneck_hidden_dim,
            bottleneck_regularizer=bottleneck_regularizer,
            post_norm_after_decode=bottleneck_post_norm,
        )
        self.tc_bottleneck = common.FullGridTokenBottleneck(
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
            text_dim=self.model_dim,  # placeholder dim; text branch is frozen+unused below
            num_steps=self.full_num_steps,
            source_num_patches=self.teacher.teacher_num_patches,
            reduced_num_patches=self.reduced_num_patches,
            dec_blocks=self.dec_blocks,
            num_heads=self.num_heads,
            dropout=self.lam_dropout,
            max_text_tokens=max_decoder_text_tokens,
        )
        # Stage-2 has no text in the decoder sequence: freeze text branch params.
        self.decoder.text_proj.requires_grad_(False)
        self.decoder.text_position_embed.requires_grad_(False)

    @torch.no_grad()
    def task_relevant_from_context(self, reduced_context: torch.Tensor) -> torch.Tensor:
        """Encoded teacher context -> task-relevant latent ``z_rel`` for every WAN latent step.

        Single source of truth for the Stage-2 TC branch (queryset_cross_v1 + separate/shared
        encoder split + KL bottleneck), shared by the public encoder API and the diffusion
        handoff so no caller re-derives it from private modules (audit B2). Returns ``z_rel`` of
        shape ``[B, T_lat, K, D]`` (K = query_num_tokens, D = bottleneck_dim; 4 x 32 for q4).
        """
        if self.separate_tc_encoder:
            tc_encoded = self.tc_encoder(reduced_context)
        else:
            shared_tokens = self.shared_query_encoder(reduced_context)
            _, tc_encoded = torch.split(
                shared_tokens, [self.reduced_num_patches, self.reduced_num_patches], dim=2
            )
        return self.tc_bottleneck(tc_encoded)['latent']

    @torch.no_grad()
    def encode_task_relevant(self, target_videos: torch.Tensor) -> torch.Tensor:
        """Public encoder API: a target-window clip ``[B, T, 3, H, W]`` in [-1, 1] -> ``z_rel``.

        Returns the task-relevant latent for every WAN step, ``[B, T_lat, K, D]``. Used by the
        diffusion teacher and by downstream consumers (e.g. the reward / RL repo) so encoding
        always matches training.
        """
        teacher_latents = self.teacher.encode(target_videos)
        reduced_context = self._encode_teacher_context(teacher_latents)
        return self.task_relevant_from_context(reduced_context)

    def initialize_from_stage_one_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        if self.separate_tc_encoder:
            self._initialize_separate_tc_from_stage_one_state_dict(state_dict)
        else:
            self._flat_initialize_from_stage_one_state_dict(state_dict)
        if self.init_tc_from_ti:
            self._initialize_tc_from_ti_branch()
        if self.freeze_ti_branch_enabled or self.tc_bootstrap_freeze:
            self.freeze_ti_branch()
        if self.tc_bootstrap_freeze:
            self._apply_tc_bootstrap_freeze()
        self.validate_ti_frozen()

    def _initialize_separate_tc_from_stage_one_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        current_state = self.state_dict()
        init_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key[len('lam.') :] if key.startswith('lam.') else key
            if normalized_key in current_state and tuple(current_state[normalized_key].shape) == tuple(value.shape):
                if normalized_key.startswith(('input_proj.', 'reducer.', 'ti_bottleneck.', 'decoder.')):
                    init_state[normalized_key] = value
                continue
            if not normalized_key.startswith('ti_encoder.'):
                continue
            encoder_key = normalized_key[len('ti_encoder.') :]
            if encoder_key.startswith('text_'):
                continue
            for target_prefix in ('ti_encoder', 'tc_encoder'):
                mapped_key = f'{target_prefix}.{encoder_key}'
                if mapped_key in current_state and tuple(current_state[mapped_key].shape) == tuple(value.shape):
                    init_state[mapped_key] = value
        if not init_state:
            raise KeyError('Stage-1 checkpoint does not contain WAN queryset TI weights for TC bootstrap.')
        incompatible = self.load_state_dict(init_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                'Unexpected keys while initializing WAN queryset stage-2 TC bootstrap from stage-1: '
                + ', '.join(unexpected)
            )

    def _initialize_tc_from_ti_branch(self) -> None:
        with torch.no_grad():
            if self.separate_tc_encoder:
                self.tc_encoder.load_state_dict(self.ti_encoder.state_dict())
            else:
                query_embed = self.shared_query_encoder.query_embed
                query_count = int(self.reduced_num_patches)
                if query_embed.shape[2] < query_count * 2:
                    raise ValueError(
                        'Expected shared_query_encoder.query_embed to contain TI+TC query slots, '
                        f'got shape={tuple(query_embed.shape)} query_count={query_count}.'
                    )
                query_embed[:, :, query_count : query_count * 2].copy_(query_embed[:, :, :query_count])

            self.tc_bottleneck.load_state_dict(self.ti_bottleneck.state_dict())

            if hasattr(self.decoder, 'ti_proj') and hasattr(self.decoder, 'tc_proj'):
                self.decoder.tc_proj.load_state_dict(self.decoder.ti_proj.state_dict())
            if hasattr(self.decoder, 'ti_patch_embed') and hasattr(self.decoder, 'tc_patch_embed'):
                self.decoder.tc_patch_embed.copy_(self.decoder.ti_patch_embed)
            if hasattr(self.decoder, 'token_type_embed'):
                token_weight = self.decoder.token_type_embed.weight
                if token_weight.shape[0] > self.decoder.TC_TOKEN_TYPE:
                    token_weight[self.decoder.TC_TOKEN_TYPE].copy_(token_weight[self.decoder.TI_TOKEN_TYPE])
        print(
            '[wan-vae queryset predictive stage2] initialized TC query/bottleneck/decoder branch from TI',
            flush=True,
        )

    def freeze_ti_branch(self) -> None:
        self._flat_freeze_ti_branch()
        if self.separate_tc_encoder and self.ti_encoder is not None:
            self.ti_encoder.requires_grad_(False)

    def _apply_tc_bootstrap_freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        if self.tc_encoder is None:
            raise RuntimeError('TC bootstrap requires a trainable tc_encoder.')
        self.tc_encoder.requires_grad_(True)
        self.tc_bottleneck.requires_grad_(True)

    def validate_ti_frozen(self) -> None:
        if not (getattr(self, 'freeze_ti_branch_enabled', True) or self.tc_bootstrap_freeze):
            return
        self._flat_validate_ti_frozen()
        frozen_prefixes = ('ti_encoder.', 'ti_bottleneck.')
        trainable_ti = [
            name
            for name, param in self.named_parameters()
            if name.startswith(frozen_prefixes) and param.requires_grad
        ]
        if trainable_ti:
            raise RuntimeError('WAN TC bootstrap requires frozen TI params: ' + ', '.join(trainable_ti))
        if self.tc_bootstrap_freeze:
            allowed_prefixes = ('tc_encoder.', 'tc_bottleneck.')
            unexpected = [
                name
                for name, param in self.named_parameters()
                if param.requires_grad and not name.startswith(allowed_prefixes)
            ]
            if unexpected:
                raise RuntimeError(
                    'WAN TC bootstrap should train only TC encoder/bottleneck, but found trainable params: '
                    + ', '.join(unexpected)
                )

    def _maybe_drop_ti_tokens_for_decoder(self, ti_tokens_t: torch.Tensor) -> torch.Tensor:
        if self.ti_decoder_dropout_p <= 0.0:
            return ti_tokens_t
        if not self.training and not self.tc_bootstrap_freeze:
            return ti_tokens_t
        if self.ti_decoder_dropout_p >= 1.0:
            return torch.zeros_like(ti_tokens_t)
        keep_prob = 1.0 - self.ti_decoder_dropout_p
        mask = torch.empty(
            ti_tokens_t.shape[0],
            ti_tokens_t.shape[1],
            1,
            device=ti_tokens_t.device,
            dtype=ti_tokens_t.dtype,
        ).bernoulli_(keep_prob)
        return ti_tokens_t * mask / keep_prob

    def _target_step_indices(self, batch, teacher_latents):
        return _wan_vae_target_step_indices(self, batch, teacher_latents)

    @staticmethod
    def _extract_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], Mapping):
            return checkpoint['state_dict']  # type: ignore[return-value]
        return checkpoint  # type: ignore[return-value]

    def _flat_initialize_from_stage_one_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        current_state = self.state_dict()
        init_state: Dict[str, torch.Tensor] = {}
        stage_one_query_embed: Optional[torch.Tensor] = None
        for key, value in state_dict.items():
            normalized_key = key[len('lam.') :] if key.startswith('lam.') else key
            if normalized_key in current_state and tuple(current_state[normalized_key].shape) == tuple(value.shape):
                if normalized_key.startswith(('input_proj.', 'reducer.', 'ti_bottleneck.', 'decoder.')):
                    init_state[normalized_key] = value
                elif self.encoder_arch == SPATIAL_GRID_ENCODER and normalized_key.startswith('ti_encoder.'):
                    init_state[normalized_key] = value
                continue
            if self.encoder_arch != QUERYSET_CROSS_ENCODER or not normalized_key.startswith('ti_encoder.'):
                continue
            encoder_key = normalized_key[len('ti_encoder.') :]
            if encoder_key.startswith('text_'):
                continue
            if encoder_key == 'query_embed':
                stage_one_query_embed = value
                continue
            mapped_key = f'shared_query_encoder.{encoder_key}'
            if mapped_key in current_state and tuple(current_state[mapped_key].shape) == tuple(value.shape):
                init_state[mapped_key] = value
        if not init_state:
            raise KeyError('Stage-1 checkpoint does not contain predictive compact TI/shared decoder weights.')
        incompatible = self.load_state_dict(init_state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith('t5_conditioner.')]
        if unexpected:
            raise RuntimeError(
                'Unexpected keys while initializing predictive compact stage-2 from stage-1 checkpoint: '
                + ', '.join(unexpected)
            )
        if self.encoder_arch == QUERYSET_CROSS_ENCODER:
            if stage_one_query_embed is None:
                raise KeyError('Stage-1 query-set checkpoint is missing ti_encoder.query_embed for TI query init.')
            expected_shape = (1, 1, self.reduced_num_patches, self.model_dim)
            if tuple(stage_one_query_embed.shape) != expected_shape:
                raise ValueError(
                    'Stage-1 query-set query_embed shape does not match current TI query slice: '
                    f'{tuple(stage_one_query_embed.shape)} vs {expected_shape}.'
                )
            with torch.no_grad():
                self.shared_query_encoder.query_embed[:, :, : self.reduced_num_patches].copy_(stage_one_query_embed)
        if self.freeze_ti_branch_enabled:
            self.freeze_ti_branch()
        self.validate_ti_frozen()

    def _flat_freeze_ti_branch(self) -> None:
        self.ti_bottleneck.requires_grad_(False)
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            self.input_proj.requires_grad_(False)
            self.reducer.requires_grad_(False)
            self.ti_encoder.requires_grad_(False)

    def _flat_validate_ti_frozen(self) -> None:
        if not getattr(self, 'freeze_ti_branch_enabled', True):
            return
        frozen_prefixes = ('ti_bottleneck.',)
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            frozen_prefixes = ('input_proj.', 'reducer.', 'ti_encoder.', 'ti_bottleneck.')
        trainable = [
            name
            for name, param in self.named_parameters()
            if name.startswith(frozen_prefixes) and param.requires_grad
        ]
        if trainable:
            raise RuntimeError(
                'Predictive compact stage-2 TI branch must be frozen, but found trainable parameters: '
                + ', '.join(trainable)
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'videos' not in batch:
            raise KeyError('Predictive compact stage-2 path requires batch["videos"].')
        if self.encoder_arch == SPATIAL_GRID_ENCODER and 'task_instruction' not in batch:
            raise KeyError('Predictive compact stage-2 path requires batch["task_instruction"].')

        with torch.no_grad():
            teacher_latents = self.teacher.encode(batch['videos'])
        reduced_teacher_tokens = self._encode_teacher_context(teacher_latents)
        freeze_ti_branch = bool(getattr(self, 'freeze_ti_branch_enabled', True)) or bool(
            getattr(self, 'tc_bootstrap_freeze', False)
        )
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            if freeze_ti_branch:
                with torch.no_grad():
                    ti_encoded = self.ti_encoder(reduced_teacher_tokens)
                    ti_outputs = self.ti_bottleneck(ti_encoded)
                    ti_tokens = ti_outputs['decoded'].detach()
            else:
                ti_encoded = self.ti_encoder(reduced_teacher_tokens)
                ti_outputs = self.ti_bottleneck(ti_encoded)
                ti_tokens = ti_outputs['decoded']
            with torch.no_grad():
                text_tokens, text_mask = self.t5_conditioner(batch['task_instruction'])
            tc_encoded = self.tc_encoder(
                reduced_teacher_tokens.detach(),
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
            tc_outputs = self.tc_bottleneck(tc_encoded)
            tc_tokens = tc_outputs['decoded']
            target_step_indices = self._target_step_indices(batch, teacher_latents)
            if freeze_ti_branch:
                ti_kl_loss = teacher_latents.new_zeros(())
                ti_kl_logs = {'kl_target_loss': ti_kl_loss, 'kl_all_loss': ti_kl_loss}
            else:
                ti_kl_loss, ti_kl_logs = self._aggregate_bottleneck_kl(ti_outputs, target_step_indices)
            tc_kl_loss, tc_kl_logs = self._aggregate_bottleneck_kl(tc_outputs, target_step_indices)
        else:
            text_tokens = None
            text_mask = None
            if getattr(self, 'separate_tc_encoder', False):
                if freeze_ti_branch:
                    with torch.no_grad():
                        ti_encoded = self.ti_encoder(reduced_teacher_tokens)
                        ti_outputs = self.ti_bottleneck(ti_encoded)
                else:
                    ti_encoded = self.ti_encoder(reduced_teacher_tokens)
                    ti_outputs = self.ti_bottleneck(ti_encoded)
                tc_context_tokens = (
                    reduced_teacher_tokens.detach()
                    if getattr(self, 'tc_bootstrap_freeze', False)
                    else reduced_teacher_tokens
                )
                tc_encoded = self.tc_encoder(tc_context_tokens)
            else:
                shared_tokens = self.shared_query_encoder(reduced_teacher_tokens)
                ti_encoded, tc_encoded = torch.split(
                    shared_tokens,
                    [self.reduced_num_patches, self.reduced_num_patches],
                    dim=2,
                )
                ti_outputs = self.ti_bottleneck(ti_encoded)
            tc_outputs = self.tc_bottleneck(tc_encoded)
            ti_tokens = ti_outputs['decoded']
            tc_tokens = tc_outputs['decoded']
            target_step_indices = self._target_step_indices(batch, teacher_latents)
            ti_kl_loss, ti_kl_logs = self._aggregate_bottleneck_kl(ti_outputs, target_step_indices)
            tc_kl_loss, tc_kl_logs = self._aggregate_bottleneck_kl(tc_outputs, target_step_indices)

        source_tokens = teacher_latents[:, 0]
        ti_tokens_t = gather_step_tokens(ti_tokens, target_step_indices)
        tc_tokens_t = gather_step_tokens(tc_tokens, target_step_indices)
        ti_tokens_t_for_decoder = ti_tokens_t
        maybe_drop_ti = getattr(self, '_maybe_drop_ti_tokens_for_decoder', None)
        if maybe_drop_ti is not None:
            ti_tokens_t_for_decoder = maybe_drop_ti(ti_tokens_t)
        target_latent_t = gather_step_tokens(teacher_latents, target_step_indices)
        pred_latent_t, decoder_aux_outputs = self._unpack_decoder_outputs(self.decoder(
            source_tokens=source_tokens,
            ti_tokens_t=ti_tokens_t_for_decoder,
            tc_tokens_t=tc_tokens_t,
            target_step_indices=target_step_indices,
        ))
        return {
            'pred_latent_t': pred_latent_t,
            'target_latent_t': target_latent_t,
            'target_step_indices': target_step_indices,
            'source_latent_l0': source_tokens,
            'teacher_latents': teacher_latents,
            'reduced_teacher_tokens': reduced_teacher_tokens,
            'ti_tokens': ti_tokens,
            'tc_tokens': tc_tokens,
            'ti_tokens_t': ti_tokens_t,
            'ti_tokens_t_for_decoder': ti_tokens_t_for_decoder,
            'tc_tokens_t': tc_tokens_t,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'kl_loss': tc_kl_loss,
            'kl_target_loss': tc_kl_logs['kl_target_loss'],
            'kl_all_loss': tc_kl_logs['kl_all_loss'],
            'ti_kl_loss': ti_kl_loss,
            'ti_kl_target_loss': ti_kl_logs['kl_target_loss'],
            'ti_kl_all_loss': ti_kl_logs['kl_all_loss'],
            'tc_kl_loss': tc_kl_loss,
            'tc_kl_target_loss': tc_kl_logs['kl_target_loss'],
            'tc_kl_all_loss': tc_kl_logs['kl_all_loss'],
            'bottleneck_latent': tc_outputs['latent'],
            'bottleneck_mu': tc_outputs['mu'],
            'bottleneck_logvar': tc_outputs['logvar'],
            'bottleneck_per_dim_kl': tc_outputs['per_dim_kl'],
            'ti_bottleneck_latent': ti_outputs['latent'],
            'ti_bottleneck_mu': ti_outputs['mu'],
            'ti_bottleneck_logvar': ti_outputs['logvar'],
            'ti_bottleneck_per_dim_kl': ti_outputs['per_dim_kl'],
            'tc_bottleneck_latent': tc_outputs['latent'],
            'tc_bottleneck_mu': tc_outputs['mu'],
            'tc_bottleneck_logvar': tc_outputs['logvar'],
            'tc_bottleneck_per_dim_kl': tc_outputs['per_dim_kl'],
            **decoder_aux_outputs,
        }
