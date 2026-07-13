from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from ditar_encoder.lam.defaults import DEFAULT_BOTTLENECK_DIM, DEFAULT_BOTTLENECK_HIDDEN_DIM, DEFAULT_DEC_BLOCKS, DEFAULT_ENC_BLOCKS, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS
from ditar_encoder.lam.vjepa2_wanconv_compact_predictive_lam import DECODER_ARCH_ANCHOR_FULL_ATTN, DECODER_ARCH_SOURCE_QUERY_SELF_ATTN, KL_SCOPE_TARGET, QUERYSET_CROSS_ENCODER, SPATIAL_GRID_ENCODER, TARGET_MODE_ABSOLUTE
from ditar_encoder.lam.wan_vae_queryset_predictive_lam import WanVaeQuerysetPredictiveStage2LatentActionModel


OptimizerCallable = Callable[[Iterable], Optimizer]


class WanVaeQuerysetPredictiveStage2LAM(LightningModule):
    """Lightning wrapper for WAN VAE queryset predictive stage-2.

    Inherits all loss / logging / state-dict logic from the V-JEPA2 sibling.
    Only the LAM module instantiation and the WAN-specific ctor args differ.
    """

    _CHECKPOINT_EXCLUDED_PREFIXES = ('lam.teacher.',)

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
        target_mode: str = TARGET_MODE_ABSOLUTE,
        encoder_arch: str = QUERYSET_CROSS_ENCODER,
        decoder_arch: str = DECODER_ARCH_SOURCE_QUERY_SELF_ATTN,
        query_num_tokens: int = 64,
        query_blocks: int = 8,
        kl_scope: str = KL_SCOPE_TARGET,
        ti_kl_weight: float = 1e-3,
        tc_kl_weight: float = 1e-3,
        tc_kl_warmup_steps: int = 0,
        stage_one_ckpt: str = '',
        log_interval: int = 1000,
        log_path: str = 'ditar_encoder/logs',
        task_name: str = 'stage2_droid_wan_vae_queryset_predictive_tc',
        optimizer: OptimizerCallable = AdamW,
        wan_vae_model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        wan_vae_model_path: str = '',
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
        use_wandb: bool = False,
    ) -> None:
        from lightning import LightningModule

        LightningModule.__init__(self)
        normalized_target_mode = str(target_mode or TARGET_MODE_ABSOLUTE).strip().lower()
        if normalized_target_mode != TARGET_MODE_ABSOLUTE:
            raise ValueError('WAN VAE queryset predictive stage-2 only supports target_mode="absolute".')
        stage_one_ckpt = os.path.expanduser(os.path.expandvars(str(stage_one_ckpt or '')))
        if not stage_one_ckpt or '$' in stage_one_ckpt:
            raise ValueError(
                'stage_one_ckpt is required to construct WAN VAE queryset predictive stage-2: it seeds and '
                'freezes the task-irrelevant branch from Stage 1. Pass --model.stage_one_ckpt <stage1.ckpt>, '
                'or set the DITAR_STAGE1_CKPT env var consumed by the shipped configs. NOTE: --init_ckpt is a '
                'post-construction full-weights warm-start (used for the Phase-B/alltrain transition) and does '
                'NOT satisfy this constructor requirement.'
            )

        self.lam = WanVaeQuerysetPredictiveStage2LatentActionModel(
            bottleneck_dim=bottleneck_dim,
            bottleneck_hidden_dim=bottleneck_hidden_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            bottleneck_regularizer=bottleneck_regularizer,
            bottleneck_post_norm=bottleneck_post_norm,
            lam_dropout=lam_dropout,
            encoder_arch=encoder_arch,
            decoder_arch=decoder_arch,
            query_num_tokens=query_num_tokens,
            query_blocks=query_blocks,
            kl_scope=kl_scope,
            wan_vae_model_id=wan_vae_model_id,
            wan_vae_model_path=wan_vae_model_path or None,
            wan_vae_subfolder=wan_vae_subfolder,
            wan_vae_dtype=wan_vae_dtype,
            wan_vae_sample_mode=wan_vae_sample_mode,
            wan_vae_normalize_latents=wan_vae_normalize_latents,
            teacher_input_resolution=teacher_input_resolution,
            num_frames=num_frames,
            max_decoder_text_tokens=max_decoder_text_tokens,
            separate_tc_encoder=separate_tc_encoder,
            init_tc_from_ti=init_tc_from_ti,
            tc_bootstrap_freeze=tc_bootstrap_freeze,
            ti_decoder_dropout_p=ti_decoder_dropout_p,
            freeze_ti_branch=freeze_ti_branch,
        )
        self.target_mode = normalized_target_mode
        self.encoder_arch = self.lam.encoder_arch
        self.decoder_arch = self.lam.decoder_arch
        self.kl_scope = self.lam.kl_scope
        self.ti_kl_weight = float(ti_kl_weight)
        self.tc_kl_weight = float(tc_kl_weight)
        self.tc_kl_warmup_steps = max(int(tc_kl_warmup_steps), 0)
        self.stage_one_ckpt = stage_one_ckpt
        self.log_interval = log_interval
        self.log_path = log_path
        self.task_name = task_name
        self.optimizer = optimizer
        self.use_wandb = use_wandb
        self._initialize_from_stage_one_ckpt(self.stage_one_ckpt)
        self.param_report = self._build_param_report()
        self._log_param_report()
        self.save_hyperparameters()

    def _log_param_report(self) -> None:
        summary = ', '.join(f'{key}={value}' for key, value in sorted(self.param_report.items()))
        print(
            '[wan-vae queryset predictive stage2] '
            f'target_mode={self.target_mode} encoder_arch={self.encoder_arch} '
            f'decoder_arch={self.decoder_arch} kl_scope={self.kl_scope} '
            f'reduced_steps={self.lam.reduced_num_steps} '
            f'reduced_patches={self.lam.reduced_num_patches} {summary}',
            flush=True,
        )

    _ACTIVE_DIM_VAR_THRESHOLD = 1e-2

    _ACTIVE_DIM_KL_THRESHOLD = 1e-3

    @staticmethod
    def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], Mapping):
            return checkpoint['state_dict']
        return checkpoint

    def _initialize_from_stage_one_ckpt(self, checkpoint_path: str) -> None:
        checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_path))
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = self._extract_state_dict(checkpoint)
        self.lam.initialize_from_stage_one_state_dict(state_dict)  # type: ignore[arg-type]

    def _build_param_report(self) -> Dict[str, int]:
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            return {
                'ti_encoder_total': sum(param.numel() for param in self.lam.ti_encoder.parameters()),
                'tc_encoder_trainable': sum(param.numel() for param in self.lam.tc_encoder.parameters() if param.requires_grad),
                'decoder_trainable': sum(param.numel() for param in self.lam.decoder.parameters() if param.requires_grad),
                'conv_frozen': sum(param.numel() for param in self.lam.input_proj.parameters())
                + sum(param.numel() for param in self.lam.reducer.parameters()),
                'tc_bottleneck_trainable': sum(
                    param.numel() for param in self.lam.tc_bottleneck.parameters() if param.requires_grad
                ),
                'total_trainable': sum(param.numel() for param in self.lam.parameters() if param.requires_grad),
            }
        shared_query_encoder = getattr(self.lam, 'shared_query_encoder', None)
        ti_encoder = getattr(self.lam, 'ti_encoder', None)
        tc_encoder = getattr(self.lam, 'tc_encoder', None)
        return {
            'shared_query_encoder_trainable': sum(
                param.numel() for param in shared_query_encoder.parameters() if param.requires_grad
            )
            if shared_query_encoder is not None
            else 0,
            'ti_encoder_trainable': sum(param.numel() for param in ti_encoder.parameters() if param.requires_grad)
            if ti_encoder is not None
            else 0,
            'tc_encoder_trainable': sum(param.numel() for param in tc_encoder.parameters() if param.requires_grad)
            if tc_encoder is not None
            else 0,
            'frontend_trainable': sum(param.numel() for param in self.lam.input_proj.parameters() if param.requires_grad)
            + sum(param.numel() for param in self.lam.reducer.parameters() if param.requires_grad),
            'decoder_trainable': sum(param.numel() for param in self.lam.decoder.parameters() if param.requires_grad),
            'ti_bottleneck_frozen': sum(
                param.numel() for param in self.lam.ti_bottleneck.parameters() if not param.requires_grad
            ),
            'ti_bottleneck_trainable': sum(
                param.numel() for param in self.lam.ti_bottleneck.parameters() if param.requires_grad
            ),
            'tc_bottleneck_trainable': sum(
                param.numel() for param in self.lam.tc_bottleneck.parameters() if param.requires_grad
            ),
            'total_trainable': sum(param.numel() for param in self.lam.parameters() if param.requires_grad),
        }

    @staticmethod
    def _mean_cosine_similarity(prediction: Tensor, target: Tensor) -> Tensor:
        prediction_flat = prediction.reshape(prediction.shape[0], -1).float()
        target_flat = target.reshape(target.shape[0], -1).float()
        return F.cosine_similarity(prediction_flat, target_flat, dim=-1, eps=1e-8).mean()

    @classmethod
    def _active_dim_fraction_from_mu(cls, mu: Optional[Tensor]) -> Optional[Tensor]:
        if mu is None:
            return None
        flat_mu = mu.reshape(-1, mu.shape[-1]).float()
        if flat_mu.shape[0] < 2:
            return flat_mu.new_zeros(())
        dim_var = flat_mu.var(dim=0, unbiased=False)
        return (dim_var > cls._ACTIVE_DIM_VAR_THRESHOLD).float().mean()

    @classmethod
    def _active_dim_fraction_from_kl(cls, per_dim_kl: Optional[Tensor]) -> Optional[Tensor]:
        if per_dim_kl is None:
            return None
        flat_kl = per_dim_kl.reshape(-1, per_dim_kl.shape[-1]).float()
        mean_per_dim_kl = flat_kl.mean(dim=0)
        return (mean_per_dim_kl > cls._ACTIVE_DIM_KL_THRESHOLD).float().mean()

    @classmethod
    def _collect_bottleneck_logs(cls, outputs: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
        logs: Dict[str, Tensor] = {}
        latent = outputs.get('bottleneck_latent')
        if latent is not None:
            latent_float = latent.float()
            logs[f'{prefix}_latent_rms'] = latent_float.pow(2).mean().sqrt()
            logs[f'{prefix}_latent_abs_mean'] = latent_float.abs().mean()
        mu = outputs.get('bottleneck_mu')
        if mu is not None:
            mu_float = mu.float()
            logs[f'{prefix}_mu_abs_mean'] = mu_float.abs().mean()
            logs[f'{prefix}_mu_rms'] = mu_float.pow(2).mean().sqrt()
            active_dim_frac_var = cls._active_dim_fraction_from_mu(mu_float)
            if active_dim_frac_var is not None:
                logs[f'{prefix}_active_dim_frac_var'] = active_dim_frac_var
        logvar = outputs.get('bottleneck_logvar')
        if logvar is not None:
            logvar_float = logvar.float()
            logs[f'{prefix}_logvar_mean'] = logvar_float.mean()
            logs[f'{prefix}_posterior_std_mean'] = torch.exp(0.5 * logvar_float).mean()
        per_dim_kl = outputs.get('bottleneck_per_dim_kl')
        if per_dim_kl is not None:
            flat_kl = per_dim_kl.reshape(-1, per_dim_kl.shape[-1]).float()
            mean_per_dim_kl = flat_kl.mean(dim=0)
            logs[f'{prefix}_per_dim_kl_mean'] = mean_per_dim_kl.mean()
            logs[f'{prefix}_per_dim_kl_max'] = mean_per_dim_kl.max()
            if per_dim_kl.ndim >= 3:
                per_query_kl = per_dim_kl.float().sum(dim=-1).reshape(-1, per_dim_kl.shape[-2]).mean(dim=0)
                logs[f'{prefix}_active_query_frac_kl'] = (
                    per_query_kl > cls._ACTIVE_DIM_KL_THRESHOLD
                ).float().mean()
            active_dim_frac_kl = cls._active_dim_fraction_from_kl(per_dim_kl)
            if active_dim_frac_kl is not None:
                logs[f'{prefix}_active_dim_frac_kl'] = active_dim_frac_kl
        return logs

    @staticmethod
    def _collect_scale_logs(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        logs: Dict[str, Tensor] = {}
        for source_key, log_key in (
            ('target_latent_t', 'target_latent_t_rms'),
            ('pred_latent_t', 'pred_latent_t_rms'),
            ('source_latent_l0', 'source_latent_l0_rms'),
            ('reduced_teacher_tokens', 'reduced_teacher_tokens_rms'),
            ('ti_tokens', 'ti_tokens_rms'),
            ('tc_tokens', 'tc_tokens_rms'),
            ('ti_tokens_t', 'ti_tokens_t_rms'),
            ('ti_tokens_t_for_decoder', 'ti_tokens_t_for_decoder_rms'),
            ('tc_tokens_t', 'tc_tokens_t_rms'),
        ):
            value = outputs.get(source_key)
            if value is not None:
                logs[log_key] = value.float().pow(2).mean().sqrt()
        return logs

    def _effective_tc_kl_weight(self) -> float:
        if self.tc_kl_warmup_steps <= 0:
            return self.tc_kl_weight
        return self.tc_kl_weight * min(1.0, float(self.global_step) / float(self.tc_kl_warmup_steps))

    def shared_step(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor, Tuple[Tuple[str, Tensor], ...]]:
        outputs = self.lam(batch)
        recon_error = outputs['target_latent_t'] - outputs['pred_latent_t']
        recon_loss = recon_error.pow(2).mean()
        recon_rmse = recon_loss.sqrt()
        recon_mae = recon_error.abs().mean()
        target_rms = outputs['target_latent_t'].float().pow(2).mean().sqrt().clamp_min(1e-8)
        tc_kl_loss = outputs.get('tc_kl_loss', outputs['kl_loss'])
        effective_tc_kl_weight = self._effective_tc_kl_weight()
        weighted_tc_kl_loss = effective_tc_kl_weight * tc_kl_loss
        ti_kl_loss = outputs.get('ti_kl_loss')
        if ti_kl_loss is None:
            ti_kl_loss = target_rms.new_zeros(())
        weighted_ti_kl_loss = self.ti_kl_weight * ti_kl_loss
        loss = recon_loss + weighted_ti_kl_loss + weighted_tc_kl_loss
        loss_logs: Dict[str, Tensor] = {
            'total_loss': loss,
            'recon_loss': recon_loss,
            'recon_rmse': recon_rmse,
            'recon_relative_rmse': recon_rmse / target_rms,
            'recon_mae': recon_mae,
            'recon_cosine': self._mean_cosine_similarity(outputs['pred_latent_t'], outputs['target_latent_t']),
            'ti_kl_loss': ti_kl_loss,
            'weighted_ti_kl_loss': weighted_ti_kl_loss,
            'tc_kl_loss': tc_kl_loss,
            'weighted_tc_kl_loss': weighted_tc_kl_loss,
            'tc_kl_effective_weight': tc_kl_loss.new_tensor(float(effective_tc_kl_weight)),
            'target_step_mean': outputs['target_step_indices'].float().mean(),
        }
        if outputs.get('ti_kl_target_loss') is not None:
            loss_logs['ti_kl_target_loss'] = outputs['ti_kl_target_loss']
        if outputs.get('ti_kl_all_loss') is not None:
            loss_logs['ti_kl_all_loss'] = outputs['ti_kl_all_loss']
        elif self.encoder_arch == SPATIAL_GRID_ENCODER:
            zero = loss.new_zeros(())
            loss_logs['ti_kl_target_loss'] = zero
            loss_logs['ti_kl_all_loss'] = zero
        if outputs.get('tc_kl_target_loss') is not None:
            loss_logs['tc_kl_target_loss'] = outputs['tc_kl_target_loss']
        elif outputs.get('kl_target_loss') is not None:
            loss_logs['tc_kl_target_loss'] = outputs['kl_target_loss']
        if outputs.get('tc_kl_all_loss') is not None:
            loss_logs['tc_kl_all_loss'] = outputs['tc_kl_all_loss']
        elif outputs.get('kl_all_loss') is not None:
            loss_logs['tc_kl_all_loss'] = outputs['kl_all_loss']
        if self.encoder_arch == SPATIAL_GRID_ENCODER:
            loss_logs.update(self._collect_bottleneck_logs(outputs, prefix='tc'))
        else:
            loss_logs.update(
                self._collect_bottleneck_logs(
                    {
                        'bottleneck_latent': outputs.get('ti_bottleneck_latent'),
                        'bottleneck_mu': outputs.get('ti_bottleneck_mu'),
                        'bottleneck_logvar': outputs.get('ti_bottleneck_logvar'),
                        'bottleneck_per_dim_kl': outputs.get('ti_bottleneck_per_dim_kl'),
                    },
                    prefix='ti',
                )
            )
            loss_logs.update(
                self._collect_bottleneck_logs(
                    {
                        'bottleneck_latent': outputs.get('tc_bottleneck_latent'),
                        'bottleneck_mu': outputs.get('tc_bottleneck_mu'),
                        'bottleneck_logvar': outputs.get('tc_bottleneck_logvar'),
                        'bottleneck_per_dim_kl': outputs.get('tc_bottleneck_per_dim_kl'),
                    },
                    prefix='tc',
                )
            )
        loss_logs.update(self._collect_scale_logs(outputs))
        return outputs, loss, tuple(loss_logs.items())

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        _, loss, aux_losses = self.shared_step(batch)
        batch_size = int(batch['videos'].shape[0])
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f'train/{key}': value for key, value in aux_losses},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        _, loss, aux_losses = self.shared_step(batch)
        batch_size = int(batch['videos'].shape[0])
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f'val/{key}': value for key, value in aux_losses},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        _, loss, aux_losses = self.shared_step(batch)
        batch_size = int(batch['videos'].shape[0])
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f'test/{key}': value for key, value in aux_losses},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self) -> Optimizer:
        self.lam.validate_ti_frozen()
        return self.optimizer(parameter for parameter in self.parameters() if parameter.requires_grad)

    @classmethod
    def _is_excluded_checkpoint_key(cls, key: str) -> bool:
        return key.startswith(cls._CHECKPOINT_EXCLUDED_PREFIXES)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if self._is_excluded_checkpoint_key(key):
                del state[key]
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing_keys = [key for key in incompatible.missing_keys if not self._is_excluded_checkpoint_key(key)]
            unexpected_keys = [key for key in incompatible.unexpected_keys if not self._is_excluded_checkpoint_key(key)]
            if missing_keys or unexpected_keys:
                problems = []
                if missing_keys:
                    problems.append(f'missing keys: {missing_keys}')
                if unexpected_keys:
                    problems.append(f'unexpected keys: {unexpected_keys}')
                raise RuntimeError(
                    'Error(s) in loading state_dict for VJEPA2WanConvCompactPredictiveStage2LAM: '
                    + '; '.join(problems)
                )
        return incompatible
