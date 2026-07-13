from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from ditar_encoder.lam.defaults import DEFAULT_BOTTLENECK_DIM, DEFAULT_BOTTLENECK_HIDDEN_DIM, DEFAULT_DEC_BLOCKS, DEFAULT_ENC_BLOCKS, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS
from ditar_encoder.lam.vjepa2_wanconv_compact_predictive_lam import DECODER_ARCH_ANCHOR_FULL_ATTN, DECODER_ARCH_SOURCE_QUERY_SELF_ATTN, KL_SCOPE_TARGET, QUERYSET_CROSS_ENCODER, SPATIAL_GRID_ENCODER, TARGET_MODE_ABSOLUTE
from ditar_encoder.lam.wan_vae_queryset_predictive_t5_lam import WanVaeQuerysetPredictiveT5Stage1LatentActionModel


OptimizerCallable = Callable[[Iterable], Optimizer]


class WanVaeQuerysetPredictiveT5Stage1LAM(LightningModule):
    """Lightning wrapper for the WAN VAE queryset predictive stage-1 LAM (T5)."""

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
        target_mode: str = TARGET_MODE_ABSOLUTE,
        encoder_arch: str = QUERYSET_CROSS_ENCODER,
        decoder_arch: str = DECODER_ARCH_SOURCE_QUERY_SELF_ATTN,
        query_num_tokens: int = 64,
        query_blocks: int = 8,
        kl_scope: str = KL_SCOPE_TARGET,
        ti_kl_weight: float = 1e-3,
        log_interval: int = 1000,
        log_path: str = 'ditar_encoder/logs',
        task_name: str = 'stage1_droid_wan_vae_queryset_predictive_t5',
        optimizer: OptimizerCallable = AdamW,
        wan_vae_model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        wan_vae_model_path: str = '',
        wan_vae_subfolder: str = 'vae',
        wan_vae_dtype: str = 'float32',
        wan_vae_sample_mode: str = 'mode',
        wan_vae_normalize_latents: bool = True,
        teacher_input_resolution: int = 256,
        num_frames: int = 17,
        t5_model_id: str = 't5-base',
        t5_model_path: str = '',
        t5_max_length: int = 64,
        text_cache_size: int = 0,
        max_decoder_text_tokens: int = 256,
        use_wandb: bool = False,
    ) -> None:
        from lightning import LightningModule

        LightningModule.__init__(self)

        normalized_target_mode = str(target_mode or TARGET_MODE_ABSOLUTE).strip().lower()
        if normalized_target_mode != TARGET_MODE_ABSOLUTE:
            raise ValueError('WAN VAE queryset predictive T5 LAM only supports target_mode="absolute".')

        self.lam = WanVaeQuerysetPredictiveT5Stage1LatentActionModel(
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
            t5_model_id=t5_model_id,
            t5_model_path=t5_model_path or None,
            t5_max_length=t5_max_length,
            text_cache_size=text_cache_size,
            max_decoder_text_tokens=max_decoder_text_tokens,
        )
        self.target_mode = normalized_target_mode
        self.encoder_arch = self.lam.encoder_arch
        self.decoder_arch = self.lam.decoder_arch
        self.kl_scope = self.lam.kl_scope
        self.ti_kl_weight = float(ti_kl_weight)
        self.log_interval = log_interval
        self.log_path = log_path
        self.task_name = task_name
        self.optimizer = optimizer
        self.use_wandb = use_wandb
        self.param_report = self._build_param_report()
        self._log_param_report()
        self.save_hyperparameters()

    def _log_param_report(self) -> None:
        summary = ', '.join(f'{key}={value}' for key, value in sorted(self.param_report.items()))
        print(
            '[wan-vae queryset predictive stage1 (t5)] '
            f'target_mode={self.target_mode} encoder_arch={self.encoder_arch} '
            f'decoder_arch={self.decoder_arch} kl_scope={self.kl_scope} '
            f'reduced_steps={self.lam.reduced_num_steps} '
            f'reduced_patches={self.lam.reduced_num_patches} {summary}',
            flush=True,
        )

    _ACTIVE_DIM_VAR_THRESHOLD = 1e-2

    _ACTIVE_DIM_KL_THRESHOLD = 1e-3

    _MAX_QUERY_SLOT_LOGS = 32

    _MAX_SORTED_KL_DIM_LOGS = 64

    def _build_param_report(self) -> Dict[str, int]:
        return {
            'encoder_total': sum(param.numel() for param in self.lam.ti_encoder.parameters() if param.requires_grad),
            'decoder_total': sum(param.numel() for param in self.lam.decoder.parameters() if param.requires_grad),
            'conv_total': sum(param.numel() for param in self.lam.input_proj.parameters() if param.requires_grad)
            + sum(param.numel() for param in self.lam.reducer.parameters() if param.requires_grad),
            'bottleneck_total': sum(param.numel() for param in self.lam.ti_bottleneck.parameters() if param.requires_grad),
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

    @staticmethod
    def _distribution_gini(values: Tensor) -> Tensor:
        values = values.reshape(-1).float().clamp_min(0.0)
        if values.numel() == 0:
            return values.new_zeros(())
        total = values.sum()
        if total <= 0:
            return values.new_zeros(())
        sorted_values = values.sort().values
        ranks = torch.arange(1, sorted_values.numel() + 1, device=values.device, dtype=values.dtype)
        n = values.new_tensor(float(sorted_values.numel()))
        return ((2.0 * ranks * sorted_values).sum() / (n * total) - (n + 1.0) / n).clamp_min(0.0)

    @staticmethod
    def _distribution_entropy(values: Tensor) -> Tensor:
        values = values.reshape(-1).float().clamp_min(0.0)
        if values.numel() == 0:
            return values.new_zeros(())
        total = values.sum()
        if total <= 0:
            return values.new_zeros(())
        probs = values / total.clamp_min(1e-12)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        if values.numel() <= 1:
            return values.new_zeros(())
        normalizer = values.new_tensor(float(values.numel())).log().clamp_min(1e-12)
        return entropy / normalizer

    @staticmethod
    def _participation_ratio(values: Tensor) -> Tensor:
        values = values.reshape(-1).float().clamp_min(0.0)
        if values.numel() == 0:
            return values.new_zeros(())
        numerator = values.sum().pow(2)
        denominator = values.pow(2).sum()
        return numerator / denominator.clamp_min(1e-12)

    @classmethod
    def _collect_mu_variance_decomposition_logs(cls, mu: Tensor, prefix: str) -> Dict[str, Tensor]:
        logs: Dict[str, Tensor] = {}
        if mu.ndim < 3:
            return logs
        mu_flat = mu.float().reshape(-1, mu.shape[-2], mu.shape[-1])
        if mu_flat.shape[0] < 2:
            return logs

        per_query_dim_var = mu_flat.var(dim=0, unbiased=False)
        per_query_active_frac = (per_query_dim_var > cls._ACTIVE_DIM_VAR_THRESHOLD).float().mean(dim=-1)
        logs[f'{prefix}_within_query_active_dim_frac_var'] = per_query_active_frac.mean()
        logs[f'{prefix}_per_query_active_dim_frac_var_mean'] = per_query_active_frac.mean()
        logs[f'{prefix}_per_query_active_dim_frac_var_std'] = per_query_active_frac.std(unbiased=False)
        logs[f'{prefix}_per_query_active_dim_frac_var_min'] = per_query_active_frac.min()
        logs[f'{prefix}_per_query_active_dim_frac_var_max'] = per_query_active_frac.max()

        query_slot_mean_by_dim = mu_flat.mean(dim=0)
        across_query_dim_var = query_slot_mean_by_dim.var(dim=0, unbiased=False)
        logs[f'{prefix}_across_query_slot_var'] = across_query_dim_var.mean()
        logs[f'{prefix}_across_query_slot_active_dim_frac_var'] = (
            across_query_dim_var > cls._ACTIVE_DIM_VAR_THRESHOLD
        ).float().mean()

        for query_idx in range(min(int(mu_flat.shape[1]), cls._MAX_QUERY_SLOT_LOGS)):
            logs[f'{prefix}_per_query_active_dim_frac_var_slot_{query_idx:02d}'] = per_query_active_frac[query_idx]
        return logs

    @classmethod
    def _collect_kl_distribution_logs(cls, outputs: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
        logs: Dict[str, Tensor] = {}
        per_dim_kl = outputs.get('bottleneck_per_dim_kl')
        if per_dim_kl is None or per_dim_kl.ndim < 3:
            return logs

        per_dim_kl_float = per_dim_kl.float()
        query_count = int(per_dim_kl_float.shape[-2])
        logs[f'{prefix}_query_count'] = per_dim_kl_float.new_tensor(float(query_count))

        kl_target_loss = outputs.get('kl_target_loss')
        if kl_target_loss is not None:
            logs[f'{prefix}_target_total_kl'] = kl_target_loss.float() * query_count
        kl_all_loss = outputs.get('kl_all_loss')
        if kl_all_loss is not None:
            logs[f'{prefix}_all_total_kl'] = kl_all_loss.float() * query_count

        per_token_kl = per_dim_kl_float.sum(dim=-1)
        all_per_query_values = per_token_kl.reshape(-1, query_count)
        all_per_query_mean = all_per_query_values.mean(dim=0)
        logs[f'{prefix}_all_per_query_kl_mean'] = all_per_query_mean.mean()
        logs[f'{prefix}_all_per_query_kl_std_across_slots'] = all_per_query_mean.std(unbiased=False)
        logs[f'{prefix}_all_per_query_kl_min'] = all_per_query_mean.min()
        logs[f'{prefix}_all_per_query_kl_max'] = all_per_query_mean.max()
        logs[f'{prefix}_all_per_query_kl_gini'] = cls._distribution_gini(all_per_query_mean)
        logs[f'{prefix}_all_per_query_kl_entropy'] = cls._distribution_entropy(all_per_query_mean)
        logs[f'{prefix}_all_eff_query_kl'] = cls._participation_ratio(all_per_query_mean)

        all_per_dim_mean = per_dim_kl_float.reshape(-1, per_dim_kl_float.shape[-1]).mean(dim=0)
        all_sorted_per_dim = all_per_dim_mean.sort(descending=True).values
        logs[f'{prefix}_all_eff_dim_kl'] = cls._participation_ratio(all_per_dim_mean)
        logs[f'{prefix}_all_active_dim_frac_kl'] = (
            all_per_dim_mean > cls._ACTIVE_DIM_KL_THRESHOLD
        ).float().mean()

        target_step_indices = outputs.get('target_step_indices')
        if target_step_indices is None or per_dim_kl_float.ndim != 4:
            sorted_limit = min(int(all_sorted_per_dim.numel()), cls._MAX_SORTED_KL_DIM_LOGS)
            for dim_idx in range(sorted_limit):
                logs[f'{prefix}_all_per_dim_kl_sorted_{dim_idx:02d}'] = all_sorted_per_dim[dim_idx]
            return logs

        batch_size = int(per_dim_kl_float.shape[0])
        batch_arange = torch.arange(batch_size, device=per_dim_kl_float.device)
        target_step_indices = target_step_indices.to(device=per_dim_kl_float.device, dtype=torch.long)
        target_per_dim_kl = per_dim_kl_float[batch_arange, target_step_indices]
        target_per_query_values = target_per_dim_kl.sum(dim=-1)
        target_per_query_mean = target_per_query_values.mean(dim=0)
        target_per_query_std = target_per_query_values.std(dim=0, unbiased=False)
        target_per_query_min = target_per_query_values.min(dim=0).values
        target_per_query_max = target_per_query_values.max(dim=0).values

        logs[f'{prefix}_target_per_query_kl_mean'] = target_per_query_mean.mean()
        logs[f'{prefix}_target_per_query_kl_std_across_slots'] = target_per_query_mean.std(unbiased=False)
        logs[f'{prefix}_target_per_query_kl_min'] = target_per_query_mean.min()
        logs[f'{prefix}_target_per_query_kl_max'] = target_per_query_mean.max()
        logs[f'{prefix}_target_per_query_kl_gini'] = cls._distribution_gini(target_per_query_mean)
        logs[f'{prefix}_target_per_query_kl_entropy'] = cls._distribution_entropy(target_per_query_mean)
        logs[f'{prefix}_target_eff_query_kl'] = cls._participation_ratio(target_per_query_mean)

        for query_idx in range(min(query_count, cls._MAX_QUERY_SLOT_LOGS)):
            slot_prefix = f'{prefix}_target_per_query_kl_slot_{query_idx:02d}'
            logs[f'{slot_prefix}_mean'] = target_per_query_mean[query_idx]
            logs[f'{slot_prefix}_std'] = target_per_query_std[query_idx]
            logs[f'{slot_prefix}_min'] = target_per_query_min[query_idx]
            logs[f'{slot_prefix}_max'] = target_per_query_max[query_idx]

        target_per_dim_mean = target_per_dim_kl.reshape(-1, target_per_dim_kl.shape[-1]).mean(dim=0)
        target_sorted_per_dim = target_per_dim_mean.sort(descending=True).values
        logs[f'{prefix}_target_eff_dim_kl'] = cls._participation_ratio(target_per_dim_mean)
        logs[f'{prefix}_target_active_dim_frac_kl'] = (
            target_per_dim_mean > cls._ACTIVE_DIM_KL_THRESHOLD
        ).float().mean()
        for dim_idx in range(min(int(target_sorted_per_dim.numel()), cls._MAX_SORTED_KL_DIM_LOGS)):
            logs[f'{prefix}_target_per_dim_kl_sorted_{dim_idx:02d}'] = target_sorted_per_dim[dim_idx]
        for dim_idx in range(min(int(all_sorted_per_dim.numel()), cls._MAX_SORTED_KL_DIM_LOGS)):
            logs[f'{prefix}_all_per_dim_kl_sorted_{dim_idx:02d}'] = all_sorted_per_dim[dim_idx]
        return logs

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
            logs.update(cls._collect_mu_variance_decomposition_logs(mu_float, prefix=prefix))
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
            logs.update(cls._collect_kl_distribution_logs(outputs, prefix=prefix))
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
            ('ti_tokens_t', 'ti_tokens_t_rms'),
        ):
            value = outputs.get(source_key)
            if value is not None:
                logs[log_key] = value.float().pow(2).mean().sqrt()
        return logs

    @classmethod
    def _collect_reconstruction_diagnostics(cls, outputs: Dict[str, Tensor], target_rms: Tensor) -> Dict[str, Tensor]:
        source_tokens = outputs.get('source_latent_l0')
        pred_tokens = outputs.get('pred_latent_t')
        target_tokens = outputs.get('target_latent_t')
        if source_tokens is None or pred_tokens is None or target_tokens is None:
            return {}

        source_float = source_tokens.float()
        pred_float = pred_tokens.float()
        target_float = target_tokens.float()
        source_copy_error = target_float - source_float
        pred_delta = pred_float - source_float
        target_delta = target_float - source_float
        delta_error = target_delta - pred_delta
        source_copy_rmse = source_copy_error.pow(2).mean().sqrt()
        delta_rmse = delta_error.pow(2).mean().sqrt()
        logs = {
            'source_copy_cosine': cls._mean_cosine_similarity(source_float, target_float),
            'source_copy_rmse': source_copy_rmse,
            'source_copy_relative_rmse': source_copy_rmse / target_rms.clamp_min(1e-8),
            'delta_recon_cosine': cls._mean_cosine_similarity(pred_delta, target_delta),
            'delta_rmse': delta_rmse,
            'pred_delta_rms': pred_delta.pow(2).mean().sqrt(),
            'target_delta_rms': target_delta.pow(2).mean().sqrt(),
        }
        for source_key, log_key in (
            ('decoder_ti_delta', 'decoder_ti_delta_rms'),
            ('decoder_text_delta', 'decoder_text_delta_rms'),
            ('decoder_tc_delta', 'decoder_tc_delta_rms'),
        ):
            value = outputs.get(source_key)
            if value is not None:
                logs[log_key] = value.float().pow(2).mean().sqrt()
        return logs

    @staticmethod
    def _distributed_sum(value: Tensor) -> Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value

    def _collect_target_step_logs(self, outputs: Dict[str, Tensor], recon_error: Tensor) -> Dict[str, Tensor]:
        target_step_indices = outputs.get('target_step_indices')
        pred_tokens = outputs.get('pred_latent_t')
        target_tokens = outputs.get('target_latent_t')
        source_tokens = outputs.get('source_latent_l0')
        if target_step_indices is None or pred_tokens is None or target_tokens is None or source_tokens is None:
            return {}

        target_step_indices = target_step_indices.to(device=recon_error.device, dtype=torch.long)
        batch_size = int(target_step_indices.shape[0])
        if batch_size == 0:
            return {}

        device = recon_error.device
        pred_tokens = pred_tokens.detach()
        target_tokens = target_tokens.detach()
        source_tokens = source_tokens.detach()
        recon_error = recon_error.detach()

        pred_flat = pred_tokens.reshape(batch_size, -1).float()
        target_flat = target_tokens.reshape(batch_size, -1).float()
        source_flat = source_tokens.reshape(batch_size, -1).float()
        recon_flat = recon_error.reshape(batch_size, -1).float()
        source_error_flat = (target_tokens - source_tokens).reshape(batch_size, -1).float()
        pred_delta_flat = pred_flat - source_flat
        target_delta_flat = target_flat - source_flat
        delta_error_flat = target_delta_flat - pred_delta_flat

        per_sample: Dict[str, Tensor] = {
            'recon_loss': recon_flat.pow(2).mean(dim=-1),
            'recon_cosine': F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8),
            'source_copy_loss': source_error_flat.pow(2).mean(dim=-1),
            'source_copy_cosine': F.cosine_similarity(source_flat, target_flat, dim=-1, eps=1e-8),
            'delta_recon_cosine': F.cosine_similarity(pred_delta_flat, target_delta_flat, dim=-1, eps=1e-8),
            'delta_loss': delta_error_flat.pow(2).mean(dim=-1),
        }

        per_dim_kl = outputs.get('bottleneck_per_dim_kl')
        if per_dim_kl is not None:
            batch_arange = torch.arange(batch_size, device=device)
            target_token_kl = per_dim_kl.detach().float().sum(dim=-1)[batch_arange, target_step_indices]
            per_sample['ti_kl_loss'] = target_token_kl.mean(dim=-1)

        mu = outputs.get('bottleneck_mu')
        if mu is not None:
            batch_arange = torch.arange(batch_size, device=device)
            target_mu = mu.detach().float()[batch_arange, target_step_indices]
            per_sample['ti_mu_rms'] = target_mu.pow(2).mean(dim=(-1, -2)).sqrt()

        logvar = outputs.get('bottleneck_logvar')
        if logvar is not None:
            batch_arange = torch.arange(batch_size, device=device)
            target_logvar = logvar.detach().float()[batch_arange, target_step_indices]
            per_sample['ti_logvar_mean'] = target_logvar.mean(dim=(-1, -2))
            per_sample['ti_posterior_std_mean'] = torch.exp(0.5 * target_logvar).mean(dim=(-1, -2))

        full_num_steps = int(getattr(self.lam, 'full_num_steps', int(target_step_indices.max().item()) + 1))
        total_count = self._distributed_sum(
            torch.tensor(float(batch_size), device=device, dtype=torch.float32)
        )
        logs: Dict[str, Tensor] = {}
        for step_idx in range(1, full_num_steps):
            mask = target_step_indices == step_idx
            count = self._distributed_sum(mask.float().sum())
            prefix = f'target_step_{step_idx}'
            logs[f'{prefix}_count'] = count.detach()
            logs[f'{prefix}_fraction'] = count / total_count.clamp_min(1.0)
            for metric_name, sample_values in per_sample.items():
                numerator = self._distributed_sum(torch.where(mask, sample_values, sample_values.new_zeros(())).sum())
                value = numerator / count.clamp_min(1.0)
                logs[f'{prefix}_{metric_name}'] = value
                if metric_name in {'recon_loss', 'source_copy_loss', 'delta_loss'}:
                    rmse_name = metric_name[:-4] + 'rmse'
                    logs[f'{prefix}_{rmse_name}'] = value.clamp_min(0.0).sqrt()
        return logs

    def shared_step(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor, Tuple[Tuple[str, Tensor], ...]]:
        outputs = self.lam(batch)
        recon_error = outputs['target_latent_t'] - outputs['pred_latent_t']
        recon_loss = recon_error.pow(2).mean()
        recon_rmse = recon_loss.sqrt()
        recon_mae = recon_error.abs().mean()
        target_rms = outputs['target_latent_t'].float().pow(2).mean().sqrt().clamp_min(1e-8)
        kl_loss = outputs['kl_loss']
        weighted_kl_loss = self.ti_kl_weight * kl_loss
        loss = recon_loss + weighted_kl_loss
        loss_logs: Dict[str, Tensor] = {
            'total_loss': loss,
            'recon_loss': recon_loss,
            'recon_rmse': recon_rmse,
            'recon_relative_rmse': recon_rmse / target_rms,
            'recon_mae': recon_mae,
            'recon_cosine': self._mean_cosine_similarity(outputs['pred_latent_t'], outputs['target_latent_t']),
            'ti_kl_loss': kl_loss,
            'weighted_ti_kl_loss': weighted_kl_loss,
            'target_step_mean': outputs['target_step_indices'].float().mean(),
        }
        if outputs.get('kl_target_loss') is not None:
            loss_logs['ti_kl_target_loss'] = outputs['kl_target_loss']
        if outputs.get('kl_all_loss') is not None:
            loss_logs['ti_kl_all_loss'] = outputs['kl_all_loss']
        query_count = int(getattr(self.lam, 'reduced_num_patches', 0) or 0)
        per_dim_kl = outputs.get('bottleneck_per_dim_kl')
        if per_dim_kl is not None:
            query_count = int(per_dim_kl.shape[-2])
        if query_count > 0:
            loss_logs['kl_query_count'] = loss.new_tensor(float(query_count))
            loss_logs['ti_kl_effective_total_weight'] = loss.new_tensor(float(self.ti_kl_weight) / query_count)
            if outputs.get('kl_target_loss') is not None:
                loss_logs['target_total_kl'] = outputs['kl_target_loss'] * query_count
            if outputs.get('kl_all_loss') is not None:
                loss_logs['all_total_kl'] = outputs['kl_all_loss'] * query_count
        loss_logs.update(self._collect_bottleneck_logs(outputs, prefix='ti'))
        loss_logs.update(self._collect_scale_logs(outputs))
        loss_logs.update(self._collect_reconstruction_diagnostics(outputs, target_rms=target_rms))
        return outputs, loss, tuple(loss_logs.items())

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        outputs, loss, aux_losses = self.shared_step(batch)
        recon_error = outputs['target_latent_t'] - outputs['pred_latent_t']
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
        target_step_logs = self._collect_target_step_logs(outputs, recon_error)
        if target_step_logs:
            self.log_dict(
                {f'train/{key}': value for key, value in target_step_logs.items()},
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch_size,
            )
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        outputs, loss, aux_losses = self.shared_step(batch)
        recon_error = outputs['target_latent_t'] - outputs['pred_latent_t']
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
        target_step_logs = self._collect_target_step_logs(outputs, recon_error)
        if target_step_logs:
            self.log_dict(
                {f'val/{key}': value for key, value in target_step_logs.items()},
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
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
                    'Error(s) in loading state_dict for VJEPA2WanConvCompactPredictiveStage1LAM: '
                    + '; '.join(problems)
                )
        return incompatible
