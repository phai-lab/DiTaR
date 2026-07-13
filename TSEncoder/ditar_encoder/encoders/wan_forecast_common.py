import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat



def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = (dtype_name or 'float16').lower()
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    if dtype_name == 'float32':
        return torch.float32
    return torch.float16


def resolve_wan_autoencoder():
    os.environ.setdefault('_CHECK_PEFT', '0')
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise ImportError(
            'diffusers with AutoencoderKLWan support is required for the WAN forecast path.'
        ) from exc
    return AutoencoderKLWan






class FrozenWanLatentTeacher(nn.Module):
    def __init__(
        self,
        wan_vae_model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        wan_vae_model_path: Optional[str] = None,
        wan_vae_subfolder: str = 'vae',
        wan_vae_dtype: str = 'float32',
        wan_latent_sample_mode: str = 'mode',
        wan_normalize_latents: bool = True,
        teacher_num_frames: Optional[int] = None,
        teacher_input_resolution: Optional[int] = None,
    ) -> None:
        super().__init__()
        if wan_latent_sample_mode not in {'mode', 'sample'}:
            raise ValueError(f'Unsupported wan_latent_sample_mode={wan_latent_sample_mode!r}.')

        AutoencoderKLWan = resolve_wan_autoencoder()
        wan_source = wan_vae_model_path or wan_vae_model_id
        wan_dtype = resolve_torch_dtype(wan_vae_dtype)
        wan_subfolder = wan_vae_subfolder or None
        self.wan_vae = AutoencoderKLWan.from_pretrained(
            wan_source,
            subfolder=wan_subfolder,
            torch_dtype=wan_dtype,
            low_cpu_mem_usage=True,
        )
        self.wan_vae.requires_grad_(False)
        self.wan_vae.eval()

        self.wan_latent_sample_mode = wan_latent_sample_mode
        self.wan_normalize_latents = bool(wan_normalize_latents)

        wan_config = self.wan_vae.config
        self.scale_factor_spatial = int(getattr(wan_config, 'scale_factor_spatial', 16))
        self.scale_factor_temporal = int(getattr(wan_config, 'scale_factor_temporal', 4))
        self.latent_dim = int(getattr(wan_config, 'z_dim', 0))
        if self.latent_dim <= 0:
            raise ValueError(f'Invalid WAN latent dimension: {self.latent_dim}.')

        # Optional duck-type aliases so this teacher can stand in for FrozenVJEPA2Teacher.
        # Both are required when this teacher is used by the queryset predictive LAM.
        self._teacher_num_frames = int(teacher_num_frames) if teacher_num_frames is not None else None
        self._teacher_input_resolution = int(teacher_input_resolution) if teacher_input_resolution is not None else None
        if self._teacher_num_frames is not None and self._teacher_num_frames <= 0:
            raise ValueError(f'teacher_num_frames must be positive when set, got {self._teacher_num_frames}.')
        if self._teacher_num_frames is not None and (self._teacher_num_frames - 1) % self.scale_factor_temporal != 0:
            raise ValueError(
                f'teacher_num_frames={self._teacher_num_frames} must satisfy the WAN VAE 4k+1 rule '
                f'(scale_factor_temporal={self.scale_factor_temporal}).'
            )
        if self._teacher_input_resolution is not None and self._teacher_input_resolution % self.scale_factor_spatial != 0:
            raise ValueError(
                f'teacher_input_resolution={self._teacher_input_resolution} must be divisible by '
                f'scale_factor_spatial={self.scale_factor_spatial}.'
            )

        latents_mean = torch.tensor(getattr(wan_config, 'latents_mean', [0.0] * self.latent_dim), dtype=torch.float32)
        latents_std = torch.tensor(getattr(wan_config, 'latents_std', [1.0] * self.latent_dim), dtype=torch.float32)
        if latents_mean.numel() != self.latent_dim or latents_std.numel() != self.latent_dim:
            raise ValueError(
                'WAN config latents_mean/std do not match z_dim: '
                f'{latents_mean.numel()} / {latents_std.numel()} vs {self.latent_dim}.'
            )
        self.register_buffer('wan_latents_mean', latents_mean.view(1, self.latent_dim, 1, 1, 1), persistent=False)
        self.register_buffer(
            'wan_latents_std_inv',
            latents_std.reciprocal().view(1, self.latent_dim, 1, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f'Expected videos to have shape [B, T, C, H, W], got {tuple(videos.shape)}.')

        model_parameter = next(self.wan_vae.parameters())
        wan_videos = videos.permute(0, 2, 1, 3, 4)
        wan_videos = wan_videos.mul(2.0).sub(1.0).to(model_parameter.device, dtype=model_parameter.dtype)

        encoded = self.wan_vae.encode(wan_videos)
        if hasattr(encoded, 'latents') and encoded.latents is not None:
            latents = encoded.latents
        elif hasattr(encoded, 'latent_dist') and encoded.latent_dist is not None:
            if self.wan_latent_sample_mode == 'sample':
                latents = encoded.latent_dist.sample()
            else:
                latents = encoded.latent_dist.mode()
        else:
            raise ValueError('WAN encoder output does not expose latents or latent_dist.')

        latents = latents.float()
        if self.wan_normalize_latents:
            latents = (latents - self.wan_latents_mean.to(latents.device)) * self.wan_latents_std_inv.to(latents.device)
        return latents.permute(0, 2, 3, 4, 1).reshape(latents.shape[0], latents.shape[2], -1, latents.shape[1])

    @property
    def teacher_num_frames(self) -> int:
        if self._teacher_num_frames is None:
            raise AttributeError(
                'teacher_num_frames is not configured on this FrozenWanLatentTeacher; '
                'pass teacher_num_frames in the constructor when using as a teacher protocol.'
            )
        return self._teacher_num_frames

    @property
    def teacher_input_resolution(self) -> int:
        if self._teacher_input_resolution is None:
            raise AttributeError(
                'teacher_input_resolution is not configured on this FrozenWanLatentTeacher; '
                'pass teacher_input_resolution in the constructor when using as a teacher protocol.'
            )
        return self._teacher_input_resolution

    @property
    def teacher_num_steps(self) -> int:
        return (self.teacher_num_frames - 1) // self.scale_factor_temporal + 1

    @property
    def teacher_grid_size(self) -> int:
        return self.teacher_input_resolution // self.scale_factor_spatial

    @property
    def teacher_num_patches(self) -> int:
        grid = self.teacher_grid_size
        return grid * grid

    @property
    def teacher_feature_dim(self) -> int:
        return self.latent_dim

    @property
    def tubelet_size(self) -> int:
        return self.scale_factor_temporal














