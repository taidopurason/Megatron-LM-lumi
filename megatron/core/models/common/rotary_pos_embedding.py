# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import logging
import math
from typing import Optional

import torch
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']
logger = logging.getLogger(__name__)

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            seq_len_interpolation_factor: Optional[float] = None,
            theta: int = 10000,
            scaling_type: Optional[str] = None,
            scaling_factor: Optional[float] = None,
            high_freq_factor: Optional[float] = None,
            low_freq_factor: Optional[float] = None,
            original_max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.theta = theta
        logger.info(f"rope theta: {self.theta}")
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))

        if scaling_type == "llama3":
            assert scaling_factor is not None, "rope_factor must be provided for llama3 RoPE"
            assert low_freq_factor is not None, "low_freq_factor must be provided for llama3 RoPE"
            assert high_freq_factor is not None, "high_freq_factor must be provided for llama3 RoPE"
            assert original_max_position_embeddings is not None, (
                "original_max_position_embeddings must be provided for llama3 RoPE"
            )
            logger.info(
                f"Llama-3 rope scaling: scaling_factor={scaling_factor} "
                f"low_freq_factor={low_freq_factor} high_freq_factor={high_freq_factor} "
                f"original_max_position_embeddings={original_max_position_embeddings}"
            )

            inv_freq = self._llama3_rope_scaling(
                inv_freq,
                factor=scaling_factor,
                high_freq_factor=high_freq_factor,
                low_freq_factor=low_freq_factor,
                original_max_position_embeddings=original_max_position_embeddings
            )
        elif scaling_type != "default" and scaling_type is not None:
            raise ValueError(f"Invalid RoPE type: {scaling_type}")

        self.register_buffer('inv_freq', inv_freq, persistent=False)


    def _llama3_rope_scaling(
            self,
            inv_freq: torch.Tensor,
            factor: float,
            high_freq_factor: float,
            low_freq_factor: float,
            original_max_position_embeddings: int
    ) -> torch.Tensor:
        # https://github.com/huggingface/transformers/blob/d5a99dfcee6e94065cb7c83cc8ab6fc5daa0cc4e/src/transformers/modeling_rope_utils.py#L298
        old_context_len = original_max_position_embeddings
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb[:, None, None, :]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1).bfloat16()
