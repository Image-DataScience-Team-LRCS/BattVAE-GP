from __future__ import annotations

from typing import Optional

import torch


def build_condition_vector(
    norm_cycle_numbers: torch.Tensor,
    charging_rate: torch.Tensor,
    norm_nominal_capacity: torch.Tensor,
    cond_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build the decoder conditioning vector using the same layout as training.

    Supported layouts:
      dim=0: None
      dim=1: [norm_cycle]
      dim=2: [norm_cycle, charging_rate]
      dim=3: [norm_cycle, charging_rate, norm_nominal_capacity]
    """
    cond_dim = int(cond_dim)
    if cond_dim <= 0:
        return None

    if cond_dim == 1:
        parts = [norm_cycle_numbers.unsqueeze(-1)]
    elif cond_dim == 2:
        parts = [
            norm_cycle_numbers.unsqueeze(-1),
            charging_rate.unsqueeze(-1),
        ]
    elif cond_dim == 3:
        parts = [
            norm_cycle_numbers.unsqueeze(-1),
            charging_rate.unsqueeze(-1),
            norm_nominal_capacity.unsqueeze(-1),
        ]
    else:
        raise ValueError(
            f"Unsupported conditional_vector_dim={cond_dim}. "
            "Use one of {0,1,2,3}."
        )

    return torch.cat(parts, dim=1).to(device).contiguous()
