import math
import torch
from torch import nn, Tensor


def pad(kernel_size, dilation=1) -> int:
    pad = (kernel_size - 1) // 2 * dilation
    return pad


def _make_divisible(value: float, divisor=8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def round_repeats(num_repeats: int, depth_mult: float) -> int:
    if depth_mult == 1.0:
        return num_repeats
    return int(math.ceil(num_repeats * depth_mult))


def round_filters(filters: int, width_mult: float) -> int:
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _stochastic_depth(x: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    assert 0.0 < p < 1.0, f"drop probability has to be between 0 and 1, but got {p}"
    assert mode in ["batch", "row"], f"mode has to be either 'batch' or 'row', but got {mode}"
    if not training or p == 0.0:
        return x

    survival_rate = 1.0 - p
    size = [x.shape[0]] + [1] * (x.ndim - 1)

    noise = torch.empty(size, dtype=x.dtype, device=x.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return x * noise


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return _stochastic_depth(x, self.p, self.mode, self.training)
