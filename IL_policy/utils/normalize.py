import torch
from torch import Tensor, nn

def create_stats_buffers(
    keys: list[str],
    stats: dict,
) -> dict:
    stats_buffers = {}
    for key, stat in stats.items():
        if key not in keys:
            continue
        vmin = torch.tensor(stat["min"], dtype=torch.float32)
        vmax = torch.tensor(stat["max"], dtype=torch.float32)
        buffer = nn.ParameterDict(
            {
                "min": nn.Parameter(vmin, requires_grad=False),
                "max": nn.Parameter(vmax, requires_grad=False),
            }
        )
        stats_buffers[key] = buffer
    return stats_buffers

class NormalizeMinMax(nn.Module):
    """MinMax normalization: (x - min) / (max - min) * 2 - 1  →  [-1, 1]"""
    def __init__(self, keys: list[str], stats: dict):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key in self.keys:
            if key in batch:
                buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                vmin = buffer["min"]
                vmax = buffer["max"]
                batch[key] = (batch[key] - vmin) / (vmax - vmin + 1e-8) * 2 - 1
        return batch

class UnnormalizeMinMax(nn.Module):
    """MinMax unnormalization: (x + 1) / 2 * (max - min) + min"""
    def __init__(self, keys: list[str], stats: dict):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key in self.keys:
            if key in batch:
                buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                vmin = buffer["min"]
                vmax = buffer["max"]
                batch[key] = (batch[key] + 1) / 2 * (vmax - vmin) + vmin
        return batch
