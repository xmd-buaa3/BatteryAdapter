"""Multi-scale Gaussian MMD ported from flow_final, preserving feature gradients."""
import torch
from torch import nn


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2., kernel_num=5):
        super().__init__()
        if kernel_mul <= 0 or kernel_num < 1:
            raise ValueError('MMD kernel parameters must be positive')
        self.kernel_mul = float(kernel_mul)
        self.kernel_num = int(kernel_num)

    def forward(self, source, target):
        if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
                or not len(source) or not len(target)):
            raise ValueError('MMD requires nonempty [batch, shared_features] tensors')
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError('MMD features must be finite')
        values = torch.cat((source, target), 0)
        distance = (values[:, None] - values[None, :]).square().sum(-1)
        count = len(values)
        bandwidth = (distance.detach().sum() / (count * count - count)).clamp_min(torch.finfo(values.dtype).eps)
        bandwidth = bandwidth / self.kernel_mul ** (self.kernel_num // 2)
        kernels = sum(torch.exp(-distance / (bandwidth * self.kernel_mul ** index))
                      for index in range(self.kernel_num))
        n = len(source)
        return (kernels[:n, :n].mean() + kernels[n:, n:].mean()
                - kernels[:n, n:].mean() - kernels[n:, :n].mean())
