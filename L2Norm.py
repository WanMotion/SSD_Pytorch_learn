from torch import nn
import torch


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        """
        :param n_channels: 卷积后维度
        :param scale: 放大倍数
        """
        super(L2Norm, self).__init__()
        self.eps = 1e-10
        self.n_channels = n_channels
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        power = torch.pow(x, 2)
        sum = torch.sum(power, dim=1, keepdim=True)
        norm = sum.sqrt() + self.eps # 加eps为了避免在除的时候除以0
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x  # unsqueeze拓展维度，等于就是x中的每个值乘以了scale
