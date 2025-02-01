import math

import torch
import numpy as np

from torch import nn, einsum

from .ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        self.n_heads = n_heads
        inner_dim = hidden_dim * n_heads

        self.ln = nn.LayerNorm(in_channels)
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1).float()
        return attn

class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        # out = None
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)

        feature = rearrange(x, 'n c t v -> (n t) v c')
        z = einsum('N h v y, N y c -> N h v c', [A, feature])
        z = rearrange(z, '(n t) h v c-> n h c t v', t=T).contiguous()
        out = [conv(z[:, i, :, :, :]) for i, conv in enumerate(self.conv_d)]
        out = torch.stack(out, dim=1)
        out = torch.sum(out, dim=1, keepdim=False)

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MS_TCN(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, attn=None):
        y = self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))
        return y

