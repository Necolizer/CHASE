import torch
import torch.nn as nn
from einops import rearrange


class CHASWrapper(nn.Module):
    def __init__(self, backbone, in_channels=3, num_frame=64, num_point=25, pooling_seg=[1,1,1], num_entity=2, c1=64, c2=8):
        super(CHASWrapper, self).__init__()

        out_channel = num_frame * num_point * num_entity

        self.pooling_seg = pooling_seg
        self.seg = self.pooling_seg[0]*self.pooling_seg[1]*self.pooling_seg[2]
        self.seg_num_list = [(num_frame//self.pooling_seg[0]), (num_point//self.pooling_seg[1]), (num_entity//self.pooling_seg[2])]
        self.seg_num = self.seg_num_list[0] * self.seg_num_list[1] * self.seg_num_list[2]
        self.shift = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=c1, kernel_size=1),
            nn.AdaptiveAvgPool3d((self.pooling_seg[0], self.pooling_seg[1], self.pooling_seg[2])),
            nn.Conv3d(c1, c2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(c2, out_channel, 1, bias=False),
        )

        self.backbone = backbone
    
    def forward(self, x):
        N, C, T, V, M = x.size()

        sf = self.shift(x).view(N, T*M*V, -1)
        tx = rearrange(x, 'n c t v m -> n c (t v m)', t=T, m=M, v=V).contiguous()
        sf = (tx @ sf.softmax(dim=1)).unsqueeze(-1).expand(-1, -1, self.seg, self.seg_num)
        sf = rearrange(sf, 'n c (T V M) (t v m) -> n c (T t) (V v) (M m)', 
                    T=self.pooling_seg[0], V=self.pooling_seg[1], M=self.pooling_seg[2], 
                    t=self.seg_num_list[0], v=self.seg_num_list[1], m=self.seg_num_list[2]).contiguous()
        x = x - sf

        if self.training:
            xt = rearrange(x, 'n c t v m -> n (t v) c m', t=T, v=V).contiguous()
            sorted_indices, _ = torch.sort(torch.randperm(xt.size(-1))[:2].to(xt.device))
            xt = xt.index_select(-1, sorted_indices.to(xt.device))
            xt1, _ = xt.max(1, keepdim=True)
            xt2, _ = xt.min(1, keepdim=True)
            xt = torch.cat([xt1, xt2], dim=1)
            x1 = xt[:,:,:,0]
            x2 = xt[:,:,:,1]

        out = self.backbone(x)

        if self.training:
            if isinstance(out, tuple):
                return out + (x1, x2)
            else:
                return (out, x1, x2)
        else:
            if isinstance(out, tuple):
                return out[0]
            else:
                return out