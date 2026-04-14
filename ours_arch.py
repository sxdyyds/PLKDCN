import math
import numbers
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

# PCFN
class B(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        # self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        # DWConv
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1, groups=p_dim)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x


# 部分大核卷积
class A(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim*2, 1, 1, 0)

        # LKDWConv
        self.fc = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        # LKConv
        # self.fc = nn.Conv2d(dim//2, dim//2, kernel_size, 1, kernel_size // 2)

        self.conv2 = nn.Conv2d(dim*2, dim, 1, 1, 0)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.act(self.fc(x1))
        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)
        x = self.act(self.conv2(x))
        return x


class EA(nn.Module):
    " Element-wise Attention "
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.f(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        # return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# LN
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Shuffle Mixing layer
class OurBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        # self.norm1 = LayerNorm(dim)
        # self.norm2 = LayerNorm(dim)

        self.mlp1 = A(dim, kernel_size)
        self.mlp2 = B(dim, mlp_ratio)
        # x2
        # self.ea = EA(dim)

        # TODO 已弃用
        # x3x4
        # self.attn = EA(dim)

    def forward(self, x):
        # x = self.mlp1(self.norm1(x)) + x
        # x = self.mlp2(self.norm2(x)) + x
        x = self.mlp1(F.normalize(x)) + x
        x = self.mlp2(F.normalize(x)) + x

        # DIV2K x2
        # x = self.ea(x)

        # TODO 已弃用
        # DIV2K x3x4 DF2K x2x3x4
        # x = self.attn(x)
        return x

@ARCH_REGISTRY.register()
class OURS(nn.Module):
    """
    Args:
        n_feats (int): Number of channels. Default: 64 (32 for the tiny model).
        n_blocks (int): Number of feature mixing blocks. Default: 5.
        mlp_ratio (int): The expanding factor of point-wise MLP. Default: 2.
        upscaling_factor: The upscaling factor. [2, 3, 4]
    """
    def __init__(self, n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor=4):
        super().__init__()

        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, n_feats, 3, 1, 1, bias=False)

        self.blocks = nn.Sequential(
            *[OurBlock(n_feats, kernel_size, mlp_ratio) for _ in range(n_blocks)]
        )

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_feats, n_feats * 4 , 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale),
                nn.SiLU(inplace=True)
            )

        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        base = x
        x = self.to_feat(x)
        x = self.blocks(x)
        x = self.upsapling(x)
        x = self.tail(x)
        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + base

if __name__ == '__main__':

    from fvcore.nn import FlopCountAnalysis

    '''
    base
    network_g:
      type: Ours
      n_feats: 48
      kernel_size: 17
      n_blocks: 8
      mlp_ratio: 2
      upscaling_factor: 4
    '''

    model = OURS(n_feats=64, kernel_size=9, n_blocks=10, mlp_ratio=2, upscaling_factor=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = torch.randn(1, 3, 180, 320).to(device)

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print("params(K)", num_parameters / 10 ** 3)

    flops = FlopCountAnalysis(model.to(device), input)
    print("FLOPs(G)", flops.total() / 10 ** 9)



