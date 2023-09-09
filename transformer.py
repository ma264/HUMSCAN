import torch
from torch import nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from einops import repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)   # 判断变量类型


# 层标准化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        '''
        :param dim: D维度  1*D的维度
        :param num_heads:  num_heads为hn
        :param qkv_bias: q,k,v是否存在偏置
        :param qk_scale: q,k的scale
        :param attn_drop: 丢包率
        :param proj_drop: 丢包率
        '''
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)  # 无偏置
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # B=1, N=362 C=600
        # B1C -> B1H(C/H) -> BH1(C/H)  x[:, 0:1, ...]==xcls.
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("q=", q.shape)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("k=", k.shape)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("v=", v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # print("attn=", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # print("x=", x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:   # 是否存在全连接层
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x[:, 1:, :]))) 
        x = x[:, 0:1, ...] + self.drop_path(self.attn(x))  # Better result
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # depth层
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # i = 0
        for attn, ff in self.layers:   # 循环depth次
            # print(i)
            x = torch.cat((attn(x), self.norm(x[:, 1:, :])), dim=1)
            x = ff(x) + x   # 残差学习
            # i = i + 1
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim,
                 pool='cls', dropout=0., emb_dropout=0.):
        '''
        :param image_size: 输入图像大小
        :param patch_size: patch处理大小
        :param dim: 潜在维度
        :param depth: 深度
        :param heads: hn的大小
        :param mlp_dim: 全连接层维度
        :param pool: 池化类型
        :param dropout: 参数丢包率
        :param emb_dropout:
        '''
        super().__init__()
        image_height, image_width = pair(image_size)  # 将image_size转化为行列
        patch_height, patch_width = pair(patch_size)  # 将patch转化为行列
        # 保证图像能被分为整数个块
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size. '
        # m=image_height // patch_height  n=image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        # 确定池化类型
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # rearrange用于对张量的维度进行重新变换排序
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # nn.Linear(patch_dim, dim),
        )
        # nn.Parameter将输入转化为可以训练的网络数据
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 实现对数据的块处理
        # print("编码器输出后的patch操作", x.shape)
        # b=1, n=m*n, _=p1*p2*C
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # print("嵌入数据准备", cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)   # 进行数据嵌套
        # print("编码器结果与嵌套数据相结合", x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        # print("随机生成的与嵌套后结果相同的相加", x.shape)
        x = self.dropout(x)

        x = self.transformer(x)
        # print("经过多头注意力机制后", x.shape)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # print("取均值结果", x.shape)
        x = self.to_latent(x)
        
        return x
