
import torch
import torch.nn as nn

from gumbel_vector_quantizer import GumbelVectorQuantizer


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(self.in_channels,
                            self.embed_dim,
                            kernel_size=self.patch_size,
                            stride=self.patch_size
                           )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.qk_norm = False
        # self.use_activation = False
        # self.activation = nn.ReLU() if self.use_activation else nn.Identity()
        self.q_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(0)

    def forward(self, x):
        batch_si, seq_len, emb_dim = x.shape
        qkv = self.qkv(x).reshape(batch_si, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attention = q @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_dropout(attention)

        z = attention @ v
        z = z.transpose(1, 2).reshape(batch_si, seq_len, emb_dim)
        z = self.proj(z)
#         z = self.proj_dropout(z)
        return z
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(mlp_dim, embed_dim)
                                  # nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        res = x
        x = self.layer_norm(x)
        x = self.attention(x)

        x = x + res # residual connection
        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + res
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.embed_len = self.patch_embed.num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, embed_dim//2),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(embed_dim // 2, num_classes),
                                )

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        # x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        # x = self.norm(x)
        logits = self.cls_head(x[:, 0])

        return logits


class VitWithConvs(nn.Module):
    def __init__(self, flags, in_channels, num_classes):
        super().__init__()

        edim = flags.embed_dim
        drop = flags.dropout

        # set up feature extractor
        self.conv2d = nn.Conv2d(in_channels,
                                edim * flags.conv_groups,
                                flags.conv_kernel,
                                groups=flags.conv_groups,
                                stride=flags.conv_stride,
                                dilation=flags.conv_dilate)
        self.proj_feats = nn.Linear(edim * flags.conv_groups, edim)
        self.feat_drop = nn.Dropout(drop)
        self.feat_norm = nn.LayerNorm(edim)

        self.embed_len = int(((flags.image_size - flags.conv_dilate * (flags.conv_kernel - 1) + 1) / flags.conv_stride + 1) ** 2) + 1  # max seqlen
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, edim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, edim))
        self.dropout = nn.Dropout(drop)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(edim, flags.num_heads, flags.mlp_dim, drop) for i in range(flags.num_layers)
        ])
        self.norm = nn.LayerNorm(edim)
        self.cls_head = nn.Sequential(nn.Linear(edim, edim // 2),
                                nn.GELU(),
                                nn.Dropout(drop),
                                nn.Linear(edim // 2, num_classes),
                                )

    def forward(self, x):
        # feature extract
        x = self.conv2d(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj_feats(x)
        x = self.feat_drop(x)
        x = self.feat_norm(x)

        # add class embedding
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # add positional embedding
        x = x + self.pos_embed[:, -x.size(1):, :]

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.cls_head(x[:, 0])

        return logits

class VitWithVQ(nn.Module):
    def __init__(self, flags, in_channels, num_classes):
        super().__init__()

        edim = flags.embed_dim
        drop = flags.dropout

        # set up feature extractor
        self.conv2d = nn.Conv2d(in_channels,
                                edim * flags.conv_groups,
                                flags.conv_kernel,
                                groups=flags.conv_groups,
                                stride=flags.conv_stride,
                                dilation=flags.conv_dilate)
        self.proj_feats = nn.Linear(edim * flags.conv_groups, edim)
        self.feat_drop = nn.Dropout(drop)
        self.feat_norm = nn.LayerNorm(edim)

        # set up quantizer
        vq_dim = flags.vq_dim if flags.vq_dim > 0 else edim
        self.vq = GumbelVectorQuantizer(
            dim=edim,  # input dimension
            num_vars=flags.vq_vars,
            temp=flags.vq_temp,
            groups=flags.vq_groups,
            combine_groups=False,
            vq_dim=vq_dim,  # output dimension
            time_first=True,
            weight_proj_depth=flags.vq_depth,
            weight_proj_factor=flags.vq_factor,
        )
        self.project_vs = nn.Linear(vq_dim, edim)

        self.embed_len = int(((flags.image_size - flags.conv_dilate * (flags.conv_kernel - 1) + 1) / flags.conv_stride + 1) ** 2) + 1  # max seqlen
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, edim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, edim))
        self.dropout = nn.Dropout(drop)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(edim, flags.num_heads, flags.mlp_dim, drop) for i in range(flags.num_layers)
        ])
        self.norm = nn.LayerNorm(edim)
        self.cls_head = nn.Sequential(nn.Linear(edim, edim // 2),
                                nn.GELU(),
                                nn.Dropout(drop),
                                nn.Linear(edim // 2, num_classes),
                                )

    def forward(self, x):
        # feature extract
        x = self.conv2d(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj_feats(x)
        x = self.feat_drop(x)
        x = self.feat_norm(x)
    
        vs = self.vq(x)
        x = self.project_vs(vs)

        # add class embedding
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # add positional embedding
        x = x + self.pos_embed[:, -x.size(1):, :]

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.cls_head(x[:, 0])

        return logits