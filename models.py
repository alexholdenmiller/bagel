
import torch
import torch.nn as nn

from mixed_res.patch_scorers.feature_based_patch_scorer import FeatureBasedPatchScorer
from mixed_res.quadtree_impl.quadtree_z_curve import ZCurveQuadtreeRunner
from mixed_res.tokenization.patch_embed import FlatPatchEmbed
from mixed_res.tokenization.tokenizers import QuadtreeTokenizer

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
                              stride=self.patch_size)
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
    def __init__(self, flags, in_channels, num_classes):
        super().__init__()

        edim = flags.embed_dim
        drop = flags.dropout

        self.patch_embed = PatchEmbedding(flags.image_size, flags.patch_size, in_channels, flags.embed_dim)
        self.embed_len = self.patch_embed.num_patches + 1
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
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        # x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        # x = self.norm(x)
        logits = self.cls_head(x[:, 0])

        return logits, {}


class MixedResViT(nn.Module):
    def __init__(self, flags, in_channels, num_classes):
        super().__init__()

        edim = flags.embed_dim
        drop = flags.dropout

        # self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, viz)
        # self.embed_len = self.patch_embed.num_patches + 1
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, edim))

        self.patch_embed = FlatPatchEmbed(img_size=flags.image_size, patch_size=flags.min_patch_size, embed_dim=edim)
        self.quadtree_runner = ZCurveQuadtreeRunner(flags.quadtree_num_patches, flags.min_patch_size, flags.max_patch_size)
        self.patch_scorer = FeatureBasedPatchScorer()
        self.quadtree_tokenizer = QuadtreeTokenizer(self.patch_embed, self.cls_token,
                                                    self.quadtree_runner, self.patch_scorer)

        self.dropout = nn.Dropout(drop)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(edim, flags.num_heads, flags.mlp_dim, drop) for i in range(flags.num_layers)
        ])
        self.norm = nn.LayerNorm(edim)
        self.cls_head = nn.Sequential(nn.Linear(edim, edim//2),
                                nn.GELU(),
                                nn.Dropout(drop),
                                nn.Linear(edim // 2, num_classes),
                                )

    def forward(self, x):
        x = self.quadtree_tokenizer.tokenize(x)
        # x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        # x = self.norm(x)
        logits = self.cls_head(x[:, 0])

        return logits, {}