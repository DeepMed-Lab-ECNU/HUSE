# --------------------------------------------------------
# HUSE: Histocomponent-driven Universal Model for Virtual IHC
#       Multiplex Staining via Joint Manifold Evolution
# References:
#   JiT: https://github.com/LTH14/JiT
#   SiT: https://github.com/willisma/SiT
#   Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


# ---------------- Histocomponent-driven MoE & Representation Conflict Gating ----------------
class ProPathoRouter(nn.Module):
    """Histocomponent-driven router (Eq. 3): hard assignment by cosine similarity
    between tokens and learnable histocomponent prototypes."""
    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_experts, dim))
        self.temperature = 0.07

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.matmul(x_norm, p_norm.transpose(0, 1)) / self.temperature
        indices = torch.argmax(logits, dim=-1, keepdim=True)
        return indices


class ProPathoMoE(nn.Module):
    """Hi-MoE + RCG (Eq. 4, 5): shared expert keeps the global histological scaffold,
    RCG dynamically gates the contribution of the specialized histocomponent expert."""
    def __init__(self, dim, hidden_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.shared_expert = SwiGLUFFN(dim, hidden_dim)
        self.specialized_experts = nn.ModuleList([
            SwiGLUFFN(dim, hidden_dim) for _ in range(num_experts)
        ])
        self.router = ProPathoRouter(dim, num_experts)
        self.rcg_predictor = nn.Linear(dim * 2, 1)
        nn.init.constant_(self.rcg_predictor.weight, 0)
        nn.init.constant_(self.rcg_predictor.bias, -2.0)

    def forward(self, x, num_cls_tokens=0):
        ffn_out = self.shared_expert(x)
        delta = ffn_out
        indices = self.router(x)
        out_special = torch.zeros_like(x)
        for i, expert in enumerate(self.specialized_experts):
            mask = (indices == i).float()
            if mask.sum() > 0:
                out_special += mask * expert(x)

        # RCG: stop-gradient observer over [x ; delta]
        gate_input = torch.cat([x.detach(), delta.detach()], dim=-1)
        gate_value = torch.sigmoid(self.rcg_predictor(gate_input))
        out = ffn_out + gate_value * out_special
        return out, indices, gate_value


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=6, pca_dim=128, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Maps a discrete marker index into a learnable token (class embedder)."""
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = ProPathoMoE(hidden_size, mlp_hidden_dim, num_experts=3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None, num_cls_tokens=0):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope
        )

        tokens = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out, indices, gate_value = self.mlp(tokens, num_cls_tokens=num_cls_tokens)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x, indices, gate_value


class JiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=6,        # Joint Manifold Anchoring: X_t = [H&E || z_t] in R^{H x W x 6}
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        label_dropout_prob=0.1,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        clip_dim=512,
        clip_anchor_dir="PATH/TO/CLIP_ANCHOR_FEATURES",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # predict 6 channels, the last 3 are the virtual IHC result
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.clip_dim = clip_dim
        self.clip_anchor_dir = clip_anchor_dir

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Joint Manifold Anchoring: the 6-channel [H&E || z_t] input is embedded jointly.
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len)

        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # projects CLIP (ViT-B/32, 512-d) concept features into the prototype space
        self.proto_init_proj = nn.Linear(clip_dim, hidden_size)

        self.initialize_weights()

    def init_moe_prototypes(self):
        """Initialize the histocomponent prototypes with biological concept features
        extracted offline by a pre-trained CLIP model (ViT-B/32, 512-d).

        Expected files under ``clip_anchor_dir`` (each a 1x512 / 512 tensor):
            anchor_nuclear.pt, anchor_cytoplasm.pt, anchor_background.pt
        Falls back to random initialization if the files are not found.
        """
        names = ["anchor_nuclear.pt", "anchor_cytoplasm.pt", "anchor_background.pt"]
        paths = [os.path.join(self.clip_anchor_dir, n) for n in names]
        if not all(os.path.exists(p) for p in paths):
            print(f"[init_moe_prototypes] CLIP anchor files not found under "
                  f"'{self.clip_anchor_dir}'. Keeping random prototype init.")
            return
        with torch.no_grad():
            anchors = torch.stack([torch.load(p, map_location="cpu").float() for p in paths])
            anchors = anchors.reshape(len(names), -1).to(self.proto_init_proj.weight.device)
            p_init = self.proto_init_proj(anchors)
            for block in self.blocks:
                block.mlp.router.prototypes.copy_(p_init)
        print(f"[init_moe_prototypes] Prototypes initialized from CLIP anchors in '{self.clip_anchor_dir}'.")

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            if hasattr(block.mlp, 'rcg_predictor'):
                nn.init.constant_(block.mlp.rcg_predictor.weight, 0)
                nn.init.constant_(block.mlp.rcg_predictor.bias, -2.0)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # init x_embedder
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        # x is the 6-channel Joint Manifold Anchoring input: [H&E || z_t]
        x = self.x_embedder(x) + self.pos_embed

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        all_indices = []
        all_gates = []

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x, indices, gate_value = block(x, c, feat_rope=rope)

            all_indices.append(indices)
            all_gates.append(gate_value)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len:]
        x = self.final_layer(x, c)
        out = self.unpatchify(x, self.patch_size)

        return out, all_indices, all_gates


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
}
