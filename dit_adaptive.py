import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F

class DiffConvAdaptive(nn.Module):
    def __init__(self,
                 in_dim,
                 kernel_size=3,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.groups = in_dim

        self.p1 = nn.Linear(in_dim, in_dim, bias=bias)
        self.pool = nn.AdaptiveAvgPool1d(output_size=kernel_size * kernel_size)
        self.kernel_gen = nn.Linear(kernel_size * kernel_size,
                                    kernel_size * kernel_size,
                                    bias=bias)

        self.p2 = nn.Linear(in_dim, in_dim, bias=bias)
        self.proj = nn.Linear(in_dim, in_dim, bias=bias)
        self.act = nn.SiLU()
        self.beta_residual_kernels = nn.Parameter(torch.zeros(in_dim))

    def no_weight_decay(self):
        return {"beta_residual_kernels"}

    def gen_kernel_weights(self, x: torch.Tensor):
        B, N, C = x.shape
        x = self.p1(x).transpose(-2, -1)
        x = self.pool(x)  # B,C,k²
        x = self.act(x)  # B,C,k²
        x = self.kernel_gen(x)  # B,C,k²
        x = x.view(B, C, self.kernel_size, self.kernel_size)
        return x

    def get_effective_residual_weight(self, kernels: torch.Tensor):
        mean_per_kernel = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(self.beta_residual_kernels).view(1, -1, 1, 1)
        return kernels - factor * mean_per_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x1 = self.p2(x).permute(0, 2, 1).reshape(B, C, H, W)

        kernels = self.gen_kernel_weights(x)
        kernels = self.get_effective_residual_weight(kernels)

        B, C, H, W = x1.shape
        x1_f = x1.reshape(1, B * C, H, W)
        ker_f = kernels.reshape(B * C, 1,
                                self.kernel_size,
                                self.kernel_size)
        out = F.conv2d(x1_f, ker_f,
                       padding=self.padding,
                       groups=B * C)
        out = out.reshape(B, C, H * W).permute(0, 2, 1)
        out = self.proj(out)
        return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_type, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == 'attn':
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.attn = DiffConvAdaptive(in_dim=hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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


# -------------------------- [MODIFIED] DiT Class --------------------------
class DiT(nn.Module):
    """
    An adaptive DiT model that can be conditional or unconditional.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,  # If num_classes <= 1, model is unconditional
            learn_sigma=False,
            attn_type='attn',
    ):
        super().__init__()

        assert attn_type in ["attn", "diff"]

        # [NEW] Determine if the model is conditional
        self.is_conditional = num_classes > 1
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # [NEW] Conditionally create the label embedder
        if self.is_conditional:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = None
            print("DiT is in UNCONDITIONAL mode.")

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_type=attn_type) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # ... (other initializations remain the same) ...
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # [NEW] Conditionally initialize label embedding table
        if self.is_conditional:
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

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (optional)
        """
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)

        # [NEW] Create the conditioning signal `c` adaptively
        c = t_emb  # Start with time embedding
        if self.is_conditional:
            if y is None:
                raise ValueError("Class labels `y` must be provided for a conditional DiT model.")
            y_emb = self.y_embedder(y, self.training)
            c = c + y_emb  # Add class embedding

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with Classifier-Free Guidance.
        Only valid for conditional models.
        """
        # [NEW] Add a check to prevent misuse
        if not self.is_conditional:
            raise RuntimeError("Classifier-Free Guidance is only applicable to conditional models.")

        # The rest of the CFG logic remains the same
        y_uncond = torch.full_like(y, self.y_embedder.num_classes)
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        y_in = torch.cat([y, y_uncond])

        model_out = self.forward(x_in, t_in, y_in)

        cond_output, uncond_output = torch.split(model_out, len(x), dim=0)
        guided_velocity = uncond_output + cfg_scale * (cond_output - uncond_output)
        return guided_velocity

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_4_diff(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, attn_type='diff', **kwargs)

def DiT_S_2_diff(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, attn_type='diff', **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_XS_2_diff(**kwargs):
    return DiT(depth=8, hidden_size=240, patch_size=2, num_heads=6, attn_type='diff', **kwargs)


def DiT_XS_2(**kwargs):
    return DiT(depth=8, hidden_size=240, patch_size=2, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XS/2': DiT_XS_2,
    'DiT-XS/2-diff': DiT_XS_2_diff,
    'DiT-S/4': DiT_S_4,
    'DiT-S/2': DiT_S_2,
    'DiT-S/4-diff': DiT_S_4_diff,
    'DiT-S/2-diff': DiT_S_2_diff,
}