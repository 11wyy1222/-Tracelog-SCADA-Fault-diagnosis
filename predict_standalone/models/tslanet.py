"""TSLANet模型架构组件"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class ICB(nn.Module):
    """反向卷积块 (Inverted Convolutional Block)"""
    
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """补丁嵌入层"""
    
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class AdaptiveSpectralBlock(nn.Module):
    """自适应频谱块 - 用于频域处理"""
    
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        """创建自适应高频掩码"""
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - 
                        self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class TSLANetLayer(nn.Module):
    """TSLANet层 - 结合ASB和ICB"""
    
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.norm1 = norm_layer(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    """TSLANet时间序列分类模型"""
    
    def __init__(self, seq_len, num_channels, num_classes, patch_size=8, 
                 emb_dim=128, depth=2, dropout_rate=0.15, 
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=seq_len, patch_size=patch_size,
            in_chans=num_channels, embed_dim=emb_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANetLayer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i],
                        use_icb=use_icb, use_asb=use_asb, adaptive_filter=adaptive_filter)
            for i in range(depth)
        ])
        self.head = nn.Linear(emb_dim, num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """前向传播"""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        x = x.mean(1)
        x = self.head(x)
        return x
    
    def pretrain_forward(self, x, mask_ratio=0.4):
        """预训练前向传播 - 使用掩码"""
        from utils.masking import random_masking_3D
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)
        x_masked, _, mask, _ = random_masking_3D(x, mask_ratio=mask_ratio)
        mask = mask.bool()
        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)
        return x_masked, x_patched, mask
