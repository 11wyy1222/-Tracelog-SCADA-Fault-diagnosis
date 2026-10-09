"""
TS-TCC模型实现
包含基础CNN模型和时间对比模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.attention import Seq_Transformer
from models.model_factory import ModelFactory


@ModelFactory.register('TS-TCC')
class base_Model(nn.Module):
    """TS-TCC基础CNN模型"""
    
    def __init__(self, config):
        super(base_Model, self).__init__()
        
        input_channels = self._get_config(config, 'input_channels', 1)
        kernel_size = self._get_config(config, 'kernel_size', 25)
        stride = self._get_config(config, 'stride', 6)
        final_out_channels = self._get_config(config, 'final_out_channels', 128)
        features_len = self._get_config(config, 'features_len', 127)
        num_classes = self._get_config(config, 'num_classes', 5)
        dropout = self._get_config(config, 'dropout', 0.35)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = features_len
        self.logits = nn.Linear(model_output_dim * final_out_channels, num_classes)

    def _get_config(self, config, key, default):
        """从config中获取参数,支持字典或对象"""
        if isinstance(config, dict):
            return config.get(key, default)
        else:
            return getattr(config, key, default)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


class TC(nn.Module):
    """时间对比(Temporal Contrast)模型"""
    
    def __init__(self, config, device):
        super(TC, self).__init__()
        final_out_channels = self._get_config(config, 'final_out_channels', 128)
        if isinstance(config, dict):
            tc_config = config.get('TC', {})
            if isinstance(tc_config, dict):
                hidden_dim = tc_config.get('hidden_dim', 64)
                timesteps = tc_config.get('timesteps', 15)
            else:
                hidden_dim = getattr(tc_config, 'hidden_dim', 64)
                timesteps = getattr(tc_config, 'timesteps', 15)
        else:
            hidden_dim = getattr(config.TC, 'hidden_dim', 64)
            timesteps = getattr(config.TC, 'timesteps', 15)
        
        self.num_channels = final_out_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(
            patch_size=self.num_channels, 
            dim=hidden_dim, 
            depth=4, 
            heads=4, 
            mlp_dim=64
        )

    def _get_config(self, config, key, default):
        """从config中获取参数"""
        if isinstance(config, dict):
            return config.get(key, default)
        else:
            return getattr(config, key, default)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)

        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
