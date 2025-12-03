import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------- 基础模块 --------------------------
class LC_Block(nn.Module):
    """局部特征提取块：空间卷积+深度可分离卷积+池化"""
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.3, activation='elu', AveragePooling=True):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.dwconv = nn.Conv2d(F1, F1*D, kernel_size=(Chans, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(F1*D)
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        pool_size = (1, kernLength // 8)
        self.pool1 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sep_conv = nn.Conv2d(F1*D, F1*D, kernel_size=(1, kernLength//4), padding='same', groups=F1*D)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        return x.squeeze(2)


class SE_Block(nn.Module):
    """注意力模块：通道/频段注意力"""
    def __init__(self, activation1='relu', activation2='sigmoid', BandSE=True):
        super(SE_Block, self).__init__()
        self.BandSE = BandSE
        self.activation1 = nn.ReLU() if activation1 == 'relu' else nn.ELU()
        self.activation2 = nn.Sigmoid() if activation2 == 'sigmoid' else nn.ReLU()

    def forward(self, x):
        if self.BandSE:
            x_avg = torch.mean(x, dim=2).unsqueeze(1)
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            x_se = self.activation2(fc2(self.activation1(fc1(x_avg))))
            return x * x_se.permute(0, 2, 1)
        else:
            x_avg = torch.mean(x, dim=1).unsqueeze(1)
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            x_se = self.activation2(fc2(self.activation1(fc1(x_avg))))
            return x * x_se


class GC_Block(nn.Module):
    """原始全局卷积块（保留用于对比）"""
    def __init__(self, depth=2, kernel_size=4, n_windows=5, step=4, activation='elu', TimeConv=True):
        super(GC_Block, self).__init__()
        self.depth = depth
        self.n_windows = n_windows
        self.step = step
        self.TimeConv = TimeConv
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.conv_layers = nn.ModuleList()

    def forward(self, x):
        F1, F2 = x.size(1), x.size(2)
        if self.TimeConv:
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(F1, F1, kernel_size=4, dilation=i+1, padding='same').to(x.device)
                for i in range(self.depth)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(F2, F2, kernel_size=4, dilation=i+1, padding='same').to(x.device)
                for i in range(self.depth)
            ])
        
        sw_concat = []
        for j in range(self.n_windows):
            if self.TimeConv:
                st, end = j*self.step, F2 - (self.n_windows-j-1)*self.step
                sw = x[:, :, st:end]
                se_block = SE_Block(BandSE=False)(sw)
                last_block = se_block
                for i in range(self.depth):
                    block = self.conv_layers[i](last_block)
                    block = F.batch_norm(block, torch.zeros_like(block.mean(dim=[0,2])).to(x.device),
                                         torch.ones_like(block.var(dim=[0,2])).to(x.device), training=self.training)
                    block = self.activation(block)
                    block = F.dropout(block, p=0.3, training=self.training)
                    block += se_block
                    last_block = self.activation(block)
                sw_concat.append(last_block.flatten(1))
            else:
                st, end = j*self.step, F1 - (self.n_windows-j-1)*self.step
                sw = x[:, st:end, :]
                se_block = SE_Block(BandSE=True)(sw)
                last_block = se_block
                for i in range(self.depth):
                    block = self.conv_layers[i](last_block.transpose(1,2)).transpose(1,2)
                    block = F.batch_norm(block, torch.zeros_like(block.mean(dim=[0,2])).to(x.device),
                                         torch.ones_like(block.var(dim=[0,2])).to(x.device), training=self.training)
                    block = self.activation(block)
                    block = F.dropout(block, p=0.3, training=self.training)
                    block += se_block
                    last_block = self.activation(block)
                sw_concat.append(last_block.flatten(1))
        
        return torch.cat(sw_concat, dim=1)


# -------------------------- 注意力模块 --------------------------
class ChannelAttention1D(nn.Module):
    """1D通道注意力"""
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ELU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# -------------------------- 改进Mamba核心模块 --------------------------
def calc_temporal_corr(x, window_size=50):
    """计算时序相关性"""
    batch, chans, time = x.shape
    corr_scores = []
    
    for b in range(batch):
        chan_corr = []
        for c in range(chans):
            sig = x[b, c]
            if len(sig) > window_size:
                corr_val = torch.corrcoef(torch.stack([sig[:-1], sig[1:]]))[0, 1]
                if torch.isnan(corr_val):
                    corr_val = torch.tensor(0.0, device=x.device)
                chan_corr.append(corr_val.item())
            else:
                chan_corr.append(0.0)
        corr_scores.append(np.mean(chan_corr))
    
    return torch.tensor(np.mean(corr_scores), device=x.device)


class ImprovedMambaStyleBlock(nn.Module):
    """改进Mamba时序块"""
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_local_blocks = n_local_blocks
        
        # 残差连接
        self.res_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=1) if input_channels != hidden_dim else nn.Identity()
        
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1, padding=0)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
        
        self.channel_att = ChannelAttention1D(hidden_dim)
        
        # 动态卷积层
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.activation = nn.ELU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)

    def adjust_dilation(self, corr_score):
        """根据时序相关性动态调整扩张系数"""
        if corr_score > 0.6:
            return 1, 2
        elif corr_score < 0.3:
            return 2, 4
        else:
            return 1, 3

    def forward(self, x, dropout_p=0.3):
        batch, _, time = x.shape
        
        residual = self.res_conv(x)
        
        x = self.input_proj(x)
        x = self.bn_proj(x)
        x = self.activation(x)
        
        # 多头注意力
        x_attn = x.permute(0, 2, 1)
        attn_out, _ = self.multihead_attn(x_attn, x_attn, x_attn)
        x = x + attn_out.permute(0, 2, 1)
        
        x = self.channel_att(x)
        
        # 动态调整扩张
        corr_score = calc_temporal_corr(x).item()
        dilation2, dilation3 = self.adjust_dilation(corr_score)
        
        # 应用卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        # 残差连接
        x = x + residual[:, :, :x.size(2)]
        
        self.dropout.p = dropout_p
        x = self.dropout(x)
        
        # 局部池化
        local_window = max(1, time // self.n_local_blocks)
        if local_window > 0 and time >= local_window:
            local_pool = x.unfold(dimension=2, size=local_window, step=local_window)
            local_pool = local_pool.mean(dim=3)
            local_pool_flat = local_pool.flatten(1)
        else:
            local_pool_flat = self.global_pool(x).squeeze(2)
        
        global_pool = self.global_pool(x).squeeze(2)
        
        return torch.cat([local_pool_flat, global_pool], dim=1)


class ImprovedMambaStyleGC_Block(nn.Module):
    """Mamba风格GC块"""
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3):
        super().__init__()
        self.mamba_block = ImprovedMambaStyleBlock(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            n_local_blocks=n_local_blocks
        )
        self.output_dim = None
        self.dim_initialized = False

    def forward(self, x, dropout_p=0.3):
        output = self.mamba_block(x, dropout_p=dropout_p)
        
        if not self.dim_initialized:
            self.output_dim = output.shape[1]
            self.dim_initialized = True
        
        return output


# -------------------------- 完整模型 --------------------------
class EEG_DBNet(nn.Module):
    """原始EEG模型（使用GC_Block，用于对比）"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125):
        super(EEG_DBNet, self).__init__()
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3)
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, AveragePooling=False)
        
        self.gc_block1 = GC_Block(TimeConv=True, depth=4, n_windows=6, step=1)
        self.gc_block2 = GC_Block(TimeConv=False, depth=4, n_windows=6, step=1)
        
        self.fc = nn.Linear(5250, nb_classes)

    def forward(self, x):
        x1 = self.lc_block1(x)
        x2 = self.lc_block2(x)
        
        x1 = self.gc_block1(x1)
        x2 = self.gc_block2(x2)
        
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)


class EEG_DBNet_ImprovedMamba(nn.Module):
    """改进Mamba模型（基础架构，无NAS）"""
    def __init__(self, nb_classes=4, Chans=22, hidden_dim1=64, hidden_dim2=128, n_local_blocks=4):
        super().__init__()
        self.nb_classes = nb_classes
        self.fc_initialized = False
        
        # 使用基础的LC_Block
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3)
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, AveragePooling=False)
        
        # Mamba块
        self.mamba_block1 = ImprovedMambaStyleGC_Block(
            input_channels=16,
            hidden_dim=hidden_dim1,
            n_local_blocks=n_local_blocks
        )
        self.mamba_block2 = ImprovedMambaStyleGC_Block(
            input_channels=32,
            hidden_dim=hidden_dim2,
            n_local_blocks=n_local_blocks
        )
        
        self.fc = None

    def forward(self, x, dropout_p=0.3):
        x1 = self.lc_block1(x)
        x2 = self.lc_block2(x)
        
        x1_mamba = self.mamba_block1(x1, dropout_p=dropout_p)
        x2_mamba = self.mamba_block2(x2, dropout_p=dropout_p)
        
        if not self.fc_initialized:
            fc_input_dim = x1_mamba.shape[1] + x2_mamba.shape[1]
            print(f"🔧 初始化全连接层：输入维度={fc_input_dim}，输出维度={self.nb_classes}")
            self.fc = nn.Linear(fc_input_dim, self.nb_classes).to(x1_mamba.device)
            self.fc_initialized = True
        
        x_concat = torch.cat([x1_mamba, x2_mamba], dim=1)
        return self.fc(x_concat)
