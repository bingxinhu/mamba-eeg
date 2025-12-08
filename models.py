import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class WidebandEEGMambaNet(nn.Module):
    """融合Mamba的宽态EEG网络：局部卷积+全局长时序建模"""
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.3, mamba_dim=64):
        super(WidebandEEGMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        self.mamba_dim = mamba_dim
        
        # 计算每个频段的基础通道数（多频段时总通道数是基础通道×频段数）
        base_channels = n_channels // self.n_bands if use_freq else n_channels
        
        # 1. 空间特征提取（跨通道卷积）
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(base_channels, 1),  # 对每个频段的通道单独卷积
            stride=1,
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # 2. 多尺度时间卷积分支
        self.time_branch1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 11), stride=1, padding=(0, 5)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),  # 时间维度降采样
            nn.Dropout(dropout)
        )
        
        self.time_branch2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 21), stride=1, padding=(0, 10)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 3. 频率注意力机制（对多频段特征加权）
        if use_freq:
            self.freq_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # 压缩空间和时间维度
                nn.Flatten(),
                nn.Linear(128, self.n_bands),  # 输出每个频段的权重
                nn.Softmax(dim=1)
            )
        
        # 4. 在Mamba之前的时间维度降采样
        self.time_downsample = nn.Sequential(
            nn.Conv2d(128, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU()
        )
        
        # 5. Mamba全局长时序处理模块
        self.mamba = Mamba(
            d_model=mamba_dim,  # 输入特征维度（减小到64）
            d_state=16,         # 状态维度
            d_conv=4,           # 卷积核维度
            expand=2            # 扩展因子
        )
        
        # 6. 分类头
        # 使用自适应池化来避免维度计算错误
        self.classifier = nn.Sequential(
            nn.Conv2d(mamba_dim, 128, kernel_size=(1, 13), padding=(0, 6)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应池化到1x1
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(128, n_classes)  # 全连接分类
        )

    def forward(self, x):
        # 输入形状: (batch, 1, n_channels, n_timepoints)
        #print(f"0. Input shape: {x.shape}")
        batch_size = x.size(0)
        
        # 空间特征提取：跨通道卷积捕获空间相关性
        x = self.spatial_conv(x)  # (batch, 32, 1, n_timepoints)
        #print(f"1. After spatial_conv: {x.shape}")
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        # 多尺度时间特征学习：不同卷积核捕获不同时间尺度模式
        x1 = self.time_branch1(x)
        #print(f"2. After time_branch1: {x1.shape}")
        x2 = self.time_branch2(x)
        #print(f"3. After time_branch2: {x2.shape}")
        x = torch.cat([x1, x2], dim=1)  # (batch, 128, 1, time/2) 合并特征
        #print(f"4. After concat: {x.shape}")
        
        # 频率注意力加权：动态调整不同频段的重要性
        if self.use_freq:
            freq_weights = self.freq_attention(x)  # (batch, n_bands) 得到每个频段的权重
            band_size = x.size(1) // self.n_bands  # 按特征通道维度分割（128/5=25.6→25，最后一个频段多1）
            weighted_bands = []
            for i in range(self.n_bands):
                # 从特征通道维度(dim=1)分割对应频段
                start = i * band_size
                end = start + band_size if i < self.n_bands - 1 else x.size(1)
                band = x[:, start:end, :, :]
                # 应用权重（扩展维度以匹配广播）
                weighted_bands.append(band * freq_weights[:, i].view(-1, 1, 1, 1))
            x = torch.cat(weighted_bands, dim=1)  # 合并加权后的频段特征
            #print(f"5. After freq attention: {x.shape}")
        
        # 时间维度降采样（减少Mamba处理的序列长度）
        x = self.time_downsample(x)  # (batch, mamba_dim, 1, time/4)
        #print(f"6. After time_downsample: {x.shape}")
        
        # Mamba全局长时序处理：转换为Mamba输入格式
        # 检查当前张量维度
        if x.dim() == 4:
            batch_size, d_model, height, seq_len = x.shape
            #print(f"7. x is 4D: batch={batch_size}, d_model={d_model}, height={height}, seq_len={seq_len}")
            
            # 如果高度为1，则去除该维度
            if height == 1:
                x = x.squeeze(2)  # (batch, d_model, seq_len)
                #print(f"8. After squeeze: {x.shape}")
            else:
                # 高度不为1，将高度和序列维度合并
                x = x.reshape(batch_size, d_model, height * seq_len)
                #print(f"8. After reshape: {x.shape}")
        
        # 确保形状为 (batch, seq_len, d_model) - Mamba期望的格式
        if x.dim() == 3:
            # 检查当前维度顺序
            if x.shape[1] == self.mamba_dim:  # 如果第二个维度是d_model，则转置
                x = x.transpose(1, 2)  # (batch, seq_len, d_model)
                #print(f"9. After transpose: {x.shape}")
        
        x = x.contiguous()
        #print(f"10. Before Mamba: shape={x.shape}, dim={x.dim()}")
        
        try:
            mamba_out = self.mamba(x)  # (batch, seq_len, d_model)
            #print(f"11. After Mamba: {mamba_out.shape}")
        except Exception as e:
            print(f"Error in Mamba: {e}")
            print(f"Input to Mamba shape: {x.shape}")
            print(f"Input to Mamba dim: {x.dim()}")
            raise
        
        # 恢复卷积操作所需的维度格式
        mamba_out = mamba_out.transpose(1, 2).unsqueeze(2)  # (batch, d_model, 1, seq_len)
        #print(f"12. After transpose and unsqueeze: {mamba_out.shape}")
        
        # 分类头：最终预测
        out = self.classifier(mamba_out)
        #print(f"13. Output shape: {out.shape}")
        return out


class WidebandEEGNet(nn.Module):
    """原始的WidebandEEGNet实现（如果不使用，请保持占位实现）"""
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True):
        super(WidebandEEGNet, self).__init__()
        self.use_freq = use_freq
        
        # 1. 空间特征提取
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(n_channels, 1),
            stride=1,
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(16)
        
        # 2. 时间卷积
        self.time_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 25), stride=1, padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.3)
        )
        
        # 3. 分类头
        time_dim = n_timepoints // 4  # 经过池化后的时间长度
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * (time_dim // 2), n_classes)
        )

    def forward(self, x):
        # 输入形状: (batch, 1, n_channels, n_timepoints)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        x = self.time_conv(x)
        
        out = self.classifier(x)
        return out


class BaselineEEGNet(nn.Module):
    """基线EEGNet实现（Shallow ConvNet）"""
    def __init__(self, n_channels, n_classes, n_timepoints):
        super(BaselineEEGNet, self).__init__()
        
        # 1. 时间卷积
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=(0, 12)),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), stride=1),
            nn.BatchNorm2d(40),
        )
        
        # 2. 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(40, 40, kernel_size=(1, 15), stride=1, padding=(0, 7), groups=40)
        self.bn2 = nn.BatchNorm2d(40)
        
        # 3. 平均池化
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        
        # 4. 分类头
        # 计算时间维度变化
        time_dim = n_timepoints  # 输入时间长度
        time_dim = time_dim  # 时间卷积不改变长度
        time_dim = (time_dim - 15 + 14) // 1 + 1  # 深度卷积后: (W - K + 2P)/S + 1
        time_dim = (time_dim - 75 + 0) // 15 + 1  # 池化后
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(40 * time_dim, n_classes)
        )

    def forward(self, x):
        # 输入形状: (batch, 1, n_channels, n_timepoints)
        x = self.time_conv(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = x * x  # 平方激活
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # 对数激活
        
        out = self.classifier(x)
        return out


def get_model(model_name, n_channels, n_classes, n_timepoints, use_freq=False, mamba_dim=64):
    """模型选择接口（增加Mamba模型支持）"""
    if model_name == "wideband":
        return WidebandEEGNet(n_channels, n_classes, n_timepoints, use_freq)
    elif model_name == "wideband_mamba":
        return WidebandEEGMambaNet(n_channels, n_classes, n_timepoints, use_freq, mamba_dim=mamba_dim)
    elif model_name == "baseline":
        return BaselineEEGNet(n_channels, n_classes, n_timepoints)
    else:
        raise ValueError(f"未知模型: {model_name}")
