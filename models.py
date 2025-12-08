import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

"""融合Mamba的宽态EEG网络：局部卷积+全局长时序建模"""
class WidebandEEGMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.3, mamba_dim=64):
        super(WidebandEEGMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        self.mamba_dim = mamba_dim
        
        # 对于多频段数据，通道数 = 原始通道数 × 频段数
        # 空间卷积应该处理所有通道，而不是每个频段
        # 因此我们不需要除以频段数
        if use_freq:
            # 多频段模式：输入通道数已经是原始通道数 × 5
            # 空间卷积核大小应该等于原始通道数
            raw_channels = n_channels // self.n_bands
            assert n_channels % self.n_bands == 0, f"多频段模式下总通道数 {n_channels} 必须是频段数 {self.n_bands} 的整数倍"
            spatial_kernel = (raw_channels, 1)
        else:
            # 单频段模式：直接处理所有通道
            spatial_kernel = (n_channels, 1)
        
        # 空间特征提取
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=spatial_kernel,
            stride=1,
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # 多尺度时间卷积分支
        self.time_branch1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 11), stride=1, padding=(0, 5)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout)
        )
        
        self.time_branch2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 21), stride=1, padding=(0, 10)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 频率注意力机制（仅多频段模式）
        if use_freq:
            self.freq_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, self.n_bands),
                nn.Softmax(dim=1)
            )
        
        # 时间维度降采样
        self.time_downsample = nn.Sequential(
            nn.Conv2d(128, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU()
        )
        
        # Mamba模块
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(mamba_dim, 128, kernel_size=(1, 13), padding=(0, 6)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 空间特征提取
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        # 多尺度时间特征
        x1 = self.time_branch1(x)
        x2 = self.time_branch2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # 频率注意力加权（仅多频段模式）
        if self.use_freq:
            freq_weights = self.freq_attention(x)
            band_size = x.size(1) // self.n_bands
            weighted_bands = []
            for i in range(self.n_bands):
                start = i * band_size
                end = start + band_size if i < self.n_bands - 1 else x.size(1)
                band = x[:, start:end, :, :]
                weighted_bands.append(band * freq_weights[:, i].view(-1, 1, 1, 1))
            x = torch.cat(weighted_bands, dim=1)
        
        # 时间降采样
        x = self.time_downsample(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)  # 移除空间维度
        if x.size(1) == self.mamba_dim:
            x = x.transpose(1, 2)  # (batch, seq_len, dim)
        
        # Mamba前向传播
        mamba_output = self.mamba(x)
        
        # 处理Mamba输出（兼容不同版本）
        if isinstance(mamba_output, tuple):
            # 对于元组，通常第一个元素是输出
            x = mamba_output[0]
        elif isinstance(mamba_output, list):
            # 对于列表，取第一个元素
            x = mamba_output[0] if len(mamba_output) > 0 else mamba_output
        else:
            # 对于单个张量
            x = mamba_output
        
        # 恢复维度用于分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out


"""稳定的Mamba网络：简化架构"""
class StableWidebandEEGMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.3, mamba_dim=64):
        super(StableWidebandEEGMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        
        # 对于多频段数据，通道数 = 原始通道数 × 频段数
        if use_freq:
            raw_channels = n_channels // self.n_bands
            assert n_channels % self.n_bands == 0, f"多频段模式下总通道数 {n_channels} 必须是频段数 {self.n_bands} 的整数倍"
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        # 空间特征提取
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=spatial_kernel,
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # 时间卷积
        self.time_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )
        
        self.time_after_pool = n_timepoints // 2
        
        # Mamba前降维
        self.downsample = nn.Sequential(
            nn.Conv2d(64, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU()
        )
        
        self.seq_len = self.time_after_pool // 2
        
        # Mamba模块
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, n_classes)
        )
        
    def forward(self, x):
        # 空间特征
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        # 时间特征
        x = self.time_conv(x)
        
        # 降维
        x = self.downsample(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        mamba_output = self.mamba(x)
        
        # 处理Mamba输出（兼容不同版本）
        if isinstance(mamba_output, tuple):
            x = mamba_output[0]
        elif isinstance(mamba_output, list):
            x = mamba_output[0] if len(mamba_output) > 0 else mamba_output
        else:
            x = mamba_output
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out


"""简单的Mamba网络（调试用）"""
class SimpleMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, mamba_dim=64):
        super(SimpleMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        
        # 对于多频段数据，通道数 = 原始通道数 × 频段数
        if use_freq:
            raw_channels = n_channels // self.n_bands
            assert n_channels % self.n_bands == 0, f"多频段模式下总通道数 {n_channels} 必须是频段数 {self.n_bands} 的整数倍"
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=spatial_kernel, padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, mamba_dim, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )
        
        self.seq_len = n_timepoints // 4
        
        # Mamba模块
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, n_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        mamba_output = self.mamba(x)
        
        # 处理Mamba输出（兼容不同版本）
        if isinstance(mamba_output, tuple):
            mamba_output = mamba_output[0]
        elif isinstance(mamba_output, list):
            mamba_output = mamba_output[0] if len(mamba_output) > 0 else mamba_output
        
        # 分类
        mamba_output = mamba_output.transpose(1, 2).unsqueeze(2)
        out = self.classifier(mamba_output)
        return out


"""针对多频段数据的专用Mamba网络"""
class MultiBandMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, dropout=0.3, mamba_dim=64):
        super(MultiBandMambaNet, self).__init__()
        # 假设输入是多频段数据，通道数 = 原始通道数 × 5
        self.n_bands = 5
        raw_channels = n_channels // self.n_bands
        assert n_channels % self.n_bands == 0, f"多频段模式下总通道数 {n_channels} 必须是频段数 {self.n_bands} 的整数倍"
        
        # 频段分离卷积：分别处理每个频段
        self.band_convs = nn.ModuleList()
        for _ in range(self.n_bands):
            conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(raw_channels, 1), padding=0),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=(1, 25), padding=(0, 12)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d((1, 2)),
                nn.Dropout(dropout)
            )
            self.band_convs.append(conv)
        
        # 频段融合
        self.band_fusion = nn.Sequential(
            nn.Conv2d(32 * self.n_bands, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU()
        )
        
        self.seq_len = n_timepoints // 4
        
        # Mamba模块
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, n_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        raw_channels = x.size(2) // self.n_bands
        seq_len = x.size(3)
        
        # 分离频段
        band_features = []
        for i in range(self.n_bands):
            start_idx = i * raw_channels
            end_idx = (i + 1) * raw_channels
            band_data = x[:, :, start_idx:end_idx, :]
            band_feat = self.band_convs[i](band_data)
            band_features.append(band_feat)
        
        # 合并频段特征
        x = torch.cat(band_features, dim=1)
        
        # 频段融合
        x = self.band_fusion(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        mamba_output = self.mamba(x)
        
        # 处理Mamba输出（兼容不同版本）
        if isinstance(mamba_output, tuple):
            x = mamba_output[0]
        elif isinstance(mamba_output, list):
            x = mamba_output[0] if len(mamba_output) > 0 else mamba_output
        else:
            x = mamba_output
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out


"""针对单频段数据的专用Mamba网络"""
class SingleBandMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, dropout=0.3, mamba_dim=64):
        super(SingleBandMambaNet, self).__init__()
        # 单频段模式，直接处理所有通道
        
        # 空间特征提取
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(n_channels, 1),
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # 时间特征提取
        self.time_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )
        
        # Mamba前降维
        self.downsample = nn.Sequential(
            nn.Conv2d(64, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU()
        )
        
        # Mamba模块
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, n_classes)
        )
        
    def forward(self, x):
        # 空间特征
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        # 时间特征
        x = self.time_conv(x)
        
        # 降维
        x = self.downsample(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        mamba_output = self.mamba(x)
        
        # 处理Mamba输出（兼容不同版本）
        if isinstance(mamba_output, tuple):
            x = mamba_output[0]
        elif isinstance(mamba_output, list):
            x = mamba_output[0] if len(mamba_output) > 0 else mamba_output
        else:
            x = mamba_output
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out


"""原始WidebandEEGNet"""
class WidebandEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True):
        super(WidebandEEGNet, self).__init__()
        self.use_freq = use_freq
        
        # 对于多频段数据，调整空间卷积核大小
        if use_freq:
            raw_channels = n_channels // 5
            assert n_channels % 5 == 0, f"多频段模式下总通道数 {n_channels} 必须是5的整数倍"
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=spatial_kernel,
            stride=1,
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(16)
        
        self.time_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 25), stride=1, padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.3)
        )
        
        time_dim = n_timepoints // 4
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
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.time_conv(x)
        out = self.classifier(x)
        return out


"""基线EEGNet（Shallow ConvNet）"""
class BaselineEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints):
        super(BaselineEEGNet, self).__init__()
        
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=(0, 12)),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), stride=1),
            nn.BatchNorm2d(40),
        )
        
        self.depthwise_conv = nn.Conv2d(40, 40, kernel_size=(1, 15), stride=1, padding=(0, 7), groups=40)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        
        time_dim = n_timepoints
        time_dim = (time_dim - 15 + 14) // 1 + 1
        time_dim = (time_dim - 75 + 0) // 15 + 1
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(40 * time_dim, n_classes)
        )

    def forward(self, x):
        x = self.time_conv(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = x * x  # 平方激活
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        out = self.classifier(x)
        return out


"""模型选择接口"""
def get_model(model_name, n_channels, n_classes, n_timepoints, use_freq=False, mamba_dim=64):
    if model_name == "wideband":
        return WidebandEEGNet(n_channels, n_classes, n_timepoints, use_freq)
    elif model_name == "wideband_mamba":
        return WidebandEEGMambaNet(n_channels, n_classes, n_timepoints, use_freq, mamba_dim=mamba_dim)
    elif model_name == "stable_mamba":
        return StableWidebandEEGMambaNet(n_channels, n_classes, n_timepoints, use_freq, mamba_dim=mamba_dim)
    elif model_name == "simple_mamba":
        return SimpleMambaNet(n_channels, n_classes, n_timepoints, use_freq, mamba_dim=mamba_dim)
    elif model_name == "multiband_mamba":
        # 专门为多频段数据设计的Mamba网络
        return MultiBandMambaNet(n_channels, n_classes, n_timepoints, mamba_dim=mamba_dim)
    elif model_name == "singleband_mamba":
        # 专门为单频段数据设计的Mamba网络
        return SingleBandMambaNet(n_channels, n_classes, n_timepoints, mamba_dim=mamba_dim)
    elif model_name == "baseline":
        return BaselineEEGNet(n_channels, n_classes, n_timepoints)
    else:
        raise ValueError(f"未知模型: {model_name}")
