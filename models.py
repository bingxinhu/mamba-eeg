import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查Mamba是否可用
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("警告: mamba_ssm 不可用，将使用替代方案")

class SimpleMamba(nn.Module):
    """简单的Mamba替代方案，用于CPU环境"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(SimpleMamba, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # 简化的线性变换替代Mamba
        self.in_proj = nn.Linear(d_model, d_model * expand)
        self.conv1d = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model * expand,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=d_model * expand
        )
        self.out_proj = nn.Linear(d_model * expand, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.in_proj(x)  # (batch, seq_len, d_model*expand)
        x = x.transpose(1, 2)  # (batch, d_model*expand, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model*expand)
        x = F.gelu(x)
        x = self.out_proj(x)  # (batch, seq_len, d_model)
        return x + residual  # 残差连接

class MambaWrapper(nn.Module):
    """Mamba包装器，自动选择GPU/CPU版本"""
    def __init__(self, d_model=32, d_state=16, d_conv=4, expand=2):
        super(MambaWrapper, self).__init__()
        
        if MAMBA_AVAILABLE and torch.cuda.is_available():
            # 使用真正的Mamba（需要GPU）
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_real_mamba = True
            print("使用真正的Mamba模块 (GPU)")
        else:
            # 使用简化版本（支持CPU）
            self.mamba = SimpleMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_real_mamba = False
            print("使用简化Mamba替代方案 (CPU兼容)")
    
    def forward(self, x):
        # 确保Mamba在正确的设备上
        if self.use_real_mamba and not x.is_cuda and torch.cuda.is_available():
            x = x.cuda()
        
        output = self.mamba(x)
        
        # 处理不同的输出格式
        if isinstance(output, tuple):
            return output[0]  # 通常第一个元素是隐藏状态
        elif isinstance(output, list):
            return output[0] if len(output) > 0 else output
        else:
            return output

"""融合Mamba的宽态EEG网络（完全修复）"""
class WidebandEEGMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.5, mamba_dim=32):
        super(WidebandEEGMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        self.mamba_dim = mamba_dim
        
        if use_freq:
            raw_channels = n_channels // self.n_bands
            assert n_channels % self.n_bands == 0, f"多频段模式下总通道数 {n_channels} 必须是频段数 {self.n_bands} 的整数倍"
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=spatial_kernel, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 多尺度时间卷积分支
        self.time_branch1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 11), stride=1, padding=(0, 5)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        self.time_branch2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 21), stride=1, padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        # 频率注意力机制
        if use_freq:
            self.freq_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, self.n_bands),
                nn.Dropout(dropout),
                nn.Softmax(dim=1)
            )
        
        # 时间维度降采样
        self.time_downsample = nn.Sequential(
            nn.Conv2d(64, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 使用Mamba包装器（自动选择GPU/CPU版本）
        self.mamba = MambaWrapper(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Mamba后添加Dropout
        self.mamba_dropout = nn.Dropout(dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(mamba_dim, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # 确保输入在正确设备上
        device = next(self.parameters()).device
        x = x.to(device)
        
        # 空间特征提取
        x = self.spatial_conv(x)
        
        # 多尺度时间特征
        x1 = self.time_branch1(x)
        x2 = self.time_branch2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # 频率注意力加权
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
        
        # Mamba前向传播（通过包装器）
        x = self.mamba(x)
        
        # Mamba后Dropout
        x = self.mamba_dropout(x)
        
        # 恢复维度用于分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out

"""稳定的Mamba网络：简化架构"""
class StableWidebandEEGMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.5, mamba_dim=32):
        super(StableWidebandEEGMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        
        if use_freq:
            raw_channels = n_channels // self.n_bands
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=spatial_kernel, padding=0),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 时间卷积
        self.time_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d((1, 2))
        )
        
        # Mamba前降维
        self.downsample = nn.Sequential(
            nn.Conv2d(32, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 使用Mamba包装器
        self.mamba = MambaWrapper(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        # 空间特征
        x = self.spatial_conv(x)
        
        # 时间特征
        x = self.time_conv(x)
        
        # 降维
        x = self.downsample(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        x = self.mamba(x)
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out

"""针对多频段数据的专用Mamba网络"""
class MultiBandMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, dropout=0.5, mamba_dim=32):
        super(MultiBandMambaNet, self).__init__()
        self.n_bands = 5
        raw_channels = n_channels // self.n_bands
        
        # 频段分离卷积
        self.band_convs = nn.ModuleList()
        for _ in range(self.n_bands):
            conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(raw_channels, 1), padding=0),
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 16, kernel_size=(1, 15), padding=(0, 7)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Dropout2d(dropout),
                nn.AvgPool2d((1, 2))
            )
            self.band_convs.append(conv)
        
        # 频段融合
        self.band_fusion = nn.Sequential(
            nn.Conv2d(16 * self.n_bands, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 使用Mamba包装器
        self.mamba = MambaWrapper(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        raw_channels = x.size(2) // self.n_bands
        
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
        if x.size(1) == self.mamba.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        x = self.mamba(x)
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out

"""原始WidebandEEGNet"""
class WidebandEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.5):
        super(WidebandEEGNet, self).__init__()
        self.use_freq = use_freq
        
        if use_freq:
            raw_channels = n_channels // 5
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
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=(1, 4))
        )
        
        time_dim = n_timepoints // 4
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=(1, 2)),
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
    def __init__(self, n_channels, n_classes, n_timepoints, dropout=0.5):
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
            nn.Dropout(dropout),
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

"""正则化Mamba网络"""
class RegularizedMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, use_freq=True, dropout=0.5, mamba_dim=32):
        super(RegularizedMambaNet, self).__init__()
        self.use_freq = use_freq
        self.n_bands = 5 if use_freq else 1
        
        if use_freq:
            raw_channels = n_channels // self.n_bands
            spatial_kernel = (raw_channels, 1)
        else:
            spatial_kernel = (n_channels, 1)
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=spatial_kernel, padding=0),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(16, 32, kernel_size=(1, 1)),  # 1x1卷积减少维度
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 时间特征提取
        self.time_conv = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d((1, 4))
        )
        
        # Mamba前降维
        self.downsample = nn.Sequential(
            nn.Conv2d(48, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # 使用Mamba包装器
        self.mamba = MambaWrapper(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mamba_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        # 空间特征
        x = self.spatial_conv(x)
        
        # 时间特征
        x = self.time_conv(x)
        
        # 降维
        x = self.downsample(x)
        
        # Mamba输入处理
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.size(1) == self.mamba.mamba.d_model:
            x = x.transpose(1, 2)
        
        # Mamba前向传播
        x = self.mamba(x)
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)
        out = self.classifier(x)
        return out

"""模型选择接口"""
def get_model(model_name, n_channels, n_classes, n_timepoints, use_freq=False, 
              dropout=0.5, mamba_dim=32):
    if model_name == "wideband":
        return WidebandEEGNet(n_channels, n_classes, n_timepoints, use_freq, dropout)
    elif model_name == "wideband_mamba":
        return WidebandEEGMambaNet(n_channels, n_classes, n_timepoints, use_freq, dropout, mamba_dim)
    elif model_name == "stable_mamba":
        return StableWidebandEEGMambaNet(n_channels, n_classes, n_timepoints, use_freq, dropout, mamba_dim)
    elif model_name == "regularized_mamba":
        return RegularizedMambaNet(n_channels, n_classes, n_timepoints, use_freq, dropout, mamba_dim)
    elif model_name == "multiband_mamba":
        return MultiBandMambaNet(n_channels, n_classes, n_timepoints, dropout, mamba_dim)
    elif model_name == "baseline":
        return BaselineEEGNet(n_channels, n_classes, n_timepoints, dropout)
    else:
        raise ValueError(f"未知模型: {model_name}")
