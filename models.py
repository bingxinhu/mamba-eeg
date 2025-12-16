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

"""针对多频段数据的专用Mamba网络（添加BatchNorm层）"""
class MultiBandMambaNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_timepoints, dropout=0.5, mamba_dim=32):
        super(MultiBandMambaNet, self).__init__()
        self.n_bands = 5
        raw_channels = n_channels // self.n_bands
        
        # 频段分离卷积 - 在每个卷积层后添加BatchNorm
        self.band_convs = nn.ModuleList()
        for _ in range(self.n_bands):
            conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(raw_channels, 1), padding=0),
                nn.BatchNorm2d(8),  # 新增BatchNorm
                nn.ELU(),
                nn.Conv2d(8, 16, kernel_size=(1, 15), padding=(0, 7)),
                nn.BatchNorm2d(16),  # 新增BatchNorm
                nn.ELU(),
                nn.Dropout2d(dropout),
                nn.AvgPool2d((1, 2))
            )
            self.band_convs.append(conv)
        
        # 频段融合 - 在融合层后添加BatchNorm
        self.band_fusion = nn.Sequential(
            nn.Conv2d(16 * self.n_bands, mamba_dim, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(mamba_dim),  # 新增BatchNorm
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        
        # Mamba前的BatchNorm层 - 新增
        self.before_mamba_bn = nn.BatchNorm1d(mamba_dim)
        
        # 使用Mamba包装器
        self.mamba = MambaWrapper(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Mamba后的BatchNorm层 - 新增
        self.after_mamba_bn = nn.BatchNorm1d(mamba_dim)
        
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
            x = x.squeeze(2)  # (batch, mamba_dim, seq_len)
        
        # Mamba前的BatchNorm
        x = self.before_mamba_bn(x)
        
        # 转置为Mamba需要的格式
        x = x.transpose(1, 2)  # (batch, seq_len, mamba_dim)
        
        # Mamba前向传播
        x = self.mamba(x)
        
        # Mamba后的BatchNorm（需要在转置后应用）
        x = x.transpose(1, 2)  # (batch, mamba_dim, seq_len)
        x = self.after_mamba_bn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, mamba_dim)
        
        # 分类
        x = x.transpose(1, 2).unsqueeze(2)  # (batch, mamba_dim, 1, seq_len)
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

class BandAwareInterpretableMamba(nn.Module):
    """
    频段感知可解释Mamba网络
    设计原则：
    1. 分离处理五个频段，每个频段使用独立的轻量级Mamba
    2. 引入跨频段注意力机制，以捕捉频段间的相互作用
    3. 使用注意力池化来聚合时间和通道维度，提高可解释性
    4. 整体轻量化，减少过拟合风险
    """
    def __init__(self, n_channels=22, n_classes=4, n_timepoints=1125, 
                 d_model=32, d_state=16, n_bands=5, dropout=0.5, use_freq=False):
        super().__init__()
        
        # 根据是否使用多频段滤波调整参数
        if use_freq:
            # 多频段模式下，输入通道数 = 原始通道数 × 频段数
            self.n_bands = n_bands
            self.raw_channels = n_channels // n_bands
            print(f"多频段模式：原始通道数={self.raw_channels}, 频段数={n_bands}")
        else:
            # 单频段模式下，将所有通道作为一个频段处理
            self.n_bands = 1
            self.raw_channels = n_channels
            print(f"单频段模式：通道数={self.raw_channels}")
        
        self.d_model = d_model
        self.n_bands = n_bands if use_freq else 1
        
        # 1. 每个频段的独立处理
        self.band_projections = nn.ModuleList()
        self.band_mambas = nn.ModuleList()
        
        for i in range(self.n_bands):
            # 将每个频段的通道投影到d_model
            proj = nn.Sequential(
                nn.Conv1d(self.raw_channels, d_model, kernel_size=1, stride=1),
                nn.BatchNorm1d(d_model),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            self.band_projections.append(proj)
            
            # 每个频段一个Mamba（如果可用，否则用替代）
            mamba = MambaWrapper(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.band_mambas.append(mamba)
        
        # 2. 频段间注意力机制（可选）
        if self.n_bands > 1:
            self.cross_band_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.cross_band_attention = None
        
        # 3. 时间注意力池化
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.n_bands, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x: (batch, 1, n_channels, n_timepoints)
        batch_size, _, total_channels, timepoints = x.shape
        
        # 分割频段
        band_features = []
        for i in range(self.n_bands):
            if self.n_bands > 1:
                start = i * self.raw_channels
                end = (i + 1) * self.raw_channels
                band_data = x[:, :, start:end, :]  # (batch, 1, raw_channels, timepoints)
            else:
                # 单频段模式，使用所有通道
                band_data = x
            
            band_data = band_data.squeeze(1)    # (batch, raw_channels, timepoints)
            
            # 频段投影
            proj = self.band_projections[i](band_data)  # (batch, d_model, timepoints)
            proj = proj.transpose(1, 2)                 # (batch, timepoints, d_model)
            
            # Mamba处理
            mamba_out = self.band_mambas[i](proj)      # (batch, timepoints, d_model)
            band_features.append(mamba_out)
        
        # 如果有多频段且使用跨频段注意力
        if self.n_bands > 1 and self.cross_band_attention is not None:
            # 将每个频段在时间维度上平均，然后进行跨频段注意力
            band_vectors = [bf.mean(dim=1) for bf in band_features]  # 每个形状: (batch, d_model)
            band_vectors_stacked = torch.stack(band_vectors, dim=1)  # (batch, n_bands, d_model)
            
            # 跨频段注意力
            attended_bands, _ = self.cross_band_attention(
                band_vectors_stacked, band_vectors_stacked, band_vectors_stacked
            )  # (batch, n_bands, d_model)
            
            # 将注意力后的频段特征重新分配到时间维度
            for i in range(self.n_bands):
                # 将注意力权重应用到原始特征
                attention_weight = attended_bands[:, i:i+1, :]  # (batch, 1, d_model)
                band_features[i] = band_features[i] * attention_weight
        
        # 对每个频段进行时间注意力池化
        band_vectors = []
        for i in range(self.n_bands):
            attn_weights = self.temporal_attention(band_features[i])  # (batch, timepoints, 1)
            band_vector = torch.sum(band_features[i] * attn_weights, dim=1)  # (batch, d_model)
            band_vectors.append(band_vector)
        
        # 拼接所有频段的特征向量
        combined = torch.cat(band_vectors, dim=1)  # (batch, d_model * n_bands)
        
        # 分类
        out = self.classifier(combined)
        return out

# 修改 get_model 函数
def get_model(model_name, n_channels, n_classes, n_timepoints, use_freq=False, 
              dropout=0.5, mamba_dim=32):
    if model_name == "wideband":
        return WidebandEEGNet(n_channels, n_classes, n_timepoints, use_freq, dropout)
    elif model_name == "multiband_mamba":
        return MultiBandMambaNet(n_channels, n_classes, n_timepoints, dropout, mamba_dim)
    elif model_name == "Interpretable_mamba":
        return BandAwareInterpretableMamba(
            n_channels=n_channels, 
            n_classes=n_classes, 
            n_timepoints=n_timepoints,
            d_model=mamba_dim,
            dropout=dropout,
            use_freq=use_freq
        )
    elif model_name == "baseline":
        return BaselineEEGNet(n_channels, n_classes, n_timepoints, dropout)
    else:
        raise ValueError(f"未知模型: {model_name}")
