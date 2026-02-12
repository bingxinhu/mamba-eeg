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
        else:
            # 使用简化版本（支持CPU）
            self.mamba = SimpleMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_real_mamba = False
    
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

class MultiLayerMambaBlock(nn.Module):
    """多层Mamba块，带有残差连接和层归一化"""
    def __init__(self, d_model, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        for i in range(n_layers):
            self.layers.append(MambaWrapper(d_model, d_state, d_conv, expand))
            self.norms.append(nn.LayerNorm(d_model))
    
    def forward(self, x):
        """前向传播，每层都有残差连接"""
        for i in range(self.n_layers):
            # 残差连接
            residual = x
            x = self.layers[i](x)
            x = self.dropout(x)
            x = self.norms[i](x)
            x = x + residual  # 残差连接
        return x

class MultiHeadCrossBandAttention(nn.Module):
    """改进的多头跨频段注意力机制"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性变换层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 层归一化和Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的相对位置编码
        self.rel_pos_embedding = nn.Parameter(torch.randn(1, n_heads, 1, self.head_dim))
        
        # 注意力权重保存（用于可视化）
        self.attention_weights = None
    
    def forward(self, x):
        """
        x: (batch, n_bands, d_model)
        返回: (batch, n_bands, d_model), 注意力权重
        """
        batch_size, n_bands, d_model = x.shape
        
        # 保存输入用于残差连接
        residual = x
        
        # 应用层归一化
        x = self.norm(x)
        
        # 线性变换得到Q, K, V
        q = self.q_proj(x).reshape(batch_size, n_bands, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, n_bands, head_dim)
        k = self.k_proj(x).reshape(batch_size, n_bands, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, n_bands, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, n_bands, n_bands)
        
        # 添加相对位置编码
        rel_pos = self.rel_pos_embedding.repeat(batch_size, 1, n_bands, 1)
        attn_scores = attn_scores + torch.matmul(q, rel_pos.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码（可选，这里添加因果掩码）
        mask = torch.triu(torch.ones(n_bands, n_bands, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 保存注意力权重用于可视化
        self.attention_weights = attn_weights.detach().cpu().mean(dim=1)  # 平均多头
        
        # 应用注意力权重到V
        attended = torch.matmul(attn_weights, v)  # (batch, n_heads, n_bands, head_dim)
        
        # 合并多头
        attended = attended.transpose(1, 2).reshape(batch_size, n_bands, d_model)  # (batch, n_bands, d_model)
        
        # 输出投影
        output = self.out_proj(attended)
        
        # 残差连接
        output = output + residual
        
        return output

class BandAwareInterpretableMamba(nn.Module):
    """
    频段感知可解释Mamba网络（增强版，支持多层Mamba）
    
    新增功能：
    1. 可配置的Mamba层数
    2. 支持多种特征提取方式
    3. 集成注意力可视化
    4. 支持向量数据库的特征输出
    5. 改进的多头跨频段注意力
    """
    def __init__(self, n_channels=22, n_classes=4, n_timepoints=1125, 
                 d_model=32, d_state=16, n_bands=5, dropout=0.5, use_freq=False,
                 n_attention_heads=8, n_mamba_layers=1, mamba_dropout=0.1):
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
        self.n_classes = n_classes
        self.use_freq = use_freq
        self.n_attention_heads = n_attention_heads
        self.n_mamba_layers = n_mamba_layers
        
        print(f"Mamba层数: {n_mamba_layers}, 维度: {d_model}, Dropout: {dropout}")
        
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
            
            # 每个频段一个多层Mamba块
            mamba_block = MultiLayerMambaBlock(
                d_model=d_model,
                n_layers=n_mamba_layers,
                d_state=d_state,
                d_conv=4,
                expand=2,
                dropout=mamba_dropout
            )
            self.band_mambas.append(mamba_block)
        
        # 2. 改进的多头跨频段注意力机制
        if self.n_bands > 1:
            self.cross_band_attention = MultiHeadCrossBandAttention(
                d_model=d_model,
                n_heads=n_attention_heads,
                dropout=dropout
            )
        else:
            self.cross_band_attention = None
        
        # 3. 时间注意力池化（多头）
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, self.n_attention_heads),
            nn.Softmax(dim=1)
        )
        self.temporal_projection = nn.Linear(d_model * self.n_attention_heads, d_model)
        self.temporal_attention_weights = None  # 用于存储注意力权重
        
        # 4. 分类头
        classifier_input_dim = d_model * self.n_bands
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes)
        )
        
        # 特征提取层 - 固定输出维度为128以匹配向量数据库
        self.feature_extractor = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256, momentum=0.1, track_running_stats=True),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),  # 固定输出128维以匹配向量数据库
            nn.BatchNorm1d(128, momentum=0.1, track_running_stats=True),
            nn.Tanh()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
            
            # 多层Mamba处理
            mamba_out = self.band_mambas[i](proj)      # (batch, timepoints, d_model)
            band_features.append(mamba_out)
        
        # 如果有多频段且使用跨频段注意力
        if self.n_bands > 1 and self.cross_band_attention is not None:
            # 将每个频段在时间维度上平均，然后进行跨频段注意力
            band_vectors = [bf.mean(dim=1) for bf in band_features]  # 每个形状: (batch, d_model)
            band_vectors_stacked = torch.stack(band_vectors, dim=1)  # (batch, n_bands, d_model)
            
            # 应用改进的多头跨频段注意力
            attended_bands = self.cross_band_attention(band_vectors_stacked)  # (batch, n_bands, d_model)
            
            # 将注意力后的频段特征重新分配到时间维度
            for i in range(self.n_bands):
                # 使用注意力后的频段特征增强原始特征
                attention_weight = attended_bands[:, i:i+1, :]  # (batch, 1, d_model)
                # 应用缩放因子，避免过度修改
                band_features[i] = band_features[i] + band_features[i] * attention_weight * 0.1
        
        # 对每个频段进行时间注意力池化（多头）
        band_vectors = []
        temporal_weights_list = []
        
        for i in range(self.n_bands):
            # 计算多头时间注意力权重
            attn_weights = self.temporal_attention(band_features[i])  # (batch, timepoints, n_heads)
            
            # 保存注意力权重
            temporal_weights_list.append(attn_weights)
            
            # 将每个头视为不同的时间表示
            attended_features = []
            for head_idx in range(self.n_attention_heads):
                head_weights = attn_weights[:, :, head_idx].unsqueeze(-1)  # (batch, timepoints, 1)
                head_vector = torch.sum(band_features[i] * head_weights, dim=1)  # (batch, d_model)
                attended_features.append(head_vector)
            
            # 合并多头表示
            multi_head_vector = torch.cat(attended_features, dim=1)  # (batch, d_model * n_heads)
            
            # 投影回原始维度
            band_vector = self.temporal_projection(multi_head_vector)  # (batch, d_model)
            band_vectors.append(band_vector)
        
        # 保存时间注意力权重
        self.temporal_attention_weights = temporal_weights_list
        
        # 拼接所有频段的特征向量
        combined = torch.cat(band_vectors, dim=1)  # (batch, d_model * n_bands)
        
        # 提取特征（用于向量数据库）- 现在输出固定128维
        self.extracted_features = self.feature_extractor(combined)  # (batch, 128)
        
        # 分类
        out = self.classifier(combined)
        return out
    
    def extract_features(self, x, feature_type='raw'):
        """
        提取特征向量（用于向量数据库）
        
        参数:
        x: 输入数据
        feature_type: 特征类型
            - 'pre_classifier': 分类器前的特征 (d_model * n_bands)
            - 'mamba': Mamba模块输出特征
            - 'band_attention': 频段注意力特征
            - 'temporal_attention': 时间注意力特征
            - 'raw': 原始特征提取器输出（固定128维）
        
        返回:
        特征向量
        """
        # 执行前向传播以计算所有中间特征
        _ = self.forward(x)
        
        if feature_type == 'pre_classifier':
            # 分类器前的特征
            batch_size, _, total_channels, timepoints = x.shape
            band_vectors = []
            
            for i in range(self.n_bands):
                if self.n_bands > 1:
                    start = i * self.raw_channels
                    end = (i + 1) * self.raw_channels
                    band_data = x[:, :, start:end, :]
                else:
                    band_data = x
                
                band_data = band_data.squeeze(1)
                proj = self.band_projections[i](band_data)
                proj = proj.transpose(1, 2)
                mamba_out = self.band_mambas[i](proj)
                
                # 时间注意力池化（多头）
                attn_weights = self.temporal_attention(mamba_out)
                attended_features = []
                for head_idx in range(self.n_attention_heads):
                    head_weights = attn_weights[:, :, head_idx].unsqueeze(-1)
                    head_vector = torch.sum(mamba_out * head_weights, dim=1)
                    attended_features.append(head_vector)
                
                multi_head_vector = torch.cat(attended_features, dim=1)
                band_vector = self.temporal_projection(multi_head_vector)
                band_vectors.append(band_vector)
            
            features = torch.cat(band_vectors, dim=1)
            return features
            
        elif feature_type == 'mamba':
            # Mamba模块输出特征（最后一个频段）
            if self.n_bands > 1:
                start = (self.n_bands - 1) * self.raw_channels
                end = self.n_bands * self.raw_channels
                band_data = x[:, :, start:end, :]
            else:
                band_data = x
            
            band_data = band_data.squeeze(1)
            proj = self.band_projections[-1](band_data)
            proj = proj.transpose(1, 2)
            mamba_out = self.band_mambas[-1](proj)
            
            # 在时间维度上平均
            features = mamba_out.mean(dim=1)
            return features
            
        elif feature_type == 'band_attention' and self.n_bands > 1:
            # 频段注意力特征
            band_vectors = []
            for i in range(self.n_bands):
                if self.n_bands > 1:
                    start = i * self.raw_channels
                    end = (i + 1) * self.raw_channels
                    band_data = x[:, :, start:end, :]
                else:
                    band_data = x
                
                band_data = band_data.squeeze(1)
                proj = self.band_projections[i](band_data)
                proj = proj.transpose(1, 2)
                mamba_out = self.band_mambas[i](proj)
                band_vector = mamba_out.mean(dim=1)
                band_vectors.append(band_vector)
            
            # 应用跨频段注意力
            band_vectors_stacked = torch.stack(band_vectors, dim=1)
            attended_bands = self.cross_band_attention(band_vectors_stacked)
            
            # 展平
            features = attended_bands.flatten(1)
            return features
            
        elif feature_type == 'temporal_attention':
            # 时间注意力权重作为特征
            batch_size, _, total_channels, timepoints = x.shape
            temporal_features = []
            
            for i in range(self.n_bands):
                if self.n_bands > 1:
                    start = i * self.raw_channels
                    end = (i + 1) * self.raw_channels
                    band_data = x[:, :, start:end, :]
                else:
                    band_data = x
                
                band_data = band_data.squeeze(1)
                proj = self.band_projections[i](band_data)
                proj = proj.transpose(1, 2)
                mamba_out = self.band_mambas[i](proj)
                
                # 获取时间注意力权重
                attn_weights = self.temporal_attention(mamba_out)  # (batch, timepoints, n_heads)
                # 取第一个头作为代表
                temporal_features.append(attn_weights[:, :, 0])
            
            # 拼接所有频段的时间注意力
            features = torch.cat(temporal_features, dim=1)
            return features
            
        else:
            # 默认返回特征提取器输出（固定128维）
            return self.extracted_features
    
    def get_attention_weights(self):
        """
        获取注意力权重（用于可视化）
        
        返回:
        包含注意力权重的字典
        """
        attention_weights = {}
        
        if self.cross_band_attention is not None and hasattr(self.cross_band_attention, 'attention_weights'):
            attention_weights['cross_band'] = self.cross_band_attention.attention_weights
        
        if self.temporal_attention_weights is not None:
            # 只取第一个样本的第一个频段的注意力权重作为代表
            if len(self.temporal_attention_weights) > 0:
                sample_weights = self.temporal_attention_weights[0][0].detach().cpu()  # (timepoints, n_heads)
                attention_weights['temporal'] = sample_weights
        
        return attention_weights

# 修改get_model函数以支持n_mamba_layers参数
def get_model(model_name, n_channels, n_classes, n_timepoints, use_freq=False, 
              dropout=0.5, mamba_dim=32, n_attention_heads=8, n_mamba_layers=1, mamba_dropout=0.1):
    """获取模型，支持多层Mamba配置"""
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
            n_bands=5 if use_freq else 1,
            dropout=dropout,
            use_freq=use_freq,
            n_attention_heads=n_attention_heads,
            n_mamba_layers=n_mamba_layers,
            mamba_dropout=mamba_dropout
        )
    elif model_name == "baseline":
        return BaselineEEGNet(n_channels, n_classes, n_timepoints, dropout)
    else:
        raise ValueError(f"未知模型: {model_name}")
