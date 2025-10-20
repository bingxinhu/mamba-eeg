import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Any


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


# -------------------------- 增强的NAS架构发现模块 --------------------------
class InnovationTracker:
    """跟踪和评估架构创新性"""
    
    def __init__(self):
        self.discovered_architectures = []
        self.performance_history = []
        self.innovation_scores = []
        
    def record_innovation(self, architecture, performance, novelty_score):
        """记录新发现的架构"""
        innovation_data = {
            'architecture': architecture,
            'performance': performance,
            'novelty_score': novelty_score,
            'innovation_score': 0.7 * performance + 0.3 * novelty_score,
            'discovery_time': len(self.discovered_architectures)
        }
        self.discovered_architectures.append(innovation_data)
        self.performance_history.append(performance)
        self.innovation_scores.append(innovation_data['innovation_score'])
        
        # 按创新分数排序
        self.discovered_architectures.sort(key=lambda x: x['innovation_score'], reverse=True)
        
    def get_best_architecture(self):
        """获取最佳架构"""
        if self.discovered_architectures:
            return self.discovered_architectures[0]['architecture']
        return None
    
    def get_innovation_statistics(self):
        """获取创新统计"""
        if not self.discovered_architectures:
            return {}
        
        return {
            'total_discovered': len(self.discovered_architectures),
            'best_performance': max(self.performance_history),
            'best_innovation': max(self.innovation_scores),
            'avg_performance': np.mean(self.performance_history),
            'avg_innovation': np.mean(self.innovation_scores)
        }


class ArchitectureMutator:
    """架构突变器：生成新的架构变体"""
    
    def __init__(self):
        self.mutation_operations = [
            self._add_skip_connection,
            self._change_activation,
            self._modify_kernel_size,
            self._add_attention,
            self._change_pooling,
            self._add_dilation,
            self._modify_channels
        ]
        
    def mutate_architecture(self, base_architecture):
        """对基础架构进行突变"""
        mutation_type = random.choice(self.mutation_operations)
        return mutation_type(base_architecture)
    
    def _add_skip_connection(self, arch):
        """添加跳跃连接"""
        arch['use_skip_connection'] = True
        arch['skip_connection_type'] = random.choice(['identity', 'conv1x1', 'additive'])
        return arch
    
    def _change_activation(self, arch):
        """改变激活函数"""
        arch['activation'] = random.choice(['elu', 'relu', 'leaky_relu', 'gelu'])
        return arch
    
    def _modify_kernel_size(self, arch):
        """修改卷积核大小"""
        arch['kernel_sizes'] = [random.choice([3, 5, 7, 9]) for _ in range(2)]
        return arch
    
    def _add_attention(self, arch):
        """添加注意力机制"""
        arch['use_attention'] = True
        arch['attention_type'] = random.choice(['channel', 'spatial', 'temporal', 'cross_channel'])
        return arch
    
    def _change_pooling(self, arch):
        """改变池化策略"""
        arch['pooling_type'] = random.choice(['avg', 'max', 'adaptive_avg', 'adaptive_max'])
        return arch
    
    def _add_dilation(self, arch):
        """添加扩张卷积"""
        arch['use_dilation'] = True
        arch['dilation_rates'] = [random.randint(1, 4) for _ in range(2)]
        return arch
    
    def _modify_channels(self, arch):
        """修改通道数"""
        arch['channel_multipliers'] = [random.choice([0.5, 1.0, 1.5, 2.0]) for _ in range(2)]
        return arch


class Advanced_NAS_LC_Block(nn.Module):
    """增强的NAS局部块：支持架构发现"""
    
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.3, architecture_config=None):
        super().__init__()
        
        # 默认架构配置
        self.architecture_config = architecture_config or {
            'operation_weights': None,
            'use_skip_connection': False,
            'activation': 'elu',
            'kernel_sizes': [3, 5],
            'use_attention': False,
            'pooling_type': 'avg',
            'use_dilation': False,
            'channel_multipliers': [1.0, 1.0]
        }
        
        self.F1 = F1
        self.kernLength = kernLength
        self.Chans = Chans
        self.D = D
        
        # 扩展的候选操作集合
        self.candidate_ops = nn.ModuleList([
            # 基础操作
            nn.Identity(),
            nn.Conv2d(F1, F1, (1, 3), padding='same'),
            nn.Conv2d(F1, F1, (1, 5), padding='same'),
            nn.Conv2d(F1, F1, (1, 7), padding='same'),
            nn.AvgPool2d((1, 3), stride=1, padding=(0, 1)),
            nn.MaxPool2d((1, 3), stride=1, padding=(0, 1)),
            
            # 高级操作
            nn.Conv2d(F1, F1, (1, 1)),  # 1x1卷积
            nn.Dropout2d(0.1),  # 空间dropout
            
            # 新增：深度可分离卷积
            nn.Sequential(
                nn.Conv2d(F1, F1, (1, 3), padding='same', groups=F1),
                nn.Conv2d(F1, F1, 1)
            ),
            
            # 新增：扩张卷积
            nn.Conv2d(F1, F1, (1, 3), padding='same', dilation=2),
        ])
        
        # 架构参数
        if self.architecture_config['operation_weights'] is None:
            self.alpha = nn.Parameter(torch.ones(len(self.candidate_ops)))
        else:
            self.alpha = nn.Parameter(torch.tensor(self.architecture_config['operation_weights']))
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 根据架构配置构建网络
        self._build_from_config()
        
    def _build_from_config(self):
        """根据架构配置构建网络"""
        config = self.architecture_config
        
        # 输入投影层
        self.input_conv = nn.Conv2d(1, self.F1, (1, self.kernLength), padding='same')
        self.bn_input = nn.BatchNorm2d(self.F1)
        
        # 激活函数
        if config['activation'] == 'elu':
            self.activation = nn.ELU()
        elif config['activation'] == 'relu':
            self.activation = nn.ReLU()
        elif config['activation'] == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:  # gelu
            self.activation = nn.GELU()
        
        # 注意力机制
        if config['use_attention']:
            if config['attention_type'] == 'channel':
                self.attention = ChannelAttention(self.F1)
            elif config['attention_type'] == 'spatial':
                self.attention = SpatialAttention()
            else:
                self.attention = CrossChannelAttention(self.F1)
        
        # 输出层（考虑通道倍增器）
        output_channels = int(self.F1 * self.D * config['channel_multipliers'][0])
        self.output_conv = nn.Conv2d(self.F1, output_channels, (self.Chans, 1), groups=self.F1)
        self.bn_output = nn.BatchNorm2d(output_channels)
        
        # 池化层
        pool_size = (1, self.kernLength // 8)
        if config['pooling_type'] == 'avg':
            self.pool1 = nn.AvgPool2d(pool_size)
            self.pool2 = nn.AvgPool2d(pool_size)
        elif config['pooling_type'] == 'max':
            self.pool1 = nn.MaxPool2d(pool_size)
            self.pool2 = nn.MaxPool2d(pool_size)
        else:  # adaptive
            self.pool1 = nn.AdaptiveAvgPool2d((None, pool_size[1]))
            self.pool2 = nn.AdaptiveAvgPool2d((None, pool_size[1]))
        
        self.dropout = nn.Dropout(0.3)
        
        # 跳跃连接
        if config['use_skip_connection']:
            if config['skip_connection_type'] == 'identity':
                self.skip_connection = nn.Identity()
            elif config['skip_connection_type'] == 'conv1x1':
                self.skip_connection = nn.Conv2d(self.F1, output_channels, 1)
            else:  # additive
                self.skip_connection = nn.Identity()
    
    def forward(self, x):
        # 输入投影
        x_input = self.input_conv(x)
        x_input = self.bn_input(x_input)
        x_input = self.activation(x_input)
        
        # 操作搜索
        weights = F.softmax(self.alpha / self.temperature, dim=0)
        mixed_output = 0
        
        for i, op in enumerate(self.candidate_ops):
            mixed_output += weights[i] * op(x_input)
        
        # 注意力机制
        if hasattr(self, 'attention'):
            mixed_output = self.attention(mixed_output)
        
        # 输出处理
        x = self.output_conv(mixed_output)
        x = self.bn_output(x)
        x = self.activation(x)
        
        # 跳跃连接
        if hasattr(self, 'skip_connection'):
            if self.architecture_config['skip_connection_type'] == 'additive':
                x = x + self.skip_connection(x_input)
            else:
                x = self.skip_connection(x)
        
        # 池化操作
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.pool2(x)
        x = self.dropout(x)
        
        return x.squeeze(2)
    
    def get_arch_weights(self):
        """获取架构权重"""
        return F.softmax(self.alpha / self.temperature, dim=0).detach().cpu().numpy()
    
    def get_architecture_config(self):
        """获取当前架构配置"""
        return self.architecture_config.copy()


# -------------------------- 注意力模块 --------------------------
class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class CrossChannelAttention(nn.Module):
    """跨通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 简化实现
        channel_weights = torch.sigmoid(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        return x * channel_weights


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
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3, architecture_config=None):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_local_blocks = n_local_blocks
        
        # 架构配置
        self.arch_config = architecture_config or {}
        
        # 残差连接
        self.res_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=1) if input_channels != hidden_dim else nn.Identity()
        
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1, padding=0)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        # 多头注意力
        if self.arch_config.get('use_multihead_attn', True):
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
        if hasattr(self, 'multihead_attn'):
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


class ImprovedMambaStyleGC_Block(nn.Module):
    """Mamba风格GC块"""
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3, architecture_config=None):
        super().__init__()
        self.mamba_block = ImprovedMambaStyleBlock(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            n_local_blocks=n_local_blocks,
            architecture_config=architecture_config
        )
        self.output_dim = None
        self.dim_initialized = False

    def forward(self, x, dropout_p=0.3):
        output = self.mamba_block(x, dropout_p=dropout_p)
        
        if not self.dim_initialized:
            self.output_dim = output.shape[1]
            self.dim_initialized = True
        
        return output


# -------------------------- 架构发现引擎 --------------------------
class ArchitectureDiscoveryEngine:
    """架构发现引擎：主动寻找新的有效架构"""
    
    def __init__(self, nb_classes=4, Chans=22, Samples=1125):
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        
        self.innovation_tracker = InnovationTracker()
        self.architecture_mutator = ArchitectureMutator()
        self.exploration_history = []
        
        # 探索参数
        self.exploration_rate = 0.3  # 30%的资源用于探索
        self.mutation_rate = 0.2     # 20%的概率进行架构突变
        self.performance_threshold = 0.65  # 性能阈值
        
    def discover_novel_architectures(self, train_loader, val_loader, device, num_explorations=10):
        """发现新架构的主函数"""
        print("🔍 开始架构发现探索...")
        
        best_performance = 0
        best_architecture = None
        
        for exploration_idx in range(num_explorations):
            print(f"\n🎯 探索轮次 {exploration_idx + 1}/{num_explorations}")
            
            # 生成新架构
            novel_arch = self._generate_novel_architecture()
            
            # 快速评估架构
            performance, novelty_score = self._evaluate_architecture(
                novel_arch, train_loader, val_loader, device
            )
            
            # 记录创新
            self.innovation_tracker.record_innovation(novel_arch, performance, novelty_score)
            
            # 更新最佳架构
            if performance > best_performance and performance > self.performance_threshold:
                best_performance = performance
                best_architecture = novel_arch
                print(f"✨ 发现优秀新架构！性能: {performance:.4f}, 创新度: {novelty_score:.4f}")
            
            # 记录探索历史
            self.exploration_history.append({
                'exploration_idx': exploration_idx,
                'architecture': novel_arch,
                'performance': performance,
                'novelty_score': novelty_score
            })
        
        # 输出发现统计
        stats = self.innovation_tracker.get_innovation_statistics()
        print(f"\n📊 架构发现统计:")
        print(f"   - 总发现架构数: {stats['total_discovered']}")
        print(f"   - 最佳性能: {stats['best_performance']:.4f}")
        print(f"   - 最佳创新分数: {stats['best_innovation']:.4f}")
        
        return best_architecture, best_performance
    
    def _generate_novel_architecture(self):
        """生成新架构"""
        # 基础架构配置
        base_config = {
            'operation_weights': None,
            'use_skip_connection': random.random() > 0.5,
            'activation': random.choice(['elu', 'relu', 'leaky_relu', 'gelu']),
            'kernel_sizes': [random.choice([3, 5, 7]) for _ in range(2)],
            'use_attention': random.random() > 0.7,
            'attention_type': random.choice(['channel', 'spatial', 'cross_channel']),
            'pooling_type': random.choice(['avg', 'max', 'adaptive_avg']),
            'use_dilation': random.random() > 0.6,
            'dilation_rates': [random.randint(1, 3) for _ in range(2)],
            'channel_multipliers': [random.choice([0.8, 1.0, 1.2, 1.5]) for _ in range(2)],
            'use_multihead_attn': random.random() > 0.4
        }
        
        # 应用突变
        if random.random() < self.mutation_rate:
            base_config = self.architecture_mutator.mutate_architecture(base_config)
        
        return base_config
    
    def _evaluate_architecture(self, architecture_config, train_loader, val_loader, device):
        """快速评估架构性能"""
        try:
            # 创建临时模型
            model = self._create_model_from_architecture(architecture_config)
            model = model.to(device)
            
            # 快速训练和评估
            performance = self._fast_training_evaluation(model, train_loader, val_loader, device)
            
            # 计算创新度
            novelty_score = self._calculate_novelty_score(architecture_config)
            
            return performance, novelty_score
            
        except Exception as e:
            print(f"⚠️ 架构评估失败: {e}")
            return 0.0, 0.0
    
    def _create_model_from_architecture(self, architecture_config):
        """根据架构配置创建模型"""
        class DiscoveredModel(nn.Module):
            def __init__(self, nb_classes, Chans, architecture_config):
                super().__init__()
                self.nb_classes = nb_classes
                self.fc_initialized = False
                
                # 使用发现的架构
                self.lc_block1 = Advanced_NAS_LC_Block(
                    F1=8, kernLength=48, Chans=Chans, 
                    architecture_config=architecture_config
                )
                
                self.lc_block2 = Advanced_NAS_LC_Block(
                    F1=16, kernLength=64, Chans=Chans,
                    architecture_config=architecture_config
                )
                
                self.mamba_block1 = ImprovedMambaStyleGC_Block(
                    input_channels=16, hidden_dim=64, n_local_blocks=4,
                    architecture_config=architecture_config
                )
                
                self.mamba_block2 = ImprovedMambaStyleGC_Block(
                    input_channels=32, hidden_dim=128, n_local_blocks=4,
                    architecture_config=architecture_config
                )
                
                self.fc = None
            
            def forward(self, x, dropout_p=0.3):
                x1 = self.lc_block1(x)
                x2 = self.lc_block2(x)
                
                x1_mamba = self.mamba_block1(x1, dropout_p=dropout_p)
                x2_mamba = self.mamba_block2(x2, dropout_p=dropout_p)
                
                if not self.fc_initialized:
                    fc_input_dim = x1_mamba.shape[1] + x2_mamba.shape[1]
                    self.fc = nn.Linear(fc_input_dim, self.nb_classes).to(x1_mamba.device)
                    self.fc_initialized = True
                
                x_concat = torch.cat([x1_mamba, x2_mamba], dim=1)
                return self.fc(x_concat)
        
        return DiscoveredModel(self.nb_classes, self.Chans, architecture_config)
    
    def _fast_training_evaluation(self, model, train_loader, val_loader, device, fast_epochs=5):
        """快速训练评估"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(fast_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # 快速验证
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def _calculate_novelty_score(self, architecture_config):
        """计算架构新颖度"""
        if not self.exploration_history:
            return 1.0  # 第一个架构最具新颖性
        
        # 计算与历史架构的差异
        dissimilarities = []
        for history in self.exploration_history[-10:]:  # 最近10个架构
            dissim = self._architecture_dissimilarity(architecture_config, history['architecture'])
            dissimilarities.append(dissim)
        
        return np.mean(dissimilarities)
    
    def _architecture_dissimilarity(self, arch1, arch2):
        """计算两个架构的差异度"""
        dissim = 0
        
        # 比较配置参数
        for key in arch1.keys():
            if key in arch2:
                if isinstance(arch1[key], list) and isinstance(arch2[key], list):
                    if arch1[key] != arch2[key]:
                        dissim += 0.2
                else:
                    if arch1[key] != arch2[key]:
                        dissim += 0.1
        
        return min(dissim, 1.0)
    
    def get_discovery_report(self):
        """获取发现报告"""
        stats = self.innovation_tracker.get_innovation_statistics()
        
        report = {
            'exploration_summary': stats,
            'top_architectures': self.innovation_tracker.discovered_architectures[:5],
            'exploration_trends': {
                'performance_trend': [exp['performance'] for exp in self.exploration_history],
                'novelty_trend': [exp['novelty_score'] for exp in self.exploration_history]
            }
        }
        
        return report


# -------------------------- 完整模型（集成架构发现） --------------------------
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
    """改进Mamba模型（支持架构发现）"""
    def __init__(self, nb_classes=4, Chans=22, hidden_dim1=64, hidden_dim2=128, 
                 n_local_blocks=4, use_nas=True, discovered_architecture=None):
        super().__init__()
        self.nb_classes = nb_classes
        self.fc_initialized = False
        self.use_nas = use_nas
        self.discovered_architecture = discovered_architecture
        
        # 使用发现的架构或默认NAS
        if use_nas:
            if discovered_architecture:
                print("🚀 使用发现的架构！")
                self.lc_block1 = Advanced_NAS_LC_Block(
                    F1=8, kernLength=48, Chans=Chans, 
                    architecture_config=discovered_architecture
                )
                self.lc_block2 = Advanced_NAS_LC_Block(
                    F1=16, kernLength=64, Chans=Chans,
                    architecture_config=discovered_architecture
                )
            else:
                print("🔬 使用NAS_LC_Block（神经架构搜索）")
                self.lc_block1 = Advanced_NAS_LC_Block(F1=8, kernLength=48, Chans=Chans)
                self.lc_block2 = Advanced_NAS_LC_Block(F1=16, kernLength=64, Chans=Chans)
        else:
            self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3)
            self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, AveragePooling=False)
        
        self.mamba_block1 = ImprovedMambaStyleGC_Block(
            input_channels=16,
            hidden_dim=hidden_dim1,
            n_local_blocks=n_local_blocks,
            architecture_config=discovered_architecture
        )
        self.mamba_block2 = ImprovedMambaStyleGC_Block(
            input_channels=32,
            hidden_dim=hidden_dim2,
            n_local_blocks=n_local_blocks,
            architecture_config=discovered_architecture
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
    
    def get_nas_weights(self):
        """获取NAS架构权重"""
        if self.use_nas:
            weights1 = self.lc_block1.get_arch_weights()
            weights2 = self.lc_block2.get_arch_weights()
            return {'lc_block1': weights1, 'lc_block2': weights2}
        else:
            return None


class EEG_DBNet_NAS(nn.Module):
    """纯NAS模型"""
    def __init__(self, nb_classes=4, Chans=22, hidden_dim1=32, hidden_dim2=64):
        super().__init__()
        self.nb_classes = nb_classes
        self.fc_initialized = False
        
        self.lc_block1 = Advanced_NAS_LC_Block(F1=8, kernLength=48, Chans=Chans)
        self.lc_block2 = Advanced_NAS_LC_Block(F1=16, kernLength=64, Chans=Chans)
        
        self.mamba_proj1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, hidden_dim1)
        )
        self.mamba_proj2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_dim2)
        )
        
        self.fc = None

    def forward(self, x, dropout_p=0.3):
        x1 = self.lc_block1(x)
        x2 = self.lc_block2(x)
        
        x1_mamba = self.mamba_proj1(x1)
        x2_mamba = self.mamba_proj2(x2)
        
        if not self.fc_initialized:
            fc_input_dim = x1_mamba.shape[1] + x2_mamba.shape[1]
            print(f"🔧 [NAS模型] 初始化全连接层：输入维度={fc_input_dim}")
            self.fc = nn.Linear(fc_input_dim, self.nb_classes).to(x1_mamba.device)
            self.fc_initialized = True
        
        x_concat = torch.cat([x1_mamba, x2_mamba], dim=1)
        return self.fc(x_concat)
