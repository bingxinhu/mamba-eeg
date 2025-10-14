import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- 基础模块（保留原有功能，适配改进逻辑） --------------------------
class LC_Block(nn.Module):
    """局部特征提取块：空间卷积+深度可分离卷积+池化（EEG局部特征捕获）"""
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.3, activation='elu', AveragePooling=True):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')  # 1×K空间卷积
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.dwconv = nn.Conv2d(F1, F1*D, kernel_size=(Chans, 1), groups=F1)  # 深度可分离卷积（通道交互）
        self.bn2 = nn.BatchNorm2d(F1*D)
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        pool_size = (1, kernLength // 8)  # 时序下采样（缩小8倍）
        self.pool1 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sep_conv = nn.Conv2d(F1*D, F1*D, kernel_size=(1, kernLength//4), padding='same', groups=F1*D)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, 1, Chans, Samples) → 经过LC_Block后：(batch, F1*D, time)
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
        
        return x.squeeze(2)  # 移除通道维度（Chans已通过深度卷积融合）


class SE_Block(nn.Module):
    """注意力模块：通道/频段注意力（增强关键特征）"""
    def __init__(self, activation1='relu', activation2='sigmoid', BandSE=True):
        super(SE_Block, self).__init__()
        self.BandSE = BandSE  # True=频段注意力，False=通道注意力
        self.activation1 = nn.ReLU() if activation1 == 'relu' else nn.ELU()
        self.activation2 = nn.Sigmoid() if activation2 == 'sigmoid' else nn.ReLU()

    def forward(self, x):
        if self.BandSE:
            # x: (batch, bands, time) → 频段注意力
            x_avg = torch.mean(x, dim=2).unsqueeze(1)  # (batch, 1, bands)
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            x_se = self.activation2(fc2(self.activation1(fc1(x_avg))))
            return x * x_se.permute(0, 2, 1)  # (batch, bands, 1)
        else:
            # x: (batch, channels, time) → 通道注意力
            x_avg = torch.mean(x, dim=1).unsqueeze(1)  # (batch, 1, time)
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            x_se = self.activation2(fc2(self.activation1(fc1(x_avg))))
            return x * x_se  # (batch, 1, time)


class GC_Block(nn.Module):
    """原始全局卷积块（保留，用于与改进Mamba模型对比）"""
    def __init__(self, depth=2, kernel_size=4, n_windows=5, step=4, activation='elu', TimeConv=True):
        super(GC_Block, self).__init__()
        self.depth = depth
        self.n_windows = n_windows
        self.step = step
        self.TimeConv = TimeConv
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.conv_layers = nn.ModuleList()

    def forward(self, x):
        # 动态初始化卷积层（适配输入维度）
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
        
        # 滑动窗口处理时序数据
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


# -------------------------- 改进Mamba核心模块 --------------------------
class ChannelAttention(nn.Module):
    """通道注意力：聚焦高信息量脑区通道（如运动皮层C3/C4）"""
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 时序维度压缩，保留通道特征
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),  # 降维减少参数
            nn.ELU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),  # 升维回原通道数
            nn.Sigmoid()  # 输出0-1注意力权重
        )

    def forward(self, x):
        # x: (batch, hidden_dim, time) → 输出：(batch, hidden_dim, time)（加权后）
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)  # (batch, hidden_dim)
        y = self.fc(y).view(b, c, 1)     # (batch, hidden_dim, 1)
        return x * y


def calc_temporal_corr(x, window_size=50):
    """计算时序相关性（动态调整Mamba扩张系数的依据）
    返回：整个批次的平均相关性（单元素张量，形状(1,1)）
    """
    batch, chans, time = x.shape
    corr_scores = []
    
    for b in range(batch):
        chan_corr = []
        for c in range(chans):
            sig = x[b, c]  # 单通道时序信号
            corr = []
            # 滑动窗口计算相邻时间点相关性（避免除以0）
            for t in range(time - window_size):
                win = sig[t:t+window_size]
                cov = torch.cov(torch.stack([win[:-1], win[1:]]))[0, 1]
                std1, std2 = win[:-1].std(), win[1:].std()
                if std1 > 1e-6 and std2 > 1e-6:
                    corr_val = cov / (std1 * std2)
                else:
                    corr_val = torch.tensor(0.0, device=x.device)
                corr.append(corr_val)
            chan_corr.append(torch.tensor(corr).mean().item())  # 单通道平均相关性
        corr_scores.append(chan_corr)
    
    # 对通道和批次求均值，返回单元素张量（适配.item()调用）
    corr_tensor = torch.tensor(corr_scores, device=x.device)  # (batch, chans)
    batch_avg_corr = corr_tensor.mean(dim=1).mean(dim=0)       # 标量（整个批次平均）
    return batch_avg_corr.unsqueeze(-1).unsqueeze(-1)          # (1,1)


class ImprovedMambaStyleBlock(nn.Module):
    """改进Mamba时序块：动态扩张卷积+通道注意力+动态时序分块"""
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_local_blocks = n_local_blocks  # 局部块数量（3-5为宜）
        
        # 1. 输入通道投影（适配Mamba隐藏维度）
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1, padding=0)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        # 2. 通道注意力（增强关键脑区特征）
        self.channel_att = ChannelAttention(hidden_dim)
        
        # 3. 动态扩张卷积（捕获长/短程时序依赖）
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)  # dilation=1（固定）
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)  # 可动态调整
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)  # 可动态调整
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.activation = nn.ELU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局特征聚合
        self.dropout = nn.Dropout(p=0.3)  # 自适应Dropout（训练时动态调整p）

    def adjust_dilation(self, corr_score):
        """根据时序相关性动态调整扩张系数：
        - 高相关性（>0.6）：局部特征主导 → 小感受野（dilation=1,2）
        - 低相关性（<0.3）：长程依赖主导 → 大感受野（dilation=2,4）
        - 中间值：混合感受野（dilation=1,3）
        """
        if corr_score > 0.6:
            return 1, 2
        elif corr_score < 0.3:
            return 2, 4
        else:
            return 1, 3

    def forward(self, x, dropout_p=0.3):
        # x: (batch, input_channels, time) → 输出：(batch, 局部特征+全局特征)
        batch, _, time = x.shape
        
        # 1. 输入投影与标准化
        x = self.input_proj(x)  # (batch, hidden_dim, time)
        x = self.bn_proj(x)
        x = self.activation(x)
        
        # 2. 通道注意力加权
        x = self.channel_att(x)
        
        # 3. 动态调整扩张卷积（适配当前时序相关性）
        corr_score = calc_temporal_corr(x).item()
        dilation2, dilation3 = self.adjust_dilation(corr_score)
        self.conv2.dilation = (dilation2,)
        self.conv2.padding = (dilation2,)  # 保证输出时序长度不变
        self.conv3.dilation = (dilation3,)
        self.conv3.padding = (dilation3,)
        
        # 三级扩张卷积提取时序特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        # 4. 自适应Dropout（根据过拟合程度调整p）
        self.dropout.p = dropout_p
        x = self.dropout(x)
        
        # 5. 动态时序分块（局部特征+全局特征）
        local_window = time // self.n_local_blocks  # 动态窗口大小（避免超过时序长度）
        if local_window > 0:
            # 时序分块（unfold）：(batch, hidden_dim, n_blocks, window)
            local_pool = x.unfold(dimension=2, size=local_window, step=local_window)
            local_pool = local_pool.mean(dim=3)  # 每块求平均（压缩窗口维度）
            local_pool_flat = local_pool.flatten(1)  # (batch, hidden_dim * n_blocks)
        else:
            # 极端情况（时序过短）：局部特征=全局特征
            local_pool_flat = self.global_pool(x).squeeze(2)
        
        # 全局特征聚合
        global_pool = self.global_pool(x).squeeze(2)  # (batch, hidden_dim)
        
        # 特征拼接（局部+全局）
        return torch.cat([local_pool_flat, global_pool], dim=1)


class ImprovedMambaStyleGC_Block(nn.Module):
    """Mamba风格GC块（替代原始GC_Block，确保输出维度稳定）"""
    def __init__(self, input_channels, hidden_dim=32, n_local_blocks=3):
        super().__init__()
        self.mamba_block = ImprovedMambaStyleBlock(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            n_local_blocks=n_local_blocks
        )
        self.output_dim = None  # 记录稳定的输出维度
        self.dim_initialized = False  # 维度初始化标志位

    def forward(self, x, dropout_p=0.3):
        output = self.mamba_block(x, dropout_p=dropout_p)
        
        # 首次传播记录维度，后续校验稳定性（避免同一被试内维度波动）
        if not self.dim_initialized:
            self.output_dim = output.shape[1]
            self.dim_initialized = True
        else:
            assert output.shape[1] == self.output_dim, \
                f"❌ Mamba块输出维度异常：当前{output.shape[1]} != 记录{self.output_dim}"
        
        return output


# -------------------------- 完整模型（原始+改进Mamba） --------------------------
class EEG_DBNet(nn.Module):
    """原始EEG模型（使用GC_Block，用于对比）"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125):
        super(EEG_DBNet, self).__init__()
        # LC_Block参数：F1=8/16，kernLength=48/64（适配BCI2a时序长度）
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3)
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, AveragePooling=False)
        
        # GC_Block参数：depth=4，n_windows=6（原始论文设置）
        self.gc_block1 = GC_Block(TimeConv=True, depth=4, n_windows=6, step=1)
        self.gc_block2 = GC_Block(TimeConv=False, depth=4, n_windows=6, step=1)
        
        # 全连接层（原始模型固定输入维度5250）
        self.fc = nn.Linear(5250, nb_classes)

    def forward(self, x):
        # x: (batch, 1, Chans, Samples)
        x1 = self.lc_block1(x)  # (batch, 16, time1)
        x2 = self.lc_block2(x)  # (batch, 32, time2)
        
        x1 = self.gc_block1(x1)
        x2 = self.gc_block2(x2)
        
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)


class EEG_DBNet_ImprovedMamba(nn.Module):
    """改进Mamba模型（动态全连接层+维度校验）"""
    def __init__(self, nb_classes=4, Chans=22, hidden_dim1=32, hidden_dim2=64, n_local_blocks=3):
        super().__init__()
        self.nb_classes = nb_classes
        self.fc_initialized = False  # 全连接层初始化标志位（避免重复初始化）
        
        # 1. LC_Block（与原始模型一致，确保局部特征提取能力）
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3)  # 输出16通道
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, AveragePooling=False)  # 输出32通道
        
        # 2. 改进Mamba块（输入通道匹配LC_Block输出）
        self.mamba_block1 = ImprovedMambaStyleGC_Block(
            input_channels=16,  # LC_Block1输出通道数（8×2）
            hidden_dim=hidden_dim1,
            n_local_blocks=n_local_blocks
        )
        self.mamba_block2 = ImprovedMambaStyleGC_Block(
            input_channels=32,  # LC_Block2输出通道数（16×2）
            hidden_dim=hidden_dim2,
            n_local_blocks=n_local_blocks
        )
        
        # 3. 全连接层暂不初始化（等待首次前向传播确定维度）
        self.fc = None

    def forward(self, x, dropout_p=0.3):
        # x: (batch, 1, Chans, Samples)
        x1 = self.lc_block1(x)  # (batch, 16, time1)
        x2 = self.lc_block2(x)  # (batch, 32, time2)
        
        # Mamba块前向传播（首次传播会记录输出维度）
        x1_mamba = self.mamba_block1(x1, dropout_p=dropout_p)
        x2_mamba = self.mamba_block2(x2, dropout_p=dropout_p)
        
        # 首次传播：初始化全连接层（确保维度匹配）
        if not self.fc_initialized:
            fc_input_dim = x1_mamba.shape[1] + x2_mamba.shape[1]
            print(f"🔧 初始化全连接层：输入维度={fc_input_dim}，输出维度={self.nb_classes}")
            self.fc = nn.Linear(fc_input_dim, self.nb_classes).to(x1_mamba.device)
            self.fc_initialized = True
        
        # 特征拼接与分类（维度校验，避免不匹配）
        x_concat = torch.cat([x1_mamba, x2_mamba], dim=1)
        assert x_concat.shape[1] == self.fc.in_features, \
            f"❌ 特征拼接维度异常：{x_concat.shape[1]} != {self.fc.in_features}"
        
        return self.fc(x_concat)
