import torch
import torch.nn as nn
import torch.nn.functional as F

class LC_Block(nn.Module):
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.25, activation='elu', AveragePooling=True):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.dwconv = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        
        pool_size = (1, kernLength // 8)
        self.pool1 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sep_conv = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, kernLength // 4), 
                                 padding='same', groups=F1 * D)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, 1, Chans, Samples)
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
        
        # 移除通道维度
        x = x.squeeze(2)  # (batch, F1*D, time)
        return x

class SE_Block(nn.Module):
    def __init__(self, seize=2, activation1='relu', activation2='sigmoid', BandSE=True):
        super(SE_Block, self).__init__()
        self.BandSE = BandSE
        self.activation1 = nn.ReLU() if activation1 == 'relu' else nn.ELU()
        self.activation2 = nn.Sigmoid() if activation2 == 'sigmoid' else nn.ReLU()

    def forward(self, x):
        if self.BandSE:
            # x shape: (batch, bands, time)
            x_avg = torch.mean(x, dim=2)  # (batch, bands)
            x_avg = x_avg.unsqueeze(1)    # (batch, 1, bands)
            
            # 动态创建并移动到输入数据所在设备
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            
            x_se = fc1(x_avg)
            x_se = self.activation1(x_se)
            x_se = fc2(x_se)
            x_se = self.activation2(x_se)
            x_se = x_se.permute(0, 2, 1)  # (batch, bands, 1)
        else:
            # x shape: (batch, channels, time)
            x_avg = torch.mean(x, dim=1)  # (batch, time)
            x_avg = x_avg.unsqueeze(1)    # (batch, 1, time)
            
            # 动态创建并移动到输入数据所在设备
            fc1 = nn.Linear(x_avg.size(2), 2).to(x.device)
            fc2 = nn.Linear(2, x_avg.size(2)).to(x.device)
            
            x_se = fc1(x_avg)
            x_se = self.activation1(x_se)
            x_se = fc2(x_se)
            x_se = self.activation2(x_se)  # (batch, 1, time)
        
        return x * x_se

class GC_Block(nn.Module):
    def __init__(self, depth=2, kernel_size=4, n_windows=5, step=4, seize=2, activation='elu', TimeConv=True):
        super(GC_Block, self).__init__()
        self.depth = depth
        self.n_windows = n_windows
        self.step = step
        self.TimeConv = TimeConv
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.conv_layers = nn.ModuleList()  # 动态卷积层

    def forward(self, x):
        # 根据输入形状初始化卷积层，并移动到输入数据所在设备
        if self.TimeConv:
            # x shape: (batch, channels, time)
            F1, F2 = x.size(1), x.size(2)
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(F1, F1, kernel_size=4, dilation=i+1, padding='same').to(x.device)
                for i in range(self.depth)
            ])
        else:
            # x shape: (batch, bands, time)
            F1, F2 = x.size(1), x.size(2)
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(F2, F2, kernel_size=4, dilation=i+1, padding='same').to(x.device)
                for i in range(self.depth)
            ])
        
        sw_concat = []
        for j in range(self.n_windows):
            if self.TimeConv:
                st = j * self.step
                end = F2 - (self.n_windows - j - 1) * self.step
                sw = x[:, :, st:end]
                se_block = SE_Block(seize=2, BandSE=False)(sw)
                last_block = se_block
                
                for i in range(self.depth):
                    block = self.conv_layers[i](last_block)
                    # 使用输入数据所在设备的均值和方差进行批归一化
                    block = F.batch_norm(
                        block, 
                        torch.zeros_like(block.mean(dim=[0, 2])).to(x.device), 
                        torch.ones_like(block.var(dim=[0, 2])).to(x.device), 
                        training=self.training
                    )
                    block = self.activation(block)
                    block = F.dropout(block, p=0.3, training=self.training)
                    block = block + se_block
                    last_block = self.activation(block)
                
                sw_concat.append(last_block.flatten(1))
            else:
                st = j * self.step
                end = F1 - (self.n_windows - j - 1) * self.step
                sw = x[:, st:end, :]
                se_block = SE_Block(seize=2, BandSE=True)(sw)
                last_block = se_block
                
                for i in range(self.depth):
                    block = self.conv_layers[i](last_block.transpose(1, 2)).transpose(1, 2)
                    # 使用输入数据所在设备的均值和方差进行批归一化
                    block = F.batch_norm(
                        block, 
                        torch.zeros_like(block.mean(dim=[0, 2])).to(x.device), 
                        torch.ones_like(block.var(dim=[0, 2])).to(x.device), 
                        training=self.training
                    )
                    block = self.activation(block)
                    block = F.dropout(block, p=0.3, training=self.training)
                    block = block + se_block
                    last_block = self.activation(block)
                
                sw_concat.append(last_block.flatten(1))
        
        return torch.cat(sw_concat, dim=1)

# 原始EEG_DBNet（使用GC_Block）
class EEG_DBNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, regRate=0.25, d=4, k=4, n=6, s=1, se=2):
        super(EEG_DBNet, self).__init__()
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=True)
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=False)
        
        self.gc_block1 = GC_Block(TimeConv=True, depth=d, kernel_size=k, n_windows=n, step=s, seize=se, activation='elu')
        self.gc_block2 = GC_Block(TimeConv=False, depth=d, kernel_size=k, n_windows=n, step=s, seize=se, activation='elu')
        
        # 修正全连接层输入维度（根据实际计算结果调整为5250）
        self.fc = nn.Linear(5250, nb_classes)

    def forward(self, x):
        # x shape: (batch, 1, Chans, Samples)
        x1 = self.lc_block1(x)  # (batch, 16, 31)
        x2 = self.lc_block2(x)  # (batch, 32, 17)
        
        x1 = self.gc_block1(x1)
        x2 = self.gc_block2(x2)
        
        x = torch.cat([x1, x2], dim=1)  # 拼接后维度为 (batch, 5250)
        x = self.fc(x)
        return x

# 极简但有效的Mamba替代方案 - 直接改进GC_Block
class MambaStyleBlock(nn.Module):
    """Mamba风格的时序建模块，专注于长程依赖"""
    def __init__(self, input_channels, output_features, hidden_dim=64):
        super().__init__()
        self.input_channels = input_channels
        self.output_features = output_features
        
        # 1. 时序特征提取 - 使用扩张卷积捕获长程依赖
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )
        
        # 2. 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 3. 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_features)
        
    def forward(self, x):
        # x: (batch, channels, time)
        
        # 时序特征提取
        temporal_features = self.temporal_conv(x)  # (batch, hidden_dim, time)
        
        # 全局特征聚合
        global_features = self.global_pool(temporal_features)  # (batch, hidden_dim, 1)
        global_features = global_features.squeeze(-1)  # (batch, hidden_dim)
        
        # 输出投影
        output = self.output_proj(global_features)  # (batch, output_features)
        
        return output

class MambaStyleGC_Block(nn.Module):
    """Mamba风格的GC Block - 直接替代原有GC_Block"""
    def __init__(self, input_channels, output_features, n_windows=5):
        super().__init__()
        self.input_channels = input_channels
        self.output_features = output_features
        self.n_windows = n_windows
        
        # Mamba风格模块
        self.mamba_block = MambaStyleBlock(
            input_channels=input_channels,
            output_features=output_features,
            hidden_dim=64
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        # 直接通过Mamba风格块处理
        output = self.mamba_block(x)  # (batch, output_features)
        return output

# 使用Mamba风格的EEG_DBNet
class EEG_DBNet_MambaStyle(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, regRate=0.25, d=4, k=4, n=6, s=1, se=2):
        super(EEG_DBNet_MambaStyle, self).__init__()
        
        # 保持原有的LC_Block不变
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=True)
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=False)
        
        # 用MambaStyleGC_Block替换原有的GC_Block
        # 保持与原始相同的输出维度
        self.gc_block1 = MambaStyleGC_Block(
            input_channels=16, 
            output_features=n*16,  # 与原始GC_Block相同的输出维度
            n_windows=n
        )
        self.gc_block2 = MambaStyleGC_Block(
            input_channels=32, 
            output_features=n*32,  # 与原始GC_Block相同的输出维度
            n_windows=n
        )
        
        # 全连接层输入维度
        fc_input_dim = n * (16 + 32)
        self.fc = nn.Linear(fc_input_dim, nb_classes)

    def forward(self, x):
        # x shape: (batch, 1, Chans, Samples)
        x1 = self.lc_block1(x)  # (batch, 16, time1)
        x2 = self.lc_block2(x)  # (batch, 32, time2)
        
        # 通过Mamba风格块处理
        x1 = self.gc_block1(x1)  # (batch, n*16)
        x2 = self.gc_block2(x2)  # (batch, n*32)
        
        # 拼接特征
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        
        return x
