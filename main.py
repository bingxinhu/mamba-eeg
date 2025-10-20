import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# 导入自定义模块
from preprocess import get_data
from models import (EEG_DBNet, EEG_DBNet_ImprovedMamba, EEG_DBNet_NAS, 
                   ArchitectureDiscoveryEngine, Advanced_NAS_LC_Block)


# -------------------------- 全局配置 --------------------------
# 设备设置（GPU优先）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备：{device}")

# 路径配置
DATASET_PATH = "./dataset/2a/"  # BCI2a数据集路径
RESULTS_PATH = "./results_enhanced"  # 结果保存路径
os.makedirs(RESULTS_PATH, exist_ok=True)  # 自动创建文件夹


# -------------------------- 工具函数 --------------------------
def draw_learning_curves(history, model_name, save_path):
    """绘制学习曲线（准确率+损失），保存高清图片"""
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='训练集', color='#1f77b4', linewidth=1.5)
    plt.plot(history['val_acc'], label='验证集', color='#ff7f0e', linewidth=1.5)
    plt.title(f'{model_name} - 准确率', fontsize=12)
    plt.ylabel('准确率', fontsize=10)
    plt.xlabel('训练轮次（Epoch）', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练集', color='#1f77b4', linewidth=1.5)
    plt.plot(history['val_loss'], label='验证集', color='#ff7f0e', linewidth=1.5)
    plt.title(f'{model_name} - 损失', fontsize=12)
    plt.ylabel('交叉熵损失', fontsize=10)
    plt.xlabel('训练轮次（Epoch）', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 学习曲线已保存：{save_path}/{model_name}_learning_curves.png")


def draw_nas_weights_heatmap(nas_weights_all_subjects, save_path):
    """绘制所有被试的NAS权重热力图"""
    if not nas_weights_all_subjects:
        return
    
    plt.figure(figsize=(15, 8))
    
    # 提取所有被试的权重
    n_subjects = len(nas_weights_all_subjects)
    n_ops = len(nas_weights_all_subjects[0]['lc_block1'])
    
    # 创建权重矩阵
    weights_lc1 = np.zeros((n_subjects, n_ops))
    weights_lc2 = np.zeros((n_subjects, n_ops))
    
    for i, weights in enumerate(nas_weights_all_subjects):
        weights_lc1[i] = weights['lc_block1']
        weights_lc2[i] = weights['lc_block2']
    
    # 绘制热力图
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(weights_lc1, cmap='YlOrRd', aspect='auto')
    plt.title('LC_Block1 NAS权重分布', fontsize=12)
    plt.xlabel('操作索引', fontsize=10)
    plt.ylabel('被试编号', fontsize=10)
    plt.colorbar(im1)
    
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(weights_lc2, cmap='YlOrRd', aspect='auto')
    plt.title('LC_Block2 NAS权重分布', fontsize=12)
    plt.xlabel('操作索引', fontsize=10)
    plt.ylabel('被试编号', fontsize=10)
    plt.colorbar(im2)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/nas_weights_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 NAS权重热力图已保存：{save_path}/nas_weights_heatmap.png")


def draw_architecture_discovery_results(discovery_reports, save_path):
    """绘制架构发现结果"""
    if not discovery_reports:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 性能趋势
    plt.subplot(2, 3, 1)
    for i, report in enumerate(discovery_reports):
        if 'exploration_trends' in report and 'performance_trend' in report['exploration_trends']:
            plt.plot(report['exploration_trends']['performance_trend'], 
                    label=f'被试{i+1}', alpha=0.7, linewidth=2)
    plt.title('架构发现性能趋势', fontsize=12)
    plt.ylabel('准确率', fontsize=10)
    plt.xlabel('探索轮次', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    
    # 创新度趋势
    plt.subplot(2, 3, 2)
    for i, report in enumerate(discovery_reports):
        if 'exploration_trends' in report and 'novelty_trend' in report['exploration_trends']:
            plt.plot(report['exploration_trends']['novelty_trend'], 
                    label=f'被试{i+1}', alpha=0.7, linewidth=2)
    plt.title('架构创新度趋势', fontsize=12)
    plt.ylabel('创新度分数', fontsize=10)
    plt.xlabel('探索轮次', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    
    # 最佳架构性能对比
    plt.subplot(2, 3, 3)
    best_performances = []
    subject_ids = []
    for i, report in enumerate(discovery_reports):
        if 'exploration_summary' in report and 'best_performance' in report['exploration_summary']:
            best_performances.append(report['exploration_summary']['best_performance'])
            subject_ids.append(f'Sub{i+1}')
    
    if best_performances:
        plt.bar(subject_ids, best_performances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('各被试最佳架构性能', fontsize=12)
        plt.ylabel('准确率', fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
    
    # 架构发现数量
    plt.subplot(2, 3, 4)
    discovery_counts = []
    for i, report in enumerate(discovery_reports):
        if 'exploration_summary' in report and 'total_discovered' in report['exploration_summary']:
            discovery_counts.append(report['exploration_summary']['total_discovered'])
    
    if discovery_counts:
        plt.bar(subject_ids, discovery_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('各被试发现架构数量', fontsize=12)
        plt.ylabel('发现数量', fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
    
    # 创新分数分布
    plt.subplot(2, 3, 5)
    innovation_scores = []
    for report in discovery_reports:
        if 'top_architectures' in report:
            for arch in report['top_architectures'][:3]:  # 取前3个
                innovation_scores.append(arch['innovation_score'])
    
    if innovation_scores:
        plt.hist(innovation_scores, bins=10, alpha=0.7, color='#1f77b4', edgecolor='black')
        plt.title('创新分数分布', fontsize=12)
        plt.ylabel('频次', fontsize=10)
        plt.xlabel('创新分数', fontsize=10)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/architecture_discovery_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 架构发现总结图已保存：{save_path}/architecture_discovery_summary.png")


def adjust_dropout_p(train_loss, val_loss, current_p=0.3):
    """自适应Dropout：根据过拟合程度调整p值（避免过拟合）"""
    if val_loss > 2 * train_loss + 1e-6:  # 轻微过拟合：p=0.4
        return min(current_p + 0.1, 0.5)
    elif val_loss < 1.2 * train_loss + 1e-6:  # 正常拟合：p=0.3
        return 0.3
    else:  # 无过拟合：p=0.3
        return 0.3


def set_layerwise_lr(model, base_lr=1e-4, mamba_lr_scale=0.5):
    """分层学习率：Mamba层用更低的学习率（避免参数震荡）"""
    params = []
    for name, param in model.named_parameters():
        if 'mamba_block' in name:  # Mamba层参数：学习率=base_lr×0.5
            params.append({'params': param, 'lr': base_lr * mamba_lr_scale})
        else:  # 其他层（LC_Block、全连接）：基础学习率
            params.append({'params': param, 'lr': base_lr})
    return params


def ensemble_predict(models, test_loader, device):
    """模型集成预测"""
    all_preds = []
    all_probs = []
    
    for model in models:
        model.eval()
        model_probs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())
        
        model_probs = np.concatenate(model_probs, axis=0)
        all_probs.append(model_probs)
    
    # 平均概率
    avg_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    return final_preds


def time_jitter(data, max_shift=10):
    """时间维度抖动数据增强 - 支持四维输入 (batch, 1, channels, time)"""
    if len(data.shape) == 4:
        # 四维数据: (batch, 1, channels, time)
        batch, _, channels, time = data.shape
        jittered_data = []
        
        for i in range(batch):
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:
                # 在时间维度上向右移动，左侧填充0
                jittered = np.pad(data[i, 0, :, shift:], 
                                ((0,0), (0,shift)), mode='constant')
            elif shift < 0:
                # 在时间维度上向左移动，右侧填充0
                jittered = np.pad(data[i, 0, :, :shift], 
                                ((0,0), (-shift,0)), mode='constant')
            else:
                jittered = data[i, 0]
            jittered_data.append(jittered)
        
        # 重新添加第二维度 (1)
        jittered_array = np.stack(jittered_data)
        jittered_array = jittered_array[:, np.newaxis, :, :]  # 添加第二维度
        return jittered_array
    
    elif len(data.shape) == 3:
        # 三维数据: (batch, channels, time)
        batch, channels, time = data.shape
        jittered_data = []
        
        for i in range(batch):
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:
                jittered = np.pad(data[i, :, shift:], 
                                ((0,0), (0,shift)), mode='constant')
            elif shift < 0:
                jittered = np.pad(data[i, :, :shift], 
                                ((0,0), (-shift,0)), mode='constant')
            else:
                jittered = data[i]
            jittered_data.append(jittered)
        
        return np.stack(jittered_data)
    
    else:
        raise ValueError(f"不支持的输入维度: {data.shape}")


# -------------------------- 训练与测试函数 --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs, patience, min_epochs, val_loss_ratio, model_name, device, use_warmup=True):
    """改进训练函数：解决早停太早问题"""
    # 初始化训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'dropout_p': [], 'learning_rate': []
    }
    best_val_acc = 0.0  # 最佳验证准确率
    best_val_loss = float('inf')  # 最佳验证损失
    early_stop_counter = 0  # 早停计数器
    best_model_state = None  # 最佳模型权重
    current_dropout_p = 0.3  # 初始Dropout p值
    
    print(f"\n📌 开始训练 {model_name}")
    print(f"   - 总轮次：{epochs} | 最小训练轮次：{min_epochs} | 早停耐心值：{patience} | 过拟合阈值：{val_loss_ratio}倍")
    
    # 学习率热身
    warmup_epochs = 100
    if use_warmup:
        warmup_scheduler = LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs)
        )
    
    for epoch in range(epochs):
        # 学习率热身
        if use_warmup and epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # -------------------------- 训练阶段 --------------------------
        model.train()
        train_total = 0
        train_correct = 0
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 数据增强：时间抖动（50%概率）
            if np.random.random() > 0.5:
                inputs_np = inputs.cpu().numpy()
                inputs_jittered = time_jitter(inputs_np, max_shift=8)
                inputs = torch.FloatTensor(inputs_jittered).to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播（改进Mamba模型需传递Dropout p值）
            if hasattr(model, 'use_nas') or hasattr(model, 'fc_initialized'):
                outputs = model(inputs, dropout_p=current_dropout_p)
            else:
                outputs = model(inputs)
            
            # 计算损失与反向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计训练指标
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 计算平均训练指标
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # -------------------------- 验证阶段 --------------------------
        model.eval()
        val_total = 0
        val_correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if hasattr(model, 'use_nas') or hasattr(model, 'fc_initialized'):
                    outputs = model(inputs, dropout_p=current_dropout_p)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算平均验证指标
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # -------------------------- 动态调整策略 --------------------------
        # 1. 自适应Dropout
        current_dropout_p = adjust_dropout_p(avg_train_loss, avg_val_loss, current_dropout_p)
        
        # 2. 动态调整NAS温度参数
        if hasattr(model, 'lc_block1') and hasattr(model.lc_block1, 'temperature'):
            # 随着训练进行，逐渐降低温度（从探索到利用）
            new_temp = max(0.5, 1.0 - epoch / epochs * 0.8)
            model.lc_block1.temperature.data.fill_(new_temp)
            model.lc_block2.temperature.data.fill_(new_temp)
        
        # 3. 学习率调度（余弦退火）
        if epoch >= warmup_epochs or not use_warmup:
            scheduler.step()
        
        # -------------------------- 记录与打印 --------------------------
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['dropout_p'].append(current_dropout_p)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 打印当前轮次信息（突出显示最小训练轮次进度）
        min_epochs_progress = f"{epoch+1}/{min_epochs}" if epoch+1 < min_epochs else f"✅ {min_epochs}"
        print(f"[{epoch+1:03d}/{epochs:03d}] "
              f"最小轮次进度：{min_epochs_progress} | "
              f"训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
              f"Dropout p: {current_dropout_p:.1f} | "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # -------------------------- 最佳模型保存与早停判断（核心修改） --------------------------
        # 1. 更新最佳模型（无论是否达最小轮次，都记录最佳权重）
        if (val_acc > best_val_acc + 1e-6) or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0  # 重置早停计数器
            print(f"✨ 找到更佳模型！验证准确率: {best_val_acc:.4f}（轮次{epoch+1}）")
        else:
            early_stop_counter += 1
        
        # 2. 早停判断：仅当达到最小训练轮次后，才允许触发早停
        if epoch + 1 < min_epochs:
            # 未达最小轮次：跳过早停，继续训练
            continue
        else:
            # 达最小轮次：检查早停条件
            stop_condition1 = early_stop_counter >= patience  # 耐心值耗尽
            stop_condition2 = avg_val_loss > val_loss_ratio * avg_train_loss + 1e-6  # 过拟合严重
            
            if stop_condition1 or stop_condition2:
                # 打印早停原因，方便调试
                stop_reason = f"早停计数器达到{patience}（耐心值耗尽）" if stop_condition1 else \
                              f"验证损失超过{val_loss_ratio}倍训练损失（过拟合）"
                print(f"\n🛑 早停触发！原因：{stop_reason}")
                print(f"   - 最佳验证准确率: {best_val_acc:.4f}（对应轮次{epoch+1 - early_stop_counter}）")
                break
    
    # 加载最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ 加载最佳模型权重（验证准确率: {best_val_acc:.4f}）")
    else:
        print("⚠️  未找到最佳模型（所有轮次未改进）")
    
    return model, history, best_val_acc


def test_model(model, test_loader, device):
    """测试函数：计算准确率、Kappa系数、混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播（测试时关闭Dropout）
            if hasattr(model, 'use_nas') or hasattr(model, 'fc_initialized'):
                outputs = model(inputs, dropout_p=0.0)
            else:
                outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)  # 抗类别不平衡
    cf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')  # 行归一化（真实标签视角）
    
    return acc, kappa, cf_matrix


# -------------------------- 架构发现函数 --------------------------
def architecture_discovery_phase(data_path, subject, device, discovery_config):
    """架构发现阶段：主动寻找新的有效架构"""
    print(f"\n🚀 启动架构发现阶段 - 被试 {subject+1}")
    
    # 加载数据
    X_train, y_train, X_test, y_test = get_data(
        data_path=data_path,
        subject=subject,
        loso=False,
        is_standard=True,
        fre_filter=discovery_config['fre_filter'],
        dataset='BCI2a'
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)  # 使用测试集作为验证集
    train_loader = DataLoader(train_dataset, batch_size=discovery_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=discovery_config['batch_size'], shuffle=False)
    
    # 计算通道数（考虑多频段滤波）
    n_chans = 22 * 5 if discovery_config['fre_filter'] else 22
    
    # 初始化架构发现引擎
    discovery_engine = ArchitectureDiscoveryEngine(
        nb_classes=4, 
        Chans=n_chans,
        Samples=1125
    )
    
    # 执行架构发现
    best_architecture, best_performance = discovery_engine.discover_novel_architectures(
        train_loader, val_loader, device, 
        num_explorations=discovery_config['num_explorations']
    )
    
    # 生成发现报告
    discovery_report = discovery_engine.get_discovery_report()
    
    print(f"\n🎉 架构发现完成！")
    print(f"   - 最佳架构性能: {best_performance:.4f}")
    print(f"   - 总发现架构数: {discovery_report['exploration_summary']['total_discovered']}")
    
    # 打印最佳架构配置
    if best_architecture:
        print(f"   - 最佳架构配置:")
        for key, value in best_architecture.items():
            print(f"     {key}: {value}")
    
    return best_architecture, discovery_report


# -------------------------- 主运行函数（集成架构发现） --------------------------
def main():
    # 超参数配置（核心调整：早停相关）
    BATCH_SIZE = 16  # 减小批量大小，提高梯度更新频率
    EPOCHS = 3000    # 增加总训练轮次
    PATIENCE = 800   # 增加早停耐心值
    MIN_EPOCHS = 1500  # 增加最小训练轮次
    VAL_LOSS_RATIO = 5.0  # 进一步放宽过拟合阈值
    BASE_LR = 8e-4   # 适当提高学习率
    N_SUBJECTS = 9   # BCI2a共9个被试
    FRE_FILTER = True  # 启用多频段滤波
    LOSO = False     # False=被试内验证，True=留一法交叉验证
    
    # NAS配置
    USE_NAS = True   # 是否使用NAS架构
    NAS_ONLY = False  # 是否使用纯NAS模型（不包含Mamba）
    ENSEMBLE_SIZE = 3  # 集成学习模型数量
    
    # 架构发现配置
    ENABLE_ARCHITECTURE_DISCOVERY = True  # 是否启用架构发现
    USE_DISCOVERED_ARCHITECTURE = True    # 是否使用发现的架构
    DISCOVERY_SUBJECTS = 3                # 在前几个被试上进行架构发现
    
    # 架构发现参数
    discovery_config = {
        'batch_size': 16,
        'num_explorations': 8,  # 每个被试的探索次数
        'fre_filter': FRE_FILTER
    }
    
    print(f"🧠 增强配置：")
    print(f"   - NAS架构：USE_NAS={USE_NAS}, NAS_ONLY={NAS_ONLY}")
    print(f"   - 集成学习：ENSEMBLE_SIZE={ENSEMBLE_SIZE}")
    print(f"   - 架构发现：ENABLE_DISCOVERY={ENABLE_ARCHITECTURE_DISCOVERY}, USE_DISCOVERED={USE_DISCOVERED_ARCHITECTURE}")
    print(f"   - 训练策略：BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
    
    # 结果存储（按被试统计）
    results = {
        'original': {'acc': [], 'kappa': []},
        'improved_mamba': {'acc': [], 'kappa': []},
        'ensemble': {'acc': [], 'kappa': []},
        'discovered_arch': {'acc': [], 'kappa': []}  # 新增：使用发现架构的模型
    }
    
    # NAS权重存储（用于分析）
    nas_weights_all_subjects = []
    
    # 架构发现存储
    discovered_architectures = []
    discovery_reports = []
    
    # 循环处理每个被试
    for sub_idx in range(N_SUBJECTS):
        print(f"\n{'='*70}")
        print(f"🔍 处理被试 {sub_idx+1}/{N_SUBJECTS}")
        print(f"{'='*70}")
        
        # 架构发现阶段（可选）
        current_discovered_arch = None
        if ENABLE_ARCHITECTURE_DISCOVERY and sub_idx < DISCOVERY_SUBJECTS:
            discovered_arch, discovery_report = architecture_discovery_phase(
                DATASET_PATH, sub_idx, device, discovery_config
            )
            discovered_architectures.append(discovered_arch)
            discovery_reports.append(discovery_report)
            current_discovered_arch = discovered_arch
        
        # 1. 加载预处理数据
        X_train, y_train, X_test, y_test = get_data(
            data_path=DATASET_PATH,
            subject=sub_idx,
            loso=LOSO,
            is_standard=True,
            fre_filter=FRE_FILTER,
            dataset='BCI2a'
        )
        
        # 计算通道数
        n_chans = 22 * 5 if FRE_FILTER else 22
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        val_loader = test_loader  # 被试内验证：测试集作为验证集
    
        # 2. 训练原始模型（EEG_DBNet）
        print(f"\n[1/4] 训练原始GC_Block模型")
        model_original = EEG_DBNet(nb_classes=4, Chans=n_chans, Samples=1125).to(device)
        
        # 优化器与调度器
        criterion = nn.CrossEntropyLoss()
        optimizer_original = optim.Adam(model_original.parameters(), lr=BASE_LR, weight_decay=1e-4)
        scheduler_original = CosineAnnealingLR(optimizer_original, T_max=EPOCHS)
        
        # 训练原始模型
        model_original_best, history_original, _ = train_model(
            model=model_original,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_original,
            scheduler=scheduler_original,
            epochs=EPOCHS,
            patience=PATIENCE,
            min_epochs=MIN_EPOCHS,
            val_loss_ratio=VAL_LOSS_RATIO,
            model_name=f"Original_Subject_{sub_idx+1}",
            device=device,
            use_warmup=True
        )
        
        # 测试原始模型
        acc_original, kappa_original, _ = test_model(
            model=model_original_best,
            test_loader=test_loader,
            device=device
        )
        results['original']['acc'].append(acc_original)
        results['original']['kappa'].append(kappa_original)
        
        # 保存原始模型与学习曲线
        torch.save(model_original_best.state_dict(), 
                  f"{RESULTS_PATH}/original_subject_{sub_idx+1}.pth")
        draw_learning_curves(history_original, f"Original_Subject_{sub_idx+1}", RESULTS_PATH)
        
        # 3. 训练改进模型（集成学习）
        mamba_models = []
        mamba_histories = []
        
        # 使用发现的架构（如果启用且已发现）
        use_discovered_arch = None
        if USE_DISCOVERED_ARCHITECTURE and discovered_architectures:
            use_discovered_arch = discovered_architectures[min(sub_idx, len(discovered_architectures)-1)]
            print(f"🎯 使用发现的架构进行训练")
        
        for ensemble_idx in range(ENSEMBLE_SIZE):
            print(f"\n[2.{ensemble_idx+1}/{ENSEMBLE_SIZE}] 训练改进Mamba模型{'（发现架构）' if use_discovered_arch else '（NAS架构）' if USE_NAS else ''}")
            
            if NAS_ONLY:
                model_mamba = EEG_DBNet_NAS(
                    nb_classes=4,
                    Chans=n_chans,
                    hidden_dim1=32,
                    hidden_dim2=64
                ).to(device)
                model_suffix = "NAS"
            else:
                model_mamba = EEG_DBNet_ImprovedMamba(
                    nb_classes=4,
                    Chans=n_chans,
                    hidden_dim1=64,  # 增加隐藏层维度
                    hidden_dim2=128,
                    n_local_blocks=4,  # 增加局部块数量
                    use_nas=USE_NAS,
                    discovered_architecture=use_discovered_arch  # 传入发现的架构
                ).to(device)
                model_suffix = "DiscoveredArch" if use_discovered_arch else "ImprovedMamba_NAS" if USE_NAS else "ImprovedMamba"
            
            # 分层学习率（Mamba层学习率更低）
            if not NAS_ONLY:
                params_mamba = set_layerwise_lr(model_mamba, base_lr=BASE_LR, mamba_lr_scale=0.5)
                optimizer_mamba = optim.Adam(params_mamba, weight_decay=1e-4)
            else:
                optimizer_mamba = optim.Adam(model_mamba.parameters(), lr=BASE_LR, weight_decay=1e-4)
                
            scheduler_mamba = CosineAnnealingLR(optimizer_mamba, T_max=EPOCHS)
            
            # 训练改进模型
            model_mamba_best, history_mamba, _ = train_model(
                model=model_mamba,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_mamba,
                scheduler=scheduler_mamba,
                epochs=EPOCHS,
                patience=PATIENCE,
                min_epochs=MIN_EPOCHS,
                val_loss_ratio=VAL_LOSS_RATIO,
                model_name=f"{model_suffix}_Subject_{sub_idx+1}_Ensemble{ensemble_idx+1}",
                device=device,
                use_warmup=True
            )
            
            mamba_models.append(model_mamba_best)
            mamba_histories.append(history_mamba)
            
            # 保存单个模型
            torch.save(model_mamba_best.state_dict(), 
                      f"{RESULTS_PATH}/{model_suffix.lower()}_subject_{sub_idx+1}_ensemble{ensemble_idx+1}.pth")
            draw_learning_curves(history_mamba, f"{model_suffix}_Subject_{sub_idx+1}_Ensemble{ensemble_idx+1}", RESULTS_PATH)
            
            # 收集NAS权重（如果使用NAS且没有使用发现架构）
            if USE_NAS and not NAS_ONLY and ensemble_idx == 0 and not use_discovered_arch:
                nas_weights = model_mamba_best.get_nas_weights()
                if nas_weights is not None:
                    nas_weights_all_subjects.append(nas_weights)
                    print(f"📊 被试{sub_idx+1} NAS架构权重：")
                    print(f"   - LC_Block1: {nas_weights['lc_block1']}")
                    print(f"   - LC_Block2: {nas_weights['lc_block2']}")
        
        # 4. 集成学习测试
        print(f"\n[3/4] 集成学习测试")
        ensemble_preds = ensemble_predict(mamba_models, test_loader, device)
        acc_ensemble = accuracy_score(y_test.numpy(), ensemble_preds)
        kappa_ensemble = cohen_kappa_score(y_test.numpy(), ensemble_preds)
        
        results['ensemble']['acc'].append(acc_ensemble)
        results['ensemble']['kappa'].append(kappa_ensemble)
        
        # 单个改进模型测试（取第一个集成模型）
        acc_mamba, kappa_mamba, _ = test_model(
            model=mamba_models[0],
            test_loader=test_loader,
            device=device
        )
        results['improved_mamba']['acc'].append(acc_mamba)
        results['improved_mamba']['kappa'].append(kappa_mamba)
        
        # 5. 使用发现架构的模型测试（如果使用了发现架构）
        if use_discovered_arch:
            acc_discovered, kappa_discovered, _ = test_model(
                model=mamba_models[0],  # 使用第一个模型
                test_loader=test_loader,
                device=device
            )
            results['discovered_arch']['acc'].append(acc_discovered)
            results['discovered_arch']['kappa'].append(kappa_discovered)
        
        # 6. 打印当前被试对比结果
        print(f"\n{'='*60}")
        print(f"被试 {sub_idx+1} 结果对比")
        print(f"{'='*60}")
        print(f"原始GC_Block     | 准确率: {acc_original:.4f} | Kappa: {kappa_original:.4f}")
        print(f"{model_suffix:15} | 准确率: {acc_mamba:.4f} | Kappa: {kappa_mamba:.4f}")
        print(f"集成学习({ENSEMBLE_SIZE}模型) | 准确率: {acc_ensemble:.4f} | Kappa: {kappa_ensemble:.4f}")
        
        if use_discovered_arch:
            print(f"发现架构模型    | 准确率: {acc_discovered:.4f} | Kappa: {kappa_discovered:.4f}")
            print(f"改进幅度(发现架构) | 准确率: {acc_discovered - acc_original:+.4f} | Kappa: {kappa_discovered - kappa_original:+.4f}")
        else:
            print(f"改进幅度         | 准确率: {acc_ensemble - acc_original:+.4f} | Kappa: {kappa_ensemble - kappa_original:+.4f}")
        print(f"{'='*60}")
    
    # -------------------------- 最终结果汇总 --------------------------
    # 计算平均性能
    avg_original_acc = np.mean(results['original']['acc'])
    avg_original_kappa = np.mean(results['original']['kappa'])
    avg_mamba_acc = np.mean(results['improved_mamba']['acc'])
    avg_mamba_kappa = np.mean(results['improved_mamba']['kappa'])
    avg_ensemble_acc = np.mean(results['ensemble']['acc'])
    avg_ensemble_kappa = np.mean(results['ensemble']['kappa'])
    
    # 绘制NAS权重热力图
    if nas_weights_all_subjects:
        draw_nas_weights_heatmap(nas_weights_all_subjects, RESULTS_PATH)
    
    # 绘制架构发现结果
    if discovery_reports:
        draw_architecture_discovery_results(discovery_reports, RESULTS_PATH)
    
    # 保存结果到NPZ文件（数值格式）
    final_results = {
        'original_acc_per_subject': np.array(results['original']['acc']),
        'original_kappa_per_subject': np.array(results['original']['kappa']),
        'improved_mamba_acc_per_subject': np.array(results['improved_mamba']['acc']),
        'improved_mamba_kappa_per_subject': np.array(results['improved_mamba']['kappa']),
        'ensemble_acc_per_subject': np.array(results['ensemble']['acc']),
        'ensemble_kappa_per_subject': np.array(results['ensemble']['kappa']),
        'average_original_acc': avg_original_acc,
        'average_original_kappa': avg_original_kappa,
        'average_improved_acc': avg_mamba_acc,
        'average_improved_kappa': avg_mamba_kappa,
        'average_ensemble_acc': avg_ensemble_acc,
        'average_ensemble_kappa': avg_ensemble_kappa
    }
    
    # 添加发现架构的结果（如果存在）
    if results['discovered_arch']['acc']:
        final_results['discovered_arch_acc_per_subject'] = np.array(results['discovered_arch']['acc'])
        final_results['discovered_arch_kappa_per_subject'] = np.array(results['discovered_arch']['kappa'])
        final_results['average_discovered_arch_acc'] = np.mean(results['discovered_arch']['acc'])
        final_results['average_discovered_arch_kappa'] = np.mean(results['discovered_arch']['kappa'])
    
    np.savez(f"{RESULTS_PATH}/final_results.npz", **final_results)
    print(f"\n📥 最终结果已保存：{RESULTS_PATH}/final_results.npz")
    
    # 保存架构发现结果
    if discovered_architectures:
        discovery_results = {
            'architectures': discovered_architectures,
            'reports': discovery_reports
        }
        torch.save(discovery_results, f"{RESULTS_PATH}/architecture_discovery_results.pth")
        print(f"💾 架构发现结果已保存")
    
    # 打印整体对比结果
    print(f"\n{'='*80}")
    print(f"📋 所有被试平均结果对比（集成学习{ENSEMBLE_SIZE}模型）")
    print(f"{'='*80}")
    print(f"原始GC_Block模型     | 平均准确率: {avg_original_acc:.4f} | 平均Kappa: {avg_original_kappa:.4f}")
    print(f"{model_suffix:18}模型 | 平均准确率: {avg_mamba_acc:.4f} | 平均Kappa: {avg_mamba_kappa:.4f}")
    print(f"集成学习模型         | 平均准确率: {avg_ensemble_acc:.4f} | 平均Kappa: {avg_ensemble_kappa:.4f}")
    
    if results['discovered_arch']['acc']:
        avg_discovered_acc = np.mean(results['discovered_arch']['acc'])
        avg_discovered_kappa = np.mean(results['discovered_arch']['kappa'])
        print(f"发现架构模型         | 平均准确率: {avg_discovered_acc:.4f} | 平均Kappa: {avg_discovered_kappa:.4f}")
        print(f"平均改进幅度(发现架构) | 准确率: {avg_discovered_acc - avg_original_acc:+.4f} | Kappa: {avg_discovered_kappa - avg_original_kappa:+.4f}")
    else:
        print(f"平均改进幅度         | 准确率: {avg_ensemble_acc - avg_original_acc:+.4f} | Kappa: {avg_ensemble_kappa - avg_original_kappa:+.4f}")
    
    print(f"{'='*80}")
    
    # 打印架构发现总结
    if discovery_reports:
        print(f"\n{'='*50}")
        print(f"🏆 架构发现总结")
        print(f"{'='*50}")
        for i, report in enumerate(discovery_reports):
            if 'exploration_summary' in report:
                stats = report['exploration_summary']
                print(f"被试 {i+1}:")
                print(f"  - 发现架构数: {stats['total_discovered']}")
                print(f"  - 最佳性能: {stats['best_performance']:.4f}")
                print(f"  - 最佳创新分数: {stats['best_innovation']:.4f}")
        
        # 计算总体发现统计
        total_discovered = sum(report['exploration_summary']['total_discovered'] for report in discovery_reports if 'exploration_summary' in report)
        avg_best_performance = np.mean([report['exploration_summary']['best_performance'] for report in discovery_reports if 'exploration_summary' in report])
        
        print(f"\n总体统计:")
        print(f"  - 总发现架构数: {total_discovered}")
        print(f"  - 平均最佳性能: {avg_best_performance:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
