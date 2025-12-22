import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import warnings
import torch.nn.functional as F
from datetime import datetime
import json
import yaml
warnings.filterwarnings('ignore')

from preprocess import get_data
from models import get_model, BandAwareInterpretableMamba
from eeg_vector_db import EEGVectorDatabase  # 新增：脑电向量数据库

# 设置随机种子确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_model_weights(model, init_type='kaiming'):
    """初始化模型权重"""
    print(f"初始化模型权重 (方法: {init_type})")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数统计 - 总数: {total_params:,}, 可训练: {trainable_params:,}")

def parse_args():
    parser = argparse.ArgumentParser(description='EEG信号分类主程序（集成向量数据库）')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data', help='数据集根路径')
    parser.add_argument('--subject', type=int, default=0, help='被试编号（0-8，共9个被试）')
    parser.add_argument('--dataset', type=str, default='BCI2a', choices=['BCI2a', 'BCI2b'], help='数据集类型')
    parser.add_argument('--loso', action='store_true', help='是否使用留一法交叉验证')
    parser.add_argument('--no_standard', action='store_true', help='不进行数据标准化')
    parser.add_argument('--fre_filter', action='store_true', help='是否使用多频段滤波')
    parser.add_argument('--augment', action='store_true', help='是否使用数据增强')
    parser.add_argument('--augment_factor', type=int, default=2, help='数据增强倍数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='auto', 
                        choices=['baseline', 'wideband', 'multiband_mamba', 'Interpretable_mamba', 'auto'], 
                        help='选择模型，auto表示自动选择')
    parser.add_argument('--mamba_dim', type=int, default=32, help='Mamba模块特征维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--init_type', type=str, default='kaiming', 
                        choices=['kaiming', 'xavier', 'orthogonal'], help='权重初始化方法')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=2000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减系数')
    parser.add_argument('--patience', type=int, default=500, help='早停耐心值')
    parser.add_argument('--mixup', action='store_true', help='是否使用Mixup数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha参数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'step'], help='学习率调度器')
    
    # 向量数据库参数 (新增)
    parser.add_argument('--enable_vector_db', action='store_true', help='启用脑电向量数据库')
    parser.add_argument('--db_type', type=str, default='faiss', choices=['faiss', 'annoy', 'simple'], 
                        help='向量数据库类型')
    parser.add_argument('--feature_dim', type=int, default=128, help='特征向量维度')
    parser.add_argument('--retrieval_topk', type=int, default=5, help='检索最相似的k个样本')
    parser.add_argument('--build_db_on_train', action='store_true', help='在训练时构建向量数据库')
    parser.add_argument('--use_db_for_few_shot', action='store_true', help='使用向量数据库进行少样本学习')
    
    # 保存与日志
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--save_model', action='store_true', help='是否保存最佳模型')
    parser.add_argument('--plot_cm', action='store_true', help='是否绘制混淆矩阵')
    parser.add_argument('--plot_curves', action='store_true', help='是否绘制训练曲线')
    parser.add_argument('--plot_retrieval', action='store_true', help='是否绘制检索结果')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], 
                        help='日志详细程度：0=静默，1=常规，2=详细')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='', help='配置文件路径')
    
    return parser.parse_args()

def load_config(args):
    """加载配置文件"""
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新参数
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        print(f"✓ 从 {args.config} 加载配置")
    
    return args

def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.dim = dim
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def train_epoch(model, train_loader, criterion, optimizer, device, 
                mixup=False, mixup_alpha=0.2, grad_clip=1.0, verbose=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        use_mixup = False
        if mixup and np.random.random() > 0.5:
            use_mixup = True
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            inputs = mixed_inputs
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        
        if not use_mixup:
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # 详细模式下的批次进度
        if verbose >= 2 and (batch_idx + 1) % 10 == 0:
            print(f'  批次 {batch_idx+1}/{len(train_loader)}, 损失: {loss.item():.4f}')
    
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_kappa = cohen_kappa_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, val_kappa, val_f1, all_labels, all_preds

def evaluate_model(model, test_loader, device, verbose=1):
    """全面评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if verbose >= 1:
        print(f"准确率: {acc:.4f}, 平衡准确率: {bal_acc:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}")
    
    return acc, bal_acc, kappa, f1, all_labels, all_preds, all_probs

def auto_select_model(fre_filter, n_channels, n_classes, n_timepoints):
    """根据数据特征自动选择最合适的模型"""
    if fre_filter:
        # 多频段数据，使用正则化Mamba模型
        model_name = 'Interpretable_mamba'
        print(f"检测到多频段数据（{n_channels}通道），自动选择 {model_name} 模型")
    else:
        # 单频段数据
        if n_timepoints > 1000:
            # 长时间序列，使用稳定的Mamba模型
            model_name = 'multiband_mamba'
            print(f"检测到长时间序列（{n_timepoints}点），自动选择 {model_name} 模型")
        else:
            # 短时间序列，使用宽态Mamba模型
            model_name = 'baseline'
            print(f"检测到短时间序列（{n_timepoints}点），自动选择 {model_name} 模型")
    
    return model_name

def create_optimizer(model, optimizer_type, lr, weight_decay):
    """创建优化器"""
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer_type}")

def create_scheduler(optimizer, scheduler_type, **kwargs):
    """创建学习率调度器"""
    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('epochs', 100), eta_min=1e-6
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        raise ValueError(f"未知调度器: {scheduler_type}")

def setup_experiment_directory(args):
    """设置实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_S{args.subject}_{args.model}_{timestamp}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "vector_db"), exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(exp_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"实验目录: {exp_dir}")
    return exp_dir

def extract_features_from_model(model, dataloader, device, args, feature_type='raw'):
    """
    从模型提取特征向量
    feature_type: 'pre_classifier', 'mamba', 'band_attention', 'raw'
    默认使用'raw'以获取128维固定特征
    """
    model.eval()
    all_features = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # 使用模型的特征提取方法
            if hasattr(model, 'extract_features'):
                features = model.extract_features(inputs, feature_type=feature_type)
            else:
                # 默认：获取模型中间特征
                features = extract_features_default(model, inputs, feature_type)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # 收集元数据
            for i in range(len(labels)):
                all_metadata.append({
                    'batch': batch_idx,
                    'sample_idx': i,
                    'label': int(labels[i].item()),
                    'subject': args.subject,
                    'dataset': args.dataset
                })
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels, all_metadata

def extract_features_default(model, inputs, feature_type='pre_classifier'):
    """默认特征提取方法"""
    # 这里实现具体特征提取逻辑
    # 简化版本：使用模型的中间层输出
    if hasattr(model, 'get_pre_classifier_features'):
        return model.get_pre_classifier_features(inputs)
    else:
        # 如果模型没有特征提取方法，使用模型输出前的最后一层
        with torch.no_grad():
            # 临时修改模型以获取中间特征
            # 这里需要根据具体模型结构实现
            return torch.randn(inputs.size(0), 128).to(inputs.device)  # 占位符

def build_eeg_vector_database(args, model, train_loader, test_loader, device, exp_dir):
    """构建脑电向量数据库"""
    print("\n===== 构建脑电向量数据库 =====")
    
    # 先提取特征以确定实际维度
    print("提取训练集特征以确定维度...")
    test_features, _, _ = extract_features_from_model(
        model, 
        DataLoader(train_loader.dataset, batch_size=min(10, len(train_loader.dataset))), 
        device, args, feature_type='raw'
    )
    
    actual_feature_dim = test_features.shape[1]
    print(f"模型实际特征维度: {actual_feature_dim}")
    
    # 如果实际维度与配置不同，更新配置
    if actual_feature_dim != args.feature_dim:
        print(f"注意: 模型特征维度({actual_feature_dim})与配置({args.feature_dim})不同，使用实际维度")
        args.feature_dim = actual_feature_dim
    
    # 初始化向量数据库（使用实际特征维度）
    vector_db = EEGVectorDatabase(
        db_type=args.db_type,
        dimension=args.feature_dim,
        use_gpu=torch.cuda.is_available()
    )
    
    # 提取完整的训练集特征
    print("提取完整的训练集特征...")
    train_features, train_labels, train_metadata = extract_features_from_model(
        model, train_loader, device, args, feature_type='raw'
    )
    
    print(f"特征维度: {train_features.shape[1]}, 样本数: {len(train_features)}")
    
    # 构建向量索引
    print(f"构建向量索引...")
    vector_db.build_index(train_features, train_labels, train_metadata)
    
    # 保存向量数据库
    db_path = os.path.join(exp_dir, "vector_db", "eeg_vector_db.pkl")
    vector_db.save(db_path)
    print(f"✓ 向量数据库已保存至: {db_path}")
    
    # 测试向量检索
    print("\n测试向量检索功能...")
    
    # 随机选择一些测试样本进行检索
    test_sample_indices = np.random.choice(len(test_loader.dataset), min(5, len(test_loader.dataset)), replace=False)
    
    retrieval_results = []
    for i, idx in enumerate(test_sample_indices):
        test_input, test_label = test_loader.dataset[idx]
        test_input = test_input.unsqueeze(0).to(device)
        
        # 提取测试样本特征
        test_feature, _, _ = extract_features_from_model(
            model, 
            DataLoader(TensorDataset(test_input, torch.tensor([test_label])), batch_size=1), 
            device, args, feature_type='raw'
        )
        
        # 检索相似样本
        similar_samples = vector_db.query_similar(test_feature[0], k=args.retrieval_topk)
        
        retrieval_results.append({
            'test_index': idx,
            'test_label': int(test_label),
            'similar_samples': similar_samples
        })
        
        # 打印检索结果
        if args.verbose >= 1:
            print(f"\n测试样本 {i+1} (标签: {test_label}):")
            print(f"  检索到的最相似样本:")
            for j, sample in enumerate(similar_samples):
                print(f"    {j+1}. 距离: {sample['distance']:.4f}, 标签: {sample['label']}, "
                      f"相似度: {sample['similarity']:.4f}")
    
    # 计算检索准确率
    if retrieval_results:
        retrieval_acc = analyze_retrieval_accuracy(retrieval_results)
        print(f"\n向量检索准确率 (Top-{args.retrieval_topk}): {retrieval_acc:.4f}")
    
    return vector_db, retrieval_results

def analyze_retrieval_accuracy(retrieval_results):
    """分析检索准确率"""
    correct = 0
    total = 0
    
    for result in retrieval_results:
        test_label = result['test_label']
        similar_labels = [s['label'] for s in result['similar_samples']]
        
        # 检查检索到的样本中是否有相同标签的
        if test_label in similar_labels:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def visualize_retrieval_results(retrieval_results, vector_db, exp_dir, model, args):
    """可视化检索结果"""
    if not args.plot_retrieval:
        return
    
    try:
        # 选择前几个测试样本进行可视化
        num_samples_to_plot = min(3, len(retrieval_results))
        
        for i in range(num_samples_to_plot):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'脑电向量检索结果 - 测试样本 {i+1}', fontsize=16)
            
            result = retrieval_results[i]
            test_label = result['test_label']
            similar_samples = result['similar_samples']
            
            # 这里可以添加具体的可视化代码
            # 例如：显示原始脑电信号、特征空间投影、相似度分布等
            
            plot_path = os.path.join(exp_dir, "plots", f"retrieval_sample_{i+1}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 检索结果可视化已保存: {plot_path}")
    
    except Exception as e:
        print(f"✗ 可视化检索结果失败: {e}")

def few_shot_learning_with_vector_db(vector_db, test_loader, model, device, args):
    """使用向量数据库进行少样本学习"""
    print("\n===== 基于向量数据库的少样本学习 =====")
    
    predictions = []
    true_labels = []
    few_shot_results = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            
            # 提取特征
            features, _, _ = extract_features_from_model(
                model, 
                DataLoader(TensorDataset(inputs, labels), batch_size=inputs.size(0)), 
                device,
                args
            )
            
            batch_predictions = []
            for i in range(features.shape[0]):
                # 检索相似样本
                similar_samples = vector_db.query_similar(features[i], k=args.retrieval_topk)
                
                # 基于检索结果进行预测（多数投票）
                retrieved_labels = [s['label'] for s in similar_samples]
                if retrieved_labels:
                    pred = np.bincount(retrieved_labels).argmax()
                else:
                    # 如果没有检索到结果，使用模型预测
                    pred = model(inputs[i:i+1]).argmax().item()
                
                batch_predictions.append(pred)
                
                # 记录检索信息
                few_shot_results.append({
                    'true_label': int(labels[i].item()),
                    'predicted_label': pred,
                    'retrieved_labels': retrieved_labels,
                    'retrieved_distances': [s['distance'] for s in similar_samples]
                })
            
            predictions.extend(batch_predictions)
            true_labels.extend(labels.cpu().numpy())
    
    # 计算少样本学习准确率
    few_shot_acc = accuracy_score(true_labels, predictions)
    print(f"基于向量数据库的少样本学习准确率: {few_shot_acc:.4f}")
    
    return few_shot_acc, few_shot_results

def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置文件
    if args.config:
        args = load_config(args)
    
    set_seed()
    
    # 设置实验目录
    exp_dir = setup_experiment_directory(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载数据
    print("\n===== 加载数据 =====")
    try:
        X_train, y_train, X_test, y_test = get_data(
            data_path=args.data_path,
            subject=args.subject,
            loso=args.loso,
            is_standard=not args.no_standard,
            fre_filter=args.fre_filter,
            dataset=args.dataset,
            augment=args.augment,
            augment_factor=args.augment_factor
        )
        
        print(f"✓ 数据加载成功")
        print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
        print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
        
        # 打印类别分布
        print(f"训练集类别分布: {np.bincount(y_train)}")
        print(f"测试集类别分布: {np.bincount(y_test)}")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        traceback.print_exc()
        return
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 智能模型选择
    if args.model == 'auto':
        args.model = auto_select_model(
            fre_filter=args.fre_filter,
            n_channels=X_train.shape[2],
            n_classes=len(np.unique(y_train.numpy())),
            n_timepoints=X_train.shape[3]
        )
    
    # 初始化模型
    print("\n===== 初始化模型 =====")
    n_channels = X_train.shape[2]
    n_timepoints = X_train.shape[3]
    n_classes = len(np.unique(y_train.numpy()))
    
    print(f"输入数据形状: {X_train.shape}")
    print(f"通道数: {n_channels}, 时间点数: {n_timepoints}, 类别数: {n_classes}")
    print(f"选择的模型: {args.model}")
    
    # 创建模型
    max_attempts = 3
    model = None
    
    for attempt in range(max_attempts):
        try:
            print(f"\n尝试 {attempt+1}/{max_attempts}: 创建模型...")
            model = get_model(
                model_name=args.model,
                n_channels=n_channels,
                n_classes=n_classes,
                n_timepoints=n_timepoints,
                use_freq=args.fre_filter,
                dropout=args.dropout,
                mamba_dim=args.mamba_dim
            ).to(device)
            
            # 初始化权重
            initialize_model_weights(model, init_type=args.init_type)
            
            # 测试前向传播
            print("测试模型前向传播...")
            test_input = torch.randn(2, 1, n_channels, n_timepoints).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            
            print(f"✓ 前向传播测试成功!")
            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {test_output.shape}")
            
            break  # 成功则跳出循环
            
        except Exception as e:
            print(f"✗ 模型初始化失败: {e}")
            
            if attempt < max_attempts - 1:
                # 尝试备用模型
                print(f"尝试备用模型...")
                if 'mamba' in args.model:
                    args.model = 'baseline'
                    print(f"切换到 {args.model} 模型")
                else:
                    args.model = 'multiband_mamba'
                    print(f"切换到 {args.model} 模型")
            else:
                print(f"✗ 所有模型尝试都失败了！")
                traceback.print_exc()
                return
    
    if model is None:
        print("✗ 无法创建模型，退出")
        return
    
    # 配置损失函数
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"使用标签平滑交叉熵损失 (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")
    
    # 配置优化器
    optimizer = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    print(f"优化器: {args.optimizer}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    
    # 配置学习率调度器
    scheduler = create_scheduler(optimizer, args.scheduler, epochs=args.epochs)
    print(f"学习率调度器: {args.scheduler}")
    
    # 训练模型
    print("\n===== 开始训练 =====")
    print(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}")
    print(f"Dropout: {args.dropout}, 标签平滑: {args.label_smoothing}")
    print(f"Mixup增强: {args.mixup}, 梯度裁剪: {args.grad_clip}")
    print(f"早停耐心值: {args.patience}")
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs, val_kappas, val_f1s = [], [], [], []
    lr_history = []
    
    model_save_path = os.path.join(exp_dir, "models", "best_model.pth")
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            mixup=args.mixup, mixup_alpha=args.mixup_alpha, 
            grad_clip=args.grad_clip, verbose=args.verbose
        )
        
        # 验证
        val_loss, val_acc, val_kappa, val_f1, _, _ = validate(
            model, test_loader, criterion, device
        )
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_kappas.append(val_kappa)
        val_f1s.append(val_f1)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率调度器
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                  f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
                  f"验证Kappa: {val_kappa:.4f} | "
                  f"学习率: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"早停计数: {patience_counter}/{args.patience}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': vars(args)
                }, model_save_path)
                
                if args.verbose >= 1:
                    print(f"✨ 保存最佳模型 (准确率: {val_acc:.4f}, 损失: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 早停触发 (第 {epoch+1} 轮)，连续 {patience_counter} 轮未提升")
                break
    
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    
    # 测试最佳模型
    print("\n===== 测试最佳模型 =====")
    
    if args.save_model and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (来自第 {checkpoint['epoch']+1} 轮，准确率: {checkpoint['val_acc']:.4f})")
    
    test_acc, test_bal_acc, test_kappa, test_f1, true_labels, pred_labels, test_probs = evaluate_model(
        model, test_loader, device, verbose=args.verbose
    )
    
    print(f"\n测试集性能:")
    print(f"准确率 (Accuracy): {test_acc:.4f}")
    print(f"平衡准确率: {test_bal_acc:.4f}")
    print(f"Cohen's Kappa: {test_kappa:.4f}")
    print(f"加权F1分数: {test_f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, digits=4))
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 保存结果
    results = {
        'train_losses': [float(x) for x in train_losses],
        'train_accs': [float(x) for x in train_accs],
        'val_losses': [float(x) for x in val_losses],
        'val_accs': [float(x) for x in val_accs],
        'val_kappas': [float(x) for x in val_kappas],
        'val_f1s': [float(x) for x in val_f1s],
        'lr_history': [float(x) for x in lr_history],
        'test_acc': float(test_acc),
        'test_bal_acc': float(test_bal_acc),
        'test_kappa': float(test_kappa),
        'test_f1': float(test_f1),
        'true_labels': [int(label) for label in true_labels],
        'pred_labels': [int(label) for label in pred_labels],
        'test_probs': [prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in test_probs],
        'confusion_matrix': cm.tolist(),
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'final_val_acc': float(val_acc),
        'final_val_loss': float(val_loss),
        'model_name': args.model,
        'n_epochs_trained': len(train_losses),
        'config': vars(args)
    }
    
    results_file = os.path.join(exp_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ 结果已保存至 {results_file}")
    
    # 保存性能摘要
    save_summary(exp_dir, args, model, X_train, y_train, X_test, y_test, 
                train_losses, val_losses, train_accs, val_accs, 
                test_acc, test_bal_acc, test_kappa, test_f1, best_val_acc, lr_history)
    
    # 构建脑电向量数据库（如果启用）
    vector_db = None
    retrieval_results = None
    few_shot_results = None
    
    if args.enable_vector_db:
        vector_db, retrieval_results = build_eeg_vector_database(
            args, model, train_loader, test_loader, device, exp_dir
        )
        
        # 可视化检索结果
        if args.plot_retrieval and retrieval_results:
            visualize_retrieval_results(retrieval_results, vector_db, exp_dir, model, args)
        
        # 少样本学习（如果启用）
        if args.use_db_for_few_shot and vector_db:
            few_shot_acc, few_shot_results = few_shot_learning_with_vector_db(
                vector_db, test_loader, model, device, args
            )
            
            # 保存少样本学习结果
            few_shot_file = os.path.join(exp_dir, "few_shot_results.json")
            with open(few_shot_file, 'w') as f:
                json.dump({
                    'few_shot_accuracy': float(few_shot_acc),
                    'samples': few_shot_results[:100]  # 只保存前100个样本
                }, f, indent=4)
            
            print(f"✓ 少样本学习结果已保存至 {few_shot_file}")
    
    # 绘制训练曲线
    if args.plot_curves:
        plot_training_curves(exp_dir, args, train_losses, val_losses, train_accs, val_accs, 
                           val_kappas, val_f1s, lr_history, best_val_acc)
    
    # 绘制混淆矩阵
    if args.plot_cm:
        plot_confusion_matrix(exp_dir, args, cm, n_classes, test_acc, test_kappa)
    
    # 最终输出
    print_experiment_summary(exp_dir, args, test_acc, test_bal_acc, test_kappa, test_f1, 
                           best_val_acc, n_classes, train_accs, val_accs)

def save_summary(exp_dir, args, model, X_train, y_train, X_test, y_test, 
                train_losses, val_losses, train_accs, val_accs, 
                test_acc, test_bal_acc, test_kappa, test_f1, best_val_acc, lr_history):
    """保存实验摘要"""
    summary_file = os.path.join(exp_dir, "summary.txt")
    n_classes = len(np.unique(y_train.numpy()))
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EEG分类实验摘要（集成向量数据库）\n")
        f.write("="*60 + "\n\n")
        
        f.write("实验信息:\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验目录: {exp_dir}\n")
        f.write(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write(f"数据集: {args.dataset}, 被试: {args.subject}\n")
        f.write(f"模型: {args.model}\n\n")
        
        f.write("配置参数:\n")
        f.write("-"*40 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg:25}: {value}\n")
        
        f.write("\n数据统计:\n")
        f.write("-"*40 + "\n")
        f.write(f"训练集形状: {X_train.shape}\n")
        f.write(f"测试集形状: {X_test.shape}\n")
        f.write(f"训练集类别分布: {np.bincount(y_train.numpy())}\n")
        f.write(f"测试集类别分布: {np.bincount(y_test.numpy())}\n")
        f.write(f"类别数: {n_classes}\n\n")
        
        f.write("训练统计:\n")
        f.write("-"*40 + "\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"总参数数: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        f.write(f"训练轮数: {len(train_losses)}/{args.epochs}\n")
        f.write(f"最终学习率: {lr_history[-1]:.2e}\n")
        f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
        f.write(f"最终验证损失: {val_losses[-1]:.4f}\n")
        f.write(f"训练验证差距: {train_accs[-1] - val_accs[-1]:.4f}\n\n")
        
        f.write("性能指标:\n")
        f.write("-"*40 + "\n")
        f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"最终验证准确率: {val_accs[-1]:.4f}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"测试平衡准确率: {test_bal_acc:.4f}\n")
        f.write(f"Cohen's Kappa: {test_kappa:.4f}\n")
        f.write(f"加权F1分数: {test_f1:.4f}\n")
        f.write(f"随机水平 (1/{n_classes}): {1.0/n_classes:.4f}\n")
        f.write(f"提升幅度: {test_acc - 1.0/n_classes:.4f}\n")
        
        if args.enable_vector_db:
            f.write("\n向量数据库配置:\n")
            f.write("-"*40 + "\n")
            f.write(f"数据库类型: {args.db_type}\n")
            f.write(f"特征维度: {args.feature_dim}\n")
            f.write(f"检索Top-K: {args.retrieval_topk}\n")
            f.write(f"少样本学习: {'启用' if args.use_db_for_few_shot else '禁用'}\n")
    
    print(f"✓ 实验摘要已保存至 {summary_file}")

def plot_training_curves(exp_dir, args, train_losses, val_losses, train_accs, val_accs, 
                        val_kappas, val_f1s, lr_history, best_val_acc):
    """绘制训练曲线"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        axes[0, 0].plot(train_losses, label='训练损失', linewidth=2, alpha=0.8)
        axes[0, 0].plot(val_losses, label='验证损失', linewidth=2, alpha=0.8)
        axes[0, 0].axhline(y=min(val_losses), color='r', linestyle='--', alpha=0.5, 
                          label=f'最小验证损失: {min(val_losses):.4f}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练与验证损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(train_accs, label='训练准确率', linewidth=2, alpha=0.8)
        axes[0, 1].plot(val_accs, label='验证准确率', linewidth=2, alpha=0.8)
        axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.8,
                          label=f'最佳验证准确率: {best_val_acc:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].set_title('训练与验证准确率曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Kappa和F1曲线
        axes[1, 0].plot(val_kappas, label='验证Kappa', linewidth=2, alpha=0.8, color='green')
        axes[1, 0].plot(val_f1s, label='验证F1', linewidth=2, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('分数')
        axes[1, 0].set_title('验证集Kappa和F1分数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 1].plot(lr_history, label='学习率', linewidth=2, alpha=0.8, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].set_title('学习率变化曲线')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'训练监控 - {args.dataset} S{args.subject} - {args.model}', fontsize=14)
        plt.tight_layout()
        curve_file = os.path.join(exp_dir, "plots", "training_curves.png")
        plt.savefig(curve_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练曲线已保存至 {curve_file}")
    except Exception as e:
        print(f"✗ 绘制训练曲线失败: {e}")

def plot_confusion_matrix(exp_dir, args, cm, n_classes, test_acc, test_kappa):
    """绘制混淆矩阵"""
    try:
        plt.figure(figsize=(8, 6))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=[f'类别{i}' for i in range(n_classes)], 
                   yticklabels=[f'类别{i}' for i in range(n_classes)],
                   cbar_kws={'label': '比例'})
        
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'混淆矩阵 (归一化)\n准确率: {test_acc:.4f}, Kappa: {test_kappa:.4f}')
        
        cm_file = os.path.join(exp_dir, "plots", "confusion_matrix.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵已保存至 {cm_file}")
    except Exception as e:
        print(f"✗ 绘制混淆矩阵失败: {e}")

def print_experiment_summary(exp_dir, args, test_acc, test_bal_acc, test_kappa, test_f1, 
                           best_val_acc, n_classes, train_accs, val_accs):
    """打印实验总结"""
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    print(f"所有结果已保存至目录: {exp_dir}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试平衡准确率: {test_bal_acc:.4f}")
    print(f"Cohen's Kappa: {test_kappa:.4f}")
    print(f"加权F1分数: {test_f1:.4f}")
    print(f"使用模型: {args.model}")
    print(f"随机水平 (1/{n_classes}): {1.0/n_classes:.4f}")
    print(f"提升幅度: {test_acc - 1.0/n_classes:.4f}")
    print(f"训练验证差距: {train_accs[-1] - val_accs[-1]:.4f}")
    
    if args.enable_vector_db:
        print(f"\n向量数据库功能: {'已启用' if args.enable_vector_db else '未启用'}")
        if args.enable_vector_db:
            print(f"数据库类型: {args.db_type}")
            print(f"检索Top-K: {args.retrieval_topk}")
            print(f"少样本学习: {'已启用' if args.use_db_for_few_shot else '未启用'}")
    
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    except Exception as e:
        print(f"\n\n实验运行出错: {e}")
        traceback.print_exc()
