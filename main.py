import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from preprocess import get_data
from models import get_model

# 设置随机种子确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='EEG信号分类主程序（含Mamba模型支持）')
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data', help='数据集根路径')
    parser.add_argument('--subject', type=int, default=0, help='被试编号（0-8，共9个被试）')
    parser.add_argument('--dataset', type=str, default='BCI2a', choices=['BCI2a', 'BCI2b'], help='数据集类型')
    parser.add_argument('--loso', action='store_true', help='是否使用留一法交叉验证')
    parser.add_argument('--no_standard', action='store_true', help='不进行数据标准化')
    parser.add_argument('--fre_filter', action='store_true', help='是否使用多频段滤波')
    parser.add_argument('--augment', action='store_true', help='是否使用数据增强')
    parser.add_argument('--augment_factor', type=int, default=1, help='数据增强倍数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='auto', 
                        choices=['baseline', 'wideband', 'wideband_mamba', 
                                 'stable_mamba', 'simple_mamba', 'simple',
                                 'multiband_mamba', 'singleband_mamba', 'auto'], 
                        help='选择模型，auto表示自动选择')
    parser.add_argument('--mamba_dim', type=int, default=64, help='Mamba模块特征维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减系数')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--mixup', action='store_true', help='是否使用Mixup数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha参数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 保存与日志
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--save_model', action='store_true', help='是否保存最佳模型')
    parser.add_argument('--plot_cm', action='store_true', help='是否绘制混淆矩阵')
    parser.add_argument('--plot_curves', action='store_true', help='是否绘制训练曲线')
    
    return parser.parse_args()

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

def train_epoch(model, train_loader, criterion, optimizer, device, mixup=False, mixup_alpha=0.2, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
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
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
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
    return val_loss, val_acc, all_labels, all_preds

def evaluate_model(model, test_loader, device):
    """全面评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return acc, kappa, f1, all_labels, all_preds

def auto_select_model(fre_filter, n_channels, n_classes, n_timepoints):
    """根据数据特征自动选择最合适的模型"""
    if fre_filter:
        # 多频段数据
        model_name = 'multiband_mamba'
        print(f"检测到多频段数据（{n_channels}通道），自动选择 {model_name} 模型")
    else:
        # 单频段数据
        if n_timepoints > 1000:
            # 长时间序列，使用稳定的Mamba模型
            model_name = 'stable_mamba'
            print(f"检测到长时间序列（{n_timepoints}点），自动选择 {model_name} 模型")
        else:
            # 短时间序列，使用简单Mamba模型
            model_name = 'singleband_mamba'
            print(f"检测到短时间序列（{n_timepoints}点），自动选择 {model_name} 模型")
    
    return model_name

def main():
    args = parse_args()
    set_seed()
    
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, f"best_model_subj{args.subject}.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n===== 加载数据 =====")
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
    
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
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
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # 智能模型选择
    if args.model == 'auto':
        args.model = auto_select_model(
            fre_filter=args.fre_filter,
            n_channels=X_train.shape[2],
            n_classes=len(np.unique(y_train.numpy())),
            n_timepoints=X_train.shape[3]
        )
    elif args.model == 'simple':  # 向后兼容
        args.model = 'simple_mamba'
    
    # 初始化模型
    print("\n===== 初始化模型 =====")
    n_channels = X_train.shape[2]
    n_timepoints = X_train.shape[3]
    n_classes = len(np.unique(y_train.numpy()))
    
    print(f"输入数据形状: {X_train.shape}")
    print(f"通道数: {n_channels}, 时间点数: {n_timepoints}, 类别数: {n_classes}")
    print(f"选择的模型: {args.model}")
    
    # 尝试创建模型
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            model = get_model(
                model_name=args.model,
                n_channels=n_channels,
                n_classes=n_classes,
                n_timepoints=n_timepoints,
                use_freq=args.fre_filter,
                mamba_dim=args.mamba_dim
            ).to(device)
            
            # 测试前向传播
            print(f"\n测试模型前向传播 (尝试 {attempt+1}/{max_attempts})...")
            test_input = torch.randn(2, 1, n_channels, n_timepoints).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            
            print(f"前向传播测试成功!")
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {test_output.shape}")
            
            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"参数统计 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
            
            break  # 成功则跳出循环
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
            
            if attempt < max_attempts - 1:
                # 尝试备用模型
                print(f"尝试备用模型...")
                if 'mamba' in args.model:
                    # 如果Mamba模型失败，尝试简单的baseline模型
                    args.model = 'baseline'
                    print(f"切换到 {args.model} 模型")
                else:
                    # 如果其他模型也失败，尝试更简单的模型
                    args.model = 'singleband_mamba'
                    print(f"切换到 {args.model} 模型")
            else:
                print(f"所有模型尝试都失败了！")
                print(f"最后错误详情:")
                traceback.print_exc()
                return
    
    # 配置训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练模型
    print("\n===== 开始训练 =====")
    print(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"Mixup增强: {args.mixup}, 梯度裁剪: {args.grad_clip}")
    print(f"早停耐心值: {args.patience}")
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            mixup=args.mixup, mixup_alpha=args.mixup_alpha, 
            grad_clip=args.grad_clip
        )
        
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                  f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
                  f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停触发 (第 {epoch+1} 轮)，连续 {patience_counter} 轮未提升")
                break
    
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    
    # 测试最佳模型
    print("\n===== 测试最佳模型 =====")
    
    if args.save_model and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (来自第 {checkpoint['epoch']+1} 轮)")
    
    test_acc, test_kappa, test_f1, true_labels, pred_labels = evaluate_model(model, test_loader, device)
    
    print(f"\n测试集性能:")
    print(f"准确率 (Accuracy): {test_acc:.4f}")
    print(f"Cohen's Kappa: {test_kappa:.4f}")
    print(f"加权F1分数: {test_f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, digits=4))
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 保存结果
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_kappa': test_kappa,
        'test_f1': test_f1,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'confusion_matrix': cm,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'model_name': args.model,
        'config': vars(args)
    }
    
    results_file = os.path.join(args.save_dir, f"results_subj{args.subject}.npy")
    np.save(results_file, results)
    print(f"结果已保存至 {results_file}")
    
    # 保存配置和性能摘要
    summary_file = os.path.join(args.save_dir, f"summary_subj{args.subject}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("EEG分类实验摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write("配置参数:\n")
        f.write("-"*30 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\n数据统计:\n")
        f.write("-"*30 + "\n")
        f.write(f"训练集形状: {X_train.shape}\n")
        f.write(f"测试集形状: {X_test.shape}\n")
        f.write(f"类别数: {n_classes}\n")
        
        f.write("\n性能指标:\n")
        f.write("-"*30 + "\n")
        f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"Cohen's Kappa: {test_kappa:.4f}\n")
        f.write(f"加权F1分数: {test_f1:.4f}\n")
        f.write(f"模型: {args.model}\n")
    
    print(f"实验摘要已保存至 {summary_file}")
    
    # 绘制训练曲线
    if args.plot_curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(train_losses, label='训练损失', linewidth=2)
        axes[0].plot(val_losses, label='验证损失', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练与验证损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(train_accs, label='训练准确率', linewidth=2)
        axes[1].plot(val_accs, label='验证准确率', linewidth=2)
        axes[1].axhline(y=best_val_acc, color='r', linestyle='--', 
                       label=f'最佳验证准确率: {best_val_acc:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('训练与验证准确率曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_file = os.path.join(args.save_dir, f"train_curves_subj{args.subject}.png")
        plt.savefig(curve_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存至 {curve_file}")
    
    # 绘制混淆矩阵
    if args.plot_cm:
        plt.figure(figsize=(8, 6))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=[f'类别{i}' for i in range(n_classes)], 
                   yticklabels=[f'类别{i}' for i in range(n_classes)])
        
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'混淆矩阵 (归一化)\n准确率: {test_acc:.4f}, Kappa: {test_kappa:.4f}')
        
        cm_file = os.path.join(args.save_dir, f"confusion_matrix_subj{args.subject}.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存至 {cm_file}")
    
    # 绘制性能对比条形图
    plt.figure(figsize=(6, 4))
    metrics = ['准确率', 'Kappa', 'F1分数']
    values = [test_acc, test_kappa, test_f1]
    
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)
    
    # 在条形上方添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.ylabel('分数')
    plt.title(f'模型性能对比 - {args.model}')
    plt.grid(axis='y', alpha=0.3)
    
    perf_file = os.path.join(args.save_dir, f"performance_subj{args.subject}.png")
    plt.savefig(perf_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("实验完成!")
    print("="*50)
    print(f"所有结果已保存至目录: {args.save_dir}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"Cohen's Kappa: {test_kappa:.4f}")
    print(f"加权F1分数: {test_f1:.4f}")
    print(f"使用模型: {args.model}")
    print("="*50)

if __name__ == '__main__':
    main()
