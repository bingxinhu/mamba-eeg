import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # 模型参数
    parser.add_argument('--model', type=str, default='wideband_mamba', 
                        choices=['baseline', 'wideband', 'wideband_mamba'], help='选择模型')
    parser.add_argument('--mamba_dim', type=int, default=64, help='Mamba模块特征维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减系数')
    parser.add_argument('--patience', type=int, default=500, help='早停耐心值')
    
    # 保存与日志
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--save_model', action='store_true', help='是否保存最佳模型')
    parser.add_argument('--plot_cm', action='store_true', help='是否绘制混淆矩阵')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
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
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 记录
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc, all_labels, all_preds

def main():
    args = parse_args()
    set_seed()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, f"best_model_subj{args.subject}.pth")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载并预处理数据
    print("\n===== 加载数据 =====")
    X_train, y_train, X_test, y_test = get_data(
        data_path=args.data_path,
        subject=args.subject,
        loso=args.loso,
        is_standard=not args.no_standard,
        fre_filter=args.fre_filter,
        dataset=args.dataset
    )
    
    # 转换为PyTorch张量
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
    
    # 2. 初始化模型
    print("\n===== 初始化模型 =====")
    n_channels = X_train.shape[2]
    n_timepoints = X_train.shape[3]
    n_classes = len(np.unique(y_train.numpy()))
    
    model = get_model(
        model_name=args.model,
        n_channels=n_channels,
        n_classes=n_classes,
        n_timepoints=n_timepoints,
        use_freq=args.fre_filter,
        mamba_dim=args.mamba_dim
    ).to(device)
    
    print(f"模型: {args.model} | 输入形状: {X_train.shape[1:]} | 输出类别: {n_classes}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. 配置训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5
    )
    
    # 4. 训练模型
    print("\n===== 开始训练 =====")
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion, device)
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印信息
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 早停与最佳模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if args.save_model:
                torch.save(model.state_dict(), model_save_path)
                print(f"📌 保存最佳模型 (Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏸️ 早停触发 (第 {epoch+1} 轮)")
                break
    
    # 5. 测试最佳模型
    print("\n===== 测试最佳模型 =====")
    if args.save_model and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_acc, true_labels, pred_labels = validate(model, test_loader, criterion, device)
    print(f"测试集性能: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, digits=4))
    
    # 6. 保存结果与可视化
    # 保存指标
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'true_labels': true_labels,
        'pred_labels': pred_labels
    }
    np.save(os.path.join(args.save_dir, f"results_subj{args.subject}.npy"), results)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练与验证损失曲线')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('训练与验证准确率曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"train_curve_subj{args.subject}.png"))
    plt.close()
    
    # 绘制混淆矩阵
    if args.plot_cm:
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(n_classes), yticklabels=range(n_classes))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'混淆矩阵 (Acc: {test_acc:.4f})')
        plt.savefig(os.path.join(args.save_dir, f"confusion_matrix_subj{args.subject}.png"))
        plt.close()
    
    print(f"\n所有结果已保存至 {args.save_dir}")

if __name__ == '__main__':
    main()
