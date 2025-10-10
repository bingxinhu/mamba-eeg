import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# 导入预处理模块
from preprocess import get_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 绘制学习曲线
def draw_learning_curves(history, model_name, results_path):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/{model_name}_learning_curves.png", dpi=300)
    plt.show()
    plt.close()

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, epochs, patience, model_name):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    counter = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"{model_name} - Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, history, best_val_acc

# 测试函数
def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cf_matrix = confusion_matrix(all_labels, all_preds, normalize='pred')
    
    return acc, kappa, cf_matrix, all_labels, all_preds

# 主运行函数
def run():
    # 配置
    dataset_path = "./dataset/2a/"
    results_path = "./results_mamba_style"
    os.makedirs(results_path, exist_ok=True)
    
    # 超参数
    batch_size = 32
    epochs = 500
    patience = 180
    lr = 0.001
    n_subjects = 1
    
    # 结果存储
    original_results = {'acc': 0, 'kappa': 0}
    mamba_results = {'acc': 0, 'kappa': 0}
    
    for sub in range(n_subjects):
        print(f"\n{'='*50}")
        print(f"Testing on subject {sub + 1}")
        print(f"{'='*50}")
        
        # 获取数据
        X_train, y_train, X_test, y_test = get_data(
            dataset_path, sub, loso=False, is_standard=True, fre_filter=False, dataset='BCI2a'
        )
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)
        
        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 测试原始模型
        print("\n1. Testing Original GC_Block Model...")
        from models import EEG_DBNet
        model_original = EEG_DBNet(nb_classes=4, Chans=22, Samples=1125).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_original = optim.Adam(model_original.parameters(), lr=lr, weight_decay=1e-4)
        
        model_original_trained, history_original, best_acc_original = train(
            model_original, train_loader, test_loader, criterion, optimizer_original, 
            epochs, patience, "Original"
        )
        
        acc_original, kappa_original, cf_original, _, _ = test(model_original_trained, test_loader)
        original_results['acc'] = acc_original
        original_results['kappa'] = kappa_original
        
        # 测试Mamba风格模型
        print("\n2. Testing Mamba Style GC_Block Model...")
        from models import EEG_DBNet_MambaStyle
        model_mamba = EEG_DBNet_MambaStyle(nb_classes=4, Chans=22, Samples=1125).to(device)
        optimizer_mamba = optim.Adam(model_mamba.parameters(), lr=lr, weight_decay=1e-4)
        
        model_mamba_trained, history_mamba, best_acc_mamba = train(
            model_mamba, train_loader, test_loader, criterion, optimizer_mamba,
            epochs, patience, "MambaStyle"
        )
        
        acc_mamba, kappa_mamba, cf_mamba, _, _ = test(model_mamba_trained, test_loader)
        mamba_results['acc'] = acc_mamba
        mamba_results['kappa'] = kappa_mamba
        
        # 绘制学习曲线
        draw_learning_curves(history_original, f"Original_Subject_{sub+1}", results_path)
        draw_learning_curves(history_mamba, f"MambaStyle_Subject_{sub+1}", results_path)
        
        # 打印对比结果
        print(f"\n{'='*50}")
        print(f"Subject {sub+1} Comparison Results:")
        print(f"{'='*50}")
        print(f"Original GC_Block - Acc: {acc_original:.4f}, Kappa: {kappa_original:.4f}")
        print(f"Mamba Style     - Acc: {acc_mamba:.4f}, Kappa: {kappa_mamba:.4f}")
        print(f"Improvement     - Acc: {acc_mamba-acc_original:+.4f}, Kappa: {kappa_mamba-kappa_original:+.4f}")
        
        # 保存模型
        torch.save(model_original_trained.state_dict(), 
                  f"{results_path}/original_subject_{sub+1}.pth")
        torch.save(model_mamba_trained.state_dict(), 
                  f"{results_path}/mamba_style_subject_{sub+1}.pth")
    
    # 最终性能对比
    print(f"\n{'='*60}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"Original GC_Block - Acc: {original_results['acc']:.4f}, Kappa: {original_results['kappa']:.4f}")
    print(f"Mamba Style     - Acc: {mamba_results['acc']:.4f}, Kappa: {mamba_results['kappa']:.4f}")
    print(f"Improvement     - Acc: {mamba_results['acc']-original_results['acc']:+.4f}, Kappa: {mamba_results['kappa']-original_results['kappa']:+.4f}")
    
    # 保存结果
    results_summary = {
        'original_acc': original_results['acc'],
        'mamba_acc': mamba_results['acc'],
        'original_kappa': original_results['kappa'],
        'mamba_kappa': mamba_results['kappa'],
        'improvement_acc': mamba_results['acc'] - original_results['acc'],
        'improvement_kappa': mamba_results['kappa'] - original_results['kappa']
    }
    
    np.savez(f"{results_path}/comparison_results.npz", **results_summary)
    
    # 写入日志文件
    with open(f"{results_path}/results_summary.txt", "w") as f:
        f.write("Mamba Style vs Original GC_Block Comparison Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Original - Acc: {original_results['acc']:.4f}, Kappa: {original_results['kappa']:.4f}\n")
        f.write(f"Mamba Style - Acc: {mamba_results['acc']:.4f}, Kappa: {mamba_results['kappa']:.4f}\n")
        f.write(f"Improvement - Acc: {results_summary['improvement_acc']:+.4f}, Kappa: {results_summary['improvement_kappa']:+.4f}\n")

if __name__ == "__main__":
    run()
