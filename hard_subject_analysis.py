import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class HardSubjectInvestigator:
    """困难被试深度调查器"""
    
    def __init__(self, subject_id, X_train, y_train, X_test, y_test, model, device):
        self.subject_id = subject_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.device = device
        
        # 训练状态跟踪
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def analyze_data_quality(self):
        """分析数据质量"""
        print(f"\n分析被试 S{self.subject_id} 的数据质量:")
        
        # 1. 检查数据分布
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
        
        # 2. 检查类别平衡性
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        
        print(f"\n训练集类别分布: {train_counts}")
        print(f"测试集类别分布: {test_counts}")
        
        # 计算不平衡比率
        train_imbalance = np.max(train_counts) / np.min(train_counts)
        test_imbalance = np.max(test_counts) / np.min(test_counts)
        
        print(f"训练集不平衡比率: {train_imbalance:.2f}")
        print(f"测试集不平衡比率: {test_imbalance:.2f}")
        
        # 3. 检查数据统计特性
        train_mean = np.mean(self.X_train)
        train_std = np.std(self.X_train)
        test_mean = np.mean(self.X_test)
        test_std = np.std(self.X_test)
        
        print(f"\n训练集均值: {train_mean:.4f}, 标准差: {train_std:.4f}")
        print(f"测试集均值: {test_mean:.4f}, 标准差: {test_std:.4f}")
        
        # 4. 检查是否存在异常值
        train_q1 = np.percentile(self.X_train, 25)
        train_q3 = np.percentile(self.X_train, 75)
        train_iqr = train_q3 - train_q1
        train_outliers = np.sum((self.X_train < train_q1 - 1.5*train_iqr) | 
                               (self.X_train > train_q3 + 1.5*train_iqr))
        
        print(f"\n训练集异常值比例: {train_outliers/self.X_train.size*100:.2f}%")
        
        # 可视化数据质量
        self.visualize_data_quality()
        
        return {
            'train_counts': train_counts.tolist(),
            'test_counts': test_counts.tolist(),
            'train_imbalance': float(train_imbalance),
            'test_imbalance': float(test_imbalance),
            'train_stats': {'mean': float(train_mean), 'std': float(train_std)},
            'test_stats': {'mean': float(test_mean), 'std': float(test_std)}
        }
    
    def visualize_data_quality(self):
        """可视化数据质量"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 类别分布
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        
        x = np.arange(len(train_counts))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_counts, width, label='训练集', alpha=0.7)
        axes[0, 0].bar(x + width/2, test_counts, width, label='测试集', alpha=0.7)
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('样本数')
        axes[0, 0].set_title(f'被试 S{self.subject_id} - 类别分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 数据分布直方图
        axes[0, 1].hist(self.X_train.flatten(), bins=50, alpha=0.5, label='训练集', density=True)
        axes[0, 1].hist(self.X_test.flatten(), bins=50, alpha=0.5, label='测试集', density=True)
        axes[0, 1].set_xlabel('信号值')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].set_title(f'被试 S{self.subject_id} - 数据分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 各通道平均值
        if len(self.X_train.shape) == 4:  # (batch, 1, channels, time)
            n_channels = self.X_train.shape[2]
            train_channel_means = np.mean(self.X_train, axis=(0, 1, 3)).flatten()
            test_channel_means = np.mean(self.X_test, axis=(0, 1, 3)).flatten()
            
            axes[0, 2].bar(np.arange(n_channels) - 0.2, train_channel_means, 0.4, 
                          label='训练集', alpha=0.7)
            axes[0, 2].bar(np.arange(n_channels) + 0.2, test_channel_means, 0.4, 
                          label='测试集', alpha=0.7)
            axes[0, 2].set_xlabel('通道')
            axes[0, 2].set_ylabel('平均值')
            axes[0, 2].set_title(f'被试 S{self.subject_id} - 各通道平均值')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 信噪比估计
        # 计算每个样本的信噪比（简化版本）
        if len(self.X_train.shape) == 4:
            # 假设信号是低频部分，噪声是高频部分
            from scipy import signal
            
            # 使用Butterworth滤波器分离信号和噪声
            b, a = signal.butter(4, 0.1, 'low')
            
            # 计算一个样本的信噪比作为示例
            sample_idx = 0
            sample_signal = self.X_train[sample_idx, 0, 0, :]
            
            # 滤波得到信号
            signal_low = signal.filtfilt(b, a, sample_signal)
            noise = sample_signal - signal_low
            
            signal_power = np.mean(signal_low**2)
            noise_power = np.mean(noise**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            axes[1, 0].plot(sample_signal, label='原始信号', alpha=0.7)
            axes[1, 0].plot(signal_low, label='低频信号', alpha=0.7)
            axes[1, 0].set_xlabel('时间点')
            axes[1, 0].set_ylabel('幅值')
            axes[1, 0].set_title(f'信号分解示例 (SNR: {snr:.2f} dB)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 相关性分析
        # 计算不同类别样本之间的相关性
        n_classes = len(np.unique(self.y_train))
        
        for label in range(n_classes):
            mask = self.y_train == label
            if np.sum(mask) > 5:  # 至少有5个样本
                class_samples = self.X_train[mask]
                # 计算类内平均相关性
                if len(class_samples) > 1:
                    # 展平样本
                    flattened = class_samples.reshape(len(class_samples), -1)
                    corr_matrix = np.corrcoef(flattened)
                    avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                    
                    axes[1, 1].scatter(label, avg_correlation, s=100, alpha=0.7)
        
        axes[1, 1].set_xlabel('类别')
        axes[1, 1].set_ylabel('类内平均相关性')
        axes[1, 1].set_title(f'被试 S{self.subject_id} - 类内相关性')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 时域特征
        if len(self.X_train.shape) == 4:
            time_features = []
            for label in range(n_classes):
                mask = self.y_train == label
                if np.sum(mask) > 0:
                    class_samples = self.X_train[mask]
                    # 计算能量特征
                    energy = np.mean(class_samples**2, axis=(1, 2, 3))
                    time_features.append(energy)
            
            # 箱线图显示各类别的能量分布
            axes[1, 2].boxplot(time_features, labels=[f'类{i}' for i in range(len(time_features))])
            axes[1, 2].set_xlabel('类别')
            axes[1, 2].set_ylabel('信号能量')
            axes[1, 2].set_title(f'被试 S{self.subject_id} - 各类别能量分布')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'hard_subject_S{self.subject_id}_data_quality.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_model_behavior(self):
        """分析模型在困难被试上的行为"""
        print(f"\n分析模型在被试 S{self.subject_id} 上的行为:")
        
        self.model.eval()
        
        # 创建数据加载器
        test_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.LongTensor(self.y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 收集预测结果
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_features = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                # 获取模型输出
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # 获取特征（如果模型支持）
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(inputs, feature_type='raw')
                    all_features.append(features.cpu().numpy())
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        all_probabilities = np.concatenate(all_probabilities)
        
        if all_features:
            all_features = np.vstack(all_features)
        
        # 分析错误类型
        correct_mask = all_predictions == all_labels
        error_mask = ~correct_mask
        
        print(f"总样本数: {len(all_labels)}")
        print(f"正确分类: {np.sum(correct_mask)} ({np.mean(correct_mask)*100:.1f}%)")
        print(f"错误分类: {np.sum(error_mask)} ({np.mean(error_mask)*100:.1f}%)")
        
        # 分析错误样本的特征
        if len(all_features) > 0:
            error_features = all_features[error_mask]
            correct_features = all_features[correct_mask]
            
            # 计算特征统计
            error_mean = np.mean(error_features, axis=0)
            error_std = np.std(error_features, axis=0)
            correct_mean = np.mean(correct_features, axis=0)
            correct_std = np.std(correct_features, axis=0)
            
            # 计算特征差异
            feature_diff = np.abs(error_mean - correct_mean)
            top_diff_indices = np.argsort(feature_diff)[-10:]  # 差异最大的10个特征
            
            print(f"\n错误样本与正确样本的特征差异 (Top 10):")
            for idx in top_diff_indices[::-1]:
                print(f"特征 {idx}: 错误={error_mean[idx]:.4f}±{error_std[idx]:.4f}, "
                      f"正确={correct_mean[idx]:.4f}±{correct_std[idx]:.4f}")
        
        # 分析置信度
        error_confidences = np.max(all_probabilities[error_mask], axis=1)
        correct_confidences = np.max(all_probabilities[correct_mask], axis=1)
        
        print(f"\n置信度分析:")
        print(f"错误样本平均置信度: {np.mean(error_confidences):.4f}")
        print(f"正确样本平均置信度: {np.mean(correct_confidences):.4f}")
        print(f"置信度差异: {np.mean(correct_confidences) - np.mean(error_confidences):.4f}")
        
        # 可视化模型行为
        self.visualize_model_behavior(all_predictions, all_labels, all_probabilities)
        
        return {
            'accuracy': float(np.mean(correct_mask)),
            'error_rate': float(np.mean(error_mask)),
            'error_confidence': float(np.mean(error_confidences)),
            'correct_confidence': float(np.mean(correct_confidences))
        }
    
    def visualize_model_behavior(self, predictions, labels, probabilities):
        """可视化模型行为"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 置信度分布对比
        correct_mask = predictions == labels
        error_mask = ~correct_mask
        
        axes[0, 0].hist(probabilities[correct_mask].max(axis=1), 
                       bins=30, alpha=0.5, label='正确', density=True)
        axes[0, 0].hist(probabilities[error_mask].max(axis=1), 
                       bins=30, alpha=0.5, label='错误', density=True)
        axes[0, 0].set_xlabel('置信度')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].set_title(f'被试 S{self.subject_id} - 置信度分布对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 各类别置信度
        n_classes = len(np.unique(labels))
        class_confidences = []
        
        for label in range(n_classes):
            mask = labels == label
            if np.sum(mask) > 0:
                class_conf = probabilities[mask].max(axis=1)
                class_confidences.append(class_conf)
        
        axes[0, 1].boxplot(class_confidences, labels=[f'类{i}' for i in range(n_classes)])
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].set_title(f'被试 S{self.subject_id} - 各类别置信度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 错误类型热力图
        error_types = np.zeros((n_classes, n_classes))
        for true_label in range(n_classes):
            for pred_label in range(n_classes):
                if true_label != pred_label:
                    mask = (labels == true_label) & (predictions == pred_label)
                    error_types[true_label, pred_label] = np.sum(mask)
        
        # 归一化
        error_types_norm = error_types / error_types.sum(axis=1, keepdims=True)
        
        im = axes[0, 2].imshow(error_types_norm, cmap='hot', aspect='auto')
        axes[0, 2].set_xlabel('预测类别')
        axes[0, 2].set_ylabel('真实类别')
        axes[0, 2].set_title(f'被试 S{self.subject_id} - 错误类型热力图')
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. 决策边界可视化（如果特征维度可降维）
        # 这里需要特征数据
        
        # 5. 训练历史分析（如果有）
        if hasattr(self, 'train_history'):
            axes[1, 0].plot(self.train_history['train_loss'], label='训练损失')
            axes[1, 0].plot(self.train_history['val_loss'], label='验证损失')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('损失')
            axes[1, 0].set_title(f'被试 S{self.subject_id} - 训练历史')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(self.train_history['train_acc'], label='训练准确率')
            axes[1, 1].plot(self.train_history['val_acc'], label='验证准确率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('准确率')
            axes[1, 1].set_title(f'被试 S{self.subject_id} - 准确率历史')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'hard_subject_S{self.subject_id}_model_behavior.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def investigate_hard_subjects(hard_subject_ids=[1, 4, 5], data_path="./dataset/2a"):
    """调查困难被试"""
    from preprocess import get_data
    from models import get_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    investigation_results = {}
    
    for subject_id in hard_subject_ids:
        print(f"\n{'='*60}")
        print(f"深入调查困难被试 S{subject_id}")
        print('='*60)
        
        try:
            # 加载数据
            X_train, y_train, X_test, y_test = get_data(
                data_path=data_path,
                subject=subject_id,
                loso=False,
                is_standard=True,
                fre_filter=True,
                dataset="BCI2a",
                augment=False
            )
            
            # 初始化模型
            n_channels = X_train.shape[2]
            n_timepoints = X_train.shape[3]
            n_classes = len(np.unique(y_train))
            
            model = get_model(
                model_name="Interpretable_mamba",
                n_channels=n_channels,
                n_classes=n_classes,
                n_timepoints=n_timepoints,
                use_freq=True,
                dropout=0.3,
                mamba_dim=128
            ).to(device)
            
            # 加载预训练权重（如果有）
            # model.load_state_dict(torch.load(...))
            
            # 初始化调查器
            investigator = HardSubjectInvestigator(
                subject_id=subject_id,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model=model,
                device=device
            )
            
            # 1. 分析数据质量
            data_quality = investigator.analyze_data_quality()
            
            # 2. 分析模型行为（需要训练模型或加载预训练模型）
            model_behavior = investigator.analyze_model_behavior()
            
            investigation_results[subject_id] = {
                'data_quality': data_quality,
                'model_behavior': model_behavior
            }
            
        except Exception as e:
            print(f"调查被试 S{subject_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成综合报告
    print(f"\n{'='*60}")
    print("困难被试调查综合报告")
    print('='*60)
    
    for subject_id in hard_subject_ids:
        if subject_id in investigation_results:
            print(f"\n被试 S{subject_id}:")
            print(f"  数据不平衡比率: {investigation_results[subject_id]['data_quality']['train_imbalance']:.2f}")
            print(f"  模型准确率: {investigation_results[subject_id]['model_behavior']['accuracy']:.4f}")
            print(f"  错误样本置信度: {investigation_results[subject_id]['model_behavior']['error_confidence']:.4f}")
    
    return investigation_results
