import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

class EEGSubjectDiagnosis:
    """脑电分类被试诊断分析系统"""
    
    def __init__(self, results_dir, subject_id):
        self.results_dir = results_dir
        self.subject_id = subject_id
        self.results = None
        self.features = None
        self.labels = None
        self.predictions = None
        
        # 颜色配置
        self.colors = {
            'correct': '#2ecc71',    # 绿色
            'incorrect': '#e74c3c',  # 红色
            'neutral': '#3498db',    # 蓝色
            'background': '#ecf0f1'  # 灰色
        }
        
    def load_results(self):
        """加载被试结果"""
        import json
        import os
        
        results_file = os.path.join(
            self.results_dir, 
            f"BCI2a_S{self.subject_id}_*", 
            "results.json"
        )
        
        # 找到最新的结果文件
        import glob
        result_files = glob.glob(results_file)
        if not result_files:
            raise FileNotFoundError(f"未找到被试 {self.subject_id} 的结果文件")
        
        with open(result_files[0], 'r') as f:
            self.results = json.load(f)
        
        # 提取关键信息
        self.labels = np.array(self.results['true_labels'])
        self.predictions = np.array(self.results['pred_labels'])
        self.confusion_matrix = np.array(self.results['confusion_matrix'])
        
        print(f"加载被试 S{self.subject_id} 结果:")
        print(f"样本数: {len(self.labels)}")
        print(f"准确率: {self.results['test_acc']:.4f}")
        print(f"Kappa系数: {self.results['test_kappa']:.4f}")
        
    def analyze_class_distribution(self, y_true, y_pred):
        """分析类别分布和预测性能"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 类别分布
        unique_labels, counts = np.unique(y_true, return_counts=True)
        axes[0, 0].bar(unique_labels, counts, color='skyblue', alpha=0.8)
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('样本数')
        axes[0, 0].set_title(f'被试 S{self.subject_id} - 类别分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数量标签
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + 0.5, str(count), ha='center', fontsize=10)
        
        # 2. 类别准确率
        class_accuracies = []
        for label in unique_labels:
            mask = y_true == label
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred[mask] == label)
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)
        
        axes[0, 1].bar(unique_labels, class_accuracies, color='lightcoral', alpha=0.8)
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].set_title(f'被试 S{self.subject_id} - 各类别准确率')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加准确率标签
        for i, acc in enumerate(class_accuracies):
            axes[0, 1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)
        
        # 3. 预测置信度分布
        if 'test_probs' in self.results:
            confidences = [max(prob) for prob in self.results['test_probs']]
            axes[1, 0].hist(confidences, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('预测置信度')
            axes[1, 0].set_ylabel('频数')
            axes[1, 0].set_title(f'被试 S{self.subject_id} - 预测置信度分布')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 计算平均置信度
            avg_confidence = np.mean(confidences)
            axes[1, 0].axvline(avg_confidence, color='red', linestyle='--', 
                              label=f'平均置信度: {avg_confidence:.3f}')
            axes[1, 0].legend()
        
        # 4. 错误类型分析
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask
        
        axes[1, 1].pie([np.sum(correct_mask), np.sum(incorrect_mask)], 
                       labels=['正确', '错误'], 
                       colors=[self.colors['correct'], self.colors['incorrect']],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(f'被试 S{self.subject_id} - 正确/错误分布')
        
        plt.tight_layout()
        plt.savefig(f'diagnosis_S{self.subject_id}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_accuracies
    
    def visualize_confusion_matrix_detailed(self):
        """详细的可视化混淆矩阵"""
        cm = self.confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=[f'类{i}' for i in range(cm.shape[0])],
                   yticklabels=[f'类{i}' for i in range(cm.shape[0])],
                   cbar_kws={'label': '样本数'})
        axes[0].set_xlabel('预测标签')
        axes[0].set_ylabel('真实标签')
        axes[0].set_title(f'被试 S{self.subject_id} - 混淆矩阵（原始计数）')
        
        # 归一化混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
                   xticklabels=[f'类{i}' for i in range(cm.shape[0])],
                   yticklabels=[f'类{i}' for i in range(cm.shape[0])],
                   cbar_kws={'label': '比例'})
        axes[1].set_xlabel('预测标签')
        axes[1].set_ylabel('真实标签')
        axes[1].set_title(f'被试 S{self.subject_id} - 混淆矩阵（归一化）')
        
        # 计算每个类别的召回率和精确率
        recall = np.diag(cm) / np.sum(cm, axis=1)
        precision = np.diag(cm) / np.sum(cm, axis=0)
        
        print(f"\n被试 S{self.subject_id} 的类别性能:")
        print("类别 | 召回率 | 精确率 | 样本数")
        print("-" * 40)
        for i in range(len(recall)):
            print(f"类{i}   | {recall[i]:.3f}  | {precision[i]:.3f}  | {np.sum(cm[i, :]):d}")
        
        plt.tight_layout()
        plt.savefig(f'diagnosis_S{self.subject_id}_confusion.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return recall, precision
    
    def visualize_feature_space(self, features, labels, predictions=None):
        """特征空间可视化"""
        if features.shape[1] > 2:
            print("使用t-SNE降维...")
            
            # 首先使用PCA降维到50维
            pca = PCA(n_components=min(50, features.shape[1]))
            features_pca = pca.fit_transform(features)
            
            # 使用t-SNE降维到2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            features_2d = tsne.fit_transform(features_pca)
        else:
            features_2d = features
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 按真实标签着色
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          color=colors[i], alpha=0.6, s=30, label=f'类{label}')
        
        axes[0].set_xlabel('t-SNE维度1')
        axes[0].set_ylabel('t-SNE维度2')
        axes[0].set_title(f'被试 S{self.subject_id} - 特征空间（按真实标签）')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 按预测正确性着色
        if predictions is not None:
            correct_mask = labels == predictions
            
            # 正确分类的点
            axes[1].scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1], 
                          color=self.colors['correct'], alpha=0.6, s=30, label='正确')
            
            # 错误分类的点
            axes[1].scatter(features_2d[~correct_mask, 0], features_2d[~correct_mask, 1], 
                          color=self.colors['incorrect'], alpha=0.6, s=30, label='错误')
            
            # 添加错误分类的标签信息
            for i in range(len(labels)):
                if not correct_mask[i]:
                    axes[1].text(features_2d[i, 0], features_2d[i, 1], 
                               f'{labels[i]}→{predictions[i]}', 
                               fontsize=8, alpha=0.8)
        
        axes[1].set_xlabel('t-SNE维度1')
        axes[1].set_ylabel('t-SNE维度2')
        axes[1].set_title(f'被试 S{self.subject_id} - 特征空间（按预测正确性）')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'diagnosis_S{self.subject_id}_featurespace.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return features_2d
    
    def analyze_signal_quality(self, X, y):
        """分析信号质量（如果原始数据可用）"""
        if X is None:
            print("原始数据不可用，跳过信号质量分析")
            return
        
        n_classes = len(np.unique(y))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 信号幅值分布
        signal_amplitudes = np.abs(X).flatten()
        axes[0, 0].hist(signal_amplitudes, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('信号幅值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title(f'被试 S{self.subject_id} - 信号幅值分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 各类别平均信号
        timepoints = np.arange(X.shape[-1])
        for label in range(n_classes):
            mask = y == label
            if np.sum(mask) > 0:
                avg_signal = np.mean(X[mask], axis=(0, 1, 2))  # 平均所有通道
                axes[0, 1].plot(timepoints, avg_signal, label=f'类{label}', alpha=0.8)
        
        axes[0, 1].set_xlabel('时间点')
        axes[0, 1].set_ylabel('幅值')
        axes[0, 1].set_title(f'被试 S{self.subject_id} - 各类别平均信号')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 信噪比估计（简化）
        signal_power = np.mean(X**2, axis=(0, 1, 2, 3))
        noise_estimate = np.std(X, axis=(0, 1, 2, 3))
        snr_estimate = signal_power / (noise_estimate**2 + 1e-8)
        
        axes[1, 0].bar(['信号功率', '噪声估计', 'SNR估计'], 
                      [signal_power, noise_estimate, snr_estimate],
                      color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title(f'被试 S{self.subject_id} - 信号质量指标')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 频域分析（如果数据足够）
        try:
            from scipy import signal as sp_signal
            
            # 选择一个样本进行频域分析
            sample_idx = 0
            sample_signal = X[sample_idx, 0, 0, :]  # 第一个样本，第一个通道
            
            freqs, psd = sp_signal.welch(sample_signal, fs=250, nperseg=256)
            
            axes[1, 1].plot(freqs, psd, color='purple', alpha=0.8)
            axes[1, 1].set_xlabel('频率 (Hz)')
            axes[1, 1].set_ylabel('功率谱密度')
            axes[1, 1].set_title(f'被试 S{self.subject_id} - 频域分析示例')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 标记关键频段
            bands = {'δ (1-4Hz)': (1, 4), 'θ (4-8Hz)': (4, 8), 
                    'α (8-12Hz)': (8, 12), 'β (12-30Hz)': (12, 30)}
            
            for band_name, (f_low, f_high) in bands.items():
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                if np.any(band_mask):
                    band_power = np.mean(psd[band_mask])
                    axes[1, 1].axvspan(f_low, f_high, alpha=0.1, color='gray')
                    axes[1, 1].text((f_low + f_high)/2, np.max(psd)*0.8, 
                                   band_name, ha='center', fontsize=8)
        except Exception as e:
            print(f"频域分析失败: {e}")
            axes[1, 1].text(0.5, 0.5, '频域分析不可用', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'diagnosis_S{self.subject_id}_signal_quality.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_subjects(self, all_subject_results):
        """比较不同被试的表现"""
        subjects = list(all_subject_results.keys())
        
        metrics = ['acc', 'kappa', 'bal_acc', 'f1']
        metric_names = ['准确率', 'Kappa系数', '平衡准确率', 'F1分数']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [all_subject_results[s][metric] for s in subjects]
            
            bars = axes[idx].bar(range(len(subjects)), values, 
                                color=['red' if s in [1, 4, 5] else 'skyblue' for s in subjects],
                                alpha=0.7, edgecolor='black')
            
            axes[idx].set_xlabel('被试编号')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'各被试{metric_name}对比')
            axes[idx].set_xticks(range(len(subjects)))
            axes[idx].set_xticklabels([f'S{s}' for s in subjects])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
            
            # 添加平均值线
            avg_value = np.mean(values)
            axes[idx].axhline(avg_value, color='red', linestyle='--', alpha=0.7, 
                            label=f'平均: {avg_value:.3f}')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig('diagnosis_subject_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建数据框进行详细分析
        df_comparison = pd.DataFrame(all_subject_results).T
        print("\n各被试详细性能对比:")
        print(df_comparison)
        
        # 识别表现最差的类别
        worst_performing = df_comparison[df_comparison['acc'] < 0.75]
        print(f"\n表现较差被试 (准确率<75%):")
        print(worst_performing)
        
        return df_comparison
    
    def generate_interactive_report(self):
        """生成交互式HTML报告"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f'被试 S{self.subject_id} - 类别分布',
                f'被试 S{self.subject_id} - 各类别准确率',
                f'被试 S{self.subject_id} - 预测置信度',
                f'被试 S{self.subject_id} - 混淆矩阵',
                f'被试 S{self.subject_id} - 特征空间(真实标签)',
                f'被试 S{self.subject_id} - 特征空间(预测正确性)',
                f'被试 S{self.subject_id} - 信号质量',
                f'被试 S{self.subject_id} - 训练曲线',
                f'被试 S{self.subject_id} - 注意力可视化'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'heatmap'}]
            ]
        )
        
        # 这里可以添加具体的绘图代码
        # 由于篇幅限制，这里展示框架
        
        fig.update_layout(
            height=1200,
            width=1600,
            title_text=f"脑电分类诊断报告 - 被试 S{self.subject_id}",
            showlegend=True
        )
        
        html_file = f'diagnosis_report_S{self.subject_id}.html'
        fig.write_html(html_file)
        print(f"交互式报告已保存至: {html_file}")

def run_detailed_diagnosis(subject_ids=[1, 4, 5], results_dir="./results"):
    """运行详细的诊断分析"""
    
    all_results = {}
    
    for subject_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"分析被试 S{subject_id}")
        print('='*60)
        
        # 初始化诊断器
        diagnosis = EEGSubjectDiagnosis(results_dir, subject_id)
        
        try:
            # 1. 加载结果
            diagnosis.load_results()
            
            # 2. 分析类别分布
            class_accuracies = diagnosis.analyze_class_distribution(
                diagnosis.labels, 
                diagnosis.predictions
            )
            
            # 3. 可视化混淆矩阵
            recall, precision = diagnosis.visualize_confusion_matrix_detailed()
            
            # 4. 特征空间可视化（需要特征数据）
            # 如果有特征数据，可以添加这里
            
            # 保存结果用于比较
            all_results[subject_id] = {
                'acc': diagnosis.results['test_acc'],
                'kappa': diagnosis.results['test_kappa'],
                'bal_acc': diagnosis.results['test_bal_acc'],
                'f1': diagnosis.results['test_f1'],
                'class_accuracies': class_accuracies,
                'recall': recall.tolist(),
                'precision': precision.tolist()
            }
            
        except Exception as e:
            print(f"分析被试 S{subject_id} 时出错: {e}")
    
    # 比较所有被试
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("比较不同被试表现")
        print('='*60)
        
        diagnosis.compare_subjects(all_results)
    
    return all_results
