import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 颜色配置
        self.band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.time_colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def extract_attention_weights(self, x):
        """
        提取注意力权重
        
        参数:
        x: 输入数据 (batch, 1, channels, timepoints)
        
        返回:
        attention_dict: 包含所有注意力权重的字典
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 执行前向传播以计算注意力权重
        with torch.no_grad():
            _ = self.model(x)
        
        # 获取注意力权重
        if hasattr(self.model, 'get_attention_weights'):
            attention_dict = self.model.get_attention_weights()
        else:
            # 如果模型没有get_attention_weights方法，尝试其他方式
            attention_dict = self._extract_attention_weights_alternative(x)
        
        return attention_dict
    
    def _extract_attention_weights_alternative(self, x):
        """备用的注意力权重提取方法"""
        attention_dict = {}
        
        # 尝试提取跨频段注意力权重
        if hasattr(self.model, 'cross_band_attention'):
            if hasattr(self.model.cross_band_attention, 'attention_weights'):
                attention_dict['cross_band'] = self.model.cross_band_attention.attention_weights
        
        # 尝试提取时间注意力权重
        if hasattr(self.model, 'temporal_attention_weights'):
            attention_dict['temporal'] = self.model.temporal_attention_weights
        
        return attention_dict
    
    def visualize_cross_band_attention(self, cross_band_weights, save_path=None):
        """
        可视化跨频段注意力权重
        
        参数:
        cross_band_weights: 跨频段注意力权重矩阵 (batch, n_bands, n_bands) 或 (n_bands, n_bands)
        save_path: 保存路径
        """
        if cross_band_weights is None:
            print("跨频段注意力权重为空")
            return None
        
        # 转换为numpy数组
        if isinstance(cross_band_weights, torch.Tensor):
            cross_band_weights = cross_band_weights.cpu().numpy()
        
        # 处理3D输入：如果形状是3D，取第一个样本或平均值
        original_shape = cross_band_weights.shape
        print(f"跨频段注意力权重形状: {original_shape}")
        
        if len(original_shape) == 3:
            # 如果是3D (batch, n_bands, n_bands)，我们可以选择：
            # 1. 取第一个样本
            # 2. 取所有样本的平均
            print(f"检测到3D输入，取所有样本的平均值")
            cross_band_weights = np.mean(cross_band_weights, axis=0)
        
        # 确保现在是2D
        if len(cross_band_weights.shape) != 2:
            print(f"错误：期望2D矩阵，但得到形状 {cross_band_weights.shape}")
            return None
        
        n_bands = cross_band_weights.shape[0]
        band_names = [f'频段{i+1}' for i in range(n_bands)]
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 注意力热力图
        try:
            sns.heatmap(cross_band_weights, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=band_names, yticklabels=band_names,
                       cbar_kws={'label': '注意力权重'}, ax=axes[0, 0])
            axes[0, 0].set_xlabel('目标频段')
            axes[0, 0].set_ylabel('源频段')
            axes[0, 0].set_title('跨频段注意力热力图')
        except Exception as e:
            print(f"绘制热力图时出错: {e}")
            axes[0, 0].text(0.5, 0.5, f'热力图错误: {e}', ha='center', va='center')
            axes[0, 0].axis('off')
        
        # 2. 每个频段的总注意力（行和）
        row_sums = np.sum(cross_band_weights, axis=1)
        axes[0, 1].bar(range(n_bands), row_sums, color=self.band_colors[:n_bands])
        axes[0, 1].set_xlabel('频段')
        axes[0, 1].set_ylabel('总注意力')
        axes[0, 1].set_title('每个频段发出的总注意力')
        axes[0, 1].set_xticks(range(n_bands))
        axes[0, 1].set_xticklabels(band_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(row_sums):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        
        # 3. 每个频段接收的注意力（列和）
        col_sums = np.sum(cross_band_weights, axis=0)
        axes[1, 0].bar(range(n_bands), col_sums, color=self.band_colors[:n_bands])
        axes[1, 0].set_xlabel('频段')
        axes[1, 0].set_ylabel('接收的注意力')
        axes[1, 0].set_title('每个频段接收的总注意力')
        axes[1, 0].set_xticks(range(n_bands))
        axes[1, 0].set_xticklabels(band_names)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(col_sums):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        
        # 4. 注意力矩阵分析
        # 计算对角线元素（频段自注意力）和平均值
        diag_values = np.diag(cross_band_weights)
        avg_diag = np.mean(diag_values)
        avg_off_diag = (np.sum(cross_band_weights) - np.sum(diag_values)) / (n_bands * (n_bands - 1))
        
        metrics_text = f"""
        注意力矩阵统计:
        自注意力平均值: {avg_diag:.4f}
        跨频段注意力平均值: {avg_off_diag:.4f}
        自注意力占比: {avg_diag/(avg_diag+avg_off_diag)*100:.1f}%
        最大注意力值: {np.max(cross_band_weights):.4f}
        最小注意力值: {np.min(cross_band_weights):.4f}
        原始形状: {original_shape}
        处理后形状: {cross_band_weights.shape}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('注意力矩阵统计')
        
        plt.tight_layout()
        plt.suptitle('跨频段注意力分析', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"跨频段注意力可视化已保存至: {save_path}")
        
        plt.show()
        
        return cross_band_weights
    
    def visualize_temporal_attention(self, temporal_weights, n_bands=5, save_path=None):
        """
        可视化时间注意力权重
        
        参数:
        temporal_weights: 时间注意力权重，可以是多种格式
        n_bands: 频段数
        save_path: 保存路径
        """
        if temporal_weights is None:
            print("时间注意力权重为空")
            return None
        
        print(f"时间注意力权重原始类型: {type(temporal_weights)}")
        
        # 处理不同类型的输入
        sample_weights = None
        
        if isinstance(temporal_weights, list):
            print(f"输入是列表，长度: {len(temporal_weights)}")
            if len(temporal_weights) > 0:
                if isinstance(temporal_weights[0], torch.Tensor):
                    print(f"列表元素是Tensor，形状: {temporal_weights[0].shape}")
                    # 如果是多频段的列表，取第一个频段第一个样本
                    if temporal_weights[0].dim() == 3:  # (batch, timepoints, n_heads)
                        sample_weights = temporal_weights[0][0].cpu().numpy()
                    else:  # (timepoints, n_heads)
                        sample_weights = temporal_weights[0].cpu().numpy()
                else:
                    # 假设已经是numpy数组
                    print(f"列表元素不是Tensor，类型: {type(temporal_weights[0])}")
                    sample_weights = temporal_weights[0]
        elif isinstance(temporal_weights, torch.Tensor):
            print(f"输入是Tensor，形状: {temporal_weights.shape}")
            if temporal_weights.dim() == 3:  # (batch, timepoints, n_heads)
                sample_weights = temporal_weights[0].cpu().numpy()
            elif temporal_weights.dim() == 2:  # (timepoints, n_heads)
                sample_weights = temporal_weights.cpu().numpy()
            else:
                print(f"不支持Tensor维度: {temporal_weights.dim()}")
                return None
        else:
            print(f"输入是其他类型，尝试作为numpy数组处理")
            sample_weights = temporal_weights
        
        if sample_weights is None:
            print("无法处理时间注意力权重")
            return None
        
        print(f"处理后样本权重形状: {sample_weights.shape}")
        
        # 确保是2D (timepoints, n_heads)
        if len(sample_weights.shape) != 2:
            print(f"错误：期望2D数组 (timepoints, n_heads)，但得到形状 {sample_weights.shape}")
            return None
        
        timepoints, n_heads = sample_weights.shape
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        
        # 1. 多注意力头热力图
        ax1 = plt.subplot(gs[0, :2])
        try:
            sns.heatmap(sample_weights.T, cmap='viridis', 
                       xticklabels=50, yticklabels=[f'头{i+1}' for i in range(n_heads)],
                       cbar_kws={'label': '注意力权重'}, ax=ax1)
            ax1.set_xlabel('时间点')
            ax1.set_ylabel('注意力头')
            ax1.set_title('时间注意力权重热力图（多注意力头）')
        except Exception as e:
            print(f"绘制热力图时出错: {e}")
            ax1.text(0.5, 0.5, f'热力图错误: {e}', ha='center', va='center')
            ax1.axis('off')
        
        # 2. 注意力头平均权重
        ax2 = plt.subplot(gs[0, 2])
        avg_per_head = np.mean(sample_weights, axis=0)
        ax2.bar(range(n_heads), avg_per_head, color=self.time_colors[:n_heads])
        ax2.set_xlabel('注意力头')
        ax2.set_ylabel('平均权重')
        ax2.set_title('各注意力头平均权重')
        ax2.set_xticks(range(n_heads))
        ax2.set_xticklabels([f'头{i+1}' for i in range(n_heads)])
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(avg_per_head):
            ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
        
        # 3. 时间注意力分布（所有头的平均）
        ax3 = plt.subplot(gs[1, 0])
        avg_over_time = np.mean(sample_weights, axis=1)
        ax3.plot(avg_over_time, linewidth=2, color='blue', alpha=0.8)
        ax3.fill_between(range(timepoints), 0, avg_over_time, alpha=0.3, color='blue')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('平均注意力权重')
        ax3.set_title('时间注意力分布（所有头平均）')
        ax3.grid(True, alpha=0.3)
        
        # 标记最高点和最低点
        max_idx = np.argmax(avg_over_time)
        min_idx = np.argmin(avg_over_time)
        ax3.scatter(max_idx, avg_over_time[max_idx], color='red', s=100, zorder=5, label=f'最高点: {avg_over_time[max_idx]:.3f}')
        ax3.scatter(min_idx, avg_over_time[min_idx], color='green', s=100, zorder=5, label=f'最低点: {avg_over_time[min_idx]:.3f}')
        ax3.legend()
        
        # 4. 关键时间点统计
        ax4 = plt.subplot(gs[1, 1])
        # 找出权重最高的10%时间点
        threshold = np.percentile(avg_over_time, 90)
        high_attention_points = avg_over_time > threshold
        
        # 计算连续高注意力区域
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(high_attention_points)
        
        stats_text = f"""
        时间注意力统计:
        总时间点: {timepoints}
        高注意力时间点 (>90%分位数): {np.sum(high_attention_points)}
        高注意力比例: {np.sum(high_attention_points)/timepoints*100:.1f}%
        连续高注意力区域数: {num_features}
        最大连续区域长度: {max([np.sum(labeled_array == i) for i in range(1, num_features+1)]) if num_features > 0 else 0}
        平均注意力权重: {np.mean(avg_over_time):.4f}
        注意力权重标准差: {np.std(avg_over_time):.4f}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, 
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('时间注意力统计')
        
        # 5. 注意力头相关性
        ax5 = plt.subplot(gs[1, 2])
        # 计算注意力头之间的相关性
        try:
            head_correlation = np.corrcoef(sample_weights.T)
            
            im = ax5.imshow(head_correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax5.set_xlabel('注意力头')
            ax5.set_ylabel('注意力头')
            ax5.set_title('注意力头间相关性')
            ax5.set_xticks(range(n_heads))
            ax5.set_yticks(range(n_heads))
            ax5.set_xticklabels([f'H{i+1}' for i in range(n_heads)])
            ax5.set_yticklabels([f'H{i+1}' for i in range(n_heads)])
            
            # 添加相关性数值
            for i in range(n_heads):
                for j in range(n_heads):
                    ax5.text(j, i, f'{head_correlation[i, j]:.2f}', 
                            ha='center', va='center', color='white' if abs(head_correlation[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax5)
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            ax5.text(0.5, 0.5, f'相关性计算错误: {e}', ha='center', va='center')
            ax5.axis('off')
        
        plt.tight_layout()
        plt.suptitle('时间注意力分析', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间注意力可视化已保存至: {save_path}")
        
        plt.show()
        
        return sample_weights
    
    def visualize_multi_band_temporal_attention(self, temporal_weights_list, save_path=None):
        """
        可视化多频段时间注意力权重
        
        参数:
        temporal_weights_list: 时间注意力权重列表，每个元素对应一个频段
        save_path: 保存路径
        """
        if not temporal_weights_list:
            print("时间注意力权重列表为空")
            return None
        
        print(f"多频段权重列表长度: {len(temporal_weights_list)}")
        
        n_bands = len(temporal_weights_list)
        
        # 提取每个频段的注意力权重（第一个样本，第一个注意力头）
        band_attentions = []
        band_names = []
        
        for i, weights in enumerate(temporal_weights_list):
            print(f"频段 {i+1} 权重类型: {type(weights)}")
            
            if isinstance(weights, torch.Tensor):
                print(f"  Tensor形状: {weights.shape}")
                # 取第一个样本的第一个注意力头
                if weights.dim() == 3:  # (batch, timepoints, n_heads)
                    band_attention = weights[0, :, 0].cpu().numpy()
                elif weights.dim() == 2:  # (timepoints, n_heads)
                    band_attention = weights[:, 0].cpu().numpy()
                else:
                    print(f"  不支持Tensor维度: {weights.dim()}")
                    continue
            elif isinstance(weights, np.ndarray):
                print(f"  NumPy数组形状: {weights.shape}")
                if len(weights.shape) == 2:  # (timepoints, n_heads)
                    band_attention = weights[:, 0]
                elif len(weights.shape) == 1:  # (timepoints,)
                    band_attention = weights
                else:
                    print(f"  不支持数组维度: {weights.shape}")
                    continue
            else:
                print(f"  不支持的类型: {type(weights)}")
                continue
            
            band_attentions.append(band_attention)
            band_names.append(f'频段{i+1}')
        
        if not band_attentions:
            print("无法提取任何频段的注意力权重")
            return None
        
        # 确保所有频段有相同的时间点数
        timepoints = len(band_attentions[0])
        for i, att in enumerate(band_attentions):
            if len(att) != timepoints:
                print(f"警告: 频段 {i} 的时间点数 {len(att)} 与频段 0 的 {timepoints} 不同")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 多频段时间注意力曲线
        ax1 = axes[0, 0]
        time_axis = np.arange(timepoints)
        
        for i in range(n_bands):
            if i < len(band_attentions):
                ax1.plot(time_axis, band_attentions[i], 
                        color=self.band_colors[i], 
                        linewidth=2, 
                        alpha=0.7, 
                        label=band_names[i])
        
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('注意力权重')
        ax1.set_title('多频段时间注意力对比')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 各频段平均注意力
        ax2 = axes[0, 1]
        band_means = [np.mean(att) for att in band_attentions]
        band_stds = [np.std(att) for att in band_attentions]
        
        bars = ax2.bar(range(n_bands), band_means, 
                      yerr=band_stds,
                      color=self.band_colors[:n_bands],
                      alpha=0.7,
                      capsize=5)
        
        ax2.set_xlabel('频段')
        ax2.set_ylabel('平均注意力权重')
        ax2.set_title('各频段平均注意力')
        ax2.set_xticks(range(n_bands))
        ax2.set_xticklabels(band_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(band_means, band_stds)):
            ax2.text(i, mean + 0.005, f'{mean:.3f}±{std:.3f}', 
                    ha='center', fontsize=9)
        
        # 3. 注意力热力图（频段×时间）
        ax3 = axes[0, 2]
        try:
            attention_matrix = np.vstack(band_attentions)  # (n_bands, timepoints)
            
            im = ax3.imshow(attention_matrix, cmap='YlOrRd', aspect='auto',
                           extent=[0, timepoints, 0, n_bands])
            ax3.set_xlabel('时间点')
            ax3.set_ylabel('频段')
            ax3.set_title('频段-时间注意力热力图')
            ax3.set_yticks(np.arange(n_bands) + 0.5)
            ax3.set_yticklabels(band_names)
            plt.colorbar(im, ax=ax3, label='注意力权重')
        except Exception as e:
            print(f"创建热力图时出错: {e}")
            ax3.text(0.5, 0.5, f'热力图错误: {e}', ha='center', va='center')
            ax3.axis('off')
        
        # 4. 注意力相关性矩阵（频段间）
        ax4 = axes[1, 0]
        try:
            # 计算频段间的注意力相关性
            band_correlation = np.corrcoef(attention_matrix)
            
            im2 = ax4.imshow(band_correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax4.set_xlabel('频段')
            ax4.set_ylabel('频段')
            ax4.set_title('频段间注意力相关性')
            ax4.set_xticks(range(n_bands))
            ax4.set_yticks(range(n_bands))
            ax4.set_xticklabels([f'B{i+1}' for i in range(n_bands)])
            ax4.set_yticklabels([f'B{i+1}' for i in range(n_bands)])
            
            # 添加相关性数值
            for i in range(n_bands):
                for j in range(n_bands):
                    ax4.text(j, i, f'{band_correlation[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if abs(band_correlation[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im2, ax=ax4)
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            ax4.text(0.5, 0.5, f'相关性计算错误: {e}', ha='center', va='center')
            ax4.axis('off')
        
        # 5. 注意力峰值分布
        ax5 = axes[1, 1]
        peak_indices = [np.argmax(att) for att in band_attentions]
        peak_values = [np.max(att) for att in band_attentions]
        
        colors = self.band_colors[:n_bands]
        for i in range(n_bands):
            if i < len(band_attentions):
                ax5.scatter(peak_indices[i], peak_values[i], 
                           color=colors[i], s=150, alpha=0.8, label=band_names[i])
                # 添加连接线
                ax5.plot([peak_indices[i], peak_indices[i]], [0, peak_values[i]], 
                        color=colors[i], linestyle='--', alpha=0.5)
        
        ax5.set_xlabel('峰值时间点')
        ax5.set_ylabel('峰值注意力权重')
        ax5.set_title('各频段注意力峰值')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 注意力分布统计
        ax6 = axes[1, 2]
        
        stats_text = "各频段注意力统计:\n\n"
        for i in range(n_bands):
            if i < len(band_attentions):
                stats_text += f"{band_names[i]}:\n"
                stats_text += f"  均值: {band_means[i]:.4f}\n"
                stats_text += f"  标准差: {band_stds[i]:.4f}\n"
                stats_text += f"  峰值: {peak_values[i]:.4f}\n"
                stats_text += f"  峰值位置: {peak_indices[i]}\n"
                if len(band_attentions[i]) > 1:
                    from scipy import stats
                    skewness = stats.skew(band_attentions[i])
                    stats_text += f"  偏度: {skewness:.4f}\n\n"
        
        ax6.text(0.1, 0.5, stats_text, fontsize=9, 
                verticalalignment='center', transform=ax6.transAxes)
        ax6.axis('off')
        ax6.set_title('注意力分布统计')
        
        plt.tight_layout()
        plt.suptitle('多频段时间注意力综合分析', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多频段时间注意力可视化已保存至: {save_path}")
        
        plt.show()
        
        return attention_matrix if 'attention_matrix' in locals() else None
    
    def create_interactive_attention_visualization(self, cross_band_weights, temporal_weights, save_path=None):
        """
        创建交互式注意力可视化
        
        参数:
        cross_band_weights: 跨频段注意力权重
        temporal_weights: 时间注意力权重
        save_path: 保存路径
        """
        # 处理跨频段注意力权重
        if isinstance(cross_band_weights, torch.Tensor):
            cross_band_weights = cross_band_weights.cpu().numpy()
        
        # 处理3D输入
        if len(cross_band_weights.shape) == 3:
            cross_band_weights = np.mean(cross_band_weights, axis=0)
        
        # 处理时间注意力权重
        temporal_sample = None
        
        if isinstance(temporal_weights, list):
            if temporal_weights and isinstance(temporal_weights[0], torch.Tensor):
                if temporal_weights[0].dim() == 3:
                    temporal_sample = temporal_weights[0][0].cpu().numpy()
                else:
                    temporal_sample = temporal_weights[0].cpu().numpy()
        elif isinstance(temporal_weights, torch.Tensor):
            if temporal_weights.dim() == 3:
                temporal_sample = temporal_weights[0].cpu().numpy()
            else:
                temporal_sample = temporal_weights.cpu().numpy()
        
        if temporal_sample is None:
            print("无法处理时间注意力权重")
            return None
        
        # 确保是2D
        if len(temporal_sample.shape) != 2:
            print(f"时间注意力权重需要2D，但得到形状 {temporal_sample.shape}")
            return None
        
        timepoints, n_heads = temporal_sample.shape
        n_bands = cross_band_weights.shape[0]
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('跨频段注意力热力图', '时间注意力热力图', '注意力头分布',
                           '频段注意力统计', '时间注意力分布', '注意力相关性'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'bar'}],
                  [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. 跨频段注意力热力图
        fig.add_trace(
            go.Heatmap(
                z=cross_band_weights,
                x=[f'频段{i+1}' for i in range(n_bands)],
                y=[f'频段{i+1}' for i in range(n_bands)],
                colorscale='YlOrRd',
                colorbar=dict(title='权重')
            ),
            row=1, col=1
        )
        
        # 2. 时间注意力热力图
        fig.add_trace(
            go.Heatmap(
                z=temporal_sample.T,
                x=list(range(timepoints)),
                y=[f'头{i+1}' for i in range(n_heads)],
                colorscale='viridis',
                colorbar=dict(title='权重')
            ),
            row=1, col=2
        )
        
        # 3. 注意力头平均权重
        head_means = np.mean(temporal_sample, axis=0)
        fig.add_trace(
            go.Bar(
                x=[f'头{i+1}' for i in range(n_heads)],
                y=head_means,
                marker_color=px.colors.qualitative.Set3[:n_heads]
            ),
            row=1, col=3
        )
        
        # 4. 频段注意力统计（跨频段注意力的行和）
        band_sums = np.sum(cross_band_weights, axis=1)
        fig.add_trace(
            go.Bar(
                x=[f'频段{i+1}' for i in range(n_bands)],
                y=band_sums,
                marker_color=self.band_colors[:n_bands]
            ),
            row=2, col=1
        )
        
        # 5. 时间注意力分布（所有头平均）
        time_avg = np.mean(temporal_sample, axis=1)
        fig.add_trace(
            go.Scatter(
                x=list(range(timepoints)),
                y=time_avg,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # 6. 注意力头相关性
        try:
            head_correlation = np.corrcoef(temporal_sample.T)
            fig.add_trace(
                go.Heatmap(
                    z=head_correlation,
                    x=[f'头{i+1}' for i in range(n_heads)],
                    y=[f'头{i+1}' for i in range(n_heads)],
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    colorbar=dict(title='相关性')
                ),
                row=2, col=3
            )
        except:
            pass
        
        # 更新布局
        fig.update_layout(
            height=800,
            width=1200,
            title_text="注意力权重交互式可视化",
            showlegend=False
        )
        
        # 保存
        if save_path:
            fig.write_html(save_path)
            print(f"交互式注意力可视化已保存至: {save_path}")
        
        return fig

def run_attention_visualization(subject_id, model_path, data_path, save_dir="./attention_viz"):
    """
    运行完整的注意力可视化流程
    
    参数:
    subject_id: 被试ID
    model_path: 模型路径
    data_path: 数据路径
    save_dir: 保存目录
    """
    import os
    import torch
    from preprocess import get_data
    from models import get_model
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    try:
        X_train, y_train, X_test, y_test = get_data(
            data_path=data_path,
            subject=subject_id,
            loso=False,
            is_standard=True,
            fre_filter=True,
            dataset="BCI2a",
            augment=False
        )
    except Exception as e:
        print(f"加载数据失败: {e}")
        # 使用模拟数据
        print("使用模拟数据...")
        X_test = np.random.randn(5, 1, 110, 1125).astype(np.float32)  # 5个样本，110通道，1125时间点
        y_test = np.random.randint(0, 4, 5)
    
    # 转换数据为张量
    X_test_tensor = torch.FloatTensor(X_test[:3])  # 只取前3个样本
    print(f"测试数据形状: {X_test_tensor.shape}")
    
    # 加载模型
    print("加载模型...")
    n_channels = X_test.shape[2]
    n_timepoints = X_test.shape[3]
    n_classes = 4  # BCI2a有4个类别
    
    try:
        model = get_model(
            model_name="Interpretable_mamba",
            n_channels=n_channels,
            n_classes=n_classes,
            n_timepoints=n_timepoints,
            use_freq=True,
            dropout=0.3,
            mamba_dim=128,
            n_attention_heads=8
        ).to(device)
        
        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型权重已从 {model_path} 加载")
        else:
            print("警告: 未找到预训练模型，使用随机初始化模型")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用模拟模型...")
        # 创建模拟模型
        model = type('MockModel', (), {
            'get_attention_weights': lambda self: {
                'cross_band': np.random.rand(5, 5, 5),  # 3D模拟数据
                'temporal': [np.random.rand(1125, 8) for _ in range(5)]  # 5个频段
            },
            'eval': lambda self: None,
            'to': lambda self, device: self
        })()
    
    # 初始化可视化器
    visualizer = AttentionVisualizer(model, device)
    
    # 提取注意力权重
    print("提取注意力权重...")
    try:
        attention_dict = visualizer.extract_attention_weights(X_test_tensor)
        print(f"提取的注意力字典键: {list(attention_dict.keys())}")
    except Exception as e:
        print(f"提取注意力权重失败: {e}")
        print("使用模拟注意力权重...")
        attention_dict = {
            'cross_band': np.random.rand(3, 5, 5),  # 3个样本，5个频段
            'temporal': [np.random.rand(1125, 8) for _ in range(5)]  # 5个频段
        }
    
    # 可视化跨频段注意力
    if 'cross_band' in attention_dict:
        print("可视化跨频段注意力...")
        cross_band_path = os.path.join(save_dir, f"cross_band_attention_S{subject_id}.png")
        visualizer.visualize_cross_band_attention(
            attention_dict['cross_band'],
            save_path=cross_band_path
        )
    
    # 可视化时间注意力
    if 'temporal' in attention_dict:
        print("可视化时间注意力...")
        
        # 先尝试多频段时间注意力
        if isinstance(attention_dict['temporal'], list):
            print("检测到多频段时间注意力列表")
            multi_band_path = os.path.join(save_dir, f"multi_band_temporal_S{subject_id}.png")
            visualizer.visualize_multi_band_temporal_attention(
                attention_dict['temporal'],
                save_path=multi_band_path
            )
        else:
            # 单频段时间注意力
            temporal_path = os.path.join(save_dir, f"temporal_attention_S{subject_id}.png")
            visualizer.visualize_temporal_attention(
                attention_dict['temporal'],
                save_path=temporal_path
            )
    
    # 创建交互式可视化
    if 'cross_band' in attention_dict and 'temporal' in attention_dict:
        print("创建交互式可视化...")
        interactive_path = os.path.join(save_dir, f"interactive_attention_S{subject_id}.html")
        visualizer.create_interactive_attention_visualization(
            attention_dict['cross_band'],
            attention_dict['temporal'],
            save_path=interactive_path
        )
    
    print(f"\n所有可视化结果已保存到: {save_dir}")
    return attention_dict

# 简化的测试函数
def simple_attention_test():
    """简单的注意力测试，不依赖真实数据和模型"""
    print("运行简单注意力测试...")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟跨频段注意力权重 (3个样本，5个频段)
    cross_band_weights = np.random.rand(3, 5, 5)
    
    # 模拟时间注意力权重 (5个频段，每个1125时间点，8个注意力头)
    temporal_weights = [np.random.rand(1125, 8) for _ in range(5)]
    
    # 创建可视化器（使用空模型）
    class MockModel:
        def __init__(self):
            self.cross_band_attention = type('MockAttention', (), {
                'attention_weights': cross_band_weights
            })()
            self.temporal_attention_weights = temporal_weights
        
        def eval(self):
            pass
    
    device = torch.device('cpu')
    visualizer = AttentionVisualizer(MockModel(), device)
    
    # 保存目录
    save_dir = "./test_attention_viz"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化跨频段注意力
    print("可视化跨频段注意力...")
    visualizer.visualize_cross_band_attention(
        cross_band_weights,
        save_path=os.path.join(save_dir, "test_cross_band.png")
    )
    
    # 可视化多频段时间注意力
    print("可视化多频段时间注意力...")
    visualizer.visualize_multi_band_temporal_attention(
        temporal_weights,
        save_path=os.path.join(save_dir, "test_multi_band_temporal.png")
    )
    
    print(f"测试完成，结果保存到: {save_dir}")

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='注意力权重可视化')
    parser.add_argument('--subject', type=int, default=1, help='被试ID')
    parser.add_argument('--model_path', type=str, default='', help='模型路径')
    parser.add_argument('--data_path', type=str, default='./dataset/2a', help='数据路径')
    parser.add_argument('--save_dir', type=str, default='./attention_viz', help='保存目录')
    parser.add_argument('--test', action='store_true', help='运行简单测试而不依赖真实数据')
    
    args = parser.parse_args()
    
    if args.test:
        simple_attention_test()
    else:
        # 运行可视化
        attention_dict = run_attention_visualization(
            subject_id=args.subject,
            model_path=args.model_path,
            data_path=args.data_path,
            save_dir=args.save_dir
        )
