import numpy as np
import torch
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
from collections import defaultdict
import warnings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("Faiss not available. Install with: pip install faiss-cpu (or faiss-gpu)")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    warnings.warn("Annoy not available. Install with: pip install annoy")

class BaseVectorIndex:
    """向量索引基类"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.is_built = False
    
    def build(self, vectors: np.ndarray):
        """构建索引"""
        raise NotImplementedError
    
    def query(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """查询相似向量"""
        raise NotImplementedError
    
    def save(self, filepath: str):
        """保存索引到文件"""
        raise NotImplementedError
    
    def load(self, filepath: str):
        """从文件加载索引"""
        raise NotImplementedError

class FaissIndex(BaseVectorIndex):
    """Faiss向量索引"""
    
    def __init__(self, dimension: int, use_gpu: bool = False, index_type: str = "flat"):
        super().__init__(dimension)
        self.use_gpu = use_gpu
        self.index_type = index_type
        
    def build(self, vectors: np.ndarray):
        """构建Faiss索引"""
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed")
        
        vectors = vectors.astype(np.float32)
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            nlist = min(100, int(np.sqrt(len(vectors))))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(vectors)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        if self.use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(vectors)
        self.is_built = True
    
    def query(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """查询相似向量"""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]
    
    def save(self, filepath: str):
        """保存Faiss索引"""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        if self.use_gpu and torch.cuda.is_available():
            # 如果是GPU索引，先转换到CPU
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, filepath)
        else:
            faiss.write_index(self.index, filepath)
    
    def load(self, filepath: str):
        """加载Faiss索引"""
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed")
        
        self.index = faiss.read_index(filepath)
        self.is_built = True

class AnnoyIndexWrapper(BaseVectorIndex):
    """Annoy向量索引包装器"""
    
    def __init__(self, dimension: int, n_trees: int = 10, metric: str = "euclidean"):
        super().__init__(dimension)
        self.n_trees = n_trees
        self.metric = metric
        
    def build(self, vectors: np.ndarray):
        """构建Annoy索引"""
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy is not installed")
        
        self.index = AnnoyIndex(self.dimension, self.metric)
        
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        
        self.index.build(self.n_trees)
        self.is_built = True
    
    def query(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """查询相似向量"""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        indices, distances = self.index.get_nns_by_vector(
            query_vector, k, include_distances=True
        )
        return np.array(distances), np.array(indices)
    
    def save(self, filepath: str):
        """保存Annoy索引"""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        self.index.save(filepath)
    
    def load(self, filepath: str):
        """加载Annoy索引"""
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy is not installed")
        
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.index.load(filepath)
        self.is_built = True

class SimpleVectorIndex(BaseVectorIndex):
    """简单的暴力搜索向量索引（用于小规模数据）"""
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.vectors = None
    
    def build(self, vectors: np.ndarray):
        """构建简单索引"""
        self.vectors = vectors.astype(np.float32)
        self.is_built = True
    
    def query(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """暴力搜索相似向量"""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # 计算所有距离
        distances = np.linalg.norm(self.vectors - query_vector, axis=1)
        
        # 获取最近的k个
        if k >= len(distances):
            indices = np.arange(len(distances))
            sorted_distances = distances
        else:
            indices = np.argpartition(distances, k)[:k]
            sorted_indices = indices[np.argsort(distances[indices])]
            indices = sorted_indices
            sorted_distances = distances[sorted_indices]
        
        return sorted_distances, indices
    
    def save(self, filepath: str):
        """保存简单索引"""
        if self.vectors is None:
            raise RuntimeError("No vectors to save")
        
        np.save(filepath, self.vectors)
    
    def load(self, filepath: str):
        """加载简单索引"""
        self.vectors = np.load(filepath)
        self.is_built = True

class EEGVectorDatabase:
    """
    脑电信号向量数据库
    
    功能：
    1. 存储脑电特征向量
    2. 快速相似性检索
    3. 支持多种索引类型
    4. 元数据管理
    5. 支持增删改查操作
    """
    
    def __init__(self, 
                 db_type: str = "faiss",
                 dimension: int = 128,
                 use_gpu: bool = False,
                 index_params: Optional[Dict] = None):
        """
        初始化向量数据库
        
        参数:
        db_type: 数据库类型，可选 'faiss', 'annoy', 'simple'
        dimension: 特征向量维度
        use_gpu: 是否使用GPU加速（仅Faiss有效）
        index_params: 索引特定参数
        """
        self.db_type = db_type
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index_params = index_params or {}
        
        # 数据存储
        self.vectors = []  # 特征向量
        self.labels = []   # 类别标签
        self.metadata = [] # 元数据
        self.feature_stats = {}  # 特征统计信息
        
        # 索引
        self.index = None
        self._initialize_index()
        
        # 统计信息
        self.stats = {
            'n_samples': 0,
            'n_classes': 0,
            'dimension': dimension,
            'created_time': datetime.now().isoformat(),
            'last_updated': None
        }
    
    def _initialize_index(self):
        """初始化向量索引"""
        original_db_type = self.db_type
        
        if self.db_type == "faiss":
            if not FAISS_AVAILABLE:
                warnings.warn("Faiss not available, falling back to simple index")
                self.db_type = "simple"
                self.index = SimpleVectorIndex(self.dimension)
            else:
                index_type = self.index_params.get('index_type', 'flat')
                self.index = FaissIndex(self.dimension, self.use_gpu, index_type)
        
        elif self.db_type == "annoy":
            if not ANNOY_AVAILABLE:
                warnings.warn("Annoy not available, falling back to simple index")
                self.db_type = "simple"
                self.index = SimpleVectorIndex(self.dimension)
            else:
                n_trees = self.index_params.get('n_trees', 10)
                metric = self.index_params.get('metric', 'euclidean')
                self.index = AnnoyIndexWrapper(self.dimension, n_trees, metric)
        
        elif self.db_type == "simple":
            self.index = SimpleVectorIndex(self.dimension)
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        # 如果数据库类型被更改，记录日志
        if original_db_type != self.db_type:
            print(f"警告: 数据库类型从 {original_db_type} 更改为 {self.db_type}")
    
    def add_samples(self, 
                   vectors: np.ndarray,
                   labels: np.ndarray,
                   metadata: Optional[List[Dict]] = None):
        """
        添加样本到数据库
        
        参数:
        vectors: 特征向量数组 (n_samples, dimension)
        labels: 标签数组 (n_samples,)
        metadata: 元数据列表，每个元素是一个字典
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")
        
        n_samples = vectors.shape[0]
        
        # 添加到存储
        self.vectors.append(vectors)
        self.labels.extend(labels.tolist())
        
        # 添加元数据
        if metadata is None:
            metadata = [{} for _ in range(n_samples)]
        self.metadata.extend(metadata)
        
        # 更新统计信息
        self.stats['n_samples'] += n_samples
        self.stats['n_classes'] = len(set(self.labels))
        self.stats['last_updated'] = datetime.now().isoformat()
        
        print(f"添加了 {n_samples} 个样本到数据库，当前总数: {self.stats['n_samples']}")
    
    def build_index(self, 
                   vectors: Optional[np.ndarray] = None,
                   labels: Optional[np.ndarray] = None,
                   metadata: Optional[List[Dict]] = None):
        """
        构建向量索引
        
        参数:
        vectors: 特征向量，如果为None则使用已存储的向量
        labels: 标签，如果为None则使用已存储的标签
        metadata: 元数据，如果为None则使用已存储的元数据
        """
        # 如果提供了新数据，先添加到存储
        if vectors is not None:
            if labels is None:
                raise ValueError("Labels must be provided with vectors")
            
            self.add_samples(vectors, labels, metadata)
        
        # 合并所有向量
        if len(self.vectors) == 0:
            raise RuntimeError("No vectors to build index from")
        
        all_vectors = np.vstack(self.vectors)
        
        # 计算特征统计
        self._compute_feature_stats(all_vectors)
        
        # 构建索引
        print(f"构建 {self.db_type} 索引，样本数: {all_vectors.shape[0]}, 维度: {all_vectors.shape[1]}")
        
        # 确保索引对象已初始化
        if self.index is None:
            self._initialize_index()
            if self.index is None:
                raise RuntimeError("Failed to initialize vector index")
        
        self.index.build(all_vectors)
        
        print(f"✓ 索引构建完成")
    
    def _compute_feature_stats(self, vectors: np.ndarray):
        """计算特征统计信息"""
        self.feature_stats = {
            'mean': np.mean(vectors, axis=0),
            'std': np.std(vectors, axis=0),
            'min': np.min(vectors, axis=0),
            'max': np.max(vectors, axis=0),
            'norm_mean': np.mean(np.linalg.norm(vectors, axis=1)),
            'norm_std': np.std(np.linalg.norm(vectors, axis=1))
        }
    
    def query_similar(self, 
                     query_vector: np.ndarray, 
                     k: int = 5,
                     include_metadata: bool = True) -> List[Dict]:
        """
        查询相似样本
        
        参数:
        query_vector: 查询向量 (dimension,)
        k: 返回的最相似样本数
        include_metadata: 是否包含元数据
        
        返回:
        相似样本列表，每个元素是包含距离、相似度、标签和元数据的字典
        """
        if not self.index.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # 查询相似向量
        distances, indices = self.index.query(query_vector, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices):
            result = {
                'index': int(idx),
                'distance': float(distances[i]),
                'similarity': float(1.0 / (1.0 + distances[i])),  # 转换为相似度
                'label': int(self.labels[idx])
            }
            
            if include_metadata and idx < len(self.metadata):
                result['metadata'] = self.metadata[idx]
            
            results.append(result)
        
        return results
    
    def query_by_label(self, 
                      label: int, 
                      n_samples: int = 10,
                      include_metadata: bool = True) -> List[Dict]:
        """
        按标签查询样本
        
        参数:
        label: 目标标签
        n_samples: 返回的样本数
        include_metadata: 是否包含元数据
        
        返回:
        指定标签的样本列表
        """
        # 找到所有指定标签的样本索引
        label_indices = [i for i, l in enumerate(self.labels) if l == label]
        
        if not label_indices:
            return []
        
        # 随机选择样本（或按某种排序）
        if len(label_indices) > n_samples:
            selected_indices = np.random.choice(label_indices, n_samples, replace=False)
        else:
            selected_indices = label_indices
        
        # 构建结果
        results = []
        for idx in selected_indices:
            result = {
                'index': int(idx),
                'label': int(label)
            }
            
            if include_metadata and idx < len(self.metadata):
                result['metadata'] = self.metadata[idx]
            
            results.append(result)
        
        return results
    
    def get_feature_space_analysis(self) -> Dict:
        """
        分析特征空间
        
        返回:
        特征空间分析结果
        """
        if len(self.vectors) == 0:
            return {}
        
        all_vectors = np.vstack(self.vectors)
        
        # 计算类别中心
        unique_labels = np.unique(self.labels)
        class_centers = {}
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(self.labels) if l == label]
            class_vectors = all_vectors[label_indices]
            class_centers[label] = np.mean(class_vectors, axis=0)
        
        # 计算类间距离
        inter_class_distances = {}
        labels_list = list(unique_labels)
        
        for i, label1 in enumerate(labels_list):
            for label2 in labels_list[i+1:]:
                dist = np.linalg.norm(class_centers[label1] - class_centers[label2])
                inter_class_distances[f"{label1}_{label2}"] = float(dist)
        
        # 计算类内距离
        intra_class_distances = {}
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(self.labels) if l == label]
            class_vectors = all_vectors[label_indices]
            
            if len(class_vectors) > 1:
                center = class_centers[label]
                distances = np.linalg.norm(class_vectors - center, axis=1)
                intra_class_distances[label] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'max': float(np.max(distances)),
                    'min': float(np.min(distances))
                }
        
        return {
            'n_classes': len(unique_labels),
            'class_centers': {k: v.tolist() for k, v in class_centers.items()},
            'inter_class_distances': inter_class_distances,
            'intra_class_distances': intra_class_distances,
            'feature_stats': self.feature_stats
        }
    
    def cluster_analysis(self, n_clusters: int = 4) -> Dict:
        """
        聚类分析
        
        参数:
        n_clusters: 聚类数
        
        返回:
        聚类分析结果
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            all_vectors = np.vstack(self.vectors)
            
            # 应用K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_vectors)
            
            # 计算轮廓系数
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(all_vectors, cluster_labels)
            else:
                silhouette = -1
            
            # 分析每个聚类的标签分布
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                # 获取该聚类中的标签分布
                cluster_sample_labels = [self.labels[i] for i in cluster_indices]
                label_counts = {}
                for label in cluster_sample_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                cluster_analysis[cluster_id] = {
                    'n_samples': len(cluster_indices),
                    'label_distribution': label_counts,
                    'dominant_label': max(label_counts, key=label_counts.get) if label_counts else None,
                    'purity': max(label_counts.values()) / len(cluster_indices) if label_counts else 0
                }
            
            return {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'silhouette_score': float(silhouette),
                'cluster_analysis': cluster_analysis,
                'inertia': float(kmeans.inertia_)
            }
        
        except ImportError:
            warnings.warn("scikit-learn not installed for clustering analysis")
            return {}
    
    def anomaly_detection(self, 
                         query_vector: np.ndarray,
                         threshold: float = 2.0) -> Dict:
        """
        异常检测
        
        参数:
        query_vector: 查询向量
        threshold: 异常阈值（基于标准差）
        
        返回:
        异常检测结果
        """
        if not self.index.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # 查询最近的样本
        similar_samples = self.query_similar(query_vector, k=5)
        
        if not similar_samples:
            return {'is_anomaly': False, 'reason': 'No similar samples found'}
        
        # 计算平均距离
        avg_distance = np.mean([s['distance'] for s in similar_samples])
        
        # 获取特征统计
        avg_norm = self.feature_stats.get('norm_mean', 1.0)
        std_norm = self.feature_stats.get('norm_std', 0.1)
        
        # 判断是否为异常
        normalized_distance = avg_distance / (avg_norm + 1e-8)
        is_anomaly = normalized_distance > threshold
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'avg_distance': float(avg_distance),
            'normalized_distance': float(normalized_distance),
            'threshold': float(threshold),
            'similar_samples': similar_samples[:3]  # 返回前3个相似样本
        }
        
        return result
    
    def few_shot_prediction(self, 
                           query_vector: np.ndarray,
                           k: int = 5,
                           method: str = 'majority_vote') -> Dict:
        """
        少样本预测
        
        参数:
        query_vector: 查询向量
        k: 使用的最近邻数量
        method: 预测方法，可选 'majority_vote', 'weighted_vote', 'nearest_centroid'
        
        返回:
        预测结果
        """
        if not self.index.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # 查询相似样本
        similar_samples = self.query_similar(query_vector, k=k)
        
        if not similar_samples:
            return {'prediction': None, 'confidence': 0.0, 'similar_samples': []}
        
        # 提取标签和距离
        labels = [s['label'] for s in similar_samples]
        distances = [s['distance'] for s in similar_samples]
        
        if method == 'majority_vote':
            # 多数投票
            unique_labels, counts = np.unique(labels, return_counts=True)
            pred_label = unique_labels[np.argmax(counts)]
            confidence = np.max(counts) / len(labels)
            
        elif method == 'weighted_vote':
            # 加权投票（距离的倒数作为权重）
            weights = 1.0 / (np.array(distances) + 1e-8)
            
            label_weights = {}
            for label, weight in zip(labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            
            pred_label = max(label_weights, key=label_weights.get)
            total_weight = sum(label_weights.values())
            confidence = label_weights[pred_label] / total_weight
            
        elif method == 'nearest_centroid':
            # 计算每个类别的平均距离
            label_distances = {}
            for label, distance in zip(labels, distances):
                if label not in label_distances:
                    label_distances[label] = []
                label_distances[label].append(distance)
            
            avg_distances = {label: np.mean(dists) for label, dists in label_distances.items()}
            pred_label = min(avg_distances, key=avg_distances.get)
            confidence = 1.0 / (min(avg_distances.values()) + 1e-8)
        
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        return {
            'prediction': int(pred_label),
            'confidence': float(confidence),
            'method': method,
            'k': k,
            'similar_samples': similar_samples,
            'label_distribution': dict(zip(*np.unique(labels, return_counts=True)))
        }
    
    def save(self, filepath: str):
        """
        保存向量数据库到文件
        
        参数:
        filepath: 文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存数据
        data = {
            'db_type': self.db_type,
            'dimension': self.dimension,
            'use_gpu': self.use_gpu,
            'index_params': self.index_params,
            'vectors': [v.tolist() for v in self.vectors] if self.vectors else [],
            'labels': self.labels,
            'metadata': self.metadata,
            'stats': self.stats,
            'feature_stats': self.feature_stats
        }
        
        # 保存索引
        index_file = filepath.replace('.pkl', '_index')
        if self.index and self.index.is_built:
            try:
                self.index.save(index_file)
                data['index_file'] = index_file
            except Exception as e:
                print(f"警告: 保存索引失败: {e}")
        
        # 保存数据
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ 向量数据库已保存至: {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载向量数据库
        
        参数:
        filepath: 文件路径
        """
        # 加载数据
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复属性
        self.db_type = data['db_type']
        self.dimension = data['dimension']
        self.use_gpu = data.get('use_gpu', False)
        self.index_params = data.get('index_params', {})
        
        # 恢复数据
        self.vectors = [np.array(v) for v in data['vectors']]
        self.labels = data['labels']
        self.metadata = data['metadata']
        self.stats = data['stats']
        self.feature_stats = data.get('feature_stats', {})
        
        # 重新初始化索引
        self._initialize_index()
        
        # 加载索引
        index_file = data.get('index_file')
        if index_file and os.path.exists(index_file):
            try:
                self.index.load(index_file)
                print(f"✓ 向量索引已从 {index_file} 加载")
            except Exception as e:
                print(f"警告: 加载向量索引失败: {e}")
        
        print(f"✓ 向量数据库已从 {filepath} 加载")
        print(f"  样本数: {self.stats['n_samples']}, 类别数: {self.stats['n_classes']}")
    
    def get_statistics(self) -> Dict:
        """
        获取数据库统计信息
        
        返回:
        统计信息字典
        """
        stats = self.stats.copy()
        
        if self.vectors:
            all_vectors = np.vstack(self.vectors)
            stats.update({
                'total_vectors': all_vectors.shape[0],
                'vector_dimension': all_vectors.shape[1],
                'index_built': self.index.is_built if self.index else False,
                'db_type': self.db_type
            })
        
        return stats
    
    def export_to_json(self, filepath: str, max_samples: int = 1000):
        """
        导出数据库到JSON文件（用于可视化）
        
        参数:
        filepath: 输出文件路径
        max_samples: 最大导出样本数（避免文件过大）
        """
        if not self.vectors:
            print("No vectors to export")
            return
        
        all_vectors = np.vstack(self.vectors)
        
        # 限制样本数
        if len(all_vectors) > max_samples:
            indices = np.random.choice(len(all_vectors), max_samples, replace=False)
            vectors = all_vectors[indices]
            labels = [self.labels[i] for i in indices]
            metadata = [self.metadata[i] for i in indices] if self.metadata else []
        else:
            vectors = all_vectors
            labels = self.labels
            metadata = self.metadata
        
        # 准备数据
        data = {
            'metadata': {
                'n_samples': len(vectors),
                'n_classes': len(set(labels)),
                'dimension': vectors.shape[1],
                'export_time': datetime.now().isoformat()
            },
            'samples': []
        }
        
        # 添加样本数据
        for i in range(min(100, len(vectors))):  # 只导出前100个样本
            sample = {
                'id': i,
                'label': int(labels[i]),
                'vector': vectors[i].tolist()
            }
            
            if metadata and i < len(metadata):
                sample['metadata'] = metadata[i]
            
            data['samples'].append(sample)
        
        # 添加特征空间分析
        if len(vectors) > 0:
            data['feature_space_analysis'] = self.get_feature_space_analysis()
        
        # 保存到JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 数据库已导出至 JSON: {filepath}")
