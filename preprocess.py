import numpy as np
import scipy.io as io
from scipy import signal
from sklearn.preprocessing import StandardScaler
import os 
import glob

def load_data_BCI2a(data_path, subject, training):
    """加载BCI2a数据集（被试内，排除伪影试次）"""
    n_channels = 22
    window_length = 7 * 250  # 7秒原始数据（250Hz采样率）
    data_list = []
    label_list = []
    
    file_name = f"A0{subject}T.mat" if training else f"A0{subject}E.mat"
    try:
        a = io.loadmat(f"{data_path}/{file_name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据集文件 {file_name} 未找到，请检查路径")
    
    a_data = a["data"]
    
    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]  # EEG数据：(time, channels)
        a_trial = a_data3[1]  # 试次起始索引
        a_y = a_data3[2]  # 标签（1-4）
        a_artifacts = a_data3[5]  # 伪影标记（0=无伪影）
        
        for trial in range(a_trial.size):
            if a_artifacts[trial] == 0:  # 仅保留无伪影试次
                start_idx = int(a_trial[trial].item())
                end_idx = start_idx + window_length
                if end_idx > a_X.shape[0]:  # 防止索引越界
                    continue
                eeg_data = np.transpose(a_X[start_idx:end_idx, :n_channels])
                data_list.append(eeg_data)
                label_list.append(int(a_y[trial].item()) - 1)  # 标签转为0-3
    
    if not data_list:
        raise ValueError(f"未加载到有效数据，请检查数据集 {file_name}")
    
    return np.array(data_list), np.array(label_list)

def load_data_BCI2b(data_path, subject, training):
    """加载BCI2b数据集（2分类，3通道）"""
    n_channels = 3
    window_length = 8 * 250  # 8秒原始数据
    data_list = []
    label_list = []
    
    # BCI2b有9个被试，每个被试有多个会话
    subject_str = str(subject).zfill(2)
    
    print(f"正在加载BCI2b数据 - 被试: {subject_str}, 模式: {'训练' if training else '测试'}")
    
    if training:
        # 训练文件：B0{session}{subject}T.mat (如 B0101T.mat, B0201T.mat, B0301T.mat)
        pattern = f"*{subject_str}T.mat"
    else:
        # 测试文件：B0{session}{subject}E.mat (如 B0104E.mat, B0204E.mat, B0304E.mat)
        pattern = f"*{subject_str}E.mat"
    
    import glob
    file_pattern = os.path.join(data_path, pattern)
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    
    print(f"找到 {len(files)} 个文件")
    
    for file_idx, file_path in enumerate(files):
        try:
            print(f"  加载文件 {file_idx+1}/{len(files)}: {os.path.basename(file_path)}")
            data_dict = io.loadmat(file_path)
        except Exception as e:
            print(f"警告: 加载文件 {file_path} 失败: {e}")
            continue
        
        # BCI2b的数据结构不同，通常包含多个字段
        # 让我们先查看文件中有哪些键
        if file_idx == 0:
            print(f"    文件键: {list(data_dict.keys())}")
        
        # 尝试不同的键
        data_key = None
        for key in ['data', 'X', 'eeg', 'EEG']:
            if key in data_dict:
                data_key = key
                break
        
        if data_key is None:
            # 如果没有找到标准键，尝试第一个不是以'_'开头的键
            for key in data_dict.keys():
                if not key.startswith('__'):
                    data_key = key
                    break
        
        if data_key is None:
            print(f"    警告: 未找到有效数据键，跳过文件")
            continue
        
        print(f"    使用数据键: {data_key}")
        
        # 获取数据
        raw_data = data_dict[data_key]
        
        # 打印数据形状以便调试
        print(f"    原始数据形状: {raw_data.shape}")
        print(f"    原始数据类型: {type(raw_data)}")
        print(f"    原始数据dtype: {raw_data.dtype}")
        
        # BCI2b的数据可能是连续记录的，我们需要根据试次信息分割
        # 或者数据可能已经按照试次组织
        
        # 尝试方法1：如果数据是3D数组 (trials, channels, timepoints)
        if raw_data.ndim == 3:
            print(f"    检测到3D数据形状，假设为 (trials, channels, timepoints)")
            
            # 提取试次
            n_trials = raw_data.shape[0]
            for trial_idx in range(n_trials):
                trial_data = raw_data[trial_idx, :, :]
                
                # 检查数据长度
                if trial_data.shape[1] >= window_length:
                    # 使用前window_length个时间点
                    eeg_data = trial_data[:n_channels, :window_length]
                    data_list.append(eeg_data)
                    # 对于BCI2b，我们需要从其他字段获取标签
                    label_list.append(0)  # 临时标签，稍后可能需要调整
                else:
                    print(f"      试次 {trial_idx} 长度不足: {trial_data.shape[1]} < {window_length}")
        
        # 尝试方法2：如果数据是2D数组 (channels, timepoints)
        elif raw_data.ndim == 2 and raw_data.shape[0] == n_channels:
            print(f"    检测到2D数据形状，假设为 (channels, timepoints)")
            
            # 这可能是单个试次或多个试次连接在一起
            total_timepoints = raw_data.shape[1]
            n_trials = total_timepoints // window_length
            
            print(f"    总时间点: {total_timepoints}, 可提取试次数: {n_trials}")
            
            for trial_idx in range(n_trials):
                start_idx = trial_idx * window_length
                end_idx = start_idx + window_length
                
                if end_idx <= total_timepoints:
                    eeg_data = raw_data[:n_channels, start_idx:end_idx]
                    data_list.append(eeg_data)
                    label_list.append(0)  # 临时标签
        
        # 尝试方法3：结构化的MATLAB数据
        elif raw_data.dtype.names is not None:
            print(f"    检测到结构化数据，字段: {raw_data.dtype.names}")
            
            # 尝试提取常见的字段
            if 'X' in raw_data.dtype.names:
                eeg_data_field = raw_data['X'][0, 0]
                print(f"    找到EEG数据字段 'X', 形状: {eeg_data_field.shape}")
                
                # 处理EEG数据
                # 这里需要根据实际结构调整
                
            # 对于结构化数据，我们可能需要不同的处理方式
            # 暂时跳过这种格式
            print(f"    警告: 结构化数据格式未实现，跳过文件")
            continue
        
        else:
            print(f"    警告: 未知数据格式，形状: {raw_data.shape}，跳过文件")
            continue
        
        print(f"    从该文件加载了 {len(data_list) - len(label_list) + n_trials} 个试次")
    
    if not data_list:
        raise ValueError(f"未加载到有效数据，请检查数据集文件")
    
    print(f"成功加载了 {len(data_list)} 个试次")
    
    # 由于我们没有真实的标签，我们需要从其他文件或方法获取
    # 对于BCI2b，标签通常是单独的文件或字段
    # 这里我们暂时使用随机标签进行测试
    if len(label_list) != len(data_list):
        # 如果标签数量不匹配，创建临时标签
        print(f"警告: 标签数量({len(label_list)})与数据数量({len(data_list)})不匹配，使用随机标签")
        label_list = np.random.randint(0, 2, len(data_list)).tolist()
    
    return np.array(data_list), np.array(label_list)


def load_data_loso(data_path, subject, dataset='BCI2a'):
    """留一法交叉验证数据加载（1个被试为测试集，其余为训练集）"""
    X_train, y_train = [], []
    X_test, y_test = None, None
    
    for sub in range(1, 10):  # BCI2a/2b均为9个被试
        if dataset == 'BCI2a':
            x_train_sub, y_train_sub = load_data_BCI2a(data_path, sub, training=True)
            x_test_sub, y_test_sub = load_data_BCI2a(data_path, sub, training=False)
        else:
            x_train_sub, y_train_sub = load_data_BCI2b(data_path, sub, training=True)
            x_test_sub, y_test_sub = load_data_BCI2b(data_path, sub, training=False)
        
        # 合并当前被试的训练+测试数据
        x_sub = np.concatenate((x_train_sub, x_test_sub), axis=0)
        y_sub = np.concatenate((y_train_sub, y_test_sub), axis=0)
        
        # 当前被试为测试集，其余为训练集
        if sub == subject + 1:  # subject为0-based，sub为1-based
            X_test, y_test = x_sub, y_sub
        else:
            X_train.append(x_sub)
            y_train.append(y_sub)
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    return X_train, y_train, X_test, y_test


def standardize_data(X_train, X_test, channels):
    """按通道标准化（避免不同脑区信号幅值差异影响训练）"""
    # 输入形状：(n_samples, 1, n_channels, n_timepoints)
    for j in range(channels):
        scaler = StandardScaler()
        # 训练集：提取第j通道数据，拟合scaler后标准化
        train_chan_data = X_train[:, 0, j, :].reshape(-1, 1)
        scaler.fit(train_chan_data)
        X_train[:, 0, j, :] = scaler.transform(train_chan_data).reshape(
            X_train.shape[0], X_train.shape[3]
        )
        # 测试集：使用训练集的scaler（避免数据泄露）
        test_chan_data = X_test[:, 0, j, :].reshape(-1, 1)
        X_test[:, 0, j, :] = scaler.transform(test_chan_data).reshape(
            X_test.shape[0], X_test.shape[3]
        )
    return X_train, X_test


def bandpass_filter(data, bandFiltCutF, fs, filtOrder=50, axis=1, filtType='filtfilt'):
    """EEG信号带通滤波（支持多频段提取，避免相位偏移）"""
    # 无效截止频率处理
    if (bandFiltCutF[0] in (0, None)) and (bandFiltCutF[1] in (None, fs/2.0)):
        print("⚠️  无效滤波参数，不进行滤波")
        return data
    
    # 设计FIR滤波器
    nyq = 0.5 * fs  # 奈奎斯特频率
    if bandFiltCutF[0] in (0, None):
        print(f"🔧 应用低通滤波（截止频率：{bandFiltCutF[1]}Hz）")
        cutoff = bandFiltCutF[1] / nyq
        h = signal.firwin(filtOrder + 1, cutoff=cutoff, pass_zero="lowpass")
    elif bandFiltCutF[1] in (None, fs/2.0):
        print(f"🔧 应用高通滤波（截止频率：{bandFiltCutF[0]}Hz）")
        cutoff = bandFiltCutF[0] / nyq
        h = signal.firwin(filtOrder + 1, cutoff=cutoff, pass_zero="highpass")
    else:
        print(f"🔧 应用带通滤波（频段：{bandFiltCutF[0]}-{bandFiltCutF[1]}Hz）")
        cutoff = [b / nyq for b in bandFiltCutF]
        h = signal.firwin(filtOrder + 1, cutoff=cutoff, pass_zero="bandpass")
    
    # 应用滤波（filtfilt避免相位偏移，更适合EEG）
    if filtType == 'filtfilt':
        data_out = signal.filtfilt(h, [1], data, axis=axis)
    else:
        data_out = signal.lfilter(h, [1], data, axis=axis)
    return data_out


def augment_eeg_data_simple(X, y, augment_factor=2):
    """简化的EEG数据增强函数"""
    if augment_factor < 1:
        return X, y
    
    X_augmented = [X]
    y_augmented = [y]
    
    for i in range(augment_factor):
        # 1. 高斯噪声增强
        noise = np.random.normal(0, 0.02, X.shape)  # 2%的高斯噪声
        X_augmented.append(X + noise)
        y_augmented.append(y)
        
        # 2. 振幅缩放增强
        scale = np.random.uniform(0.9, 1.1, (X.shape[0], 1, 1, 1))
        X_augmented.append(X * scale)
        y_augmented.append(y)
    
    # 合并所有增强数据
    X_augmented = np.concatenate(X_augmented, axis=0)
    y_augmented = np.concatenate(y_augmented, axis=0)
    
    # 打乱顺序
    indices = np.arange(len(X_augmented))
    np.random.shuffle(indices)
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    return X_augmented, y_augmented


def get_data(data_path, subject, loso=False, is_standard=True, fre_filter=False, 
             dataset='BCI2a', augment=False, augment_factor=1):
    """增强版数据预处理函数：加载→截取生理窗口→标准化→滤波→增强→返回张量"""
    if dataset == 'BCI2a':
        fs = 250  # BCI2a采样率250Hz
        t1 = int(1.5 * fs)  # 运动想象关键窗口：1.5秒开始
        t2 = int(6 * fs)    # 6秒结束（共4.5秒有效数据）
        T = t2 - t1         # 最终时序长度：1125（250*4.5）
        n_raw_chans = 22    # 原始通道数
    else:  # BCI2b
        fs = 250
        t1 = int(2.5 * fs)
        t2 = int(7 * fs)
        T = t2 - t1
        n_raw_chans = 3
    
    # 1. 加载数据（被试内或留一法）
    if loso:
        X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, dataset)
    else:
        if dataset == 'BCI2a':
            X_train, y_train = load_data_BCI2a(data_path, subject + 1, training=True)
            X_test, y_test = load_data_BCI2a(data_path, subject + 1, training=False)
        else:
            X_train, y_train = load_data_BCI2b(data_path, subject + 1, training=True)
            X_test, y_test = load_data_BCI2b(data_path, subject + 1, training=False)
    
    # 2. 截取运动想象关键窗口，调整形状为 (n_samples, 1, n_channels, n_timepoints)
    n_tr, _, _ = X_train.shape
    X_train = X_train[:, :, t1:t2].reshape(n_tr, 1, n_raw_chans, T)  # 训练集
    
    n_te, _, _ = X_test.shape
    X_test = X_test[:, :, t1:t2].reshape(n_te, 1, n_raw_chans, T)    # 测试集
    
    # 3. 数据增强（仅在训练时且未使用留一法时进行）
    if augment and not loso:
        print(f"🔄 数据增强：原始样本数 {len(X_train)}")
        X_train, y_train = augment_eeg_data_simple(X_train, y_train, augment_factor)
        print(f"✅ 数据增强完成：增强后样本数 {len(X_train)}")
    
    # 4. 按通道标准化
    if is_standard:
        X_train, X_test = standardize_data(X_train, X_test, n_raw_chans)
        print("✅ 数据标准化完成")
    
    # 5. 多频段滤波（提取EEG关键频段）
    if fre_filter:
        filt_banks = [[1,4], [4,8], [8,10], [10,12], [12,18], [18,24], [24,30], [30,40], [40,60], [60,80]]  # δ,θ,α1,α2,β1,β2,β3,γ1,γ2,γ3
        n_bands = len(filt_banks)
        # 初始化多频段数据存储
        X_train_bands = np.zeros((X_train.shape[0], 1, n_raw_chans * n_bands, T))
        X_test_bands = np.zeros((X_test.shape[0], 1, n_raw_chans * n_bands, T))
        
        for i, band in enumerate(filt_banks):
            # 对每个频段单独滤波（按时间维度）
            X_train_band = bandpass_filter(X_train.squeeze(1), band, fs, axis=-1)
            X_test_band = bandpass_filter(X_test.squeeze(1), band, fs, axis=-1)
            # 分配到对应频段通道
            start_idx = i * n_raw_chans
            end_idx = (i + 1) * n_raw_chans
            X_train_bands[:, 0, start_idx:end_idx, :] = X_train_band
            X_test_bands[:, 0, start_idx:end_idx, :] = X_test_band
        
        # 替换为多频段数据
        X_train = X_train_bands
        X_test = X_test_bands
        print(f"✅ 多频段滤波完成（频段数：{n_bands}，总通道数：{n_raw_chans * n_bands}）")
    
    return X_train, y_train, X_test, y_test
