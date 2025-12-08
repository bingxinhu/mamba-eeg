import numpy as np
import scipy.io as io
from scipy import signal
from sklearn.preprocessing import StandardScaler


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
    
    file_name = f"B0{subject}T.mat" if training else f"B0{subject}E.mat"
    try:
        a = io.loadmat(f"{data_path}/{file_name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据集文件 {file_name} 未找到，请检查路径")
    
    a_data = a["data"]
    
    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]
        
        for trial in range(a_trial.size):
            if a_artifacts[trial] == 0:
                start_idx = int(a_trial[trial].item())
                end_idx = start_idx + window_length
                if end_idx > a_X.shape[0]:
                    continue
                eeg_data = np.transpose(a_X[start_idx:end_idx, :n_channels])
                data_list.append(eeg_data)
                label_list.append(int(a_y[trial].item()) - 1)  # 标签转为0-1
    
    if not data_list:
        raise ValueError(f"未加载到有效数据，请检查数据集 {file_name}")
    
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


def get_data(data_path, subject, loso=False, is_standard=True, fre_filter=False, dataset='BCI2a'):
    """核心数据预处理函数：加载→截取生理窗口→标准化→滤波→返回张量"""
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
    
    # 3. 按通道标准化
    if is_standard:
        X_train, X_test = standardize_data(X_train, X_test, n_raw_chans)
        print("✅ 数据标准化完成")
    
    # 4. 多频段滤波（提取EEG关键频段）
    if fre_filter:
        filt_banks = [[1,4], [4,8], [8,12], [12,30], [30,40]]  # δ,θ,α,β,γ
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
