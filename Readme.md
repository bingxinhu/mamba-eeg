S0	准确率: 0.8719, 平衡准确率: 0.8717, Kappa: 0.8292, F1: 0.8722
S1	准确率: 0.7173, 平衡准确率: 0.7185, Kappa: 0.6231, F1: 0.7166
S2	准确率: 0.9084, 平衡准确率: 0.9084, Kappa: 0.8779, F1: 0.9089
S3	准确率: 0.8728, 平衡准确率: 0.8739, Kappa: 0.8305, F1: 0.8724
S4	准确率: 0.7138, 平衡准确率: 0.7192, Kappa: 0.6191, F1: 0.6952
S5	准确率: 0.6651, 平衡准确率: 0.6645, Kappa: 0.5533, F1: 0.6650
S6	准确率: 0.8450, 平衡准确率: 0.8460, Kappa: 0.7935, F1: 0.8456
S7	准确率: 0.8953, 平衡准确率: 0.8933, Kappa: 0.8602, F1: 0.8958
S8	准确率: 0.7879, 平衡准确率: 0.7880, Kappa: 0.7169, F1: 0.7867

# 1. 仅分析结果（最快）
python run_diagnosis.py --subjects 1,4,5 --mode results

# 2. 完整诊断（需要重新加载数据）
python run_diagnosis.py --subjects 1,4,5 --mode both

# 3. 与其他被试比较
python run_diagnosis.py --subjects 0,1,2,3,4,5,6,7,8 --mode results

# 4. 生成详细报告（包含交互式可视化）
python -c "
from diagnosis_analysis import EEGSubjectDiagnosis
diagnosis = EEGSubjectDiagnosis('./results', 1)
diagnosis.load_results()
diagnosis.generate_interactive_report()
