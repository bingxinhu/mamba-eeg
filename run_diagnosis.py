#!/usr/bin/env python
"""
脑电分类困难被试诊断系统
用于诊断为什么某些被试（如S1、S4、S5）表现较差
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='脑电分类困难被试诊断系统')
    parser.add_argument('--subjects', type=str, default='1,4,5', 
                       help='要诊断的被试ID，用逗号分隔')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果目录路径')
    parser.add_argument('--data_path', type=str, default='./dataset/2a',
                       help='原始数据路径')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['results', 'data', 'model', 'both'],
                       help='诊断模式: results=仅结果分析, data=仅数据分析, model=仅模型分析, both=全部')
    
    args = parser.parse_args()
    
    # 解析被试ID
    subject_ids = [int(s.strip()) for s in args.subjects.split(',')]
    
    print(f"诊断被试: {subject_ids}")
    print(f"结果目录: {args.results_dir}")
    print(f"数据路径: {args.data_path}")
    print(f"诊断模式: {args.mode}")
    
    if args.mode in ['results', 'both']:
        print(f"\n{'='*60}")
        print("阶段1: 结果分析")
        print('='*60)
        
        try:
            from diagnosis_analysis import run_detailed_diagnosis
            results_analysis = run_detailed_diagnosis(
                subject_ids=subject_ids,
                results_dir=args.results_dir
            )
        except Exception as e:
            print(f"结果分析失败: {e}")
    
    if args.mode in ['data', 'model', 'both']:
        print(f"\n{'='*60}")
        print("阶段2: 数据与模型深度调查")
        print('='*60)
        
        try:
            from hard_subject_analysis import investigate_hard_subjects
            investigation = investigate_hard_subjects(
                hard_subject_ids=subject_ids,
                data_path=args.data_path
            )
        except Exception as e:
            print(f"深度调查失败: {e}")
    
    print(f"\n{'='*60}")
    print("诊断完成")
    print('='*60)
    
    # 生成建议
    print("\n基于诊断结果的建议:")
    print("1. 对于困难被试，检查数据质量（噪声水平、信号幅度）")
    print("2. 考虑使用更强的数据增强（特别是对于S1、S4、S5）")
    print("3. 调整模型正则化参数（增加dropout、权重衰减）")
    print("4. 使用更复杂的模型架构或集成方法")
    print("5. 考虑被试特定的模型微调")

if __name__ == '__main__':
    main()
