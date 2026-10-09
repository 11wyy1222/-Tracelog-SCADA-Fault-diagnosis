"""
完整预测流程：
  步骤1: prepare_ml_data 把原始txt提取成特征 .pt
  步骤2: predict.py 加载模型预测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.prepare_ml_data import prepare_ml_data
from predict import predict
from error_codes import ErrorCode

# ============================================================
# 步骤1: 提取特征
# ============================================================
print("=" * 60)
print("步骤1: 提取预测数据特征")
print("=" * 60)

prepare_para = {
    # 数据源（目录模式，会扫描目录下所有csv/txt）
    # 如果只预测单个文件，把文件放到一个单独目录里
    'source_test_excel': r'\\192.168.100.21\风场数据\海上工程\S1-20170048 粤电湛江外罗海上风电项目200MW\主控文件\002#\Tracelog',
    'col_tracelog_path': '',

    # 输出
    'output_dir': './temp_predict',
    'output_filename': 'prediction_features.pt',

    # 特征提取参数（从变桨心跳preprocess_2D.yaml获取）
    'main_sensor_channels': [
        'ioperation_mode',
        'generator_speed',
        'converter_generator_speed',
        'rotor_speed_relay2',
        'rotor_speed_counter2',
        'breaker_on_feedback1',
        'breaker_on_feedback2',
    ],
    'timestamp_window': [-2000, 2000],
    'default_fault_time': 0,
    'missing_channel_strategy': 'zero',
    'column_mapping_file': r'.\configs\tracelog_tag_info.xlsx',
}

result_dict = {}
code = prepare_ml_data(prepare_para, result_dict)

if code != ErrorCode.SUCCESS.value:
    print(f"特征提取失败，错误码: {code}")
    sys.exit(1)

pt_path = result_dict['pt_path']
print(f"特征文件: {pt_path}")
print(f"样本数: {result_dict['sample_count']}, 维度: {result_dict['feature_dim']}")

# ============================================================
# 步骤2: 模型预测
# ============================================================
print("\n" + "=" * 60)
print("步骤2: 使用 predict.py 预测")
print("=" * 60)

predict_result = {}
code = predict(
    config='configs/config.yaml',       # 读取 checkpoint、device、batch_size、dataset_configs
    model='XGBoost',                     # 模型类型
    data_path=pt_path,                   # 步骤1生成的特征文件
    result_dict=predict_result,
    mapping=r'D:\不同故障模型设置文件\class_mapping映射文件\变桨心跳.xlsx',
)

if code != 0:
    print(f"预测失败，错误码: {code}")
    sys.exit(1)

print(f"\n预测完成，结果JSON: {predict_result['result_dir']}")

# 读取并打印预测结果
import json
with open(predict_result['result_dir'], 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"样本数: {results['num_samples']}")
print(f"预测结果: {results['predictions']}")
print(f"置信度均值: {results['confidence_stats']['mean']:.4f}")

if 'diagnosis' in results:
    print("\n故障诊断:")
    for idx, diag in list(results['diagnosis'].items())[:5]:
        print(f"  样本{idx}: 标签={diag['预测标签']}, 置信度={diag['置信度']:.3f}, 原因={diag['故障原因']}")
