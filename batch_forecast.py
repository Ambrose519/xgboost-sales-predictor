"""
批量 SKU 预测：预测每个 SKU 未来 1 个月销量
策略：月均 50-100 用 XGBoost，其他用加权平均
"""
import pandas as pd
import numpy as np
import os
from model import SalesPredictor, compute_seasonal_profile

# 输入输出路径
INPUT_PATH = r"C:\Users\Jiayang.Liu1\Downloads\BI表_20260826_094319.xlsx"
OUTPUT_PATH = r"D:\预测\SKU预测_2026年9月.xlsx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

PRED_LABEL = '2026年9月'

print('=' * 60)
print('  极氪备件 SKU 销量预测')
print(f'  预测: {PRED_LABEL}')
print('=' * 60)
print()

# 1. 加载模型
print('[1/4] 加载模型...')
predictor = SalesPredictor()
predictor.load(MODEL_PATH)
print(f'  模型已加载，RMSE: {predictor.train_rmse}')

# 2. 加载数据
print()
print('[2/4] 加载数据...')
df = pd.read_excel(INPUT_PATH, sheet_name=0, header=0)
print(f'  SKU 数量: {len(df)}')

# 3. 预测
print()
print('[3/4] 开始预测...')
results = []
total = len(df)

for idx, row in df.iterrows():
    sku_code = str(row['SKU编码'])
    sku_name = str(row['SKU名称']) if pd.notna(row['SKU名称']) else ''

    raw_months = pd.to_numeric(row[['月1','月2','月3','月4','月5','月6','月7','月8','月9','月10','月11','月12']], errors='coerce').fillna(0).values
    months = raw_months[::-1]  # 反转：最早→最近

    result_row = {'SKU编码': sku_code, 'SKU名称': sku_name}

    try:
        if months.sum() == 0:
            result_row[PRED_LABEL] = 0
        else:
            # 加权平均 = 最近3个月加权
            weighted_avg = months[-1] * 0.5 + months[-2] * 0.3 + months[-3] * 0.2
            recent_3m_avg = np.mean(months[-3:])

            # 月均 50-100：用 XGBoost；其他：用加权平均
            if 50 <= recent_3m_avg <= 100:
                seasonal_profile = predictor.sku_seasonal_profiles.get(sku_code, None)
                if seasonal_profile is None:
                    seasonal_profile = compute_seasonal_profile(months)
                pred = predictor.predict(months, seasonal_profile=seasonal_profile, first_pred_month=9)
                calib = predictor.sku_calibration_factors.get(sku_code, 1.0)
                result_row[PRED_LABEL] = round(pred['month_1'] * calib)
            else:
                result_row[PRED_LABEL] = round(weighted_avg)
    except Exception:
        result_row[PRED_LABEL] = 0

    results.append(result_row)

    if (idx + 1) % 10000 == 0:
        print(f'  进度: {idx + 1}/{total}')

print(f'  完成! 预测 SKU 数: {len(results)}')

# 4. 保存结果
print()
print('[4/4] 保存结果...')
result_df = pd.DataFrame(results)
result_df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
print(f'  已保存到: {OUTPUT_PATH}')
print(f'  文件大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB')

print()
print('=' * 60)
print('  预测完成!')
print(f'  总 SKU 数: {len(results)}')
print(f'  预测: {PRED_LABEL}')
print('=' * 60)