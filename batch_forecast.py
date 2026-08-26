"""
批量 SKU 滚动预测：预测每个 SKU 未来 10 个月销量
"""
import pandas as pd
import numpy as np
import os
from model import SalesPredictor, compute_seasonal_profile, prepare_input_features, dampen_spike_prediction

# 输入输出路径
INPUT_PATH = r"C:\Users\Jiayang.Liu1\Downloads\BI表_20260826_094319.xlsx"
OUTPUT_PATH = r"D:\预测\SKU预测_2026年9月-2027年6月.xlsx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# 预测参数
FORECAST_MONTHS = 10  # 预测 10 个月
MONTH_LABELS = [
    '2026年9月', '2026年10月', '2026年11月', '2026年12月',
    '2027年1月', '2027年2月', '2027年3月', '2027年4月',
    '2027年5月', '2027年6月'
]

print('=' * 60)
print('  极氪备件 SKU 销量预测')
print(f'  预测范围: {MONTH_LABELS[0]} ~ {MONTH_LABELS[-1]}')
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
print(f'  列: {df.columns.tolist()}')

# 3. 预测
print()
print('[3/4] 开始预测...')
results = []
total = len(df)

# 预测月份映射：列表中的每个月份对应的实际日历月 (1-12)
PRED_MONTH_CALENDAR = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]  # 2026年9月=9, ..., 2027年6月=6

for idx, row in df.iterrows():
    sku_code = str(row['SKU编码'])
    sku_name = str(row['SKU名称']) if pd.notna(row['SKU名称']) else ''

    # 提取 12 个月数据（月1=最近, 月12=最早 → 反转成从早到晚）
    raw_months = pd.to_numeric(row[['月1','月2','月3','月4','月5','月6','月7','月8','月9','月10','月11','月12']], errors='coerce').fillna(0).values
    months = raw_months[::-1]  # 反转：最早→最近

    # 跳过全零
    if months.sum() == 0:
        continue

    try:
        # 从训练数据中匹配该 SKU 的季节性特征
        seasonal_profile = predictor.sku_seasonal_profiles.get(sku_code, None)
        # 如果训练数据中没有，用该 SKU 自己的 12 个月数据粗略估算
        if seasonal_profile is None:
            seasonal_profile = compute_seasonal_profile(months)

        # SKU 校准因子
        calib = predictor.sku_calibration_factors.get(sku_code, 1.0)

        # 分块预测：每 3 个月一块，使用 3 个直接模型，减少滚动误差累积
        window = list(months)
        predictions = []

        for chunk_start in range(0, FORECAST_MONTHS, 3):
            chunk_size = min(3, FORECAST_MONTHS - chunk_start)
            first_month = PRED_MONTH_CALENDAR[chunk_start]

            X = prepare_input_features(window[-12:], seasonal_profile=seasonal_profile, first_pred_month=first_month)
            X_s = predictor.scaler.transform(X)

            if chunk_size >= 1:
                raw_pred = max(0, float(predictor.model_1m.predict(X_s)[0]))
                raw_pred = dampen_spike_prediction(raw_pred, window[-12:])
                predictions.append(round(raw_pred * calib))
                window.append(raw_pred)  # 窗口用原始预测值，避免校准因子叠加

            if chunk_size >= 2:
                raw_pred = max(0, float(predictor.model_2m.predict(X_s)[0]))
                raw_pred = dampen_spike_prediction(raw_pred, window[-12:])
                predictions.append(round(raw_pred * calib))
                window.append(raw_pred)

            if chunk_size >= 3:
                raw_pred = max(0, float(predictor.model_3m.predict(X_s)[0]))
                raw_pred = dampen_spike_prediction(raw_pred, window[-12:])
                predictions.append(round(raw_pred * calib))
                window.append(raw_pred)

        result_row = {
            'SKU编码': sku_code,
            'SKU名称': sku_name,
        }
        for i, label in enumerate(MONTH_LABELS):
            result_row[label] = predictions[i]

        results.append(result_row)

    except Exception as e:
        pass

    if (idx + 1) % 5000 == 0:
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
print(f'  预测范围: {MONTH_LABELS[0]} ~ {MONTH_LABELS[-1]}')
print('=' * 60)