"""
预训练脚本：用极氪备件历史数据训练 XGBoost 模型

训练策略：滚动窗口交叉验证
- 12 个月训练 → 3 个月验证
- 窗口每 3 个月滚动一次
"""
import os
import sys
import json
from model import SalesPredictor

# 数据路径
DATA_PATH = r'C:\Users\Jiayang.Liu1\Desktop\极氪备件月度销量表 (1).xlsx'
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), 'model.pkl')
SUMMARY_OUTPUT = os.path.join(os.path.dirname(__file__), 'model_summary.json')

print('=' * 60)
print('  极氪备件月度销量预测 — 模型训练')
print('  训练策略: 滚动窗口交叉验证 (12月→3月)')
print('=' * 60)
print()

# 1. 加载数据
print('[1/3] 加载数据...')
predictor = SalesPredictor()
months, sales = predictor.load_data_from_excel(DATA_PATH)
print(f'  数据范围: {months[0]} ~ {months[-1]}')
print(f'  数据点: {len(sales)} 个月')
print(f'  销量范围: {sales.min():,.0f} ~ {sales.max():,.0f}')
print()

# 2. 滚动窗口训练
print('[2/3] 滚动窗口交叉验证训练...')
print()
summary = predictor.train_rolling_cv(sales, n_lags=12, forecast_horizon=3, verbose=True)

# 3. 保存模型
print()
print('[3/3] 保存模型...')
predictor.save(MODEL_OUTPUT)
print(f'  模型已保存到: {MODEL_OUTPUT}')

# 保存摘要
with open(SUMMARY_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f'  摘要已保存到: {SUMMARY_OUTPUT}')

print()
print('=' * 60)
print('  训练完成!')
print(f'  滚动窗口数: {summary["cv_windows"]}')
print(f'  CV RMSE: 1m={summary["cv_rmse_1m_mean"]:.0f}±{summary["cv_rmse_1m_std"]:.0f}, '
      f'2m={summary["cv_rmse_2m_mean"]:.0f}±{summary["cv_rmse_2m_std"]:.0f}, '
      f'3m={summary["cv_rmse_3m_mean"]:.0f}±{summary["cv_rmse_3m_std"]:.0f}')
print(f'  最终 RMSE: 1m={summary["final_rmse_1m"]:.0f}, '
      f'2m={summary["final_rmse_2m"]:.0f}, 3m={summary["final_rmse_3m"]:.0f}')
print(f'  Top 5 特征: {[f[0] for f in summary["top_features"]]}')
print('=' * 60)