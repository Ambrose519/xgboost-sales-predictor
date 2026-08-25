"""
XGBoost 月度销量预测模型

训练策略：滚动窗口交叉验证
- 12 个月训练 → 3 个月验证
- 滚动窗口不断推进
- 最终模型在所有可用数据上训练
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import os

MODEL_DIR = os.path.dirname(__file__)


class SimpleScaler:
    """简易标准化器，替代 sklearn.preprocessing.SimpleScaler"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def compute_seasonal_profile(series):
    """
    计算一个 SKU 的季节性特征：每个月相对于平均值的比例。

    参数:
        series: 一维数组，按月排列的销量数据（至少 12 个月）

    返回:
        seasonal_ratios: 长度为 12 的数组，每个月的平均比例
    """
    n = len(series)
    monthly_sums = np.zeros(12)
    monthly_counts = np.zeros(12)

    for i in range(n):
        month_idx = i % 12
        monthly_sums[month_idx] += series[i]
        monthly_counts[month_idx] += 1

    monthly_avg = np.where(monthly_counts > 0, monthly_sums / monthly_counts, 0)
    overall_avg = np.mean(series[series > 0]) if np.any(series > 0) else 1.0

    if overall_avg == 0:
        overall_avg = 1.0

    seasonal_ratios = monthly_avg / overall_avg
    return seasonal_ratios


def build_features_from_series(series, n_lags=12, seasonal_profile=None):
    """
    从一维时间序列构建监督学习特征矩阵。

    参数:
        series: 一维 list/array，按月排列的销量数据
        n_lags: 滞后阶数
        seasonal_profile: 长度为 12 的季节性比率数组（可选）

    返回:
        X: 特征矩阵 (样本数, 特征数)
        y1, y2, y3: 目标值（未来 1/2/3 个月）
        feature_names: 特征名列表
    """
    n = len(series)
    rows = []

    for i in range(n_lags, n - 3):
        feats = {}

        # 滞后特征 (lag 1-12)
        for lag in range(1, n_lags + 1):
            feats[f'lag_{lag}'] = series[i - lag]

        # 滚动统计（基于最近窗口，不包含当前值）
        window_3 = series[i-3:i]
        window_6 = series[i-6:i]
        window_12 = series[i-12:i]

        feats['rolling_mean_3'] = np.mean(window_3)
        feats['rolling_mean_6'] = np.mean(window_6)
        feats['rolling_mean_12'] = np.mean(window_12)
        feats['rolling_std_3'] = np.std(window_3) if len(window_3) >= 2 else 0
        feats['rolling_std_6'] = np.std(window_6) if len(window_6) >= 2 else 0

        # 趋势：最近 3 个月 vs 前 3 个月
        if i >= 6:
            feats['trend_3m'] = np.mean(series[i-3:i]) - np.mean(series[i-6:i-3])
        else:
            feats['trend_3m'] = 0

        # 环比增长率
        feats['mom_growth'] = (series[i-1] - series[i-2]) / series[i-2] if series[i-2] != 0 else 0

        # 同比增长率（12 个月前）
        feats['yoy_growth'] = (series[i-1] - series[i-13]) / series[i-13] if i >= 13 and series[i-13] != 0 else 0

        # 月份季节性特征
        month_idx = (i % 12) + 1
        feats['month_sin'] = np.sin(2 * np.pi * month_idx / 12)
        feats['month_cos'] = np.cos(2 * np.pi * month_idx / 12)

        # 季度
        feats['quarter'] = ((month_idx - 1) // 3) + 1

        # SKU 季节性特征（该 SKU 各月的历史平均占比）
        if seasonal_profile is not None:
            for m in range(12):
                feats[f'seasonal_m{m+1}'] = float(seasonal_profile[m])

        # 目标值
        feats['target_1'] = series[i + 1]
        feats['target_2'] = series[i + 2]
        feats['target_3'] = series[i + 3]

        rows.append(feats)

    df = pd.DataFrame(rows)
    feature_names = [c for c in df.columns if c not in ['target_1', 'target_2', 'target_3']]

    X = df[feature_names].values.astype(float)
    y1 = df['target_1'].values.astype(float)
    y2 = df['target_2'].values.astype(float)
    y3 = df['target_3'].values.astype(float)

    return X, y1, y2, y3, feature_names


def prepare_input_features(recent_12_months, seasonal_profile=None):
    """
    从最近 12 个月销量构建单个预测样本的特征向量。

    参数:
        recent_12_months: 长度为 12 的 list/array
        seasonal_profile: 长度为 12 的季节性比率数组（可选）

    返回:
        features: 形状为 (1, n_features) 的 numpy array
    """
    series = list(recent_12_months)
    feats = {}

    # 滞后特征
    for lag in range(1, 13):
        feats[f'lag_{lag}'] = series[-lag]

    # 滚动统计
    s = pd.Series(series)
    feats['rolling_mean_3'] = float(s.tail(3).mean())
    feats['rolling_mean_6'] = float(s.tail(6).mean())
    feats['rolling_mean_12'] = float(s.mean())
    feats['rolling_std_3'] = float(s.tail(3).std()) if len(s) >= 3 else 0.0
    feats['rolling_std_6'] = float(s.tail(6).std()) if len(s) >= 6 else 0.0
    feats['trend_3m'] = float(s.tail(3).mean() - s.head(3).mean())
    feats['mom_growth'] = float((series[-1] - series[-2]) / series[-2]) if series[-2] != 0 else 0.0
    feats['yoy_growth'] = 0.0  # 用户输入只有 12 个月，无法计算同比

    # 月份特征（假设预测从最近一个月之后开始）
    next_month_idx = 13  # 预测的第 1 个月
    feats['month_sin'] = float(np.sin(2 * np.pi * next_month_idx / 12))
    feats['month_cos'] = float(np.cos(2 * np.pi * next_month_idx / 12))
    feats['quarter'] = float(((next_month_idx - 1) // 3) + 1)

    # SKU 季节性特征
    if seasonal_profile is not None:
        for m in range(12):
            feats[f'seasonal_m{m+1}'] = float(seasonal_profile[m])

    return np.array([[feats[k] for k in sorted(feats.keys())]])


class SalesPredictor:
    """XGBoost 销量预测器（滚动窗口交叉验证）"""

    def __init__(self):
        self.model_1m = None       # 预测 +1 月
        self.model_2m = None       # 预测 +2 月
        self.model_3m = None       # 预测 +3 月
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.cv_results = []       # 交叉验证结果
        self.train_rmse = {}
        self.feature_importance = None
        self.training_summary = {}

    def load_data_from_excel_all_skus(self, file_path, min_months=24):
        """
        从 Excel 加载所有 SKU 的月度销量数据。

        返回:
            sku_data: list of (sku_code, sku_name, sales_array)
            months: 月份标签列表
        """
        df = pd.read_excel(file_path, sheet_name=0, header=0)

        # 找到月度列
        month_cols = [c for c in df.columns if '（个）' in str(c) or
                      (isinstance(c, str) and c[:4].isdigit() and len(c) >= 6)]
        if not month_cols:
            month_cols = df.columns[4:]

        # 解析月份标签
        months_raw = []
        for c in month_cols:
            m = str(c).replace('（个）', '').strip()
            months_raw.append(m)

        # 按时间排序
        month_pairs = sorted(zip(months_raw, month_cols), key=lambda x: x[0])
        months_sorted = [m for m, _ in month_pairs]
        cols_sorted = [c for _, c in month_pairs]

        # 提取每个 SKU 的数据
        sku_data = []
        code_col = df.columns[0]
        name_col = df.columns[1]

        for idx, row in df.iterrows():
            sales = pd.to_numeric(row[cols_sorted], errors='coerce').fillna(0).values
            # 跳过数据太少的 SKU
            non_zero = np.count_nonzero(sales)
            if non_zero < min_months:
                continue
            sku_data.append((
                str(row[code_col]),
                str(row[name_col]) if pd.notna(row[name_col]) else '',
                sales.astype(float)
            ))

        print(f'加载 {len(sku_data)} 个 SKU，每个 {len(months_sorted)} 个月')
        return sku_data, months_sorted

    def train_sku_mixed(self, sku_data, n_lags=12, forecast_horizon=3, verbose=True):
        """
        混合训练：将所有 SKU 的数据合并训练一个 XGBoost 模型。

        参数:
            sku_data: list of (code, name, sales_array)
            n_lags: 滞后特征数
            forecast_horizon: 预测步数
            verbose: 是否打印进度
        """
        all_X = []
        all_y1 = []
        all_y2 = []
        all_y3 = []
        sku_count = 0

        if verbose:
            print(f'混合训练：{len(sku_data)} 个 SKU，每个 {n_lags} 个月滞后')

        for code, name, sales in sku_data:
            if len(sales) < n_lags + forecast_horizon + 1:
                continue

            # 计算该 SKU 的季节性特征
            seasonal_profile = compute_seasonal_profile(sales)

            X, y1, y2, y3, feature_names = build_features_from_series(
                sales, n_lags, seasonal_profile=seasonal_profile
            )

            if len(X) >= 5:
                all_X.append(X)
                all_y1.append(y1)
                all_y2.append(y2)
                all_y3.append(y3)
                sku_count += 1

        if sku_count == 0:
            raise ValueError('没有足够的 SKU 数据用于训练')

        X_all = np.vstack(all_X)
        y1_all = np.concatenate(all_y1)
        y2_all = np.concatenate(all_y2)
        y3_all = np.concatenate(all_y3)
        self.feature_names = feature_names

        if verbose:
            print(f'有效 SKU: {sku_count}')
            print(f'总训练样本: {len(X_all)}')
            print(f'特征数: {len(feature_names)}')

        # 标准化
        self.scaler = SimpleScaler()
        X_all_s = self.scaler.fit_transform(X_all)

        # 训练三个模型
        for label, y_all in [('1m', y1_all), ('2m', y2_all), ('3m', y3_all)]:
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            model.fit(X_all_s, y_all)

            preds = model.predict(X_all_s)
            rmse = np.sqrt(np.mean((y_all - preds) ** 2))
            self.train_rmse[label] = float(rmse)

            if label == '1m':
                self.model_1m = model
            elif label == '2m':
                self.model_2m = model
            else:
                self.model_3m = model

        self.is_trained = True

        # 特征重要性
        importances = []
        for m in [self.model_1m, self.model_2m, self.model_3m]:
            importances.append(m.feature_importances_)
        avg_importance = np.mean(importances, axis=0)
        self.feature_importance = sorted(
            zip(self.feature_names, [float(v) for v in avg_importance]),
            key=lambda x: x[1], reverse=True
        )

        self.training_summary = {
            'total_skus': int(sku_count),
            'total_samples': int(len(X_all)),
            'features': int(len(feature_names)),
            'final_rmse_1m': float(self.train_rmse.get('1m', 0)),
            'final_rmse_2m': float(self.train_rmse.get('2m', 0)),
            'final_rmse_3m': float(self.train_rmse.get('3m', 0)),
            'top_features': [(name, float(imp)) for name, imp in self.feature_importance[:5]]
        }

        if verbose:
            print(f'训练完成!')
            print(f'RMSE: 1m={self.train_rmse["1m"]:.0f}, 2m={self.train_rmse["2m"]:.0f}, 3m={self.train_rmse["3m"]:.0f}')
            print(f'Top 5 特征: {[f[0] for f in self.feature_importance[:5]]}')

        return self.training_summary
        """从 Excel 加载月度销量数据（汇总所有备件）"""
        df = pd.read_excel(file_path, sheet_name=0, header=0)

        # 找到月度列
        month_cols = [c for c in df.columns if '（个）' in str(c) or
                      (isinstance(c, str) and c[:4].isdigit() and len(c) >= 6)]
        if not month_cols:
            month_cols = df.columns[4:]  # 前 4 列是信息列

        # 解析月份标签
        months = []
        for c in month_cols:
            m = str(c).replace('（个）', '').strip()
            months.append(m)

        # 按时间排序
        month_pairs = sorted(zip(months, month_cols), key=lambda x: x[0])
        months_sorted = [m for m, _ in month_pairs]
        cols_sorted = [c for _, c in month_pairs]

        # 汇总月度总销量
        monthly_totals = []
        for c in cols_sorted:
            total = pd.to_numeric(df[c], errors='coerce').fillna(0).sum()
            monthly_totals.append(float(total))

        return months_sorted, np.array(monthly_totals)

    def train_rolling_cv(self, sales_data, n_lags=12, forecast_horizon=3, verbose=True):
        """
        滚动窗口交叉验证训练。

        训练策略：
        - 一次性构建全部特征矩阵
        - 每个窗口使用对应区间的特征行训练/验证
        - 最终模型在所有可用数据上训练

        参数:
            sales_data: 一维 numpy array，全部历史月度销量
            n_lags: 滞后特征数（默认 12）
            forecast_horizon: 预测步数（默认 3）
            verbose: 是否打印训练过程
        """
        n = len(sales_data)

        if verbose:
            print(f'数据总量: {n} 个月')

        # 一次性构建全部特征矩阵
        X_all, y1_all, y2_all, y3_all, self.feature_names = build_features_from_series(
            sales_data, n_lags
        )
        total_rows = len(X_all)
        if verbose:
            print(f'特征矩阵: {total_rows} 行 × {len(self.feature_names)} 列')
            print()

        self.cv_results = []
        all_models_1m = []
        all_models_2m = []
        all_models_3m = []

        # 滚动窗口：每次前进 3 个月
        # 特征矩阵行 k 对应原始数据位置 k+n_lags，使用 sales[k:k+n_lags] 预测 sales[k+n_lags+1:k+n_lags+4]
        for cv_start in range(0, n - n_lags - forecast_horizon, 3):
            cv_end = cv_start + n_lags  # 训练窗口结束（原始数据索引）
            val_end = cv_end + forecast_horizon  # 验证窗口结束

            # 特征矩阵索引
            train_idx_end = cv_end - n_lags  # 训练数据在特征矩阵中的结束位置
            val_idx_end = train_idx_end + forecast_horizon  # 验证数据在特征矩阵中的结束位置

            if train_idx_end < 5:
                continue
            if val_idx_end > total_rows:
                val_idx_end = total_rows
            if train_idx_end >= val_idx_end:
                continue

            X_train = X_all[:train_idx_end]
            X_val = X_all[train_idx_end:val_idx_end]
            y1_train = y1_all[:train_idx_end]
            y1_val = y1_all[train_idx_end:val_idx_end]
            y2_train = y2_all[:train_idx_end]
            y2_val = y2_all[train_idx_end:val_idx_end]
            y3_train = y3_all[:train_idx_end]
            y3_val = y3_all[train_idx_end:val_idx_end]

            if len(X_train) < 5:
                continue

            scaler = SimpleScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            wi = len(self.cv_results) + 1
            window_result = {'window': wi,
                             'train_samples': len(X_train),
                             'val_samples': len(X_val)}

            for label, y_train, y_val, Xt, Xv in [
                ('1m', y1_train, y1_val, X_train_s, X_val_s),
                ('2m', y2_train, y2_val, X_train_s, X_val_s),
                ('3m', y3_train, y3_val, X_train_s, X_val_s)
            ]:
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0
                )
                model.fit(Xt, y_train)

                preds = model.predict(Xv)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                mape = np.mean(np.abs((y_val - preds) / np.maximum(np.abs(y_val), 1))) * 100

                window_result[f'rmse_{label}'] = float(rmse)
                window_result[f'mape_{label}'] = float(mape)
                window_result[f'pred_{label}'] = preds.tolist()
                window_result[f'actual_{label}'] = y_val.tolist()

                if label == '1m':
                    all_models_1m.append(model)
                elif label == '2m':
                    all_models_2m.append(model)
                else:
                    all_models_3m.append(model)

            self.cv_results.append(window_result)

            if verbose:
                print(f'窗口 {wi}: 训练 {len(X_train)} 样本 → 验证 {len(X_val)} 样本 | '
                      f'RMSE(1m)={window_result["rmse_1m"]:.0f}, '
                      f'RMSE(2m)={window_result["rmse_2m"]:.0f}, '
                      f'RMSE(3m)={window_result["rmse_3m"]:.0f}')

        # 最终模型：在所有可用数据上训练
        if verbose:
            print()
            print(f'训练最终模型（全部 {total_rows} 个样本）...')

        self.scaler = SimpleScaler()
        X_all_s = self.scaler.fit_transform(X_all)

        for label, y_all, model_list in [
            ('1m', y1_all, all_models_1m),
            ('2m', y2_all, all_models_2m),
            ('3m', y3_all, all_models_3m)
        ]:
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            model.fit(X_all_s, y_all)

            preds = model.predict(X_all_s)
            rmse = np.sqrt(np.mean((y_all - preds) ** 2))
            self.train_rmse[label] = float(rmse)

            if label == '1m':
                self.model_1m = model
            elif label == '2m':
                self.model_2m = model
            else:
                self.model_3m = model

        self.is_trained = True

        # 特征重要性
        importances = []
        for m in [self.model_1m, self.model_2m, self.model_3m]:
            importances.append(m.feature_importances_)
        avg_importance = np.mean(importances, axis=0)
        self.feature_importance = sorted(
            zip(self.feature_names, [float(v) for v in avg_importance]),
            key=lambda x: x[1], reverse=True
        )

        # 汇总 CV 指标
        cv_rmse_1m = [r['rmse_1m'] for r in self.cv_results]
        cv_rmse_2m = [r['rmse_2m'] for r in self.cv_results]
        cv_rmse_3m = [r['rmse_3m'] for r in self.cv_results]

        self.training_summary = {
            'total_data_points': int(n),
            'cv_windows': len(self.cv_results),
            'cv_rmse_1m_mean': float(np.mean(cv_rmse_1m)),
            'cv_rmse_2m_mean': float(np.mean(cv_rmse_2m)),
            'cv_rmse_3m_mean': float(np.mean(cv_rmse_3m)),
            'cv_rmse_1m_std': float(np.std(cv_rmse_1m)),
            'cv_rmse_2m_std': float(np.std(cv_rmse_2m)),
            'cv_rmse_3m_std': float(np.std(cv_rmse_3m)),
            'final_rmse_1m': float(self.train_rmse.get('1m', 0)),
            'final_rmse_2m': float(self.train_rmse.get('2m', 0)),
            'final_rmse_3m': float(self.train_rmse.get('3m', 0)),
            'features': len(self.feature_names),
            'top_features': [(name, float(imp)) for name, imp in self.feature_importance[:5]]
        }

        if verbose:
            print(f'最终模型 RMSE: 1m={self.train_rmse["1m"]:.0f}, '
                  f'2m={self.train_rmse["2m"]:.0f}, 3m={self.train_rmse["3m"]:.0f}')
            print(f'CV 平均 RMSE: 1m={np.mean(cv_rmse_1m):.0f}±{np.std(cv_rmse_1m):.0f}, '
                  f'2m={np.mean(cv_rmse_2m):.0f}±{np.std(cv_rmse_2m):.0f}, '
                  f'3m={np.mean(cv_rmse_3m):.0f}±{np.std(cv_rmse_3m):.0f}')

        return self.training_summary

    def predict(self, recent_12_months, seasonal_profile=None):
        """
        基于最近 12 个月销量预测未来 3 个月。

        参数:
            recent_12_months: 长度为 12 的 list/array
            seasonal_profile: 长度为 12 的季节性比率数组（可选）

        返回:
            dict: {'month_1': float, 'month_2': float, 'month_3': float}
        """
        if not self.is_trained:
            raise ValueError('模型尚未训练')

        if len(recent_12_months) != 12:
            raise ValueError(f'需要 12 个月数据，当前提供了 {len(recent_12_months)} 个')

        X = prepare_input_features(recent_12_months, seasonal_profile=seasonal_profile)
        X_s = self.scaler.transform(X)

        pred_1 = max(0, float(self.model_1m.predict(X_s)[0]))
        pred_2 = max(0, float(self.model_2m.predict(X_s)[0]))
        pred_3 = max(0, float(self.model_3m.predict(X_s)[0]))

        return {
            'month_1': round(pred_1, 2),
            'month_2': round(pred_2, 2),
            'month_3': round(pred_3, 2),
            'total_3m': round(pred_1 + pred_2 + pred_3, 2),
            'avg_monthly': round((pred_1 + pred_2 + pred_3) / 3, 2)
        }

    def rolling_predict(self, recent_12_months, steps=3):
        """
        滚动预测：预测未来 N 个月，并将预测值纳入窗口继续预测。
        """
        window = list(recent_12_months)
        predictions = []

        for _ in range(steps):
            X = prepare_input_features(window[-12:])
            X_s = self.scaler.transform(X)
            pred = max(0, float(self.model_1m.predict(X_s)[0]))
            predictions.append(round(pred, 2))
            window.append(pred)

        return predictions

    def predict_sku_batch(self, df_input):
        """
        批量 SKU 预测。

        参数:
            df_input: pandas DataFrame，每行一个 SKU
                      格式: [SKU编码, SKU名称, 月1, 月2, ..., 月12]
                      或: [SKU编码, 月1, 月2, ..., 月12]

        返回:
            pandas DataFrame: [SKU编码, SKU名称, 预测月1, 预测月2, 预测月3, 3月合计]
        """
        if not self.is_trained:
            raise ValueError('模型尚未训练')

        # 自动检测列数
        n_cols = df_input.shape[1]
        if n_cols == 13:
            # [SKU编码, 月1-月12]
            name_col = None
        elif n_cols == 14:
            # [SKU编码, SKU名称, 月1-月12]
            name_col = df_input.columns[1]
        elif n_cols >= 15:
            # 多列信息列 + 12 个月数据
            # 取最后 12 列作为月度数据
            name_col = df_input.columns[1] if n_cols == 14 else None
        else:
            raise ValueError(f'数据格式错误：需要 13-14 列（当前 {n_cols} 列）')

        # 提取月度数据列（最后 12 列）
        month_cols = df_input.columns[-12:].tolist()
        sku_col = df_input.columns[0]

        results = []
        for idx, row in df_input.iterrows():
            months = pd.to_numeric(row[month_cols], errors='coerce').fillna(0).values

            # 跳过全零的 SKU
            if months.sum() == 0:
                continue

            try:
                # 计算该 SKU 的季节性特征（从输入 12 个月估算）
                seasonal_profile = compute_seasonal_profile(months)
                pred = self.predict(months, seasonal_profile=seasonal_profile)
                result_row = {sku_col: row[sku_col]}
                if name_col and name_col in df_input.columns:
                    result_row[name_col] = row[name_col]
                result_row['预测月1'] = pred['month_1']
                result_row['预测月2'] = pred['month_2']
                result_row['预测月3'] = pred['month_3']
                result_row['3月合计'] = pred['total_3m']
                results.append(result_row)
            except Exception:
                continue

        if not results:
            raise ValueError('没有有效的 SKU 数据可以预测')

        return pd.DataFrame(results)

    def save(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_1m': self.model_1m,
                'model_2m': self.model_2m,
                'model_3m': self.model_3m,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'cv_results': self.cv_results,
                'train_rmse': self.train_rmse,
                'feature_importance': self.feature_importance,
                'training_summary': self.training_summary
            }, f)

    def load(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model_1m = data['model_1m']
        self.model_2m = data['model_2m']
        self.model_3m = data['model_3m']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        self.cv_results = data.get('cv_results', [])
        self.train_rmse = data.get('train_rmse', {})
        self.feature_importance = data.get('feature_importance', None)
        self.training_summary = data.get('training_summary', {})
        return self