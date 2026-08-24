"""
XGBoost 月度销量预测 — Flask Web 应用
预训练模型已内嵌，启动时自动加载，直接进入预测
"""
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

from model import SalesPredictor

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'zeekr-xgboost-predictor-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ==================== 加载预训练模型 ====================

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), 'model_summary.json')

predictor = SalesPredictor()
model_loaded = False
model_info = {}

try:
    predictor.load(MODEL_PATH)
    model_loaded = True
    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    print(f'[OK] 预训练模型已加载: {MODEL_PATH}')
    print(f'     训练数据: {model_info.get("total_data_points", "?")} 个月')
    print(f'     CV窗口数: {model_info.get("cv_windows", "?")}')
    print(f'     最终RMSE: 1m={model_info.get("final_rmse_1m", "?"):.0f}, '
          f'2m={model_info.get("final_rmse_2m", "?"):.0f}, '
          f'3m={model_info.get("final_rmse_3m", "?"):.0f}')
except FileNotFoundError:
    print('[WARN] 未找到预训练模型文件，请先运行 train_model.py')
except Exception as e:
    print(f'[ERROR] 模型加载失败: {e}')


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页：直接进入预测"""
    if not model_loaded:
        return render_template('no_model.html')
    return render_template('predict.html', model_info=model_info)


# ==================== API 路由 ====================

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """预测 API：接收 12 个月数据，返回预测结果"""
    if not model_loaded:
        return jsonify({'error': '模型未加载，请先运行训练脚本'}), 503

    data = request.get_json()
    if not data or 'months' not in data:
        return jsonify({'error': '请提供 months 数组（12 个月销量数据）'}), 400

    months = data['months']
    if len(months) != 12:
        return jsonify({'error': f'需要正好 12 个月的数据，当前提供了 {len(months)} 个'}), 400

    try:
        months = [float(m) for m in months]
        result = predictor.predict(months)
        return jsonify({
            'success': True,
            'prediction': result,
            'input': months
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/forecast/rolling', methods=['POST'])
def forecast_rolling():
    """滚动预测 API"""
    if not model_loaded:
        return jsonify({'error': '模型未加载'}), 503

    data = request.get_json()
    if not data or 'months' not in data:
        return jsonify({'error': '请提供 months 数组'}), 400

    months = data['months']
    steps = data.get('steps', 3)

    if len(months) < 12:
        return jsonify({'error': f'需要至少 12 个月的数据'}), 400

    try:
        months = [float(m) for m in months]
        predictions = predictor.rolling_predict(months[-12:], steps=steps)
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input': months[-12:]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    if not model_loaded:
        return jsonify({'trained': False})

    return jsonify({
        'trained': True,
        **model_info
    })


# ==================== 启动 ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)