#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学生成绩预测系统 - Web应用
提供友好的用户界面，方便教师使用系统功能
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

# 导入自定义模块
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.prediction_service import init_prediction_service
from src.visualization import DataVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'student_performance_prediction_secret_key'

# 设置上传文件的保存路径
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 设置模型保存路径
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# 初始化预测服务
init_prediction_service(app, MODEL_FOLDER)

# 设置处理后数据保存路径
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# 设置可视化图表保存路径
VISUALIZATION_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
if not os.path.exists(VISUALIZATION_FOLDER):
    os.makedirs(VISUALIZATION_FOLDER)


def allowed_file(filename):
    """检查文件扩展名是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """上传数据文件"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 如果用户没有选择文件
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        # 检查文件类型并保存
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            flash(f'文件 {filename} 上传成功')
            return redirect(url_for('data_processing', filename=filename))
        else:
            flash('不支持的文件类型')
            return redirect(request.url)
    
    return render_template('upload.html')


@app.route('/data_processing/<filename>', methods=['GET', 'POST'])
def data_processing(filename):
    """数据处理页面"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据
    try:
        data = processor.load_data(file_path)
        columns = data.columns.tolist()
        
        # 显示数据前10行
        preview_data = data.head(10).to_dict('records')
        
        # 数据统计信息
        data_info = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'dtypes': data.dtypes.astype(str).to_dict()
        }
        
        if request.method == 'POST':
            # 获取表单数据
            target_column = request.form.get('target_column')
            categorical_columns = request.form.getlist('categorical_columns')
            numerical_columns = request.form.getlist('numerical_columns')
            drop_columns = request.form.getlist('drop_columns')
            
            # 保存处理配置
            processing_config = {
                'target_column': target_column,
                'categorical_columns': categorical_columns,
                'numerical_columns': numerical_columns,
                'drop_columns': drop_columns
            }
            
            # 保存配置到文件
            config_path = os.path.join(PROCESSED_FOLDER, 'processing_config.json')
            with open(config_path, 'w') as f:
                json.dump(processing_config, f)
            
            # 处理数据
            processor.set_columns(
                target_column=target_column,
                categorical_columns=categorical_columns
            )
            
            # 清洗数据
            processor.clean_data(data)
            
            # 编码分类特征
            processor.encode_categorical_features()
            
            # 缩放数值特征
            processor.scale_numerical_features()
            
            # 获取处理后的特征和目标变量
            X, y = processor.get_features_and_target()
            
            # 保存处理后的数据
            X.to_csv(os.path.join(PROCESSED_FOLDER, 'processed_features.csv'), index=False)
            pd.DataFrame(y, columns=[target_column]).to_csv(os.path.join(PROCESSED_FOLDER, 'processed_target.csv'), index=False)
            
            flash('数据处理完成')
            return redirect(url_for('feature_engineering'))
        
        return render_template('data_processing.html', 
                              filename=filename, 
                              columns=columns, 
                              preview_data=preview_data, 
                              data_info=data_info)
    
    except Exception as e:
        flash(f'数据处理出错: {str(e)}')
        return redirect(url_for('upload_file'))


@app.route('/feature_engineering', methods=['GET', 'POST'])
def feature_engineering():
    """特征工程页面"""
    # 检查处理后的数据是否存在
    features_path = os.path.join(PROCESSED_FOLDER, 'processed_features.csv')
    target_path = os.path.join(PROCESSED_FOLDER, 'processed_target.csv')
    
    if not os.path.exists(features_path) or not os.path.exists(target_path):
        flash('请先完成数据处理')
        return redirect(url_for('upload_file'))
    
    # 加载处理后的数据
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    
    # 获取特征列名
    feature_columns = X.columns.tolist()
    
    if request.method == 'POST':
        # 初始化特征工程器
        engineer = FeatureEngineer(X)
        
        # 获取表单数据
        interaction_features = request.form.getlist('interaction_features')
        polynomial_features = request.form.getlist('polynomial_features')
        binning_features = request.form.getlist('binning_features')
        feature_selection = request.form.get('feature_selection', 'none')
        n_features = int(request.form.get('n_features', 10))
        
        # 创建交互特征
        if interaction_features and len(interaction_features) >= 2:
            # 每两个特征创建一个交互特征
            for i in range(0, len(interaction_features), 2):
                if i + 1 < len(interaction_features):
                    engineer.create_interaction_features([
                        (interaction_features[i], interaction_features[i+1])
                    ])
        
        # 创建多项式特征
        if polynomial_features:
            engineer.create_polynomial_features(polynomial_features)
        
        # 创建分箱特征
        if binning_features:
            engineer.create_binning_features(binning_features)
        
        # 特征选择
        if feature_selection != 'none':
            if feature_selection == 'correlation':
                engineer.select_features_by_correlation(y.iloc[:, 0], n_features)
            elif feature_selection == 'mutual_info':
                engineer.select_features_by_mutual_info(y.iloc[:, 0], n_features)
        
        # 获取工程化后的特征
        X_engineered = engineer.get_engineered_features()
        
        # 保存工程化后的特征
        X_engineered.to_csv(os.path.join(PROCESSED_FOLDER, 'engineered_features.csv'), index=False)
        
        # 保存特征工程配置
        engineering_config = {
            'interaction_features': interaction_features,
            'polynomial_features': polynomial_features,
            'binning_features': binning_features,
            'feature_selection': feature_selection,
            'n_features': n_features
        }
        
        config_path = os.path.join(PROCESSED_FOLDER, 'engineering_config.json')
        with open(config_path, 'w') as f:
            json.dump(engineering_config, f)
        
        flash('特征工程完成')
        return redirect(url_for('model_training'))
    
    return render_template('feature_engineering.html', feature_columns=feature_columns)


@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    # 定义模型名称与实例的映射
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    model_map = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    """模型训练页面"""
    # 检查工程化后的特征是否存在
    features_path = os.path.join(PROCESSED_FOLDER, 'engineered_features.csv')
    target_path = os.path.join(PROCESSED_FOLDER, 'processed_target.csv')
    
    if not os.path.exists(features_path):
        features_path = os.path.join(PROCESSED_FOLDER, 'processed_features.csv')
        if not os.path.exists(features_path) or not os.path.exists(target_path):
            flash('请先完成数据处理和特征工程')
            return redirect(url_for('upload_file'))
    
    # 可用的模型列表
    available_models = [
        'LinearRegression', 
        'Ridge', 
        'Lasso', 
        'RandomForestRegressor', 
        'GradientBoostingRegressor',
        'SVR'
    ]
    
    if request.method == 'POST':
        # 获取表单数据
        selected_models = request.form.getlist('models')
        test_size = float(request.form.get('test_size', 0.2))
        random_state = int(request.form.get('random_state', 42))
        cv_folds = int(request.form.get('cv_folds', 5))
        
        # 加载数据
        X = pd.read_csv(features_path)
        y = pd.read_csv(target_path).iloc[:, 0]
        
        # 初始化模型训练器
        trainer = ModelTrainer(
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds
        )
        
        # 设置数据
        trainer.set_data(X, y)

        # 添加选定的模型
        for model_name in selected_models:
            # 从model_map中获取模型实例
            model = model_map[model_name]
            # 补全model参数
            trainer.add_model(model=model, model_name=model_name)
        
        # 训练模型
        trainer.train_models()
        
        # 评估模型
        evaluation_results = trainer.evaluate_models()
        
        # 保存模型评估结果
        evaluation_path = os.path.join(MODEL_FOLDER, 'evaluation_results.json')
        with open(evaluation_path, 'w') as f:
            # 将numpy数据类型转换为Python原生类型
            results_dict = {}
            for model_name, metrics in evaluation_results.items():
                results_dict[model_name] = {k: float(v) for k, v in metrics.items()}
            json.dump(results_dict, f)
        
        # 保存最佳模型
        best_model, best_score = trainer.get_best_model()
        joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'best_model.pkl'))
        
        # 保存所有模型
        for model_name, model in trainer.models.items():
            joblib.dump(model, os.path.join(MODEL_FOLDER, f'{model_name}.pkl'))
        
        # 保存训练配置
        training_config = {
            'selected_models': selected_models,
            'test_size': test_size,
            'random_state': random_state,
            'cv_folds': cv_folds,
            'best_model': trainer.best_model_name,
            'best_score': float(best_score)
        }
        
        config_path = os.path.join(MODEL_FOLDER, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(training_config, f)
        
        # 创建可视化
        visualizer = DataVisualizer(X, y)
        
        # 保存特征重要性图
        if hasattr(best_model, 'feature_importances_'):
            visualizer.plot_feature_importance(
                best_model, 
                X.columns, 
                os.path.join(VISUALIZATION_FOLDER, 'feature_importance.png')
            )
        
        # 保存预测与实际值对比图
        y_pred = best_model.predict(X)
        visualizer.plot_prediction_vs_actual(
            y,
            y_pred,
            os.path.join(VISUALIZATION_FOLDER, 'prediction_vs_actual.png')
        )
        
        # 保存残差图
        visualizer.plot_residuals(
            y,
            y_pred,
            os.path.join(VISUALIZATION_FOLDER, 'residuals.png')
        )
        
        # 保存相关性矩阵
        visualizer.plot_correlation_matrix(
            X,
            os.path.join(VISUALIZATION_FOLDER, 'correlation_matrix.png')
        )
        
        # 保存成绩分布图
        visualizer.plot_grade_distribution(
            y,
            os.path.join(VISUALIZATION_FOLDER, 'grade_distribution.png')
        )
        
        flash('模型训练完成')
        return redirect(url_for('visualization'))
    
    return render_template('model_training.html', available_models=available_models)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """预测页面"""
    # 检查模型是否已训练
    model_path = os.path.join(MODEL_FOLDER, 'best_model.pkl')
    if not os.path.exists(model_path):
        flash('请先训练模型')
        return redirect(url_for('model_training'))
    
    # 加载特征工程配置
    engineering_config_path = os.path.join(PROCESSED_FOLDER, 'engineering_config.json')
    if os.path.exists(engineering_config_path):
        with open(engineering_config_path, 'r') as f:
            engineering_config = json.load(f)
    else:
        engineering_config = {}
    
    # 加载处理配置
    processing_config_path = os.path.join(PROCESSED_FOLDER, 'processing_config.json')
    if os.path.exists(processing_config_path):
        with open(processing_config_path, 'r') as f:
            processing_config = json.load(f)
    else:
        flash('请先完成数据处理')
        return redirect(url_for('upload_file'))
    
    # 加载训练配置
    training_config_path = os.path.join(MODEL_FOLDER, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        available_models = training_config.get('selected_models', [])
    else:
        available_models = []
    
    # 加载特征列
    features_path = os.path.join(PROCESSED_FOLDER, 'engineered_features.csv')
    if not os.path.exists(features_path):
        features_path = os.path.join(PROCESSED_FOLDER, 'processed_features.csv')
    
    X = pd.read_csv(features_path)
    feature_columns = X.columns.tolist()
    
    if request.method == 'POST':
        # 获取表单数据
        input_data = {}
        for column in feature_columns:
            input_data[column] = request.form.get(column, '')
        
        # 转换为DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 预测
        prediction = model.predict(input_df)[0]
        
        # 返回预测结果
        return render_template(
            'prediction_result.html',
            prediction=prediction,
            model_name=training_config.get('best_model', 'Unknown'),
            input_data=input_data
        )
    
    return render_template('prediction.html', feature_columns=feature_columns)


@app.route('/visualization')
def visualization():
    """可视化页面"""
    # 检查模型评估结果是否存在
    evaluation_path = os.path.join(MODEL_FOLDER, 'evaluation_results.json')
    if not os.path.exists(evaluation_path):
        flash('请先训练模型')
        return redirect(url_for('model_training'))
    
    # 加载评估结果
    with open(evaluation_path, 'r') as f:
        evaluation_results = json.load(f)
    
    # 图片路径
    image_paths = {
        'grade_distribution': url_for('static', filename='images/grade_distribution.png'),
        'feature_importance': url_for('static', filename='images/feature_importance.png'),
        'prediction_vs_actual': url_for('static', filename='images/prediction_vs_actual.png'),
        'residuals': url_for('static', filename='images/residuals.png'),
        'correlation_matrix': url_for('static', filename='images/correlation_matrix.png')
    }
    
    return render_template(
        'visualization.html',
        evaluation_results=evaluation_results,
        image_paths=image_paths
    )


@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    """批量预测页面"""
    # 检查模型是否已训练
    model_path = os.path.join(MODEL_FOLDER, 'best_model.pkl')
    if not os.path.exists(model_path):
        flash('请先训练模型')
        return redirect(url_for('model_training'))
    
    # 加载训练配置
    training_config_path = os.path.join(MODEL_FOLDER, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        available_models = training_config.get('selected_models', [])
    else:
        available_models = []
    
    if request.method == 'POST':
        # 检查是否有文件
        if 'batch_file' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)
        
        file = request.files['batch_file']
        
        # 如果用户没有选择文件
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        # 检查文件类型并处理
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'batch_' + filename)
            file.save(file_path)
            
            # 获取选项
            model_select = request.form.get('model_select', 'best')
            output_format = request.form.get('output_format', 'csv')
            include_confidence = 'include_confidence' in request.form
            include_explanations = 'include_explanations' in request.form
            
            try:
                # 使用预测服务进行批量预测
                from src.prediction_service import prediction_service
                
                # 加载数据
                if file_path.endswith('.csv'):
                    batch_data = pd.read_csv(file_path)
                else:
                    batch_data = pd.read_excel(file_path)
                
                # 保存原始数据的ID列（如果存在）
                id_column = None
                for col in ['id', 'ID', 'student_id', 'student_ID', 'index']:
                    if col in batch_data.columns:
                        id_column = col
                        break
                
                # 进行预测
                predictions = prediction_service.predict(batch_data)
                
                if predictions is None:
                    flash('预测失败，请检查输入数据格式是否正确')
                    return redirect(request.url)
                
                # 添加预测结果到原始数据
                batch_data['predicted_grade'] = predictions
                
                # 添加置信区间（如果支持）
                if include_confidence:
                    # 简单估计：使用模型的平均误差作为置信区间
                    model_info = prediction_service.get_model_info()
                    rmse = model_info.get('performance', {}).get('RMSE', 5.0)
                    
                    batch_data['confidence_lower'] = batch_data['predicted_grade'] - 1.96 * rmse
                    batch_data['confidence_upper'] = batch_data['predicted_grade'] + 1.96 * rmse
                
                # 添加解释（简单版本）
                if include_explanations:
                    conditions = [
                        (batch_data['predicted_grade'] < 60, '需要额外辅导'),
                        (batch_data['predicted_grade'] < 70, '需要巩固基础'),
                        (batch_data['predicted_grade'] < 85, '良好表现'),
                        (batch_data['predicted_grade'] >= 85, '优秀表现')
                    ]
                    explanations = ['需要额外辅导', '需要巩固基础', '良好表现', '优秀表现']
                    batch_data['explanation'] = np.select([c[0] for c in conditions], explanations, default='未知')
                
                # 保存结果
                result_filename = f'prediction_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
                result_path = os.path.join(PROCESSED_FOLDER, result_filename)
                
                if output_format == 'csv':
                    batch_data.to_csv(result_path + '.csv', index=False)
                    result_file = result_filename + '.csv'
                else:
                    batch_data.to_excel(result_path + '.xlsx', index=False)
                    result_file = result_filename + '.xlsx'
                
                flash('批量预测完成')
                return redirect(url_for('download_results', filename=result_file))
            
            except Exception as e:
                logger.error(f"批量预测失败: {str(e)}")
                flash(f'批量预测失败: {str(e)}')
                return redirect(request.url)
    
    return render_template('batch_prediction.html', available_models=available_models)


@app.route('/download_template')
def download_template():
    """下载数据模板"""
    # 检查处理后的数据是否存在
    features_path = os.path.join(PROCESSED_FOLDER, 'processed_features.csv')
    
    if not os.path.exists(features_path):
        flash('请先完成数据处理')
        return redirect(url_for('upload_file'))
    
    # 加载特征列
    X = pd.read_csv(features_path)
    
    # 创建模板（只保留列名）
    template = pd.DataFrame(columns=X.columns)
    
    # 保存模板
    template_path = os.path.join(PROCESSED_FOLDER, 'batch_prediction_template.csv')
    template.to_csv(template_path, index=False)
    
    # 返回文件下载
    return send_file(template_path, as_attachment=True)


@app.route('/download_results/<filename>')
def download_results(filename):
    """下载预测结果"""
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    
    if not os.path.exists(file_path):
        flash('文件不存在')
        return redirect(url_for('batch_prediction'))
    
    return send_file(file_path, as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
    """404页面"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """500页面"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
