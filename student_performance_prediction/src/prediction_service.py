#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测服务模块
该模块负责加载训练好的模型，并提供预测服务
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Blueprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('prediction_service')

class PredictionService:
    """
    预测服务类
    负责加载模型、处理输入数据并返回预测结果
    """
    
    def __init__(self, model_dir=None):
        """
        初始化预测服务
        
        参数:
            model_dir (str, optional): 模型目录路径，如果为None则使用默认路径
        """
        if model_dir is None:
            # 使用默认模型目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, 'model')
        
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.preprocessor = None
        self.metadata = None
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载最佳模型及其相关组件
        """
        try:
            # 查找模型目录中的模型文件
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.pkl')]
            
            if not model_files:
                logger.warning(f"在 {self.model_dir} 中未找到模型文件")
                return False
            
            # 加载元数据以确定最佳模型
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    logger.info(f"已加载模型元数据: {self.metadata['model_name']}")
            
            # 确定要加载的模型文件
            if self.metadata and 'model_name' in self.metadata:
                model_file = f"{self.metadata['model_name'].replace(' ', '_')}_model.pkl"
                model_path = os.path.join(self.model_dir, model_file)
                if not os.path.exists(model_path):
                    # 如果找不到指定的模型，使用第一个可用的模型
                    model_path = os.path.join(self.model_dir, model_files[0])
                    logger.warning(f"未找到指定的模型 {model_file}，使用 {model_files[0]} 代替")
            else:
                # 如果没有元数据，使用第一个可用的模型
                model_path = os.path.join(self.model_dir, model_files[0])
                logger.info(f"使用模型: {model_files[0]}")
            
            # 加载模型
            self.model = joblib.load(model_path)
            logger.info(f"已加载模型: {model_path}")
            
            # 加载特征名称
            feature_names_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"已加载特征名称，共 {len(self.feature_names)} 个特征")
            
            # 加载预处理器
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("已加载预处理器")
            
            return True
        
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def _preprocess_data(self, data):
        """
        预处理输入数据
        
        参数:
            data (DataFrame): 输入数据
            
        返回:
            DataFrame: 预处理后的数据
        """
        try:
            # 检查是否有特征名称列表
            if self.feature_names is not None:
                # 检查输入数据是否包含所有必需的特征
                missing_features = [f for f in self.feature_names if f not in data.columns]
                if missing_features:
                    logger.warning(f"输入数据缺少以下特征: {missing_features}")
                    # 为缺失的特征填充0
                    for feature in missing_features:
                        data[feature] = 0
                
                # 确保数据只包含模型需要的特征，并按正确顺序排列
                data = data[self.feature_names]
            
            # 应用预处理器（如果存在）
            if self.preprocessor is not None:
                data = self.preprocessor.transform(data)
                logger.info("已应用预处理器")
            
            return data
        
        except Exception as e:
            logger.error(f"预处理数据失败: {str(e)}")
            return None
    
    def predict(self, data):
        """
        对输入数据进行预测
        
        参数:
            data (DataFrame): 输入数据
            
        返回:
            array: 预测结果
        """
        try:
            # 检查模型是否已加载
            if self.model is None:
                logger.error("模型未加载，无法进行预测")
                return None
            
            # 预处理数据
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return None
            
            # 进行预测
            predictions = self.model.predict(processed_data)
            logger.info(f"已完成对 {len(data)} 条数据的预测")
            
            return predictions
        
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None
    
    def predict_single(self, input_data):
        """
        对单条输入数据进行预测
        
        参数:
            input_data (dict): 输入数据字典
            
        返回:
            float: 预测结果
        """
        try:
            # 将输入数据转换为DataFrame
            df = pd.DataFrame([input_data])
            
            # 确保所有数值字段都是浮点型
            numeric_fields = ['attendance_rate', 'homework_completion', 'class_participation', 
                             'study_hours_per_week', 'previous_exam_score', 'difficulty_level']
            
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = df[field].astype(float)
            
            # 进行预测
            predictions = self.predict(df)
            if predictions is None:
                return None
            
            # 返回单个预测值
            return float(predictions[0])
        
        except Exception as e:
            logger.error(f"单条预测失败: {str(e)}")
            return None
    
    def predict_batch(self, file_path):
        """
        对批量数据进行预测
        
        参数:
            file_path (str): 输入数据文件路径，支持CSV格式
            
        返回:
            tuple: (预测结果DataFrame, 成功标志)
        """
        try:
            # 加载数据
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return None, False
            
            # 保存原始数据的ID或索引列
            id_column = None
            for col in ['id', 'ID', 'student_id', 'student_ID', 'index']:
                if col in data.columns:
                    id_column = col
                    break
            
            if id_column is None:
                # 如果没有ID列，使用行索引
                original_indices = data.index
            else:
                original_indices = data[id_column].values
            
            # 移除非特征列
            feature_data = data.copy()
            if id_column:
                feature_data = feature_data.drop(id_column, axis=1)
            
            # 进行预测
            predictions = self.predict(feature_data)
            if predictions is None:
                return None, False
            
            # 创建结果DataFrame
            results = pd.DataFrame({
                'ID': original_indices,
                'Predicted_Score': predictions
            })
            
            logger.info(f"批量预测完成，共 {len(results)} 条结果")
            return results, True
        
        except Exception as e:
            logger.error(f"批量预测失败: {str(e)}")
            return None, False
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            dict: 模型信息
        """
        if self.model is None:
            return {"error": "模型未加载"}
        
        info = {
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names) if self.feature_names else "未知",
            "loaded_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加元数据信息（如果存在）
        if self.metadata:
            info.update({
                "model_name": self.metadata.get('model_name', '未知'),
                "training_samples": self.metadata.get('training_samples', '未知'),
                "creation_date": self.metadata.get('creation_date', '未知'),
                "performance": self.metadata.get('performance', {})
            })
        
        return info


# 创建Flask Blueprint
prediction_bp = Blueprint('prediction', __name__)

# 全局预测服务实例
prediction_service = None

def init_prediction_service(app, model_dir=None):
    """
    初始化预测服务
    
    参数:
        app (Flask): Flask应用实例
        model_dir (str, optional): 模型目录路径
    """
    global prediction_service
    prediction_service = PredictionService(model_dir)
    
    # 注册Blueprint
    app.register_blueprint(prediction_bp, url_prefix='/prediction')
    
    logger.info("预测服务已初始化")


@prediction_bp.route('/info', methods=['GET'])
def model_info():
    """获取模型信息的API端点"""
    if prediction_service is None:
        return jsonify({"error": "预测服务未初始化"}), 500
    
    info = prediction_service.get_model_info()
    return jsonify(info)


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """单条预测的API端点"""
    if prediction_service is None:
        return jsonify({"error": "预测服务未初始化"}), 500
    
    try:
        # 获取JSON输入数据
        input_data = request.json
        if not input_data:
            return jsonify({"error": "未提供输入数据"}), 400
        
        # 进行预测
        prediction = prediction_service.predict_single(input_data)
        if prediction is None:
            return jsonify({"error": "预测失败"}), 500
        
        # 返回预测结果
        return jsonify({
            "prediction": prediction,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        logger.error(f"API预测失败: {str(e)}")
        return jsonify({"error": str(e)}), 500


@prediction_bp.route('/single', methods=['GET'])
def predict_single():
    """单条预测页面"""
    return render_template('single_prediction.html')


@prediction_bp.route('/batch', methods=['GET', 'POST'])
def batch_prediction():
    """批量预测页面和API端点"""
    if prediction_service is None:
        return jsonify({"error": "预测服务未初始化"}), 500
    
    if request.method == 'GET':
        # 返回批量预测页面
        return render_template('batch_prediction.html')
    
    elif request.method == 'POST':
        try:
            # 检查是否有文件上传
            if 'file' not in request.files:
                return jsonify({"error": "未上传文件"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "未选择文件"}), 400
            
            # 保存上传的文件
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            
            # 进行批量预测
            results, success = prediction_service.predict_batch(file_path)
            if not success:
                return jsonify({"error": "批量预测失败"}), 500
            
            # 保存预测结果
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            result_file = os.path.join(results_dir, f'prediction_results_{timestamp}.csv')
            results.to_csv(result_file, index=False)
            
            # 返回结果
            return jsonify({
                "message": "批量预测成功",
                "result_file": result_file,
                "row_count": len(results)
            })
        
        except Exception as e:
            logger.error(f"批量预测API失败: {str(e)}")
            return jsonify({"error": str(e)}), 500


def main():
    """
    主函数，用于测试预测服务
    """
    # 创建预测服务实例
    service = PredictionService()
    
    # 打印模型信息
    print("\n模型信息:")
    print(json.dumps(service.get_model_info(), indent=4, ensure_ascii=False))
    
    # 测试单条预测
    test_data = {
        'attendance_rate': 0.85,
        'homework_completion': 0.9,
        'class_participation': 0.7,
        'previous_exam_score': 75,
        'study_hours_per_week': 10
    }
    
    print("\n测试单条预测:")
    prediction = service.predict_single(test_data)
    if prediction is not None:
        print(f"预测分数: {prediction:.2f}")
    
    # 测试批量预测
    print("\n测试批量预测:")
    # 创建测试数据
    test_batch = pd.DataFrame([
        {
            'student_id': 1,
            'attendance_rate': 0.85,
            'homework_completion': 0.9,
            'class_participation': 0.7,
            'previous_exam_score': 75,
            'study_hours_per_week': 10
        },
        {
            'student_id': 2,
            'attendance_rate': 0.95,
            'homework_completion': 0.95,
            'class_participation': 0.9,
            'previous_exam_score': 85,
            'study_hours_per_week': 15
        }
    ])
    
    # 保存测试数据
    test_file = 'test_batch.csv'
    test_batch.to_csv(test_file, index=False)
    
    # 进行批量预测
    results, success = service.predict_batch(test_file)
    if success:
        print("批量预测结果:")
        print(results)
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    main()
