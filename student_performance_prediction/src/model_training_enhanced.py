#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版模型训练与评估模块
使用多种机器学习算法构建预测模型，并评估其性能
增加了更多高级功能和优化选项
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
from sklearn.base import clone

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """
    增强版模型训练类，负责训练、评估和选择最佳模型
    增加了更多高级功能和优化选项
    """
    
    def __init__(self, config=None):
        """
        初始化模型训练器
        
        参数:
            config (dict, optional): 配置参数字典
        """
        self.config = config or {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = None
        self.training_time = {}
        self.model_pipelines = {}
        self.preprocessor = None
        
        # 初始化默认模型
        self._init_models()
    
    def _init_models(self):
        """
        初始化模型字典，包含更多模型选项和默认参数
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        }

    def load_data(self, X_train_path, y_train_path, X_test_path=None, y_test_path=None):
        """
        加载训练和测试数据
        
        参数:
            X_train_path (str): 训练特征文件路径
            y_train_path (str): 训练目标文件路径
            X_test_path (str, optional): 测试特征文件路径
            y_test_path (str, optional): 测试目标文件路径
            
        返回:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"正在加载训练数据: {X_train_path}, {y_train_path}")
        
        # 加载训练数据
        self.X_train = pd.read_csv(X_train_path)
        self.feature_names = self.X_train.columns.tolist()
        
        # 加载训练目标
        y_train_data = pd.read_csv(y_train_path)
        if isinstance(y_train_data, pd.DataFrame) and y_train_data.shape[1] == 1:
            self.y_train = y_train_data.iloc[:, 0]
        else:
            self.y_train = y_train_data
        
        # 加载测试数据（如果提供）
        if X_test_path and y_test_path:
            logger.info(f"正在加载测试数据: {X_test_path}, {y_test_path}")
            self.X_test = pd.read_csv(X_test_path)
            
            y_test_data = pd.read_csv(y_test_path)
            if isinstance(y_test_data, pd.DataFrame) and y_test_data.shape[1] == 1:
                self.y_test = y_test_data.iloc[:, 0]
            else:
                self.y_test = y_test_data
        
        logger.info(f"数据加载完成，训练集形状: {self.X_train.shape}, {self.y_train.shape}")
        if self.X_test is not None:
            logger.info(f"测试集形状: {self.X_test.shape}, {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load_data_from_dataframe(self, X_train, y_train, X_test=None, y_test=None):
        """
        直接从DataFrame加载数据
        
        参数:
            X_train (DataFrame): 训练特征DataFrame
            y_train (Series/DataFrame): 训练目标Series或DataFrame
            X_test (DataFrame, optional): 测试特征DataFrame
            y_test (Series/DataFrame, optional): 测试目标Series或DataFrame
            
        返回:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("正在从DataFrame加载数据")
        
        # 加载训练数据
        self.X_train = X_train
        self.feature_names = self.X_train.columns.tolist()
        
        # 加载训练目标
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            self.y_train = y_train.iloc[:, 0]
        else:
            self.y_train = y_train
        
        # 加载测试数据（如果提供）
        if X_test is not None and y_test is not None:
            self.X_test = X_test
            
            if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                self.y_test = y_test.iloc[:, 0]
            else:
                self.y_test = y_test
        
        logger.info(f"数据加载完成，训练集形状: {self.X_train.shape}, {self.y_train.shape}")
        if self.X_test is not None:
            logger.info(f"测试集形状: {self.X_test.shape}, {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self, categorical_features=None, numerical_features=None, scaling='standard'):
        """
        预处理数据，包括类别特征编码和数值特征缩放
        
        参数:
            categorical_features (list, optional): 类别特征列表
            numerical_features (list, optional): 数值特征列表
            scaling (str): 缩放方法，'standard'或'minmax'
            
        返回:
            tuple: (X_train_processed, X_test_processed)
        """
        logger.info(f"正在预处理数据，缩放方法: {scaling}")
        
        if self.X_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        # 如果未指定特征类型，自动检测
        if categorical_features is None and numerical_features is None:
            categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            logger.info(f"自动检测到 {len(categorical_features)} 个类别特征和 {len(numerical_features)} 个数值特征")
        
        # 创建预处理管道
        preprocessor = self._create_preprocessor(categorical_features, numerical_features, scaling)
        
        # 应用预处理
        X_train_processed = preprocessor.fit_transform(self.X_train)
        
        # 如果有测试集，也进行预处理
        X_test_processed = None
        if self.X_test is not None:
            X_test_processed = preprocessor.transform(self.X_test)
        
        logger.info("数据预处理完成")
        
        # 保存预处理器
        self.preprocessor = preprocessor
        
        return X_train_processed, X_test_processed
    
    def _create_preprocessor(self, categorical_features, numerical_features, scaling='standard'):
        """
        创建数据预处理管道
        
        参数:
            categorical_features (list): 类别特征列表
            numerical_features (list): 数值特征列表
            scaling (str): 缩放方法，'standard'或'minmax'
            
        返回:
            ColumnTransformer: 预处理管道
        """
        from sklearn.preprocessing import OneHotEncoder
        
        # 选择缩放器
        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {scaling}")
        
        # 创建预处理管道
        transformers = []
        
        # 添加数值特征处理
        if numerical_features:
            transformers.append(('num', scaler, numerical_features))
        
        # 添加类别特征处理
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
        
        # 创建列转换器
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        
        return preprocessor

    def train_models(self, selected_models=None):
        """
        训练所有模型或指定的模型
        
        参数:
            selected_models (list, optional): 要训练的模型名称列表，如果为None则训练所有模型
            
        返回:
            dict: 训练后的模型字典
        """
        logger.info("开始训练模型")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        self.trained_models = {}
        self.training_time = {}
        
        # 确定要训练的模型
        models_to_train = {}
        if selected_models is not None:
            for name in selected_models:
                if name in self.models:
                    models_to_train[name] = self.models[name]
                else:
                    logger.warning(f"模型 {name} 不存在，将被跳过")
        else:
            models_to_train = self.models
        
        # 训练模型
        for name, model in models_to_train.items():
            logger.info(f"正在训练模型: {name}")
            try:
                # 记录训练开始时间
                start_time = time.time()
                
                # 训练模型
                model.fit(self.X_train, self.y_train)
                
                # 记录训练结束时间
                end_time = time.time()
                training_time = end_time - start_time
                
                # 保存模型和训练时间
                self.trained_models[name] = model
                self.training_time[name] = training_time
                
                logger.info(f"模型 {name} 训练完成，耗时: {training_time:.2f}秒")
            except Exception as e:
                logger.error(f"模型 {name} 训练失败: {str(e)}")
        
        logger.info(f"所有模型训练完成，共 {len(self.trained_models)} 个模型")
        return self.trained_models
    
    def evaluate_models(self, additional_metrics=False):
        """
        评估所有训练好的模型
        
        参数:
            additional_metrics (bool): 是否计算额外的评估指标
            
        返回:
            dict: 模型评估结果
        """
        logger.info("开始评估模型")
        
        if not self.trained_models:
            logger.warning("没有训练好的模型，请先调用train_models方法")
            return {}
        
        if self.X_test is None or self.y_test is None:
            logger.warning("测试数据未加载，将使用训练数据进行评估")
            X_eval = self.X_train
            y_eval = self.y_train
        else:
            X_eval = self.X_test
            y_eval = self.y_test
        
        self.model_results = {}
        
        for name, model in self.trained_models.items():
            logger.info(f"正在评估模型: {name}")
            
            try:
                # 预测
                y_pred = model.predict(X_eval)
                
                # 计算基本评估指标
                mse = mean_squared_error(y_eval, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_eval, y_pred)
                r2 = r2_score(y_eval, y_pred)
                
                # 创建结果字典
                result = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': y_pred,
                    'training_time': self.training_time.get(name, 0)
                }
                
                # 计算额外的评估指标
                if additional_metrics:
                    evs = explained_variance_score(y_eval, y_pred)
                    result['EVS'] = evs
                    
                    # 计算预测误差
                    errors = y_eval - y_pred
                    result['mean_error'] = np.mean(errors)
                    result['median_error'] = np.median(errors)
                    result['error_std'] = np.std(errors)
                
                # 保存结果
                self.model_results[name] = result
                
                logger.info(f"模型 {name} 评估完成: RMSE={rmse:.4f}, R2={r2:.4f}")
            except Exception as e:
                logger.error(f"模型 {name} 评估失败: {str(e)}")
        
        # 根据R2分数选择最佳模型
        if self.model_results:
            self.best_model_name = max(self.model_results, key=lambda x: self.model_results[x]['R2'])
            self.best_model = self.trained_models[self.best_model_name]
            logger.info(f"最佳模型: {self.best_model_name}, R2={self.model_results[self.best_model_name]['R2']:.4f}")
        
        return self.model_results
    
    def cross_validate_models(self, cv=5, scoring='r2', selected_models=None):
        """
        对模型进行交叉验证
        
        参数:
            cv (int): 交叉验证折数
            scoring (str): 评分标准
            selected_models (list, optional): 要验证的模型名称列表，如果为None则验证所有模型
            
        返回:
            dict: 交叉验证结果
        """
        logger.info(f"开始交叉验证，折数: {cv}, 评分标准: {scoring}")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        cv_results = {}
        
        # 确定要验证的模型
        models_to_validate = {}
        if selected_models is not None:
            for name in selected_models:
                if name in self.models:
                    models_to_validate[name] = self.models[name]
                else:
                    logger.warning(f"模型 {name} 不存在，将被跳过")
        else:
            models_to_validate = self.models
        
        for name, model in models_to_validate.items():
            logger.info(f"正在对模型 {name} 进行交叉验证")
            
            try:
                # 使用KFold进行交叉验证
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                
                # 存储每折的预测结果
                fold_predictions = []
                fold_scores = []
                
                # 对每折进行训练和评估
                for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                    X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                    
                    # 训练模型
                    model_clone = clone(model)
                    model_clone.fit(X_fold_train, y_fold_train)
                    
                    # 预测并评估
                    y_fold_pred = model_clone.predict(X_fold_val)
                    fold_score = r2_score(y_fold_val, y_fold_pred)
                    
                    # 存储结果
                    fold_predictions.append(y_fold_pred)
                    fold_scores.append(fold_score)
                    
                    logger.debug(f"模型 {name}, 折 {fold+1}/{cv}, R2: {fold_score:.4f}")
                
                # 计算平均分数和标准差
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                
                cv_results[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'fold_scores': fold_scores,
                    'fold_predictions': fold_predictions
                }
                
                logger.info(f"模型 {name} 交叉验证完成: 平均R2={mean_score:.4f}, 标准差={std_score:.4f}")
            except Exception as e:
                logger.error(f"模型 {name} 交叉验证失败: {str(e)}")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name, param_grid, method='grid', cv=5, scoring='r2', n_iter=10):
        """
        对指定模型进行超参数调优
        
        参数:
            model_name (str): 模型名称
            param_grid (dict): 参数网格
            method (str): 调优方法，'grid'或'random'
            cv (int): 交叉验证折数
            scoring (str): 评分标准
            n_iter (int): 随机搜索的迭代次数
            
        返回:
            object: 调优后的最佳模型
        """
        logger.info(f"开始对模型 {model_name} 进行超参数调优，方法: {method}")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        
        # 创建搜索对象
        if method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            raise ValueError(f"不支持的调优方法: {method}")
        
        # 执行搜索
        search.fit(self.X_train, self.y_train)
        
        # 获取最佳参数和模型
        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_
        
        logger.info(f"超参数调优完成，最佳参数: {best_params}, 最佳分数: {best_score:.4f}")
        
        # 更新模型
        self.models[model_name] = best_model
        
        # 如果模型已经训练过，更新训练后的模型
        if model_name in self.trained_models:
            self.trained_models[model_name] = best_model
        
        return best_model, best_params, best_score
    
    def analyze_feature_importance(self, model_name=None):
        """
        分析特征重要性
        
        参数:
            model_name (str, optional): 模型名称，如果为None则使用最佳模型
            
        返回:
            pandas.DataFrame: 特征重要性DataFrame
        """
        logger.info("开始分析特征重要性")
        
        if not self.trained_models:
            logger.warning("没有训练好的模型，请先调用train_models方法")
            return None
        
        # 确定要分析的模型
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("最佳模型未确定，请先调用evaluate_models方法")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            logger.warning(f"模型 {model_name} 不存在或未训练")
            return None
        
        model = self.trained_models[model_name]
        
        # 获取特征名称
        feature_names = self.X_train.columns.tolist()
        
        # 获取特征重要性
        importance = None
        
        # 对于线性模型
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.mean([np.abs(model.coef_[i]) for i in range(model.coef_.shape[0])], axis=0)
        # 对于树模型
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        # 对于管道模型
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            importance = model.steps[-1][1].feature_importances_
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'coef_'):
            if len(model.steps[-1][1].coef_.shape) == 1:
                importance = np.abs(model.steps[-1][1].coef_)
            else:
                importance = np.mean([np.abs(model.steps[-1][1].coef_[i]) for i in range(model.steps[-1][1].coef_.shape[0])], axis=0)
        else:
            logger.warning(f"模型 {model_name} 不支持特征重要性分析")
            
            # 尝试使用排列重要性
            try:
                from sklearn.inspection import permutation_importance
                logger.info("尝试使用排列重要性分析")
                
                # 使用训练集或测试集
                if self.X_test is not None and self.y_test is not None:
                    X_perm = self.X_test
                    y_perm = self.y_test
                else:
                    X_perm = self.X_train
                    y_perm = self.y_train
                
                # 计算排列重要性
                perm_importance = permutation_importance(model, X_perm, y_perm, n_repeats=10, random_state=42)
                importance = perm_importance.importances_mean
            except Exception as e:
                logger.error(f"排列重要性分析失败: {str(e)}")
                return None
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        logger.info("特征重要性分析完成")
        return feature_importance
    
    def visualize_feature_importance(self, model_name=None, top_n=10, output_dir=None):
        """
        可视化特征重要性
        
        参数:
            model_name (str, optional): 模型名称，如果为None则使用最佳模型
            top_n (int): 显示前N个重要特征
            output_dir (str, optional): 输出目录
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在可视化特征重要性，显示前 {top_n} 个特征")
        
        # 获取特征重要性
        feature_importance = self.analyze_feature_importance(model_name)
        
        if feature_importance is None:
            return None
        
        # 确定要使用的模型名称（用于标题）
        if model_name is None:
            model_name = self.best_model_name
        
        # 调整top_n，确保不超过特征数量
        top_n = min(top_n, len(feature_importance))
        
        # 获取前N个重要特征
        top_features = feature_importance.head(top_n)
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        
        # 添加数值标签
        for i, v in enumerate(top_features['Importance']):
            ax.text(v, i, f'{v:.4f}', va='center')
        
        plt.title(f'Top {top_n} 特征重要性 ({model_name})', fontsize=15)
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图表已保存至: {output_file}")
        
        return plt.gcf()
    
    def visualize_predictions(self, model_name=None, output_dir=None):
        """
        可视化预测结果
        
        参数:
            model_name (str, optional): 模型名称，如果为None则使用最佳模型
            output_dir (str, optional): 输出目录
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在可视化预测结果")
        
        if not self.model_results:
            logger.warning("没有模型评估结果，请先调用evaluate_models方法")
            return None
        
        # 确定要使用的模型
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("最佳模型未确定，请先调用evaluate_models方法")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.model_results:
            logger.warning(f"模型 {model_name} 的评估结果不存在")
            return None
        
        # 获取真实值和预测值
        if self.X_test is None or self.y_test is None:
            y_true = self.y_train
        else:
            y_true = self.y_test
        
        y_pred = self.model_results[model_name]['predictions']
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 散点图：真实值 vs 预测值
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolor='k')
        
        # 添加对角线（完美预测）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax1.set_xlabel('真实成绩', fontsize=12)
        ax1.set_ylabel('预测成绩', fontsize=12)
        ax1.set_title(f'真实成绩 vs 预测成绩 ({model_name})', fontsize=15)
        
        # 添加R2分数
        r2 = self.model_results[model_name]['R2']
        rmse = self.model_results[model_name]['RMSE']
        mae = self.model_results[model_name]['MAE']
        ax1.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=12)
        
        # 误差直方图
        errors = y_true - y_pred
        ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('预测误差', fontsize=12)
        ax2.set_ylabel('频率', fontsize=12)
        ax2.set_title('预测误差分布', fontsize=15)
        
        # 添加误差统计信息
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        ax2.annotate(f'平均误差 = {mean_error:.4f}\n中位数误差 = {median_error:.4f}\n标准差 = {std_error:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=12)
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'predictions_{model_name.replace(" ", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存至: {output_file}")
        
        return fig
    
    def visualize_model_comparison(self, metric='R2', output_dir=None, sort_ascending=None):
        """
        可视化模型比较
        
        参数:
            metric (str): 比较指标，可选 'R2', 'RMSE', 'MSE', 'MAE'
            output_dir (str, optional): 输出目录
            sort_ascending (bool, optional): 是否升序排序，如果为None则根据指标自动决定
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在可视化模型比较，指标: {metric}")
        
        if not self.model_results:
            logger.warning("没有模型评估结果，请先调用evaluate_models方法")
            return None
        
        # 提取指标数据
        models = []
        scores = []
        training_times = []
        
        for name, result in self.model_results.items():
            if metric in result:
                models.append(name)
                scores.append(result[metric])
                training_times.append(result.get('training_time', 0))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Model': models,
            metric: scores,
            'Training Time (s)': training_times
        })
        
        # 确定排序方向
        if sort_ascending is None:
            # R2越高越好，其他指标越低越好
            sort_ascending = False if metric == 'R2' else True
        
        # 根据指标排序
        df = df.sort_values(metric, ascending=sort_ascending)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # 创建条形图 - 性能指标
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        bars = ax1.bar(df['Model'], df[metric], color=colors)
        
        # 添加数值标签
        for bar, score in zip(bars, df[metric]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(df[metric])),
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax1.set_title(f'模型比较 ({metric})', fontsize=16)
        ax1.set_ylabel(metric, fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 创建条形图 - 训练时间
        ax2.bar(df['Model'], df['Training Time (s)'], color=colors, alpha=0.7)
        ax2.set_title('模型训练时间 (秒)', fontsize=14)
        ax2.set_ylabel('时间 (秒)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'model_comparison_{metric}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"模型比较图表已保存至: {output_file}")
        
        return fig

    def save_model(self, model_name=None, output_dir=None):
        """
        保存模型
        
        参数:
            model_name (str, optional): 模型名称，如果为None则保存最佳模型
            output_dir (str): 输出目录
            
        返回:
            str: 模型文件路径
        """
        if output_dir is None:
            logger.warning("未指定输出目录，无法保存模型")
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 确定要保存的模型
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("最佳模型未确定，请先调用evaluate_models方法")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            logger.warning(f"模型 {model_name} 不存在或未训练")
            return None
        
        model = self.trained_models[model_name]
        
        # 保存模型
        model_file = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_model.pkl')
        joblib.dump(model, model_file)
        logger.info(f"模型已保存至: {model_file}")
        
        # 保存特征名称
        feature_names_file = os.path.join(output_dir, 'feature_names.pkl')
        joblib.dump(list(self.X_train.columns), feature_names_file)
        logger.info(f"特征名称已保存至: {feature_names_file}")
        
        # 保存预处理器（如果存在）
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            preprocessor_file = os.path.join(output_dir, 'preprocessor.pkl')
            joblib.dump(self.preprocessor, preprocessor_file)
            logger.info(f"预处理器已保存至: {preprocessor_file}")
        
        # 保存模型元数据
        metadata = {
            'model_name': model_name,
            'feature_count': len(self.X_train.columns),
            'training_samples': len(self.X_train),
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance': {
                'R2': self.model_results[model_name]['R2'],
                'RMSE': self.model_results[model_name]['RMSE'],
                'MAE': self.model_results[model_name]['MAE'],
                'MSE': self.model_results[model_name]['MSE']
            },
            'training_time': self.training_time.get(model_name, 0)
        }
        
        metadata_file = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"模型元数据已保存至: {metadata_file}")
        
        return model_file
    
    def save_results(self, output_dir):
        """
        保存评估结果
        
        参数:
            output_dir (str): 输出目录
            
        返回:
            str: 结果文件路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not self.model_results:
            logger.warning("没有模型评估结果，请先调用evaluate_models方法")
            return None
        
        # 提取评估指标
        results = []
        for name, result in self.model_results.items():
            results.append({
                'Model': name,
                'MSE': result['MSE'],
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
                'R2': result['R2'],
                'Training Time (s)': result.get('training_time', 0)
            })
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 按R2分数排序
        df = df.sort_values('R2', ascending=False)
        
        # 保存结果
        results_file = os.path.join(output_dir, 'model_evaluation_results.csv')
        df.to_csv(results_file, index=False)
        logger.info(f"评估结果已保存至: {results_file}")
        
        return results_file
    
    @staticmethod
    def load_model(model_path, feature_names_path=None, preprocessor_path=None):
        """
        加载已保存的模型
        
        参数:
            model_path (str): 模型文件路径
            feature_names_path (str, optional): 特征名称文件路径
            preprocessor_path (str, optional): 预处理器文件路径
            
        返回:
            tuple: (模型, 特征名称列表, 预处理器)
        """
        logger.info(f"正在加载模型: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载特征名称（如果提供）
        feature_names = None
        if feature_names_path and os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            logger.info(f"已加载特征名称，共 {len(feature_names)} 个特征")
        
        # 加载预处理器（如果提供）
        preprocessor = None
        if preprocessor_path and os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info("已加载预处理器")
        
        return model, feature_names, preprocessor
    
    def predict(self, X, model_name=None):
        """
        使用指定模型进行预测
        
        参数:
            X (DataFrame): 输入特征
            model_name (str, optional): 模型名称，如果为None则使用最佳模型
            
        返回:
            array: 预测结果
        """
        logger.info("正在进行预测")
        
        # 确定要使用的模型
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("最佳模型未确定，请先调用evaluate_models方法")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            logger.warning(f"模型 {model_name} 不存在或未训练")
            return None
        
        model = self.trained_models[model_name]
        
        # 进行预测
        try:
            predictions = model.predict(X)
            logger.info(f"预测完成，使用模型: {model_name}")
            return predictions
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None
    
    def plot_learning_curve(self, model_name=None, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), output_dir=None):
        """
        绘制学习曲线
        
        参数:
            model_name (str, optional): 模型名称，如果为None则使用最佳模型
            cv (int): 交叉验证折数
            train_sizes (array): 训练集大小比例
            output_dir (str, optional): 输出目录
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在绘制学习曲线")
        
        # 确定要使用的模型
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("最佳模型未确定，请先调用evaluate_models方法")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.models:
            logger.warning(f"模型 {model_name} 不存在")
            return None
        
        model = self.models[model_name]
        
        # 计算学习曲线
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, cv=cv, train_sizes=train_sizes, 
                scoring='r2', n_jobs=-1
            )
            
            # 计算平均值和标准差
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # 创建可视化
            plt.figure(figsize=(10, 6))
            
            # 绘制训练集和验证集分数
            plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练集分数')
            plt.plot(train_sizes, test_mean, 'o-', color='g', label='验证集分数')
            
            # 绘制标准差区域
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
            
            # 添加标签和标题
            plt.xlabel('训练样本数', fontsize=12)
            plt.ylabel('R²分数', fontsize=12)
            plt.title(f'学习曲线 ({model_name})', fontsize=15)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图表
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_file = os.path.join(output_dir, f'learning_curve_{model_name.replace(" ", "_")}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"学习曲线图表已保存至: {output_file}")
            
            return plt.gcf()
        except Exception as e:
            logger.error(f"绘制学习曲线失败: {str(e)}")
            return None
    
    def train_and_evaluate(self, X_train_path, y_train_path, X_test_path=None, y_test_path=None, 
                          output_dir=None, selected_models=None, preprocess=True, 
                          categorical_features=None, numerical_features=None, scaling='standard'):
        """
        完整的模型训练和评估流程
        
        参数:
            X_train_path (str): 训练特征文件路径
            y_train_path (str): 训练目标文件路径
            X_test_path (str, optional): 测试特征文件路径
            y_test_path (str, optional): 测试目标文件路径
            output_dir (str, optional): 输出目录
            selected_models (list, optional): 要训练的模型名称列表
            preprocess (bool): 是否预处理数据
            categorical_features (list, optional): 类别特征列表
            numerical_features (list, optional): 数值特征列表
            scaling (str): 缩放方法，'standard'或'minmax'
            
        返回:
            tuple: (最佳模型名称, 最佳模型, 评估结果)
        """
        logger.info("开始模型训练和评估流程")
        
        # 加载数据
        self.load_data(X_train_path, y_train_path, X_test_path, y_test_path)
        
        # 预处理数据（如果需要）
        if preprocess:
            self.preprocess_data(categorical_features, numerical_features, scaling)
        
        # 训练模型
        self.train_models(selected_models)
        
        # 评估模型
        self.evaluate_models(additional_metrics=True)
        
        # 可视化结果
        if output_dir:
            # 创建可视化目录
            vis_dir = os.path.join(output_dir, 'visualizations')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            
            # 特征重要性可视化
            self.visualize_feature_importance(output_dir=vis_dir)
            
            # 预测结果可视化
            self.visualize_predictions(output_dir=vis_dir)
            
            # 模型比较可视化
            self.visualize_model_comparison(metric='R2', output_dir=vis_dir)
            self.visualize_model_comparison(metric='RMSE', output_dir=vis_dir)
            
            # 学习曲线
            self.plot_learning_curve(output_dir=vis_dir)
            
            # 保存模型
            self.save_model(output_dir=output_dir)
            
            # 保存评估结果
            self.save_results(output_dir)
        
        logger.info("模型训练和评估流程完成")
        return self.best_model_name, self.best_model, self.model_results


def main():
    """
    主函数，用于测试增强版模型训练模块
    """
    # 设置数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_data_dir = os.path.join(data_dir, 'processed')
    model_dir = os.path.join(base_dir, 'model')
    
    # 特征工程后的特征文件
    features_file = os.path.join(processed_data_dir, 'feature_engineering', 'engineered_features.csv')
    
    # 如果特征工程后的文件不存在，使用处理后的原始特征
    if not os.path.exists(features_file):
        features_file = os.path.join(processed_data_dir, 'processed_features.csv')
    
    # 目标文件
    target_file = os.path.join(processed_data_dir, 'processed_target.csv')
    
    # 创建模型训练器
    trainer = EnhancedModelTrainer()
    
    # 训练和评估模型
    best_model_name, best_model, results = trainer.train_and_evaluate(
        features_file, target_file, output_dir=model_dir
    )
    
    # 打印最佳模型信息
    print(f"\n最佳模型: {best_model_name}")
    print(f"R2分数: {results[best_model_name]['R2']:.4f}")
    print(f"RMSE: {results[best_model_name]['RMSE']:.4f}")
    
    # 打印所有模型的R2分数
    print("\n所有模型的R2分数:")
    for name, result in sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True):
        print(f"{name}: {result['R2']:.4f}")


if __name__ == "__main__":
    main()
