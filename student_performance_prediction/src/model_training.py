#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练与评估模块
使用多种机器学习算法构建预测模型，并评估其性能
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    设置模型训练所需的特征和标签数据
    """
    def set_data(self, X, y):

        self.X = X
        self.y = y
    """
    模型训练类，负责训练、评估和选择最佳模型
    """
    
    def __init__(self,cv_folds, random_state, test_size, config=None):
        """
        初始化模型训练器
        
        参数:
            config (dict, optional): 配置参数字典
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.test_size = test_size
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
        
        # 初始化默认模型
        self._init_models()
    
    def _init_models(self):
        """
        初始化模型字典
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
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
    
    def add_model(self, model, model_name):
        """
        添加自定义模型
        
        参数:
            name (str): 模型名称
            model: 模型对象
            
        返回:
            dict: 更新后的模型字典
        """
        logger.info(f"添加模型: {model_name}")
        self.models[model_name] = model
        return self.models
    
    def remove_model(self, name):
        """
        移除模型
        
        参数:
            name (str): 模型名称
            
        返回:
            dict: 更新后的模型字典
        """
        if name in self.models:
            logger.info(f"移除模型: {name}")
            del self.models[name]
        else:
            logger.warning(f"模型 {name} 不存在")
        
        return self.models
    
    def train_models(self):
        """
        训练所有模型
        
        返回:
            dict: 训练后的模型字典
        """
        logger.info("开始训练模型")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        self.trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"正在训练模型: {name}")
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                logger.info(f"模型 {name} 训练完成")
            except Exception as e:
                logger.error(f"模型 {name} 训练失败: {str(e)}")
        
        logger.info(f"所有模型训练完成，共 {len(self.trained_models)} 个模型")
        return self.trained_models
    
    def evaluate_models(self):
        """
        评估所有训练好的模型
        
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
                
                # 计算评估指标
                mse = mean_squared_error(y_eval, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_eval, y_pred)
                r2 = r2_score(y_eval, y_pred)
                
                # 保存结果
                self.model_results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"模型 {name} 评估完成: RMSE={rmse:.4f}, R2={r2:.4f}")
            except Exception as e:
                logger.error(f"模型 {name} 评估失败: {str(e)}")
        
        # 根据R2分数选择最佳模型
        if self.model_results:
            self.best_model_name = max(self.model_results, key=lambda x: self.model_results[x]['R2'])
            self.best_model = self.trained_models[self.best_model_name]
            logger.info(f"最佳模型: {self.best_model_name}, R2={self.model_results[self.best_model_name]['R2']:.4f}")
        
        return self.model_results
    
    def cross_validate_models(self, cv=5, scoring='r2'):
        """
        对模型进行交叉验证
        
        参数:
            cv (int): 交叉验证折数
            scoring (str): 评分标准
            
        返回:
            dict: 交叉验证结果
        """
        logger.info(f"开始交叉验证，折数: {cv}, 评分标准: {scoring}")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"正在对模型 {name} 进行交叉验证")
            
            try:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                logger.info(f"模型 {name} 交叉验证完成: 平均分数={scores.mean():.4f}, 标准差={scores.std():.4f}")
            except Exception as e:
                logger.error(f"模型 {name} 交叉验证失败: {str(e)}")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name, param_grid, cv=5, scoring='r2'):
        """
        对指定模型进行超参数调优
        
        参数:
            model_name (str): 模型名称
            param_grid (dict): 参数网格
            cv (int): 交叉验证折数
            scoring (str): 评分标准
            
        返回:
            object: 调优后的最佳模型
        """
        logger.info(f"开始对模型 {model_name} 进行超参数调优")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据未加载，请先调用load_data方法")
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        
        # 创建网格搜索对象
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # 执行网格搜索
        grid_search.fit(self.X_train, self.y_train)
        
        # 获取最佳参数和模型
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        logger.info(f"超参数调优完成，最佳参数: {best_params}, 最佳分数: {best_score:.4f}")
        
        # 更新模型
        self.models[model_name] = best_model
        
        return best_model
    
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
        feature_names = self.X_train.columns
        
        # 获取特征重要性
        importance = None
        
        # 对于线性模型
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_[0])
        # 对于树模型
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            logger.warning(f"模型 {model_name} 不支持特征重要性分析")
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
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} 特征重要性 ({model_name})')
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_")}.png')
            plt.savefig(output_file)
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
        plt.figure(figsize=(10, 6))
        
        # 散点图：真实值 vs 预测值
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # 添加对角线（完美预测）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('真实成绩')
        plt.ylabel('预测成绩')
        plt.title(f'真实成绩 vs 预测成绩 ({model_name})')
        
        # 添加R2分数
        r2 = self.model_results[model_name]['R2']
        rmse = self.model_results[model_name]['RMSE']
        plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'predictions_{model_name.replace(" ", "_")}.png')
            plt.savefig(output_file)
            logger.info(f"预测结果图表已保存至: {output_file}")
        
        return plt.gcf()
    
    def visualize_model_comparison(self, metric='R2', output_dir=None):
        """
        可视化模型比较
        
        参数:
            metric (str): 比较指标，可选 'R2', 'RMSE', 'MSE', 'MAE'
            output_dir (str, optional): 输出目录
            
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
        
        for name, result in self.model_results.items():
            if metric in result:
                models.append(name)
                scores.append(result[metric])
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Model': models,
            metric: scores
        })
        
        # 根据指标排序
        if metric == 'R2':  # R2越高越好
            df = df.sort_values(metric, ascending=False)
        else:  # 其他指标越低越好
            df = df.sort_values(metric, ascending=True)
        
        # 创建可视化
        plt.figure(figsize=(12, 6))
        
        # 创建条形图
        ax = sns.barplot(x='Model', y=metric, data=df)
        
        # 添加数值标签
        for i, v in enumerate(scores):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if metric == 'R2' else 'top')
        
        plt.title(f'模型比较 ({metric})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, f'model_comparison_{metric}.png')
            plt.savefig(output_file)
            logger.info(f"模型比较图表已保存至: {output_file}")
        
        return plt.gcf()
    
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
                'R2': result['R2']
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
    
    def train_and_evaluate(self, X_train_path, y_train_path, X_test_path=None, y_test_path=None, output_dir=None):
        """
        完整的模型训练和评估流程
        
        参数:
            X_train_path (str): 训练特征文件路径
            y_train_path (str): 训练目标文件路径
            X_test_path (str, optional): 测试特征文件路径
            y_test_path (str, optional): 测试目标文件路径
            output_dir (str, optional): 输出目录
            
        返回:
            tuple: (最佳模型名称, 最佳模型, 评估结果)
        """
        logger.info("开始模型训练和评估流程")
        
        # 加载数据
        self.load_data(X_train_path, y_train_path, X_test_path, y_test_path)
        
        # 训练模型
        self.train_models()
        
        # 评估模型
        self.evaluate_models()
        
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
            
            # 保存模型
            self.save_model(output_dir=output_dir)
            
            # 保存评估结果
            self.save_results(output_dir)
        
        logger.info("模型训练和评估流程完成")
        return self.best_model_name, self.best_model, self.model_results


def main():
    """
    主函数，用于测试模型训练模块
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
    trainer = ModelTrainer()
    
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
