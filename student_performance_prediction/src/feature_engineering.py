#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程模块
负责创建和选择对学生成绩预测有价值的特征
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    特征工程类，负责特征创建、选择和可视化
    """
    
    def __init__(self, config=None):
        """
        初始化特征工程器
        
        参数:
            config (dict, optional): 配置参数字典
        """
        self.config = config or {}
        self.features = None
        self.target = None
        self.selected_features = None
        self.feature_importances = None
    
    def load_data(self, features_path, target_path=None):
        """
        加载特征和目标数据
        
        参数:
            features_path (str): 特征数据文件路径
            target_path (str, optional): 目标数据文件路径
            
        返回:
            tuple: (特征, 目标变量)
        """
        logger.info(f"正在加载特征数据: {features_path}")
        self.features = pd.read_csv(features_path)
        
        if target_path:
            logger.info(f"正在加载目标数据: {target_path}")
            self.target = pd.read_csv(target_path)
            # 如果目标是DataFrame，转换为Series
            if isinstance(self.target, pd.DataFrame) and self.target.shape[1] == 1:
                self.target = self.target.iloc[:, 0]
        
        logger.info(f"数据加载完成，特征形状: {self.features.shape}")
        if self.target is not None:
            logger.info(f"目标变量形状: {self.target.shape}")
        
        return self.features, self.target
    
    def create_interaction_features(self):
        """
        创建特征交互项
        
        返回:
            pandas.DataFrame: 包含交互特征的数据
        """
        logger.info("正在创建特征交互项")
        
        # 选择数值型特征进行交互
        numerical_features = self.features.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 定义要创建交互项的特征对
        interaction_pairs = [
            ('study_time', 'previous_grade'),
            ('attendance_rate', 'assignment_completion_rate'),
            ('study_efficiency', 'balance_score')
        ]
        
        # 创建交互特征
        for feature1, feature2 in interaction_pairs:
            if feature1 in self.features.columns and feature2 in self.features.columns:
                # 乘积交互
                interaction_name = f"{feature1}_x_{feature2}"
                self.features[interaction_name] = self.features[feature1] * self.features[feature2]
                logger.info(f"已创建交互特征: {interaction_name}")
                
                # 比率交互（避免除零）
                ratio_name = f"{feature1}_div_{feature2}"
                self.features[ratio_name] = self.features[feature1] / (self.features[feature2] + 1e-5)
                logger.info(f"已创建交互特征: {ratio_name}")
        
        logger.info(f"特征交互项创建完成，当前特征数量: {self.features.shape[1]}")
        return self.features
    
    def create_polynomial_features(self, degree=2):
        """
        创建多项式特征
        
        参数:
            degree (int): 多项式的次数
            
        返回:
            pandas.DataFrame: 包含多项式特征的数据
        """
        logger.info(f"正在创建{degree}次多项式特征")
        
        # 选择适合创建多项式的特征
        poly_candidates = ['study_time', 'previous_grade', 'attendance_rate', 
                          'assignment_completion_rate', 'study_efficiency']
        
        # 筛选存在的特征
        poly_features = [f for f in poly_candidates if f in self.features.columns]
        
        # 为每个特征创建多项式
        for feature in poly_features:
            for d in range(2, degree + 1):
                poly_name = f"{feature}_pow{d}"
                self.features[poly_name] = self.features[feature] ** d
                logger.info(f"已创建多项式特征: {poly_name}")
        
        logger.info(f"多项式特征创建完成，当前特征数量: {self.features.shape[1]}")
        return self.features
    
    def create_binned_features(self, n_bins=5):
        """
        创建分箱特征
        
        参数:
            n_bins (int): 分箱数量
            
        返回:
            pandas.DataFrame: 包含分箱特征的数据
        """
        logger.info(f"正在创建分箱特征，分箱数量: {n_bins}")
        
        # 选择适合分箱的特征
        bin_candidates = ['study_time', 'previous_grade', 'attendance_rate', 
                         'assignment_completion_rate']
        
        # 筛选存在的特征
        bin_features = [f for f in bin_candidates if f in self.features.columns]
        
        # 为每个特征创建分箱
        for feature in bin_features:
            bin_name = f"{feature}_bin"
            self.features[bin_name] = pd.qcut(
                self.features[feature], 
                q=n_bins, 
                labels=False, 
                duplicates='drop'
            )
            logger.info(f"已创建分箱特征: {bin_name}")
        
        logger.info(f"分箱特征创建完成，当前特征数量: {self.features.shape[1]}")
        return self.features
    
    def select_features_correlation(self, threshold=0.1):
        """
        基于相关性选择特征
        
        参数:
            threshold (float): 相关性阈值
            
        返回:
            pandas.DataFrame: 选择后的特征
        """
        logger.info(f"正在基于相关性选择特征，阈值: {threshold}")
        
        if self.target is None:
            logger.warning("目标变量未设置，无法计算相关性")
            return self.features
        
        # 计算与目标变量的相关性
        corr = pd.DataFrame()
        corr['feature'] = self.features.columns
        corr['correlation'] = [abs(np.corrcoef(self.features[f], self.target)[0, 1]) 
                              for f in self.features.columns]
        
        # 按相关性排序
        corr = corr.sort_values('correlation', ascending=False)
        
        # 选择相关性高于阈值的特征
        selected_features = corr[corr['correlation'] > threshold]['feature'].tolist()
        
        logger.info(f"相关性选择完成，选择了 {len(selected_features)}/{self.features.shape[1]} 个特征")
        
        # 保存特征重要性
        self.feature_importances = corr
        
        # 更新选择的特征
        self.selected_features = self.features[selected_features]
        
        return self.selected_features
    
    def select_features_mutual_info(self, k=10):
        """
        基于互信息选择特征
        
        参数:
            k (int): 选择的特征数量
            
        返回:
            pandas.DataFrame: 选择后的特征
        """
        logger.info(f"正在基于互信息选择特征，选择数量: {k}")
        
        if self.target is None:
            logger.warning("目标变量未设置，无法计算互信息")
            return self.features
        
        # 调整k，确保不超过特征数量
        k = min(k, self.features.shape[1])
        
        # 使用互信息选择特征
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(self.features, self.target)
        
        # 获取特征得分
        feature_scores = pd.DataFrame()
        feature_scores['feature'] = self.features.columns
        feature_scores['score'] = selector.scores_
        
        # 按得分排序
        feature_scores = feature_scores.sort_values('score', ascending=False)
        
        # 选择得分最高的k个特征
        selected_features = feature_scores.head(k)['feature'].tolist()
        
        logger.info(f"互信息选择完成，选择了 {len(selected_features)}/{self.features.shape[1]} 个特征")
        
        # 保存特征重要性
        self.feature_importances = feature_scores
        
        # 更新选择的特征
        self.selected_features = self.features[selected_features]
        
        return self.selected_features
    
    def apply_pca(self, n_components=0.95):
        """
        应用PCA降维
        
        参数:
            n_components (float or int): 如果是小数，表示保留的方差比例；如果是整数，表示保留的主成分数量
            
        返回:
            pandas.DataFrame: PCA转换后的特征
        """
        logger.info(f"正在应用PCA降维，n_components: {n_components}")
        
        # 创建PCA对象
        pca = PCA(n_components=n_components)
        
        # 应用PCA
        pca_result = pca.fit_transform(self.features)
        
        # 创建包含主成分的DataFrame
        if isinstance(n_components, int):
            columns = [f'PC{i+1}' for i in range(n_components)]
        else:
            columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        
        pca_df = pd.DataFrame(pca_result, columns=columns)
        
        logger.info(f"PCA降维完成，从 {self.features.shape[1]} 个特征降至 {pca_df.shape[1]} 个主成分")
        logger.info(f"解释方差比例: {pca.explained_variance_ratio_}")
        
        # 更新选择的特征
        self.selected_features = pca_df
        
        return self.selected_features
    
    def visualize_feature_importance(self, top_n=10, output_dir=None):
        """
        可视化特征重要性
        
        参数:
            top_n (int): 显示前N个重要特征
            output_dir (str, optional): 输出目录
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在可视化特征重要性，显示前 {top_n} 个特征")
        
        if self.feature_importances is None:
            logger.warning("特征重要性未计算，请先运行特征选择方法")
            return None
        
        # 调整top_n，确保不超过特征数量
        top_n = min(top_n, len(self.feature_importances))
        
        # 获取前N个重要特征
        top_features = self.feature_importances.head(top_n)
        
        # 创建可视化
        plt.figure(figsize=(10, 6))
        
        # 确定要绘制的列名
        if 'correlation' in top_features.columns:
            value_col = 'correlation'
            title = '特征相关性'
        else:
            value_col = 'score'
            title = '特征重要性分数'
        
        # 绘制条形图
        sns.barplot(x=value_col, y='feature', data=top_features)
        plt.title(f'Top {top_n} {title}')
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(output_file)
            logger.info(f"特征重要性图表已保存至: {output_file}")
        
        return plt.gcf()
    
    def visualize_correlation_matrix(self, output_dir=None):
        """
        可视化特征相关性矩阵
        
        参数:
            output_dir (str, optional): 输出目录
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在可视化特征相关性矩阵")
        
        # 计算相关性矩阵
        if self.selected_features is not None:
            corr_matrix = self.selected_features.corr()
        else:
            corr_matrix = self.features.corr()
        
        # 创建可视化
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, 'correlation_matrix.png')
            plt.savefig(output_file)
            logger.info(f"相关性矩阵图表已保存至: {output_file}")
        
        return plt.gcf()
    
    def save_features(self, output_path):
        """
        保存选择的特征
        
        参数:
            output_path (str): 输出文件路径
            
        返回:
            str: 输出文件路径
        """
        logger.info(f"正在保存特征至: {output_path}")
        
        # 确定要保存的特征
        features_to_save = self.selected_features if self.selected_features is not None else self.features
        
        # 保存特征
        features_to_save.to_csv(output_path, index=False)
        
        logger.info(f"特征已保存，形状: {features_to_save.shape}")
        return output_path
    
    def process_features(self, features_path, target_path=None, output_dir=None):
        """
        完整的特征工程流程
        
        参数:
            features_path (str): 特征数据文件路径
            target_path (str, optional): 目标数据文件路径
            output_dir (str, optional): 输出目录
            
        返回:
            pandas.DataFrame: 处理后的特征
        """
        logger.info("开始特征工程流程")
        
        # 加载数据
        self.load_data(features_path, target_path)
        
        # 创建交互特征
        self.create_interaction_features()
        
        # 创建多项式特征
        self.create_polynomial_features(degree=2)
        
        # 创建分箱特征
        self.create_binned_features(n_bins=5)
        
        # 基于互信息选择特征
        if self.target is not None:
            self.select_features_mutual_info(k=15)
            
            # 可视化特征重要性
            if output_dir:
                self.visualize_feature_importance(output_dir=output_dir)
                self.visualize_correlation_matrix(output_dir=output_dir)
        
        # 保存处理后的特征
        if output_dir:
            output_path = os.path.join(output_dir, 'engineered_features.csv')
            self.save_features(output_path)
        
        logger.info("特征工程流程完成")
        return self.selected_features if self.selected_features is not None else self.features


def main():
    """
    主函数，用于测试特征工程模块
    """
    # 设置数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_data_dir = os.path.join(data_dir, 'processed')
    features_path = os.path.join(processed_data_dir, 'processed_features.csv')
    target_path = os.path.join(processed_data_dir, 'processed_target.csv')
    output_dir = os.path.join(processed_data_dir, 'feature_engineering')
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 处理特征
    processed_features = engineer.process_features(features_path, target_path, output_dir)
    
    # 打印处理结果
    print("\n特征工程完成!")
    print(f"处理后的特征形状: {processed_features.shape}")
    
    # 打印特征列表
    print("\n处理后的特征列表:")
    for i, feature in enumerate(processed_features.columns):
        print(f"{i+1}. {feature}")


if __name__ == "__main__":
    main()
