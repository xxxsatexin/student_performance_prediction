#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据可视化模块
生成直观的图表和报告，帮助教师理解预测结果和影响因素
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import io
import base64

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    数据可视化类，负责生成各种图表和报告
    """
    
    def __init__(self, config=None):
        """
        初始化数据可视化器
        
        参数:
            config (dict, optional): 配置参数字典
        """
        self.config = config or {}
        self.data = None
        self.features = None
        self.target = None
        self.predictions = None
        self.model = None
        self.feature_importance = None
        
        # 设置可视化样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def load_data(self, data_path):
        """
        加载数据
        
        参数:
            data_path (str): 数据文件路径
            
        返回:
            pandas.DataFrame: 加载的数据
        """
        logger.info(f"正在加载数据: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            self.data = pd.read_csv(data_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            self.data = pd.read_excel(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        logger.info(f"成功加载数据，共 {len(self.data)} 条记录")
        return self.data
    
    def load_model_data(self, features_path, target_path=None, model_path=None):
        """
        加载模型相关数据
        
        参数:
            features_path (str): 特征数据文件路径
            target_path (str, optional): 目标数据文件路径
            model_path (str, optional): 模型文件路径
            
        返回:
            tuple: (特征, 目标变量, 模型)
        """
        logger.info(f"正在加载模型数据")
        
        # 加载特征
        self.features = pd.read_csv(features_path)
        logger.info(f"成功加载特征数据，共 {len(self.features)} 条记录，{self.features.shape[1]} 个特征")
        
        # 加载目标变量（如果提供）
        if target_path and os.path.exists(target_path):
            target_data = pd.read_csv(target_path)
            if isinstance(target_data, pd.DataFrame) and target_data.shape[1] == 1:
                self.target = target_data.iloc[:, 0]
            else:
                self.target = target_data
            logger.info(f"成功加载目标数据，共 {len(self.target)} 条记录")
        
        # 加载模型（如果提供）
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"成功加载模型: {type(self.model).__name__}")
            
            # 使用模型进行预测
            if self.features is not None:
                self.predictions = self.model.predict(self.features)
                logger.info(f"已生成预测结果，共 {len(self.predictions)} 条记录")
        
        return self.features, self.target, self.model
    
    def plot_grade_distribution(self, output_path=None):
        """
        绘制成绩分布图
        
        参数:
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在绘制成绩分布图")
        
        if self.target is None:
            logger.warning("目标变量未加载，无法绘制成绩分布图")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # 绘制成绩分布直方图
        sns.histplot(self.target, kde=True, bins=20)
        
        plt.title("学生成绩分布")
        plt.xlabel("成绩")
        plt.ylabel("频数")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息
        mean_val = self.target.mean()
        median_val = self.target.median()
        std_val = self.target.std()
        
        plt.axvline(mean_val, color='r', linestyle='--', label=f'平均值: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='--', label=f'中位数: {median_val:.2f}')
        
        plt.legend()
        
        # 添加统计文本
        stats_text = f"平均值: {mean_val:.2f}\n中位数: {median_val:.2f}\n标准差: {std_val:.2f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top')
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"成绩分布图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_feature_correlation(self, target_col='final_grade', top_n=10, output_path=None):
        """
        绘制特征与目标变量的相关性图
        
        参数:
            target_col (str): 目标列名
            top_n (int): 显示前N个相关性最高的特征
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在绘制特征相关性图，显示前 {top_n} 个特征")
        
        if self.data is None:
            logger.warning("数据未加载，无法绘制特征相关性图")
            return None
        
        if target_col not in self.data.columns:
            logger.warning(f"目标列 {target_col} 不在数据中")
            return None
        
        # 计算相关性
        corr = self.data.corr()[target_col].drop(target_col)
        
        # 按绝对值排序
        corr_abs = corr.abs().sort_values(ascending=False)
        
        # 获取前N个特征
        top_features = corr_abs.head(top_n).index
        corr_values = corr[top_features]
        
        plt.figure(figsize=(12, 8))
        
        # 创建条形图
        colors = ['green' if x > 0 else 'red' for x in corr_values]
        bars = plt.barh(top_features, corr_values, color=colors)
        
        # 添加数值标签
        for i, v in enumerate(corr_values):
            plt.text(v + 0.01 if v > 0 else v - 0.06, i, f'{v:.3f}', 
                    va='center', color='black' if v > 0 else 'white')
        
        plt.title(f"特征与{target_col}的相关性（前{top_n}个）")
        plt.xlabel("相关系数")
        plt.ylabel("特征")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        plt.legend([plt.Rectangle((0,0),1,1, color='green'), 
                   plt.Rectangle((0,0),1,1, color='red')],
                  ['正相关', '负相关'])
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"特征相关性图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_correlation_matrix(self, output_path=None):
        """
        绘制相关性矩阵热图
        
        参数:
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在绘制相关性矩阵热图")
        
        if self.data is None:
            logger.warning("数据未加载，无法绘制相关性矩阵热图")
            return None
        
        # 计算相关性矩阵
        corr_matrix = self.data.corr()
        
        # 创建掩码，只显示下三角部分
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(14, 12))
        
        # 绘制热图
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title("特征相关性矩阵")
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"相关性矩阵热图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, model=None, top_n=10, output_path=None):
        """
        绘制特征重要性图
        
        参数:
            model (object, optional): 模型对象，如果为None则使用已加载的模型
            top_n (int): 显示前N个重要特征
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在绘制特征重要性图，显示前 {top_n} 个特征")
        
        # 确定要使用的模型
        if model is None:
            model = self.model
        
        if model is None:
            logger.warning("模型未加载，无法绘制特征重要性图")
            return None
        
        if self.features is None:
            logger.warning("特征数据未加载，无法绘制特征重要性图")
            return None
        
        # 获取特征名称
        feature_names = self.features.columns
        
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
            logger.warning(f"模型 {type(model).__name__} 不支持特征重要性分析")
            return None
        
        # 创建特征重要性DataFrame
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # 按重要性排序
        self.feature_importance = self.feature_importance.sort_values('Importance', ascending=False)
        
        # 获取前N个重要特征
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # 创建条形图
        sns.barplot(x='Importance', y='Feature', data=top_features)
        
        plt.title(f"特征重要性（前{top_n}个）")
        plt.xlabel("重要性")
        plt.ylabel("特征")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"特征重要性图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_prediction_vs_actual(self, output_path=None):
        """
        绘制预测值与实际值对比图
        
        参数:
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在绘制预测值与实际值对比图")
        
        if self.predictions is None:
            logger.warning("预测结果未生成，无法绘制预测值与实际值对比图")
            return None
        
        if self.target is None:
            logger.warning("目标变量未加载，无法绘制预测值与实际值对比图")
            return None
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        plt.scatter(self.target, self.predictions, alpha=0.6)
        
        # 添加对角线（完美预测）
        min_val = min(self.target.min(), self.predictions.min())
        max_val = max(self.target.max(), self.predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title("预测成绩 vs 实际成绩")
        plt.xlabel("实际成绩")
        plt.ylabel("预测成绩")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 计算评估指标
        mse = mean_squared_error(self.target, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.target, self.predictions)
        r2 = r2_score(self.target, self.predictions)
        
        # 添加评估指标文本
        metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}"
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top')
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"预测值与实际值对比图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_residuals(self, output_path=None):
        """
        绘制残差图
        
        参数:
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info("正在绘制残差图")
        
        if self.predictions is None:
            logger.warning("预测结果未生成，无法绘制残差图")
            return None
        
        if self.target is None:
            logger.warning("目标变量未加载，无法绘制残差图")
            return None
        
        # 计算残差
        residuals = self.target - self.predictions
        
        plt.figure(figsize=(12, 10))
        
        # 创建子图
        plt.subplot(2, 1, 1)
        
        # 绘制残差散点图
        plt.scatter(self.predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.title("残差图")
        plt.xlabel("预测成绩")
        plt.ylabel("残差 (实际 - 预测)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制残差分布图
        plt.subplot(2, 1, 2)
        sns.histplot(residuals, kde=True, bins=20)
        
        plt.title("残差分布")
        plt.xlabel("残差")
        plt.ylabel("频数")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息
        mean_val = residuals.mean()
        std_val = residuals.std()
        
        plt.axvline(mean_val, color='r', linestyle='--', label=f'平均值: {mean_val:.2f}')
        
        plt.legend()
        
        # 添加统计文本
        stats_text = f"平均值: {mean_val:.2f}\n标准差: {std_val:.2f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top')
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"残差图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_feature_scatter(self, feature_name, target_col='final_grade', output_path=None):
        """
        绘制特征与目标变量的散点图
        
        参数:
            feature_name (str): 特征名称
            target_col (str): 目标列名
            output_path (str, optional): 输出文件路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        logger.info(f"正在绘制特征 {feature_name} 与 {target_col} 的散点图")
        
        if self.data is None:
            logger.warning("数据未加载，无法绘制特征散点图")
            return None
        
        if feature_name not in self.data.columns:
            logger.warning(f"特征 {feature_name} 不在数据中")
            return None
        
        if target_col not in self.data.columns:
            logger.warning(f"目标列 {target_col} 不在数据中")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # 绘制散点图
        sns.regplot(x=feature_name, y=target_col, data=self.data, 
                   scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        
        plt.title(f"{feature_name} vs {target_col}")
        plt.xlabel(feature_name)
        plt.ylabel(target_col)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 计算相关系数
        corr = self.data[[feature_name, target_col]].corr().iloc[0, 1]
        
        # 添加相关系数文本
        corr_text = f"相关系数: {corr:.3f}"
        plt.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top')
        
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path)
            logger.info(f"特征散点图已保存至: {output_path}")
        
        return plt.gcf()
    
    def plot_dashboard(self, output_dir=None):
        """
        生成仪表板图表
        
        参数:
            output_dir (str, optional): 输出目录
            
        返回:
            dict: 图表路径字典
        """
        logger.info("正在生成仪表板图表")
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成各种图表
        charts = {}
        
        # 成绩分布图
        if self.target is not None:
            output_path = os.path.join(output_dir, 'grade_distribution.png') if output_dir else None
            charts['grade_distribution'] = self.plot_grade_distribution(output_path)
        
        # 特征重要性图
        if self.model is not None and self.features is not None:
            output_path = os.path.join(output_dir, 'feature_importance.png') if output_dir else None
            charts['feature_importance'] = self.plot_feature_importance(output_path=output_path)
        
        # 预测值与实际值对比图
        if self.predictions is not None and self.target is not None:
            output_path = os.path.join(output_dir, 'prediction_vs_actual.png') if output_dir else None
            charts['prediction_vs_actual'] = self.plot_prediction_vs_actual(output_path)
        
        # 残差图
        if self.predictions is not None and self.target is not None:
            output_path = os.path.join(output_dir, 'residuals.png') if output_dir else None
            charts['residuals'] = self.plot_residuals(output_path)
        
        # 相关性矩阵热图
        if self.data is not None:
            output_path = os.path.join(output_dir, 'correlation_matrix.png') if output_dir else None
            charts['correlation_matrix'] = self.plot_correlation_matrix(output_path)
        
        logger.info(f"仪表板图表生成完成，共 {len(charts)} 个图表")
        return charts
    
    def get_base64_chart(self, fig):
        """
        将图表转换为base64编码
        
        参数:
            fig: matplotlib图表对象
            
        返回:
            str: base64编码的图表
        """
        if fig is None:
            return None
        
        # 将图表转换为base64编码
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # 关闭图表，释放内存
        plt.close(fig)
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def get_dashboard_base64(self):
        """
        获取仪表板图表的base64编码
        
        返回:
            dict: 图表base64编码字典
        """
        logger.info("正在获取仪表板图表的base64编码")
        
        # 生成图表
        charts = self.plot_dashboard()
        
        # 转换为base64编码
        base64_charts = {}
        for name, fig in charts.items():
            base64_charts[name] = self.get_base64_chart(fig)
        
        return base64_charts


def main():
    """
    主函数，用于测试数据可视化模块
    """
    # 设置数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_data_path = os.path.join(data_dir, 'raw', 'student_data.csv')
    processed_data_dir = os.path.join(data_dir, 'processed')
    model_dir = os.path.join(base_dir, 'model')
    vis_dir = os.path.join(base_dir, 'static', 'images')
    
    # 创建可视化目录
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 创建数据可视化器
    visualizer = DataVisualizer()
    
    # 加载原始数据
    visualizer.load_data(raw_data_path)
    
    # 生成基础图表
    visualizer.plot_grade_distribution(os.path.join(vis_dir, 'grade_distribution.png'))
    visualizer.plot_feature_correlation(output_path=os.path.join(vis_dir, 'feature_correlation.png'))
    visualizer.plot_correlation_matrix(output_path=os.path.join(vis_dir, 'correlation_matrix.png'))
    
    # 尝试加载模型数据（如果存在）
    model_file = None
    for file in os.listdir(model_dir):
        if file.endswith('_model.pkl'):
            model_file = os.path.join(model_dir, file)
            break
    
    if model_file:
        features_file = os.path.join(processed_data_dir, 'processed_features.csv')
        target_file = os.path.join(processed_data_dir, 'processed_target.csv')
        
        # 加载模型数据
        visualizer.load_model_data(features_file, target_file, model_file)
        
        # 生成模型相关图表
        visualizer.plot_feature_importance(output_path=os.path.join(vis_dir, 'feature_importance.png'))
        visualizer.plot_prediction_vs_actual(output_path=os.path.join(vis_dir, 'prediction_vs_actual.png'))
        visualizer.plot_residuals(output_path=os.path.join(vis_dir, 'residuals.png'))
    
    print("\n数据可视化完成!")
    print(f"图表已保存至: {vis_dir}")


if __name__ == "__main__":
    main()
