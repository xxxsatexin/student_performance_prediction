#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据收集与预处理模块
负责从各种来源收集学生数据，并进行清洗、标准化和特征工程。
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    学生数据处理类，负责数据加载、清洗、转换和特征工程
    """
    
    def __init__(self, config=None):
        """
        初始化数据处理器
        
        参数:
            config (dict, optional): 配置参数字典
        """
        self.config = config or {}
        self.data = None
        self.features = None
        self.target = None
        self.categorical_features = ['gender', 'parent_education', 'school']
        self.numerical_features = ['age', 'study_time', 'classes_attended', 'total_classes', 
                                  'assignments_completed', 'total_assignments', 'previous_grade',
                                  'extracurricular_activities', 'study_regularity']
        self.target_column = 'final_grade'
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, file_path):
        """
        加载学生数据
        
        参数:
            file_path (str): 数据文件路径
            
        返回:
            pandas.DataFrame: 加载的数据
        """
        logger.info(f"正在加载数据: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            self.data = pd.read_csv(file_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            self.data = pd.read_excel(file_path)
        elif file_ext == '.json':
            self.data = pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        logger.info(f"成功加载数据，共 {len(self.data)} 条记录")
        return self.data
    
    def handle_missing_values(self):
        """
        处理缺失值
        
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("正在处理缺失值")
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"发现缺失值:\n{missing_values[missing_values > 0]}")
            
            # 对数值型特征使用均值填充
            numerical_data = self.data[self.numerical_features]
            self.data[self.numerical_features] = self.imputer.fit_transform(numerical_data)
            
            # 对分类特征使用众数填充
            for feature in self.categorical_features:
                if feature in self.data.columns:
                    mode_value = self.data[feature].mode()[0]
                    self.data[feature].fillna(mode_value, inplace=True)
        
        logger.info("缺失值处理完成")
        return self.data
    
    def encode_categorical_features(self):
        """
        对分类特征进行编码
        
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("正在编码分类特征")
        
        # 检查哪些分类特征存在于数据中
        available_categorical = [f for f in self.categorical_features if f in self.data.columns]
        
        if available_categorical:
            # 使用OneHotEncoder进行编码
            categorical_data = self.data[available_categorical]
            encoded_data = self.encoder.fit_transform(categorical_data)
            
            # 获取编码后的特征名称
            encoded_feature_names = []
            for i, feature in enumerate(available_categorical):
                categories = self.encoder.categories_[i][1:]  # 跳过第一个类别（drop='first'）
                for category in categories:
                    encoded_feature_names.append(f"{feature}_{category}")
            
            # 创建编码后的DataFrame
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=self.data.index)
            
            # 将编码后的特征与原始数据合并
            self.data = pd.concat([self.data.drop(available_categorical, axis=1), encoded_df], axis=1)
            
            logger.info(f"分类特征编码完成，新增 {len(encoded_feature_names)} 个特征")
        else:
            logger.info("没有找到需要编码的分类特征")
        
        return self.data
    
    def create_features(self):
        """
        创建新特征
        
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("正在创建新特征")
        
        # 计算出勤率
        if 'classes_attended' in self.data.columns and 'total_classes' in self.data.columns:
            self.data['attendance_rate'] = self.data['classes_attended'] / self.data['total_classes']
            logger.info("已创建特征: attendance_rate")
        
        # 计算作业完成率
        if 'assignments_completed' in self.data.columns and 'total_assignments' in self.data.columns:
            self.data['assignment_completion_rate'] = self.data['assignments_completed'] / self.data['total_assignments']
            logger.info("已创建特征: assignment_completion_rate")
        
        # 计算学习时间与成绩的比率（学习效率）
        if 'study_time' in self.data.columns and 'previous_grade' in self.data.columns:
            self.data['study_efficiency'] = self.data['previous_grade'] / (self.data['study_time'] + 1)  # 避免除零
            logger.info("已创建特征: study_efficiency")
        
        # 创建学习规律性指标
        if 'study_regularity' in self.data.columns and 'extracurricular_activities' in self.data.columns:
            self.data['balance_score'] = self.data['study_regularity'] - 0.5 * self.data['extracurricular_activities']
            logger.info("已创建特征: balance_score")
        
        return self.data
    
    def scale_numerical_features(self):
        """
        对数值特征进行标准化
        
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("正在标准化数值特征")
        
        # 获取当前所有的数值特征（包括新创建的特征）
        current_numerical = [col for col in self.data.columns 
                            if col != self.target_column and self.data[col].dtype in ['int64', 'float64']]
        
        if current_numerical:
            numerical_data = self.data[current_numerical]
            self.data[current_numerical] = self.scaler.fit_transform(numerical_data)
            logger.info(f"已标准化 {len(current_numerical)} 个数值特征")
        else:
            logger.info("没有找到需要标准化的数值特征")
        
        return self.data
    
    def prepare_features_target(self):
        """
        准备特征和目标变量
        
        返回:
            tuple: (特征, 目标变量)
        """
        logger.info("正在准备特征和目标变量")
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据中")
        
        self.target = self.data[self.target_column]
        self.features = self.data.drop(self.target_column, axis=1)
        
        logger.info(f"特征准备完成，共 {self.features.shape[1]} 个特征")
        return self.features, self.target
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        参数:
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        返回:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"正在划分训练集和测试集，测试集比例: {test_size}")
        
        if self.features is None or self.target is None:
            self.prepare_features_target()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"数据集划分完成，训练集: {X_train.shape[0]} 条记录，测试集: {X_test.shape[0]} 条记录")
        return X_train, X_test, y_train, y_test

    def set_columns(self, feature_cols=None, target_column=None, categorical_columns=None):
        """
        设置特征列和标签列
        :param feature_cols: 特征列列表（可选）
        :param target_col: 标签列名称（可选）
        :param categorical_columns: 分类列列表（可选）
        """
        self.feature_cols = feature_cols
        self.target_col = target_column
        self.categorical_columns = categorical_columns
        print("数据列已设置：特征列={}, 标签列={}".format(feature_cols, target_column))

    def save_processed_data(self, output_dir, prefix='processed'):
        """
        保存处理后的数据
        
        参数:
            output_dir (str): 输出目录
            prefix (str): 文件名前缀
            
        返回:
            tuple: (特征文件路径, 目标文件路径)
        """
        logger.info(f"正在保存处理后的数据到: {output_dir}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if self.features is None or self.target is None:
            self.prepare_features_target()
        
        # 保存特征
        features_file = os.path.join(output_dir, f"{prefix}_features.csv")
        self.features.to_csv(features_file, index=False)
        
        # 保存目标变量
        target_file = os.path.join(output_dir, f"{prefix}_target.csv")
        self.target.to_csv(target_file, index=False, header=True)
        
        # 保存特征名称
        feature_names_file = os.path.join(output_dir, f"{prefix}_feature_names.txt")
        with open(feature_names_file, 'w') as f:
            f.write('\n'.join(self.features.columns))
        
        # 保存缩放器
        import joblib
        scaler_file = os.path.join(output_dir, f"{prefix}_scaler.pkl")
        joblib.dump(self.scaler, scaler_file)
        
        logger.info(f"数据保存完成，特征文件: {features_file}，目标文件: {target_file}")
        return features_file, target_file
    
    def process_data(self, file_path, output_dir=None, save=True):
        """
        完整的数据处理流程
        
        参数:
            file_path (str): 数据文件路径
            output_dir (str, optional): 输出目录
            save (bool): 是否保存处理后的数据
            
        返回:
            tuple: (特征, 目标变量) 或 (X_train, X_test, y_train, y_test)
        """
        logger.info("开始数据处理流程")
        
        # 加载数据
        self.load_data(file_path)
        
        # 处理缺失值
        self.handle_missing_values()
        
        # 编码分类特征
        self.encode_categorical_features()
        
        # 创建新特征
        self.create_features()
        
        # 标准化数值特征
        self.scale_numerical_features()
        
        # 准备特征和目标变量
        self.prepare_features_target()
        
        # 保存处理后的数据
        if save and output_dir:
            self.save_processed_data(output_dir)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = self.split_train_test()
        
        logger.info("数据处理流程完成")
        return X_train, X_test, y_train, y_test

    def clean_data(self, df):
        """
        数据清洗核心方法：处理缺失值、重复值、异常值等
        :param df: 待清洗的DataFrame数据
        :return: 清洗后的DataFrame
        """
        # 1. 处理重复值
        df: object = df.drop_duplicates()

        # 2. 处理缺失值（可根据业务调整策略，这里示例为数值列填充均值，分类列填充众数）
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())  # 数值列填充均值
            else:
                df[col] = df[col].fillna(df[col].mode()[0])  # 分类列填充众数

        # 3. 可选：处理异常值（示例为数值列按3σ原则过滤）
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

        # 保存清洗后的数据到实例属性（方便后续调用）
        self.cleaned_df = df
        print("数据清洗完成，已移除重复值、填充缺失值、过滤异常值")
        return df


def main():
    """
    主函数，用于测试数据处理模块
    """
    # 设置数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_data_path = os.path.join(data_dir, 'raw', 'student_data.csv')
    processed_data_dir = os.path.join(data_dir, 'processed')
    
    # 创建数据处理器
    processor = DataProcessor()
    
    # 处理数据
    X_train, X_test, y_train, y_test = processor.process_data(
        raw_data_path, processed_data_dir, save=True
    )
    
    # 打印处理结果
    print("\n数据处理完成!")
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集目标变量形状: {y_train.shape}")
    print(f"测试集目标变量形状: {y_test.shape}")
    
    # 打印特征列表
    print("\n特征列表:")
    for i, feature in enumerate(X_train.columns):
        print(f"{i+1}. {feature}")

    # 创建DataProcessor实例
    processor = DataProcessor()
    # 打印该实例的所有可用方法和属性
    print(dir(processor))


if __name__ == "__main__":
    main()
