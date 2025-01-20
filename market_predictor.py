import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime, timedelta
import joblib

# 定义目录
DATA_DIR = 'market_data'
MODEL_DIR = 'market_models'  # 存放模型和预测结果的目录

def ensure_dirs():
    """确保所有必要的目录都存在"""
    for dir_path in [DATA_DIR, MODEL_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

def get_current_version():
    """获取当前版本号（年月格式）"""
    return datetime.now().strftime("%Y_%m")

def get_latest_data_file():
    """获取最新的数据文件"""
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith('market_data_') and f.endswith('.csv')]
    if not data_files:
        raise FileNotFoundError("未找到数据文件")
    return os.path.join(DATA_DIR, sorted(data_files)[-1])

def get_prediction_month():
    """获取预测目标月份"""
    return datetime.now().replace(day=1)

def save_prediction_results(spy_pred, qqq_pred, feature_importance, model_performance, version):
    """保存预测结果到CSV文件"""
    current_date = datetime.now()
    
    # 创建预测结果数据框
    results = {
        'Prediction_Date': [current_date.strftime("%Y-%m-%d")],
        'Target_Month': [current_date.strftime("%Y-%m")],
        'SPY_Predicted_Return': [spy_pred['mean']],
        'SPY_Confidence_Lower': [spy_pred['lower']],
        'SPY_Confidence_Upper': [spy_pred['upper']],
        'SPY_Std': [spy_pred['std']],
        'QQQ_Predicted_Return': [qqq_pred['mean']],
        'QQQ_Confidence_Lower': [qqq_pred['lower']],
        'QQQ_Confidence_Upper': [qqq_pred['upper']],
        'QQQ_Std': [qqq_pred['std']],
        'SPY_R2_Score': [model_performance['spy_r2']],
        'QQQ_R2_Score': [model_performance['qqq_r2']],
        'SPY_CV_Score': [model_performance['spy_cv_mean']],
        'QQQ_CV_Score': [model_performance['qqq_cv_mean']]
    }
    
    # 添加前5个最重要的特征及其重要性分数
    for i, (feature, importance) in enumerate(feature_importance['spy_importance'].head().values):
        results[f'SPY_Top{i+1}_Feature'] = [feature]
        results[f'SPY_Top{i+1}_Importance'] = [importance]
    
    for i, (feature, importance) in enumerate(feature_importance['qqq_importance'].head().values):
        results[f'QQQ_Top{i+1}_Feature'] = [feature]
        results[f'QQQ_Top{i+1}_Importance'] = [importance]
    
    # 创建预测结果文件
    results_df = pd.DataFrame(results)
    results_file = os.path.join(MODEL_DIR, f'prediction_results_{version}.csv')
    results_df.to_csv(results_file, index=False)
    print(f"预测结果已保存至: {results_file}")
    
    return results_df

def load_data(file_path=None):
    """加载数据，如果未指定文件则使用最新的数据文件"""
    if file_path is None:
        file_path = get_latest_data_file()
    print(f"使用数据文件: {file_path}")
    return pd.read_csv(file_path)

def prepare_features(df):
    """准备特征数据"""
    print("\n数据预处理...")
    
    # 创建基础特征的副本
    df_processed = df.copy()
    
    # 生成One-Hot编码
    # 月份 One-Hot (1-12)
    for month in range(1, 13):
        df_processed[f'Month_{month:02d}'] = (df_processed['Month'] == month).astype(int)
    
    # 季度 One-Hot (1-4)
    for quarter in range(1, 5):
        df_processed[f'Quarter_{quarter}'] = (df_processed['Quarter'] == quarter).astype(int)
    
    # 添加周期性特征
    df_processed['Month_Sin'] = np.sin(2 * np.pi * df_processed['Month']/12)
    df_processed['Month_Cos'] = np.cos(2 * np.pi * df_processed['Month']/12)
    
    # 定义特征列
    feature_columns = [
        # 时间特征
        'Year',
        # 月份 One-Hot
        'Month_01', 'Month_02', 'Month_03', 'Month_04', 'Month_05', 'Month_06',
        'Month_07', 'Month_08', 'Month_09', 'Month_10', 'Month_11', 'Month_12',
        # 季度 One-Hot
        'Quarter_1', 'Quarter_2', 'Quarter_3', 'Quarter_4',
        # 周期性特征
        'Month_Sin', 'Month_Cos',
        # 原始值
        'GDP', 'CPI', 'Unemployment', 'PMI',
        # 同比变化率
        'GDP_YOY', 'CPI_YOY', 'Unemployment_YOY', 'PMI_YOY',
        # 环比变化率
        'GDP_MOM', 'CPI_MOM', 'Unemployment_MOM', 'PMI_MOM',
        # 股票收益率趋势
        'SPY_Return', 'SPY_Return_Prev1M', 'SPY_Return_Prev2M',
        'QQQ_Return', 'QQQ_Return_Prev1M', 'QQQ_Return_Prev2M'
    ]
    
    # 处理缺失值
    df_cleaned = df_processed.copy()
    
    # 检查并打印缺失值信息
    missing_info = df_cleaned[feature_columns].isnull().sum()
    print("\n缺失值统计:")
    print(missing_info[missing_info > 0])
    
    # 1. 对于经济指标，使用前向填充，然后用后向填充处理开始的空值
    economic_indicators = ['GDP', 'CPI', 'Unemployment', 'PMI']
    df_cleaned[economic_indicators] = df_cleaned[economic_indicators].fillna(method='ffill').fillna(method='bfill')
    
    # 2. 对于变化率，先用0填充
    change_columns = [col for col in feature_columns if 'YOY' in col or 'MOM' in col]
    df_cleaned[change_columns] = df_cleaned[change_columns].fillna(0)
    
    # 3. 对于股票收益率，使用中位数填充
    returns_columns = [col for col in feature_columns if 'Return' in col]
    for col in returns_columns:
        median_value = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    # 4. 处理无穷大的值
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
    
    # 对所有剩余的NaN使用中位数填充
    for col in feature_columns:
        if df_cleaned[col].isnull().any():
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    # 检查处理后的结果
    remaining_nulls = df_cleaned[feature_columns].isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"\n警告: 仍存在 {remaining_nulls} 个缺失值")
        print(df_cleaned[feature_columns].isnull().sum()[df_cleaned[feature_columns].isnull().sum() > 0])
    else:
        print("\n所有缺失值已处理完成")
    
    # 检查无穷大值
    inf_check = np.isinf(df_cleaned[feature_columns]).sum().sum()
    if inf_check > 0:
        print(f"\n警告: 存在 {inf_check} 个无穷大值")
        print(df_cleaned[feature_columns].isin([np.inf, -np.inf]).sum()[df_cleaned[feature_columns].isin([np.inf, -np.inf]).sum() > 0])
    
    # 最后的数据验证
    final_data = df_cleaned[feature_columns]
    assert not final_data.isnull().any().any(), "仍然存在缺失值"
    assert not np.isinf(final_data).any().any(), "仍然存在无穷大值"
    
    print("\n特征数据形状:", final_data.shape)
    
    return final_data

def train_models(X, y_spy, y_qqq):
    """训练预测模型"""
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建并训练SPY模型
    spy_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    spy_model.fit(X_scaled, y_spy)
    
    # 创建并训练QQQ模型
    qqq_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    qqq_model.fit(X_scaled, y_qqq)
    
    return spy_model, qqq_model, scaler

def evaluate_models(models, X, y_spy, y_qqq):
    """评估模型性能"""
    spy_model, qqq_model, scaler = models
    X_scaled = scaler.transform(X)
    
    # 计算R²分数
    spy_r2 = r2_score(y_spy, spy_model.predict(X_scaled))
    qqq_r2 = r2_score(y_qqq, qqq_model.predict(X_scaled))
    
    # 计算交叉验证分数
    spy_cv_scores = cross_val_score(spy_model, X_scaled, y_spy, cv=5)
    qqq_cv_scores = cross_val_score(qqq_model, X_scaled, y_qqq, cv=5)
    
    # 计算特征重要性
    spy_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': spy_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    qqq_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': qqq_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'spy_r2': spy_r2,
        'qqq_r2': qqq_r2,
        'spy_cv_mean': spy_cv_scores.mean(),
        'spy_cv_std': spy_cv_scores.std(),
        'qqq_cv_mean': qqq_cv_scores.mean(),
        'qqq_cv_std': qqq_cv_scores.std(),
        'spy_importance': spy_importance,
        'qqq_importance': qqq_importance
    }

def predict_next_month(models, latest_data):
    """预测下个月的收益率"""
    spy_model, qqq_model, scaler = models
    X_scaled = scaler.transform(latest_data)
    
    # 预测收益率
    spy_pred = spy_model.predict(X_scaled)[0]
    qqq_pred = qqq_model.predict(X_scaled)[0]
    
    # 计算预测区间（使用随机森林的预测方差作为不确定性度量）
    spy_predictions = np.array([tree.predict(X_scaled) for tree in spy_model.estimators_])
    qqq_predictions = np.array([tree.predict(X_scaled) for tree in qqq_model.estimators_])
    
    spy_confidence = {
        'mean': spy_pred,
        'std': spy_predictions.std(),
        'lower': np.percentile(spy_predictions, 25),
        'upper': np.percentile(spy_predictions, 75)
    }
    
    qqq_confidence = {
        'mean': qqq_pred,
        'std': qqq_predictions.std(),
        'lower': np.percentile(qqq_predictions, 25),
        'upper': np.percentile(qqq_predictions, 75)
    }
    
    return spy_confidence, qqq_confidence

def save_processed_data(X, y_spy, y_qqq, version):
    """保存处理后的训练数据"""
    # 合并特征和目标变量
    processed_data = X.copy()
    processed_data['Next_Month_SPY'] = y_spy
    processed_data['Next_Month_QQQ'] = y_qqq
    
    # 添加时间戳
    processed_data['Processing_Date'] = datetime.now().strftime("%Y-%m-%d")
    
    # 保存到CSV文件
    processed_file = os.path.join(MODEL_DIR, f'processed_data_{version}.csv')
    processed_data.to_csv(processed_file, index=True)  # 保留索引作为日期
    print(f"处理后的训练数据已保存至: {processed_file}")
    
    # 保存数据描述统计
    stats_file = os.path.join(MODEL_DIR, f'data_statistics_{version}.csv')
    stats = processed_data.describe()
    stats.to_csv(stats_file)
    print(f"数据统计信息已保存至: {stats_file}")
    
    return processed_data

def main(training_file=None):
    """主函数"""
    # 确保所有目录存在
    ensure_dirs()
    
    # 获取当前版本
    version = get_current_version()
    current_date = datetime.now()
    
    try:
        print(f"\nPrediction Target: Market returns for {current_date.strftime('%Y-%m')}")
        print(f"Using previous month's data for prediction")
        
        # 加载数据
        df = load_data(training_file)
        
        # 检查目标变量
        if 'Next_Month_SPY' not in df.columns or 'Next_Month_QQQ' not in df.columns:
            raise ValueError("Missing target columns (Next_Month_SPY or Next_Month_QQQ)")
        
        # 准备特征和目标变量
        X = prepare_features(df)
        y_spy = df['Next_Month_SPY'].fillna(df['Next_Month_SPY'].median())
        y_qqq = df['Next_Month_QQQ'].fillna(df['Next_Month_QQQ'].median())
        
        # 保存处理后的训练数据
        processed_data = save_processed_data(X, y_spy, y_qqq, version)
        
        # 检查数据大小
        print(f"\nDataset Size: {len(df)} rows")
        print(f"Number of Features: {X.shape[1]}")
        
        # 训练模型
        print("\nTraining models...")
        models = train_models(X, y_spy, y_qqq)
        
        # 评估模型
        print("\nEvaluating model performance...")
        evaluation = evaluate_models(models, X, y_spy, y_qqq)
        
        print("\nModel Performance Metrics:")
        print(f"SPY Model R2 Score: {evaluation['spy_r2']:.4f}")
        print(f"SPY Model CV Score: {evaluation['spy_cv_mean']:.4f} (±{evaluation['spy_cv_std']:.4f})")
        print(f"QQQ Model R2 Score: {evaluation['qqq_r2']:.4f}")
        print(f"QQQ Model CV Score: {evaluation['qqq_cv_mean']:.4f} (±{evaluation['qqq_cv_std']:.4f})")
        
        print("\nTop Features (SPY):")
        print(evaluation['spy_importance'].head())
        
        print("\nTop Features (QQQ):")
        print(evaluation['qqq_importance'].head())
        
        # 使用最新数据进行预测
        latest_data = X.iloc[-1:].copy()
        spy_pred, qqq_pred = predict_next_month(models, latest_data)
        
        print(f"\nPredictions for {current_date.strftime('%Y-%m')}:")
        print(f"SPY Expected Return: {spy_pred['mean']:.2f}% (CI: {spy_pred['lower']:.2f}% to {spy_pred['upper']:.2f}%)")
        print(f"QQQ Expected Return: {qqq_pred['mean']:.2f}% (CI: {qqq_pred['lower']:.2f}% to {qqq_pred['upper']:.2f}%)")
        
        # 保存模型
        model_file = os.path.join(MODEL_DIR, f'market_models_{version}.pkl')
        joblib.dump(models, model_file)
        print(f"\nModel saved to: {model_file}")
        
        # 保存预测结果
        results_df = save_prediction_results(
            spy_pred, 
            qqq_pred, 
            {
                'spy_importance': evaluation['spy_importance'],
                'qqq_importance': evaluation['qqq_importance']
            },
            {
                'spy_r2': evaluation['spy_r2'],
                'qqq_r2': evaluation['qqq_r2'],
                'spy_cv_mean': evaluation['spy_cv_mean'],
                'qqq_cv_mean': evaluation['qqq_cv_mean']
            },
            version
        )
        
        # 显示保存的位置
        print(f"\nFiles generated for {version}:")
        print(f"1. Training Data: {os.path.basename(get_latest_data_file())}")
        print(f"2. Processed Data: processed_data_{version}.csv")
        print(f"3. Statistics: data_statistics_{version}.csv")
        print(f"4. Model File: market_models_{version}.pkl")
        print(f"5. Predictions: prediction_results_{version}.csv")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDetailed Error Information:")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 