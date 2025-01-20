import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from fredapi import Fred
import os

# 请在这里替换您的FRED API密钥
FRED_API_KEY = '138dad9cbf42352aef6caab64f1f70e1'
fred = Fred(api_key=FRED_API_KEY)

# 定义数据存储目录
DATA_DIR = 'market_data'

def ensure_data_dir():
    """确保数据存储目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"创建数据目录: {DATA_DIR}")

def get_current_month_file():
    """获取当前月份的数据文件名"""
    current_date = datetime.now()
    return os.path.join(DATA_DIR, f'market_data_{current_date.strftime("%Y_%m")}.csv')

def get_all_data_files():
    """获取所有数据文件列表"""
    ensure_data_dir()
    return sorted([f for f in os.listdir(DATA_DIR) if f.startswith('market_data_') and f.endswith('.csv')])

def get_stock_data(ticker, start_date, end_date):
    """获取股票数据并计算月度收益率"""
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        print(f"正在获取{ticker}的数据...")
        if stock.empty:
            print(f"警告: 无法获取{ticker}的数据")
            return pd.Series(dtype='float64')
        monthly_returns = stock['Adj Close'].resample('M').last().pct_change()
        return monthly_returns
    except Exception as e:
        print(f"获取{ticker}数据时出错: {str(e)}")
        return pd.Series(dtype='float64')

def get_economic_data(indicator_id, start_date, end_date):
    """从FRED获取经济数据"""
    try:
        data = fred.get_series(indicator_id, start_date, end_date)
        # 将数据转换为月度数据
        if not data.empty:
            data = data.resample('M').last()
            return data
        else:
            print(f"警告: {indicator_id}返回空数据")
            return pd.Series(dtype='float64')
    except ValueError as e:
        print(f"获取{indicator_id}数据时出错: {str(e)}")
        return pd.Series(dtype='float64')

def calculate_yoy_change(series):
    """计算同比变化率（年度同比）"""
    return ((series - series.shift(12)) / series.shift(12) * 100).round(2)

def calculate_mom_change(series):
    """计算环比变化率（月度环比）"""
    return ((series - series.shift(1)) / series.shift(1) * 100).round(2)

def collect_and_process_data(months=36):
    """收集和处理所有需要的数据"""
    # 获取当前日期
    current_date = datetime.now()
    
    # 设置结束日期为当前月份的第一天
    end_date = current_date.replace(day=1)
    
    # 计算开始日期（多获取13个月的数据用于计算年度变化率）
    start_month = end_date.month - (months + 13)
    start_year = end_date.year
    
    # 处理月份回溯时的年份变化
    while start_month <= 0:
        start_month += 12
        start_year -= 1
        
    # 设置开始日期为目标月份的第一天
    start_date = datetime(start_year, start_month, 1)

    print(f'Data Collection Range:')
    print(f'Start Date: {start_date.strftime("%Y-%m-%d")}')
    print(f'End Date: {end_date.strftime("%Y-%m-%d")}')
    print(f'Using data to predict returns for: {current_date.strftime("%Y-%m")}')

    # 获取股票数据
    spy_returns = get_stock_data('SPY', start_date, end_date)
    qqq_returns = get_stock_data('QQQ', start_date, end_date)
    
    # 获取经济指标数据
    gdp = get_economic_data('GDPC1', start_date, end_date)  # 实际GDP（季度调整）
    cpi = get_economic_data('CPIAUCSL', start_date, end_date)  # CPI
    unemployment = get_economic_data('UNRATE', start_date, end_date)  # 失业率
    pmi = get_economic_data('IPMAN', start_date, end_date)  # 工业生产指数-制造业

    # 检查是否所有数据都是空的
    if all(x.empty for x in [spy_returns, qqq_returns, gdp, cpi, unemployment, pmi]):
        print("错误: 无法获取任何数据")
        return pd.DataFrame()
    
    # 创建输入特征数据框
    df_features = pd.DataFrame({
        # 原始值
        'GDP': gdp,
        'CPI': cpi,
        'Unemployment': unemployment,
        'PMI': pmi,
        # 同比变化率
        'GDP_YOY': calculate_yoy_change(gdp),
        'CPI_YOY': calculate_yoy_change(cpi),
        'Unemployment_YOY': calculate_yoy_change(unemployment),
        'PMI_YOY': calculate_yoy_change(pmi),
        # 环比变化率
        'GDP_MOM': calculate_mom_change(gdp),
        'CPI_MOM': calculate_mom_change(cpi),
        'Unemployment_MOM': calculate_mom_change(unemployment),
        'PMI_MOM': calculate_mom_change(pmi)
    })
    
    # 创建目标变量数据框（下个月的收益率）
    df_targets = pd.DataFrame({
        'Next_Month_SPY': spy_returns.shift(-1),  # 下个月的SPY收益率
        'Next_Month_QQQ': qqq_returns.shift(-1),  # 下个月的QQQ收益率
        # 使用历史数据作为特征（t-1, t-2, t-3）
        'SPY_Return_Prev1M': spy_returns.shift(1),  # 上个月SPY收益率
        'SPY_Return_Prev2M': spy_returns.shift(2),  # 两个月前SPY收益率
        'SPY_Return_Prev3M': spy_returns.shift(3),  # 三个月前SPY收益率
        'QQQ_Return_Prev1M': qqq_returns.shift(1),  # 上个月QQQ收益率
        'QQQ_Return_Prev2M': qqq_returns.shift(2),  # 两个月前QQQ收益率
        'QQQ_Return_Prev3M': qqq_returns.shift(3),  # 三个月前QQQ收益率
    })
    
    # 合并数据框
    df = pd.concat([df_features, df_targets], axis=1)
    
    # 添加月份特征
    df.index = pd.to_datetime(df.index)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # 对经济指标进行前向填充（因为某些指标可能不是每月更新）
    df = df.fillna(method='ffill')
    
    # 将百分比转换为更易读的格式
    for col in df.columns:
        if 'SPY' in col or 'QQQ' in col:
            df[col] = df[col] * 100  # 转换为百分比
            df[col] = df[col].round(2)  # 保留两位小数
    
    # 重新排序列
    columns_order = [
        'Year', 'Month', 'Quarter',
        # 原始值
        'GDP', 'CPI', 'Unemployment', 'PMI',
        # 同比变化率
        'GDP_YOY', 'CPI_YOY', 'Unemployment_YOY', 'PMI_YOY',
        # 环比变化率
        'GDP_MOM', 'CPI_MOM', 'Unemployment_MOM', 'PMI_MOM',
        # 历史股票收益率
        'SPY_Return_Prev1M', 'SPY_Return_Prev2M', 'SPY_Return_Prev3M',
        'QQQ_Return_Prev1M', 'QQQ_Return_Prev2M', 'QQQ_Return_Prev3M',
        # 预测目标
        'Next_Month_SPY', 'Next_Month_QQQ'
    ]
    df = df[columns_order]
    
    # 删除最后一行（因为没有下个月的数据）
    df = df.iloc[:-1]
    
    return df

def format_date_index(df):
    """格式化日期索引为易读格式"""
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.strftime('%Y-%m')
    # 将Date列移到最前面
    cols = df.columns.tolist()
    cols = ['Date'] + [col for col in cols if col != 'Date']
    return df[cols]

def update_data():
    """更新数据并追加到CSV文件"""
    # 确保数据目录存在
    ensure_data_dir()
    
    # 获取当前月份的文件名
    current_file = get_current_month_file()
    
    # 获取最新一个月的数据
    df = collect_and_process_data(months=2)  # 获取最近2个月的数据以确保完整性
    
    if df.empty:
        print("错误: 无法更新数据")
        return
        
    df = df.iloc[-1:]  # 只保留最新一行
    df = format_date_index(df)  # 格式化日期
    
    if os.path.exists(current_file):
        try:
            # 如果当月文件存在，检查是否需要更新
            existing_data = pd.read_csv(current_file)
            if 'Date' in existing_data.columns:
                if not df.empty and df['Date'].iloc[0] not in existing_data['Date'].values:
                    # 追加新数据
                    df.to_csv(current_file, mode='a', header=False, index=False)
                    print(f"数据已更新到文件: {current_file}")
                else:
                    print(f"当月数据已存在于文件: {current_file}")
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
    else:
        # 如果当月文件不存在，创建新文件
        df_initial = collect_and_process_data()
        if not df_initial.empty:
            df_initial = format_date_index(df_initial)  # 格式化日期
            df_initial.to_csv(current_file, index=False)
            print(f"已创建新的数据文件: {current_file}")
            
            # 显示数据文件列表
            print("\n现有数据文件:")
            for file in get_all_data_files():
                file_path = os.path.join(DATA_DIR, file)
                file_size = os.path.getsize(file_path) / 1024  # 转换为KB
                print(f"- {file} ({file_size:.1f} KB)")
            
            print("\n数据预览:")
            print(df_initial.head())
            print("\n特征说明:")
            print("输入特征 (用于预测):")
            print("原始值:")
            print("- GDP: 国内生产总值")
            print("- CPI: 消费者物价指数")
            print("- Unemployment: 失业率")
            print("- PMI: 制造业生产指数")
            print("\n同比变化率 (%):")
            print("- GDP_YOY: GDP同比变化率")
            print("- CPI_YOY: CPI同比变化率")
            print("- Unemployment_YOY: 失业率同比变化率")
            print("- PMI_YOY: PMI同比变化率")
            print("\n环比变化率 (%):")
            print("- GDP_MOM: GDP环比变化率")
            print("- CPI_MOM: CPI环比变化率")
            print("- Unemployment_MOM: 失业率环比变化率")
            print("- PMI_MOM: PMI环比变化率")
            print("\n历史股票收益率:")
            print("- SPY_Return_Prev1M: 上月SPY收益率")
            print("- SPY_Return_Prev2M: 两个月前SPY收益率")
            print("- SPY_Return_Prev3M: 三个月前SPY收益率")
            print("- QQQ_Return_Prev1M: 上月QQQ收益率")
            print("- QQQ_Return_Prev2M: 两个月前QQQ收益率")
            print("- QQQ_Return_Prev3M: 三个月前QQQ收益率")
            print("\n预测目标:")
            print("- Next_Month_SPY: 下个月SPY收益率")
            print("- Next_Month_QQQ: 下个月QQQ收益率")
        else:
            print("错误: 无法创建初始数据文件")

if __name__ == "__main__":
    # 初始化或更新数据
    update_data() 