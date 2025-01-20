# Market Return Prediction System

这个系统用于预测SPY和QQQ的月度收益率。通过结合宏观经济指标和市场技术指标，使用机器学习模型进行预测。

## 工作流程

1. **数据采集** (`market_data_collector.py`)
   - 每月运行一次，收集最新的经济和市场数据
   - 数据存储在 `market_data/market_data_YYYY_MM.csv`
   - 包含原始数据和计算的指标（同比、环比变化率等）

2. **预测流程** (`market_predictor.py`)
   - 读取最新的数据文件
   - 进行特征工程（One-Hot编码等）
   - 训练模型并进行预测
   - 生成预测结果和模型文件

3. **报告生成** (`report_generator.py`)
   - 自动生成月度预测报告
   - 输出格式：Markdown和PDF
   - 包含预测结果、模型性能、特征重要性分析等
   - 专业的图表和表格展示

## 输出文件

### 数据文件 (在 `market_data/` 目录)
- `market_data_YYYY_MM.csv`: 每月的原始数据和计算指标

### 模型文件 (在 `market_models/` 目录)
- `processed_data_YYYY_MM.csv`: 处理后的训练数据
- `data_statistics_YYYY_MM.csv`: 数据统计信息
- `market_models_YYYY_MM.pkl`: 训练好的模型
- `prediction_results_YYYY_MM.csv`: 预测结果

### 报告文件 (在 `market_reports/` 目录)
- `market_prediction_report_YYYY_MM.md`: Markdown格式报告
- `market_prediction_report_YYYY_MM.pdf`: PDF格式报告
- `feature_importance.png`: 特征重要性可视化图表

## 特征列表

### 经济指标
- GDP (实际国内生产总值)
- CPI (消费者物价指数)
- Unemployment Rate (失业率)
- PMI (制造业生产指数)

### 变化率指标
- 同比变化率 (YOY)：与去年同期相比的变化
- 环比变化率 (MOM)：与上月相比的变化

### 市场技术指标
- 当月收益率 (SPY, QQQ)
- 上月收益率
- 前月收益率

### 时间特征
- 月份 One-Hot编码 (12个二进制特征)
- 季度 One-Hot编码 (4个二进制特征)
- 周期性特征 (sin/cos变换)

## 缺失值处理

### 处理策略
1. **经济指标**: 前向填充 (Forward Fill)
   - 优点：保持数据的时间连续性
   - 缺点：可能延续过时的数据

2. **变化率**: 填充0
   - 优点：代表"无变化"的中性假设
   - 缺点：可能低估实际波动

3. **股票收益率**: 中位数填充
   - 优点：保持数据分布特征
   - 缺点：可能掩盖极端市场行为

## 模型选择

使用**随机森林回归器** (Random Forest Regressor)

### 优点
1. 能处理非线性关系
2. 自动处理特征交互
3. 提供特征重要性评估
4. 通过集成学习减少过拟合
5. 可以估计预测不确定性

### 缺点
1. 计算成本较高
2. 模型可解释性相对较低
3. 需要足够的训练数据

## 预测结果解读

示例预测结果：
```
SPY预期收益率: 1.25% (置信区间: 0.75% 到 1.75%)
QQQ预期收益率: 1.85% (置信区间: 1.20% 到 2.50%)
```

### 关键指标
1. **预期收益率**: 模型预测的点估计值
2. **置信区间**: 预测的不确定性范围
3. **R²分数**: 模型解释数据变异的能力
4. **交叉验证分数**: 模型泛化能力的度量
5. **特征重要性**: 最具影响力的预测因子

## 下一版本计划

1. **模型增强**
   - 添加更多经济指标（如利率、货币供应量）
   - 尝试深度学习模型
   - 实现模型集成

2. **特征工程**
   - 添加市场情绪指标
   - 加入波动率指标
   - 探索更多技术指标

3. **风险管理**
   - 加入风险评估指标
   - 实现动态调整的置信区间
   - 添加极端市场条件警报

4. **系统优化**
   - 自动化数据验证
   - 模型性能监控
   - 预测结果可视化

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 设置FRED API密钥：
   - 在 `market_data_collector.py` 中设置 `FRED_API_KEY`

2. 运行数据收集：
```bash
python market_data_collector.py
```

3. 运行预测：
```bash
python market_predictor.py
```

4. 生成报告：
```bash
python report_generator.py
```

## 报告内容

每月生成的预测报告包含以下内容：

1. **预测概述**
   - SPY和QQQ的预期收益率
   - 预测置信区间
   - 目标月份说明

2. **预测方法**
   - 模型说明
   - 性能指标
   - 特征重要性分析

3. **市场环境**
   - 最新经济指标
   - 市场趋势分析
   - 风险提示

4. **可视化**
   - 特征重要性图表
   - 数据统计图表
   - 专业排版和布局 