"""
=============================================================================
项目名称：基于双均线策略的沪深300ETF量化回测与绩效分析
作者：yuzibai93-droid (经济统计学背景)
工具环境：Python 3, Jupyter Notebook (Anaconda)
依赖库：akshare, pandas, numpy, matplotlib
=============================================================================

【开发过程中的排错与解决记录 (Debug Log)】
1. 问题：安装 akshare 时报错 "[WinError 5] 拒绝访问"
   解决：因为 Anaconda 安装在 D 盘且没有管理员权限。使用 `!pip install akshare --user`，将包装在个人用户目录下，绕过系统权限限制。

2. 问题：报错 "ModuleNotFoundError: No module named 'akshare'"
   解决：Jupyter 环境冲突。之前使用的 `!pip` 调用了系统的 pip，但 Jupyter 内核使用的是另一个环境。通过改用 Jupyter 魔法命令 `%pip install akshare` 完美解决。

3. 问题：运行代码报错 "NameError: name 'np' is not defined"
   解决：在重新开一个代码单元格（或重启内核）后，内存变量被清空。在代码最开头重新导入了 `import numpy as np`。

4. 问题：画图时图表标题和坐标轴的中文显示为方块（乱码）
   解决：matplotlib 默认不支持中文，加入代码：
   plt.rcParams['font.sans-serif'] = ['SimHei'] (Windows系统)
   plt.rcParams['axes.unicode_minus'] = False (解决负号显示问题)

=============================================================================
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决画图时中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def fetch_data():
    """获取沪深300ETF历史数据"""
    print("1. 正在获取沪深300ETF(510300)数据...")
    try:
        df = ak.fund_etf_hist_em(symbol="510300", period="daily", start_date="20200101", end_date="20241231")
        # 数据清洗：保留日期和收盘价
        df = df[['日期', '收盘']]
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

def build_strategy(df):
    """构建10/60日均线策略并计算收益"""
    print("2. 正在计算均线与交易信号...")
    
    # 参数设定：10日短线，60日长线
    df['10日均线'] = df['收盘'].rolling(window=10).mean()
    df['60日均线'] = df['收盘'].rolling(window=60).mean()
    
    # 生成信号：10日均线 > 60日均线 为 1 (持有)，否则为 0 (空仓)
    df['信号'] = np.where(df['10日均线'] > df['60日均线'], 1, 0)
    
    # 计算每日真实收益率
    df['每日收益率'] = df['收盘'].pct_change()
    
    # 扣除交易成本计算策略收益
    cost = 0.0015 # 假设双边费率 0.15%
    df['交易信号'] = df['信号'].diff().abs() # 信号变化时（开仓或平仓）记为 1
    df['策略收益率'] = (df['信号'].shift(1) * df['每日收益率']) - (df['交易信号'] * cost)
    
    # 计算累计净值
    df['基准累计'] = (1 + df['每日收益率']).cumprod()
    df['策略累计'] = (1 + df['策略收益率']).cumprod()
    
    return df

def calculate_performance(df):
    """计算专业绩效指标（最大回撤与夏普比率）"""
    print("3. 正在计算绩效指标...\n")
    
    # 最大回撤计算函数
    def get_max_drawdown(cumulative_returns):
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    # 夏普比率计算函数 (无风险利率按0处理，一年252个交易日)
    def get_sharpe_ratio(returns):
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        return annual_return / annual_volatility if annual_volatility != 0 else 0

    # 提取收益序列
    benchmark_returns = df['每日收益率'].dropna()
    strategy_returns = df['策略收益率'].dropna()

    # 计算各项指标
    mdd_bench = get_max_drawdown(df['基准累计'])
    mdd_strat = get_max_drawdown(df['策略累计'])
    
    sharpe_bench = get_sharpe_ratio(benchmark_returns)
    sharpe_strat = get_sharpe_ratio(strategy_returns)
    
    total_return_bench = (df['基准累计'].iloc[-1] - 1) * 100
    total_return_strat = (df['策略累计'].iloc[-1] - 1) * 100

    # 打印报告
    print("======= 量化策略专业绩效报告 =======")
    print(f"回测区间: 2020-01-01 至 2024-12-31")
    print(f"1. 累计收益率 | 基准: {total_return_bench:.2f}% | 策略: {total_return_strat:.2f}%")
    print(f"2. 最大回撤   | 基准: {mdd_bench:.2%} | 策略: {mdd_strat:.2%}")
    print(f"3. 夏普比率   | 基准: {sharpe_bench:.2f} | 策略: {sharpe_strat:.2f}")
    print("====================================")

def plot_results(df):
    """绘制策略与基准的资金曲线对比图"""
    print("4. 正在生成资金曲线图...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['基准累计'], label='买入并持有 (基准)', color='gray')
    plt.plot(df['策略累计'], label='10/60日均线策略 (扣费)', color='red')
    plt.title('沪深300ETF (510300) 策略资金曲线对比 (2020-2024)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # 运行主程序
    data = fetch_data()
    if data is not None:
        data = build_strategy(data)
        calculate_performance(data)
        plot_results(data)
