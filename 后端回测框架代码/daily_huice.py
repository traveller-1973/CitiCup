# coding=utf-8
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
# mpl.use('TkAgg')
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns
import matplotlib.dates as mdates
import logging
import sys
import json
from json import JSONEncoder, dumps
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp, Series, DatetimeIndex
import statsmodels.api as sm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import configparser
from collections import defaultdict
import daily_factor_script

Base = declarative_base()

class Result(Base):
    __tablename__ = 'daily_factor_table'
    id = Column(Integer, primary_key=True)

    stock_pool = Column(String, default='A股全指')
    select_long_short = Column(Integer, default=1)
    select_tax = Column(Integer, default=1)
    period = Column(Integer, default=10)
    english_factor = Column(String)
    table_name = Column(String)

    cumulative_returns_with_fee = Column(Float)
    annualized_returns = Column(Float)
    annualized_volatility = Column(Float)
    sharpe_ratio = Column(Float)
    ic_mean = Column(Float)
    IR = Column(Float)


def store_results_to_database(engine ,factor_column, table_name, last_longshort_return, annualized_returns_season, annualized_volatility_season, sharpe_ratio_season, ic_mean_season, IR_season, suffix):

    # 创建Session类
    Session = sessionmaker(bind=engine)

    # 创建Session对象
    session = Session()

    cumulative_returns_with_fee_season = float(last_longshort_return-1)
    annualized_returns_season = float(annualized_returns_season["Average_Returns"])
    annualized_volatility_season = float(annualized_volatility_season["Average_Returns"])
    sharpe_ratio_season = float(sharpe_ratio_season["Average_Returns"])
    ic_mean_season = float(ic_mean_season)
    IR_season = float(IR_season)

    # 在factor_column后面添加字符串
    factor_column = factor_column + suffix

    # 创建一个结果对象并添加到数据库中
    result = Result(table_name=table_name, english_factor=factor_column, cumulative_returns_with_fee=cumulative_returns_with_fee_season, annualized_returns=annualized_returns_season, annualized_volatility=annualized_volatility_season, sharpe_ratio=sharpe_ratio_season, ic_mean=ic_mean_season, IR=IR_season)
    session.add(result)

    # 提交更改
    session.commit()

    # 关闭Session
    session.close()

####################################################### 读取基础数据模块 ################################
def ReadRawdata():
    '''所有数据的起始时间是2013-01-04 00:00:00
'''
    open1 = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_open.pkl")
    close = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_close.pkl")
    high = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_high.pkl")
    low = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_low.pkl")
    vol = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_vol.pkl")
    amt = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\stock_amt.pkl")
    score = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\title_pos_rate_converted.pkl")
    ret = close / close.shift()
    retn = ret - 1
    market = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\normalized_data_with_col_name.pkl")
    return open1, close, high, low, vol, amt, score, ret, retn, market


def find_stocks_with_value_one(df):
    # 创建一个空的 DataFrame，用于存储每天对应的股票代码
    result_df = pd.DataFrame(index=df.columns, columns=['stocks'])

    # 遍历 DataFrame 的每一列
    for column in df.columns:
        # 查找值为1的股票代码
        stocks = df[df[column] == 1].index.tolist()
        # 将找到的股票代码添加到结果 DataFrame 中
        result_df.loc[column, 'stocks'] = stocks

    return result_df

################################################## 回测框架模块 ######################################
def calculate_returns(close,window):
    returns = close.diff(window) / close.shift(window)
    # returns = returns.dropna()
    return returns

def generate_rebalance_dates(start_date, end_date, interval,open_dates):
    # 使用布尔索引选择在给定范围内的开市日期
    mask = (open_dates >= pd.to_datetime(start_date)) & (open_dates <= pd.to_datetime(end_date))
    market_open_dates = open_dates[mask]
    # 选择间隔的工作日
    rebalance_dates = market_open_dates[::interval]
    return pd.Series(rebalance_dates).astype(str)

#df_retn为调仓时点间的收益率
def calculate_previous_factor(balance_dates, df_retn, df_factor,df_market):

    '''


    :param balance_dates: 调仓时点序列
    :param df_retn: 收益率矩阵
    :param df_factor: 因子矩阵，后期财务数据都可以直接用
    :param df_market: 市场series，后期调整为中证500，沪深300等
    :return:
    '''
    balance_dates.index = pd.to_datetime(balance_dates)
    df_factor = df_factor.loc[balance_dates]
    df_retn = df_retn.loc[balance_dates]
    df_market = df_market.loc[balance_dates]
    df_ret = df_retn +1
    # 将因子值矩阵向前移动
    # df_factor = df_factor.shift(1)
    # df_ret = df_ret.drop(df_ret.columns[-1], axis=1)
    return df_factor,df_retn,df_ret,df_market

#因为score里面的股票指数数量少于df_factor里面的，这里设计对齐函数
def factor_correct(df_factor, aligned_score_df):
    # 提取factor_df列名的前6位，并创建一个映射关系
    trimmed_column_mapping = {col: col[:6] for col in df_factor.columns}
    # 重命名factor_df的列名
    trimmed_factor_df = df_factor.rename(columns=trimmed_column_mapping)

    # 获取aligned_score_df中存在的股票代码
    aligned_columns = aligned_score_df.columns

    # 选取对应的列
    aligned_columns = [col for col in trimmed_factor_df.columns if col in aligned_columns]

    aligned_factor_df = trimmed_factor_df[aligned_columns]
    return aligned_factor_df

def min_max_normalize(df):
    return (df - df.mean()) / df.std()

def calculate_long_short_returns(returns_df, factor_df,aligned_score_df,n):

    '''
    计算多头和空头收益率，算手续费

    :param returns_df:
    :param factor_df:
    :param n:
    :return:
    '''
    # # 将因子值矩阵向前移动，得到因子值矩阵2
    factor_df = factor_df.shift(1)
    factor_df = factor_df.applymap(lambda x: np.nan if x is None else x)
    #这行是保证数据型no string
    for col in factor_df.columns:
        factor_df[col] = pd.to_numeric(factor_df[col], errors='coerce')

    # 归一化(注释这条就回归单因子)
    factor_df = min_max_normalize(factor_df)

    # 取出第一行进行归一化
    first_row = aligned_score_df.iloc[0]

    # 对第一行数据应用min-max归一化(疑问，为什么没用Z方法)
    normalized_first_row = 2 * (first_row - first_row.min()) / (first_row.max() - first_row.min()) - 1
    # 将归一化的第一行复制到aligned_score_df的每一行
    aligned_score_df = pd.DataFrame([normalized_first_row] * len(factor_df), index=factor_df.index)

    # 这里要考虑factor_correct，因为矩阵格式不对齐
    # factor_df = factor_correct(factor_df, aligned_score_df)
    aligned_score_df = factor_correct(aligned_score_df, factor_df)
    double_factor_df = 0.5 * factor_df + 0.5 * aligned_score_df

    # 对因子值矩阵2进行排名，并将排名前n的货币对应的值设置为True，其他值设置为False
    long_ranked_factor = double_factor_df.rank(axis=1, ascending=False)  # 降序
    short_ranked_factor = double_factor_df.rank(axis=1, ascending=True)  # 升序
    long_bool_factor = long_ranked_factor <= n

    # 将布尔值因子矩阵转换为浮点数矩阵，True设置为1.00，False设置为0.00
    long_float_factor = long_bool_factor.astype(float)

    #截取列
    long_float_factor = factor_correct(long_float_factor,returns_df)
    start_time = long_float_factor.index[0]
    end_time = long_float_factor.index[-1]
    returns_df = returns_df.loc[start_time:end_time]

    # 因子最大的20支股票输出
    long_bool_20_factor = long_ranked_factor <= 20
    long_float_20_factor = long_bool_20_factor.astype(float)


    # 计算最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    long_final_matrix = long_float_factor * returns_df

    # 计算每行的平均收益率
    long_average_returns = long_final_matrix.sum(axis=1) / (2*n)

    # 创建一个新的DataFrame，包含平均收益率
    long_df = pd.DataFrame(long_average_returns, columns=['Average_Returns'])

    #对因子值矩阵2进行排名，并将排名后n的货币对应的值设置为True，排名前面的设置为False
    short_bool_factor = short_ranked_factor <= n

    # 将布尔值因子矩阵转换为浮点数因子矩阵，True设置为-1.00，False设置为0.00
    short_float_factor = short_bool_factor.astype(float) * -1

    # 截取列
    short_float_factor = factor_correct(short_float_factor, returns_df)
    start_time = short_float_factor.index[0]
    end_time = short_float_factor.index[-1]
    returns_df = returns_df.loc[start_time:end_time]

    #因子最小的20支股票输出
    short_bool_20_factor = short_ranked_factor <= 20
    short_float_20_factor = short_bool_20_factor.astype(float) * -1

    # 计算做空部分的最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    short_final_matrix = short_float_factor * returns_df
    short_float_factor_2 = short_bool_factor.astype(float)

    #出现了None与浮点数相乘
    factor_matrix_2 = short_float_factor_2 * double_factor_df

    # 计算做空部分每行的平均收益率
    short_average_returns = short_final_matrix.sum(axis=1) / (2*n)

    # 创建一个新的DataFrame，包含平均收益率（做空部分）
    short_df = pd.DataFrame(short_average_returns, columns=['Average_Returns'])

    # 将做空部分的平均收益率和做多部分的平均收益率相加
    longshort_df = long_df + short_df
    final_matrix = long_float_factor + short_float_factor
    final_matrix_20 = long_float_20_factor + short_float_20_factor

    #计算提调仓时间的累计收益率
    cumulative_returns_withoutfee = (1 + longshort_df).cumprod()

    return longshort_df,cumulative_returns_withoutfee,final_matrix, final_matrix_20


def calculate_long_returns(returns_df, factor_df,n):
    '''
    计算多头收益率（不算手续费）
    :param returns_df:
    :param factor_df:
    :param n:
    :return: 纯多头收益和换仓矩阵
    '''
    # # 将因子值矩阵向前移动，得到因子值矩阵2
    factor_df = factor_df.shift(1)
    factor_df = factor_df.applymap(lambda x: np.nan if x is None else x)
    for col in factor_df.columns:
        factor_df[col] = pd.to_numeric(factor_df[col], errors='coerce')

    # 对因子值矩阵2进行排名，并将排名前n的货币对应的值设置为True，其他值设置为False
    long_ranked_factor = factor_df.rank(axis=1, ascending=False)#降序
    short_ranked_factor = factor_df.rank(axis=1,ascending = True)#升序
    long_bool_factor = long_ranked_factor <= n
    # 将布尔值因子矩阵转换为浮点数矩阵，True设置为1.00，False设置为0.00
    long_float_factor = long_bool_factor.astype(float)
    # 计算最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    long_final_matrix = long_float_factor * returns_df
    # factor_matrix_1 = long_float_factor*factor_df
    # 计算每行的平均收益率
    long_average_returns = long_final_matrix.sum(axis=1) / (2*n)
    # factor_averge_1 =factor_matrix_1.sum(axis=1) / n
    # long_average_returns = long_final_matrix.mean(axis=1)
    # 创建一个新的DataFrame，包含平均收益率
    long_df = pd.DataFrame(long_average_returns, columns=['Average_Returns'])
    # factor_df_1 =  pd.DataFrame(factor_averge_1, columns=['Average Factor'])

    #对因子值矩阵2进行排名，并将排名后n的货币对应的值设置为True，排名前面的设置为False
    short_bool_factor = short_ranked_factor <= n
    # 将布尔值因子矩阵转换为浮点数因子矩阵，True设置为-1.00，False设置为0.00
    short_float_factor = short_bool_factor.astype(float) * -1
    # 计算做空部分的最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    short_final_matrix = short_float_factor * returns_df
    short_float_factor_2 = short_bool_factor.astype(float)
    factor_matrix_2 = short_float_factor_2 * factor_df

    # 计算做空部分每行的平均收益率
    # short_average_returns = short_final_matrix.mean(axis=1)
    short_average_returns = short_final_matrix.sum(axis=1) / (2*n)

    factor_averge_2 = factor_matrix_2.sum(axis=1) / n

    # 创建一个新的DataFrame，包含平均收益率（做空部分）
    short_df = pd.DataFrame(short_average_returns, columns=['Average_Returns'])
    factor_df_2 = pd.DataFrame(factor_averge_2, columns=['Average Factor'])
    # 将做空部分的平均收益率和做多部分的平均收益率相加
    longshort_df = long_df
    # averge_factor_df = factor_df_1 + factor_df_2
    final_matrix = long_float_factor

    #计算提调仓时间的累计收益率
    cumulative_returns_withoutfee = (1 + longshort_df).cumprod()
    # last_longshort_return = cumulative_returns_withoutfee.iloc[-1]['Average_Returns']
    # print("最终净值：", last_longshort_return)

    return longshort_df,cumulative_returns_withoutfee,final_matrix


def normalize_holdings_matrix(holdings_matrix, returns_matrix):
    # 对持仓矩阵和收益率矩阵进行矩阵乘法
    '''

    :param holdings_matrix:持仓矩阵
    :param returns_matrix: 收益矩阵
    :return: 换手率
    '''
    returns_matrix = returns_matrix.shift(-1)
    combined_matrix = holdings_matrix * returns_matrix
    combined_matrix2 = combined_matrix.abs()
    row_sums1 = np.sum(combined_matrix2, axis=1)
    combined_matrix = combined_matrix / np.expand_dims(row_sums1, axis=1)
    turn_matrix =holdings_matrix - holdings_matrix.shift()
    holdings_matrix2 = holdings_matrix.abs()
    row_sums2 = np.sum(holdings_matrix2, axis=1)
    holdings_matrix = holdings_matrix/ np.expand_dims(row_sums2, axis=1)
    # 计算换手矩阵
    # print(holdings_matrix)
    # print(combined_matrix)
    turnover_matrix = holdings_matrix - combined_matrix.shift()
    # 将换手矩阵的值转换为绝对值
    turnover_matrix2 = turnover_matrix.abs()
    # 计算每行的总和
    # row_sums3 = np.sum(turnover_matrix2, axis=1)

    # 将矩阵除以每行的总和实现归一化
    # turnover_matrix2 = turnover_matrix2 / np.expand_dims(row_sums3, axis=1)

    # long_average_returns = long_final_matrix.sum(axis=1) / n
    # 计算绝对值的总和并除以2

    turnover = turnover_matrix2.sum(axis=1)
    turnover[1] = 1
    turn = turnover_matrix2.sum(axis=1) / 2
    turn_mean = np.mean(turn)
    print("换手率:", turn_mean)
    return turnover, turnover_matrix,turn_matrix

def longshort_withfee(factor_name,longshort_df, turnover, cost):
    '''
    计算手续费
    :param longshort_df:
    :param turnover:
    :param cost:
    :return:
    '''
    # print(longshort_df)
    # print(turnover * cost_ * 2)
    # 将换手率转换为 DataFrame，并设置列名为 'Average_Returns'
    turnover = pd.DataFrame(turnover ,columns=['Average_Returns'])
    longshort_df_withfee = longshort_df - turnover * cost * 2
    # print(longshort_df_fee)
    # 计算提调仓时间的累计收益率
    cumulative_returns_withfee = (1 + longshort_df_withfee).cumprod()
    cumulative_returns_withfee.index = pd.to_datetime(cumulative_returns_withfee.index).normalize()
    #绘制累计收益率图像
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns_withfee.index, cumulative_returns_withfee["Average_Returns"],
             linestyle="-", label="strategy")  # 设置折线图颜色为蓝色

    # 设置横轴刻度间隔
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=15))
    plt.xlabel("Time")
    plt.ylabel("value")
    plt.title("cumulative_longshort_df")
    plt.grid(True)
    #生成文件名和保存路径
    folder_path = r'E:\pythonProject\picture'
    file_name = factor_name + '.png'

    # 检查文件夹是否存在，如果不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存图表
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close('all')  # 关闭当前创建的图形

    last_longshort_return = cumulative_returns_withfee.iloc[-1]['Average_Returns']
    print("最终净值:", last_longshort_return)
    return longshort_df_withfee, cumulative_returns_withfee, last_longshort_return


def calculate_ic(returns_df, factor_df):
    # 计算收益率和因子值之间的相关系数
    #
    factor_df = factor_df.shift()
    returns_df = returns_df.transpose()
    factor_df = factor_df.transpose()

    for col in factor_df.columns:
        factor_df[col] = pd.to_numeric(factor_df[col], errors='coerce')

    ic = returns_df.corrwith(factor_df)
    # 计算 IC 序列的均值
    ic_mean = ic.mean()
    # 计算 IC 序列的方差
    ic_var = ic.var()
    # plt.figure(figsize=(10, 6))
    # plt.plot(ic.index, ic.values, marker='o', linestyle='-')
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=15))
    # plt.title('IC Series')
    # plt.xlabel('Time')
    # plt.ylabel('IC Value')
    # plt.grid(True)
    # correlation_matrix = np.corrcoef(returns_df, factor_df)
    # ic= correlation_matrix[0:len(returns_df), len(factor_df):]
    # correlation_matrix = returns_df.corrwith(factor_df, method='pearson')
    # ic_series = correlation_matrix['BTC']

    return ic,ic_mean,ic_var

def perform_t_test(ic_series):

    # 计算样本均值和样本标准误差
    sample_mean = ic_series.mean()
    sample_std = ic_series.std()
    sample_size = len(ic_series)
    # standard_error = sample_std / np.sqrt(sample_size)
    standard_error = sample_std
    # 计算t值
    IR_value = sample_mean / standard_error
    ic_t_value = sample_mean*np.sqrt(sample_size) / standard_error
    print('ic 均值:',sample_mean)
    print('IR:',IR_value)
    print('IC t值:',ic_t_value)
    return IR_value


def calculate_max_drawdown(returns):
    # 计算累计收益率
    returns = returns.squeeze()
    cumulative_returns = (1 + returns).cumprod()
    # 计算每个时间点之前的最大净值
    peak = cumulative_returns.expanding().max()
    # 计算每个时间点的回撤幅度
    drawdown = (cumulative_returns - peak) / peak
    # 计算最大回撤
    max_drawdown = drawdown.min()
    print("最大回撤:", abs(max_drawdown))
    return abs(max_drawdown)

#年化收益率
def calculate_annualized_returns(time_series):
    # 获取时间序列的起始日期和结束日期
    start_date = pd.to_datetime(time_series.index[0])
    end_date = pd.to_datetime(time_series.index[-1])
    # 计算时间序列的长度（以年为单位）
    time_period = (end_date - start_date).days / 365
    # 计算收益率
    total_returns = time_series.iloc[-1] / time_series.iloc[0] - 1
    # 计算年化收益率
    annualized_returns = (1 + total_returns) ** (1 / time_period) - 1
    print('年化收益率:',annualized_returns['Average_Returns'])
    return annualized_returns


#年化波动率
def calculate_annualized_volatility(time_series):
    # 计算每日收益率
    returns = time_series.pct_change()
    # 计算标准差（波动率）
    volatility = returns.std()
    # 计算年化因子
    annualization_factor = np.sqrt(365)  # 假设一年有365天
    # 计算年化波动率
    annualized_volatility = volatility * annualization_factor
    print('年化波动率:',annualized_volatility['Average_Returns'])

    return annualized_volatility


#多空夏普率
def calculate_longshort_sharpe_ratio(annualized_returns,annualized_volatility):
    sharpe_ratio = annualized_returns / annualized_volatility
    print('多空夏普率', sharpe_ratio.item())

    return sharpe_ratio

################################################## 因子研究模块 ######################################
def daily_factor_calling(close, turn):
    BiasFactor10day = daily_factor_script.calculate_10_daily_biass(close)
    BiasFactor15day = daily_factor_script.calculate_15_daily_biass(close)
    BiasFactor20day = daily_factor_script.calculate_20_daily_biass(close)
    BiasFactor30day = daily_factor_script.calculate_30_daily_biass(close)
    BiasFactor50day = daily_factor_script.calculate_50_daily_biass(close)
    BiasFactor100day = daily_factor_script.calculate_100_daily_biass(close)

    MovingAverage5day = daily_factor_script.calculate_5_day_moving_average(close)
    MovingAverage10day = daily_factor_script.calculate_10_day_moving_average(close)
    MovingAverage15day = daily_factor_script.calculate_15_day_moving_average(close)
    MovingAverage30day = daily_factor_script.calculate_30_day_moving_average(close)
    MovingAverage50day = daily_factor_script.calculate_50_day_moving_average(close)
    MovingAverage100day = daily_factor_script.calculate_100_day_moving_average(close)

    MovingTurnover5day = daily_factor_script.calculate_5_day_average_turnover_factor(turn)
    MovingTurnover10day = daily_factor_script.calculate_10_day_average_turnover_factor(turn)
    MovingTurnover15day = daily_factor_script.calculate_15_day_average_turnover_factor(turn)
    MovingTurnover30day = daily_factor_script.calculate_30_day_average_turnover_factor(turn)
    MovingTurnover50day = daily_factor_script.calculate_50_day_average_turnover_factor(turn)
    MovingTurnover100day = daily_factor_script.calculate_100_day_average_turnover_factor(turn)

    BollingerBandUpperLine5day = daily_factor_script.calculate_5_day_upper_bollinger_band(close)
    BollingerBandUpperLine10day = daily_factor_script.calculate_10_day_upper_bollinger_band(close)
    BollingerBandUpperLine15day = daily_factor_script.calculate_15_day_upper_bollinger_band(close)
    BollingerBandUpperLine30day = daily_factor_script.calculate_30_day_upper_bollinger_band(close)
    BollingerBandUpperLine50day = daily_factor_script.calculate_50_day_upper_bollinger_band(close)

    BollingerBandLowerLine5day = daily_factor_script.calculate_5_day_lower_bollinger_band(close)
    BollingerBandLowerLine10day = daily_factor_script.calculate_10_day_lower_bollinger_band(close)
    BollingerBandLowerLine15day = daily_factor_script.calculate_15_day_lower_bollinger_band(close)
    BollingerBandLowerLine30day = daily_factor_script.calculate_30_day_lower_bollinger_band(close)
    BollingerBandLowerLine50day = daily_factor_script.calculate_50_day_lower_bollinger_band(close)

    return (BiasFactor10day, BiasFactor15day, BiasFactor20day, BiasFactor30day, BiasFactor50day, BiasFactor100day,
            MovingAverage5day, MovingAverage10day, MovingAverage15day, MovingAverage30day, MovingAverage50day, MovingAverage100day,
            MovingTurnover5day, MovingTurnover10day, MovingTurnover15day, MovingTurnover30day, MovingTurnover50day, MovingTurnover100day,
            BollingerBandUpperLine5day, BollingerBandUpperLine10day, BollingerBandUpperLine15day, BollingerBandUpperLine30day, BollingerBandUpperLine50day,
            BollingerBandLowerLine5day, BollingerBandLowerLine10day, BollingerBandLowerLine15day, BollingerBandLowerLine30day, BollingerBandLowerLine50day)


#股票排名：股票代码转股票名称
def result_df_to_stock_name(result_df):
    df = pd.read_excel(r"D:\桌面\HuaQiStudy\Data\stock_in_market.xls")
    # 提取第一列和第二列数据
    column1 = df.iloc[:, 0].tolist()
    column2 = df.iloc[:, 1].tolist()
    # 将两列数据合并成字典，形成一一对应的关系
    # stock_dict = dict(zip(column1, column2))
    # 创建一个字典，以股票代码的前6位为键，(完整股票代码, 股票名称)为值
    stock_dict = {str(code)[:6]: (code, name) for code, name in zip(df.iloc[:, 0], df.iloc[:, 1])}
    date_to_select = pd.to_datetime('2023-12-29')
    # 获取特定日期的股票代码数组
    stock_codes_on_date = result_df.loc[date_to_select]['stocks']
    # 通过列表推导式，为每个股票代码获取完整股票代码和股票名称
    # 如果股票代码不在字典中，返回('Unknown', 'Unknown')
    result_list = [stock_dict.get(str(code)[:6], ('Unknown', 'Unknown')) for code in stock_codes_on_date]
    # 将结果列表转换为DataFrame
    result_df = pd.DataFrame(result_list, columns=['Stock_Code', 'Stock_Name'])
    return result_df

#买卖衰退函数
def calculate_stock_occurrences(stock_data, dates):
    # stock_data为[[股票1, 股票2, ...], [股票1, 股票2, ...], ...]
    # dates为datetimeIndex

    daily_averages = {}

    for i in range(len(stock_data)):
        current_stocks = set(stock_data[i])
        stock_occurrences = defaultdict(int)

        for j in range(i + 1, len(stock_data)):
            common_stocks = current_stocks & set(stock_data[j])
            for stock in common_stocks:
                stock_occurrences[stock] += 1

        total_occurrences = sum(stock_occurrences.values())

        if len(current_stocks) > 0:
            daily_average = total_occurrences / len(current_stocks)
            daily_averages[str(dates[i])] = daily_average
        else:
            daily_averages[str(dates[i])] = 0

    return daily_averages


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, Series):
            # If the index is a DatetimeIndex, convert it to strings
            if isinstance(obj.index, DatetimeIndex):
                index_as_str = obj.index.strftime('%Y-%m-%d').tolist()
            else:
                index_as_str = obj.index.tolist()
            return dict(zip(index_as_str, obj.replace({np.nan: 0}).tolist()))
        elif isinstance(obj, DataFrame):
            # If it's a DataFrame with a single column, convert to a Series first
            if len(obj.columns) == 1:
                # Convert the DataFrame to a Series and then handle as a Series
                series = obj.squeeze()
                return self.default(series)  # Recursively call default
            else:
                # Convert the DataFrame to a list of row dictionaries
                return obj.replace({np.nan: 0}).to_dict('records')
        elif isinstance(obj, float) and np.isnan(obj):
            return 0  # Replace NaN with 0 for floating-point numbers
        return JSONEncoder.default(self, obj)

def calculate_price_liquidity( amt, retn,window):
    rolling_amt_sum = amt.rolling(window).sum()  # 计算前window个amt之和
    rolling_retn_std = retn.rolling(window).std()  # 计算retn的标准差

    price_elasticity = rolling_amt_sum / rolling_retn_std  # 计算价格弹性

    return price_elasticity

if __name__ == "__main__":
    ######################基本准备部分####################
    print('basic数据导入开始')

    open1, close, high, low, vol, amt, score_data, ret, retn, market = ReadRawdata()
    turn = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\全a指数换手率turn.pkl")
    # 开盘日
    open_date = open1.index

    #股票交集操作
    score_dict = {stock_code: float(score) for stock_code, score in score_data['data']}
    # 使用股票代码作为列名创建一个只有一行的DataFrame
    score_df = pd.DataFrame([score_dict])
    # 假设factor_df已经定义，且其索引为datetime
    # 创建一个与factor_df行数相同，列名为score_df列名的空DataFrame
    aligned_score_df = pd.DataFrame(index=open1.index, columns=score_df.columns)
    # 将score_df的得分复制到每一行
    for col in aligned_score_df.columns:
        aligned_score_df[col] = score_df[col].values[0]

    open1 = factor_correct(open1, aligned_score_df)
    close = factor_correct(close, aligned_score_df)
    ret = factor_correct(ret, aligned_score_df)
    retn = factor_correct(retn, aligned_score_df)
    amt = factor_correct(amt, aligned_score_df)
    turn = factor_correct(turn,aligned_score_df)

    ##参数导入
    if len(sys.argv) > 3:  # 检查是否有足够的参数
        interval = int(sys.argv[1])  # 将字符串参数转换为整数
        tax = float(sys.argv[2])
        calculate_long_short_returns_select = bool(sys.argv[3])
    else:
        interval = 1  # 默认值
        tax = 0.0002
        calculate_long_short_returns_select = True

    #数据库配置文件
    config_path = 'config.ini'
    # 访问数据库
    # 创建配置解析器对象
    config = configparser.ConfigParser()
    # 读取配置文件'config.ini'
    config.read(config_path)
    # try:
    # 创建数据库连接字符串
    db_uri = f"mysql+mysqlconnector://{config.get('mysql', 'user')}:{config.get('mysql', 'password')}@{config.get('mysql', 'host')}:{config.get('mysql', 'port')}/{config.get('mysql', 'database')}"
    # 创建SQLAlchemy引擎
    engine = create_engine(db_uri)

    #写入数据库以及产生json文件相关参数
    #选择大模型因子_title_score; _title_pos_rate; _predictions; _research_title_score
    suffix = '_title_pos_rate'
    table_name = 'daily_factor_table'
    output_folder = "D:/桌面/HuaQiStudy/Output_daily_factor"
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    #开始时间
    start_date = '2013-01-04 00:00:00'
    end_date = '2023-12-31 00:00:00'

    (BiasFactor10day, BiasFactor15day, BiasFactor20day, BiasFactor30day, BiasFactor50day, BiasFactor100day,
     MovingAverage5day, MovingAverage10day, MovingAverage15day, MovingAverage30day, MovingAverage50day, MovingAverage100day,
     MovingTurnover5day, MovingTurnover10day, MovingTurnover15day, MovingTurnover30day, MovingTurnover50day, MovingTurnover100day,
     BollingerBandUpperLine5day, BollingerBandUpperLine10day, BollingerBandUpperLine15day, BollingerBandUpperLine30day, BollingerBandUpperLine50day,
     BollingerBandLowerLine5day, BollingerBandLowerLine10day, BollingerBandLowerLine15day, BollingerBandLowerLine30day, BollingerBandLowerLine50day) = daily_factor_calling(close, turn)

    daily_factors_lists = [BiasFactor10day, BiasFactor15day, BiasFactor20day, BiasFactor30day, BiasFactor50day, BiasFactor100day,
                     MovingAverage5day, MovingAverage10day, MovingAverage15day, MovingAverage30day, MovingAverage50day, MovingAverage100day,
                     MovingTurnover5day, MovingTurnover10day, MovingTurnover15day, MovingTurnover30day, MovingTurnover50day, MovingTurnover100day,
                     BollingerBandUpperLine5day, BollingerBandUpperLine10day, BollingerBandUpperLine15day, BollingerBandUpperLine30day, BollingerBandUpperLine50day,
                     BollingerBandLowerLine5day, BollingerBandLowerLine10day, BollingerBandLowerLine15day, BollingerBandLowerLine30day, BollingerBandLowerLine50day]

    # 流动性因子与反转 #####
    liquidity = calculate_price_liquidity(amt,retn,window=10)
    liquidity = -liquidity

    daily_factors_names_dict = {
        'BiasFactor10day': BiasFactor10day,
        'BiasFactor15day': BiasFactor15day,
        'BiasFactor20day': BiasFactor20day,
        'BiasFactor30day': BiasFactor30day,
        'BiasFactor50day': BiasFactor50day,
        'BiasFactor100day': BiasFactor100day,

        'MovingAverage5day': MovingAverage5day,
        'MovingAverage10day': MovingAverage10day,
        'MovingAverage15day': MovingAverage15day,
        'MovingAverage30day': MovingAverage30day,
        'MovingAverage50day': MovingAverage50day,
        'MovingAverage100day': MovingAverage100day,

        'MovingTurnover5day': MovingTurnover5day,
        'MovingTurnover10day': MovingTurnover10day,
        'MovingTurnover15day': MovingTurnover15day,
        'MovingTurnover30day': MovingTurnover30day,
        'MovingTurnover50day': MovingTurnover50day,
        'MovingTurnover100day': MovingTurnover100day,

        'BollingerBandUpperLine5day': BollingerBandUpperLine5day,
        'BollingerBandUpperLine10day': BollingerBandUpperLine10day,
        'BollingerBandUpperLine15day': BollingerBandUpperLine15day,
        'BollingerBandUpperLine30day': BollingerBandUpperLine30day,
        'BollingerBandUpperLine50day': BollingerBandUpperLine50day,

        'BollingerBandLowerLine5day': BollingerBandLowerLine5day,
        'BollingerBandLowerLine10day': BollingerBandLowerLine10day,
        'BollingerBandLowerLine15day': BollingerBandLowerLine15day,
        'BollingerBandLowerLine30day': BollingerBandLowerLine30day,
        'BollingerBandLowerLine50day': BollingerBandLowerLine50day,
        'liquidity': liquidity
    }

    daily_factors_names = {

        'BiasFactor30day',
        'BiasFactor50day',
        'BiasFactor100day',

        'MovingAverage30day',
        'MovingAverage50day',
        'MovingAverage100day',

        'MovingTurnover5day',
        'MovingTurnover10day',
        'MovingTurnover15day',
        'MovingTurnover30day',
        'MovingTurnover50day',
        'MovingTurnover100day',

        'BollingerBandUpperLine15day',
        'BollingerBandUpperLine30day',
        'BollingerBandUpperLine50day',

        'BollingerBandLowerLine15day',
        'BollingerBandLowerLine30day',
        'BollingerBandLowerLine50day',
    }
    print('basic数据导入结束')

    #########################回测运行部分##################
    print('回测开始')
    # balance_dates = generate_rebalance_dates(start_date, end_date, interval, open_date)
    # #如果是季度财报数据，window=1是跨一个季度了
    # daily_returns = calculate_returns(close,window=1)#数据是一天级别数据

    #因子循环
    for daily_factor_name in daily_factors_names:
        balance_dates = generate_rebalance_dates(start_date, end_date, interval, open_date)
        # 如果是季度财报数据，window=1是跨一个季度了
        daily_returns = calculate_returns(close, window=1)  # 数据是一天级别数据
        #根据因子名称获取因子df
        print(daily_factor_name)
        daily_factor = daily_factors_names_dict[daily_factor_name]
        daily_factor = -daily_factor.astype(float)
        daily_factor = factor_correct(daily_factor, open1)
        df_factor,df_retn,df_ret,df_market = calculate_previous_factor(balance_dates,daily_returns,daily_factor,market)
        if calculate_long_short_returns_select:
            longshort_df, cumulative_returns_withoutfee, final_matrix, final_matrix_20 = calculate_long_short_returns(retn, df_factor, aligned_score_df, 50)  # 计算多空收益率
        else:
            longshort_df, cumulative_returns_withoutfee, final_matrix = calculate_long_returns(df_retn, df_factor, aligned_score_df, 50)  # 计算多头收益率

        ##回传前端
        turnover, turnover_matrix, turn_matrix = normalize_holdings_matrix(final_matrix, df_ret)

        # tax为印花税，这个函数净值偏大
        longshort_df_with_fee, cumulative_returns_with_fee, last_longshort_return = longshort_withfee(daily_factor_name,longshort_df, turnover, tax)

        ic, ic_mean, ic_var = calculate_ic(df_retn, df_factor)

        IR = perform_t_test(ic)

        max_drawdown = calculate_max_drawdown(longshort_df_with_fee)

        longshort_df_with_fee = longshort_df_with_fee.squeeze()

        annualized_returns = calculate_annualized_returns(cumulative_returns_with_fee)
        annualized_volatility = calculate_annualized_volatility(cumulative_returns_with_fee)
        sharpe_ratio = calculate_longshort_sharpe_ratio(annualized_returns, annualized_volatility)

        result_long_df = find_stocks_with_value_one(final_matrix_20.T)
        result_short_df = find_stocks_with_value_one((-final_matrix_20).T)
        result_long_return = result_df_to_stock_name(result_long_df)
        result_short_return = result_df_to_stock_name(result_short_df)
        result_df = find_stocks_with_value_one(final_matrix.T)
        all_data = result_df['stocks'].tolist()
        dates = result_df.index
        daily_averages = calculate_stock_occurrences(all_data, dates)

        # 输入到数据库(注意daily_factor是dataframe，要提取出字符串daily_factor_name)
        #    annualized_returns_season = float(annualized_returns_season["Average_Returns"])
#KeyError: 'Average_Returns'
        store_results_to_database(engine, daily_factor_name, table_name, last_longshort_return, annualized_returns, annualized_volatility, sharpe_ratio, ic_mean, IR, suffix)
        output_data = {
                "turnover": turnover,
                "longshort_df_with_fee": longshort_df_with_fee.squeeze(),  # 注意这里已经转换为Series
                "cumulative_returns_with_fee": cumulative_returns_with_fee,
                "ic": ic,
                "ic_mean": ic_mean,
                "ic_var": ic_var,
                "IR": IR,
                "max_drawdown": max_drawdown,
                "annualized_returns": annualized_returns,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "result_long_return": result_long_return,
                "result_short_return": result_short_return,
                "daily_averages": daily_averages,
                "market": market
            }
        # 指定文件名
        file_name_suffix = '.json'
        file_name = f'{daily_factor_name}_title_pos_rate{file_name_suffix}'
        # 使用os.path.join来构建文件的完整路径
        file_path = os.path.join(output_folder, file_name)
        # 打开一个文件用于写入，'w'表示写模式
        with open(file_path, 'w') as f_i_l_e:
            # 使用json.dump将数据写入文件，指定CustomEncoder处理复杂对象
            json.dump(output_data, f_i_l_e, cls=CustomEncoder, indent=4)
    print('回测结束')

