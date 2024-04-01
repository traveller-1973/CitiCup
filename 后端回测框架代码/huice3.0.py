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
#
# logging.basicConfig(filename='my_app_log.log',  # 日志文件名
#                     filemode='w',  # 'w' 覆盖现有文件，'a' 追加到现有文件
#                     level=print,  # 记录 INFO 级别及以上的日志
#                     format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
#                     datefmt='%Y-%m-%d %H:%M:%S')  # 时间格式

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

def store_results_to_database(engine ,factor_column, table_name, last_longshort_return, annualized_returns_season, annualized_volatility_season, sharpe_ratio_season, ic_mean_season, IR_season):

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
    factor_column = factor_column + "_title_score"

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
    score = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\title_score.pkl")
    # open1 = pd.read_pickle("/HuaQi/Data/stock_open.pkl")
    # close = pd.read_pickle("/HuaQi/Data/stock_close.pkl")
    # high = pd.read_pickle("/HuaQi/Data/stock_high.pkl")
    # low = pd.read_pickle("/HuaQi/Data/stock_low.pkl")
    # vol = pd.read_pickle("/HuaQi/Data/stock_vol.pkl")
    # amt = pd.read_pickle("/HuaQi/Data/stock_amt.pkl")
    # score = pd.read_pickle("/HuaQi/Data/title_score.pkl")
    ret = close / close.shift()
    retn = ret - 1
    # market = ret.mean(axis=1)#从沪深300直接读取
    market = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\ZhongZhengQuanZhi.pkl")
    return open1, close, high, low, vol, amt, score, ret, retn, market

# def Category(folder_path):
#     '''
# 暂时不用
#     :param folder_path:
#     :return:
#     '''
#     # 读取行业矩阵文件夹中的所有文件
#     files = os.listdir(folder_path)
#     # 创建一个字典来存储行业 DataFrame
#     industry_matrices_dict = {}
#     # 遍历行业矩阵文件夹中的每个文件
#     for file in files:
#         # 构建文件的完整路径
#         file_path = os.path.join(folder_path, file)
#         # 读取行业矩阵并存储到字典中，键是文件名（去掉后缀），值是 DataFrame
#         industry_matrices_dict[file.replace("_matrix.pkl", "")] = pd.read_pickle(file_path)
#     return industry_matrices_dict

# def extract_rows(df):
#     '''
#     暂时不用
#     :param df:
#     :return:
#     '''
#     # 输入月度数据df，输出季度数据df
#     # 按index提取时间数据
#     extracted_rows = []
#     for i in range(0, len(df), 3):
#         extracted_rows.append(df.iloc[i])
#     extracted_df = pd.DataFrame(extracted_rows)
#     return extracted_df

# def select_columns_by_index(zzquanzhi, data_df):
#     '''
#     暂时不用
#     :param zzquanzhi:
#     :param data_df:
#     :return:
#     '''
#     # 输入中证/沪深成分股矩阵和data
#     # 按列从data中提取中证/沪深成分股
#     selected_columns = []
#     # 遍历 zzquanzhi DataFrame 中的每个值，即要挑选的列的序号
#     for index_value in zzquanzhi.iloc[:, 0]:
#         # 根据序号从 data_df 中挑选对应的列，并添加到 selected_columns 列表中
#         selected_columns.append(data_df.iloc[:, index_value])
#         # print(index_value)
#     # 将选中的列组合成一个新的 DataFrame
#     selected_df = pd.DataFrame(selected_columns).T
#
#     return selected_df
# def fill_nan_with_row_mean(matrix):
#     matrix_without_nan = matrix.dropna()
#     # print(matrix_without_nan)
#     # 使用每一行的均值填充NaN值
#     filled_matrix = matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
#
#     return filled_matrix
# def market_neutralize(factor, mak_cp):
#     '''
#     市值中性化，以后用
#     :param factor:
#     :param mak_cp:
#     :return:
#     '''
#     # 输入因子矩阵和市值矩阵，输出中性化后的因子矩阵
#     if mak_cp.isna().any().any():
#         print("DataFrame中存在NaN值")
#         return None
#
#     print("DataFrame中没有NaN值")
#
#     # 添加截距项
#     X = sm.add_constant(mak_cp)
#
#     # 使用线性回归拟合市值因子和因子矩阵之间的关系
#     model = sm.OLS(factor.values, X.values)
#     results = model.fit()
#
#     # 提取回归的残差
#     residuals = results.resid
#
#     # 将残差转换回DataFrame
#     residuals_df = pd.DataFrame(residuals, index=factor.index, columns=factor.columns)
#
#     return residuals_df

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

# def industry_neutralize(cs_indus_code,fator):# 输入行业矩阵和因子矩阵/二者长度相同/横轴股票代码/纵轴时间序列
#     result = pd.DataFrame()
#     '''
#     行业中性化，以后用
#     '''
#
#     # 遍历每一行（时间）
#     for index in cs_indus_code.index:
#         # 提取当前时间点的行业分类数据
#         industry_data = cs_indus_code.loc[index]
#         fator_data = fator.loc[index]
#         # 使用 get_dummies() 方法将行业分类数据转换为虚拟变量矩阵
#         industry_dummies_current = pd.get_dummies(industry_data)
#         # print(industry_dummies_current)
#         X = industry_dummies_current
#         Y = fator_data.T # 存储为 DataFrame
#         # print(Y)
#         # 添加截距项
#         X = sm.add_constant(X)
#
#         # 删除包含缺失值的行
#         X = X.dropna()
#
#         # 确保所有数据类型为 float64
#         X = X.astype(float)
#         Y = Y.astype(float)
#
#         # 拟合多元线性回归模型
#         model = sm.OLS(Y, X).fit()
#         # 查看残差矩阵
#         residuals = model.resid
#         # 将当前时间点的虚拟变量矩阵添加到结果 DataFrame 中
#         result =pd.concat([result,residuals],axis=1)
#         # print('行业中性化完成')
#     result = pd.DataFrame(result)
#     return result
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

#季频因子补全算法

def season_factor_correct(factor_df, retn):
    #提取季频起始时间
    start_date = factor_df.index[0]
    end_date = factor_df.index[-1]
    middle_df_index = retn[(retn.index >= start_date) & (retn.index <= end_date)].index
    middle_df = pd.DataFrame(index=middle_df_index, columns=factor_df.columns)
    # Copy the data from 'long_float_factor' to the new DataFrame using the same index
    for date in factor_df.index:
        if date in middle_df.index:
            middle_df.loc[date] = factor_df.loc[date]

    # Forward fill the NaN values using the next valid observation
    middle_df.fillna(method='bfill', inplace=True)
    middle_df.fillna(0, inplace=True)
    return middle_df


def calculate_long_short_returns(returns_df, factor_df_test, aligned_score_df,n):

    '''
    计算多头和空头收益率，算手续费

    :param returns_df:
    :param factor_df: 因子矩阵
    :param aligned_score_df 得分矩阵
    :param n:
    :return:
    '''
    # # 将因子值矩阵向前移动，得到因子值矩阵2
    factor_df = factor_df_test.shift(1)
    factor_df = factor_df.applymap(lambda x: np.nan if x is None else x)
    for col in factor_df.columns:
        factor_df[col] = pd.to_numeric(factor_df[col], errors='coerce')

    #归一化(注释这条就回归单因子)
    factor_df = min_max_normalize(factor_df)

    # 取出第一行进行归一化
    first_row = aligned_score_df.iloc[0]
    # 对第一行数据应用min-max归一化
    normalized_first_row = 2*(first_row - first_row.min()) / (first_row.max() - first_row.min()) -1

    # 将归一化的第一行复制到aligned_score_df的每一行
    aligned_score_df = pd.DataFrame([normalized_first_row] * len(factor_df), index=factor_df.index)

    #这里要考虑factor_correct，因为矩阵格式不对齐
    # factor_df = factor_correct(factor_df, aligned_score_df)
    aligned_score_df = factor_correct(aligned_score_df, factor_df)
    double_factor_df = 0.5*factor_df+0.5*aligned_score_df

    # 对因子值矩阵2进行排名，并将排名前n的货币对应的值设置为True，其他值设置为False
    long_ranked_factor = double_factor_df.rank(axis=1, ascending=False)#降序
    short_ranked_factor = double_factor_df.rank(axis=1,ascending = True)#升序
    long_bool_factor = long_ranked_factor <= n

    #因子最大的20支股票输出
    long_bool_20_factor = long_ranked_factor <= 20
    long_float_20_factor = long_bool_20_factor.astype(float)



    # 将布尔值因子矩阵转换为浮点数矩阵，True设置为1.00，False设置为0.00
    long_float_factor = long_bool_factor.astype(float)
    long_float_factor = season_factor_correct(long_float_factor, returns_df)

    #截取列
    long_float_factor = factor_correct(long_float_factor,returns_df)
    start_time = long_float_factor.index[0]
    end_time = long_float_factor.index[-1]
    returns_df = returns_df.loc[start_time:end_time]
    # 计算最终矩阵，将浮点数因子矩阵和收益率矩阵相乘，returns_df换retn
    long_final_matrix = long_float_factor * returns_df


    # factor_matrix_1 = long_float_factor*factor_df
    # 计算每行的平均收益率
    long_average_returns = long_final_matrix.sum(axis=1) / (2*n)
    # factor_averge_1 =factor_matrix_1.sum(axis=1) / n
    # long_average_returns = long_final_matrix.mean(axis=1)
    # 创建一个新的DataFrame，包含平均收益率
    long_df = pd.DataFrame(long_average_returns, columns=['Average_Returns'])
    # factor_df_1 =  pd.DataFrame(factor_averge_1, columns=['Average_Factor'])

    #对因子值矩阵2进行排名，并将排名后n的货币对应的值设置为True，排名前面的设置为False
    short_bool_factor = short_ranked_factor <= n

    #因子最小的20支股票输出
    short_bool_20_factor = short_ranked_factor <= 20
    short_float_20_factor = short_bool_20_factor.astype(float) * -1

    # 将布尔值因子矩阵转换为浮点数因子矩阵，True设置为-1.00，False设置为0.00
    short_float_factor = short_bool_factor.astype(float) * -1
    short_float_factor = season_factor_correct(short_float_factor, returns_df)

    # 截取列
    short_float_factor = factor_correct(short_float_factor, returns_df)
    start_time = short_float_factor.index[0]
    end_time = short_float_factor.index[-1]
    returns_df = returns_df.loc[start_time:end_time]

    # 计算做空部分的最终矩阵，将浮点数因子矩阵和收益率矩阵相乘     returns_df-----retn
    short_final_matrix = short_float_factor * returns_df

    short_float_factor_2 = short_bool_factor.astype(float)

    #出现了None与浮点数相乘
    factor_matrix_2 = short_float_factor_2 * double_factor_df

    # 计算做空部分每行的平均收益率
    # short_average_returns = short_final_matrix.mean(axis=1)
    short_average_returns = short_final_matrix.sum(axis=1) / (2*n)
    factor_averge_2 = factor_matrix_2.sum(axis=1) / n

    # 创建一个新的DataFrame，包含平均收益率（做空部分）
    short_df = pd.DataFrame(short_average_returns, columns=['Average_Returns'])
    factor_df_2 = pd.DataFrame(factor_averge_2, columns=['Average_Factor'])
    # 将做空部分的平均收益率和做多部分的平均收益率相加
    longshort_df = long_df + short_df
    # averge_factor_df = factor_df_1 + factor_df_2

    ##long_float_factor和short_float_factor，用季度最后一天的因子
    final_matrix = long_float_factor + short_float_factor


    final_matrix_20 = long_float_20_factor + short_float_20_factor
    #计算提调仓时间的累计收益率
    cumulative_returns_withoutfee = (1 + longshort_df).cumprod()
    # last_longshort_return = cumulative_returns_withoutfee.iloc[-1]['Average_Returns']
    # print("最终净值：", last_longshort_return)

    return longshort_df,cumulative_returns_withoutfee,final_matrix, final_matrix_20

# returns_df-----retn
def calculate_long_returns(returns_df, factor_df, aligned_score_df, n):
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

    # 归一化
    factor_df = min_max_normalize(factor_df)
    aligned_score_df = min_max_normalize(aligned_score_df)
    double_factor_df = 0.5 * factor_df + 0.5 * aligned_score_df

    # 对因子值矩阵2进行排名，并将排名前n的货币对应的值设置为True，其他值设置为False
    long_ranked_factor = double_factor_df.rank(axis=1, ascending=False)  # 降序
    short_ranked_factor = double_factor_df.rank(axis=1, ascending=True)  # 升序

    long_bool_factor = long_ranked_factor <= n
    long_bool_20_factor = long_ranked_factor <= 20


    # 将布尔值因子矩阵转换为浮点数矩阵，True设置为1.00，False设置为0.00
    long_float_factor = long_bool_factor.astype(float)
    long_float_20_factor = long_bool_20_factor.astype(float)


    # 计算最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    long_final_matrix = long_float_factor * returns_df
    # factor_matrix_1 = long_float_factor*factor_df
    # 计算每行的平均收益率
    long_average_returns = long_final_matrix.sum(axis=1) / (2*n)
    # factor_averge_1 =factor_matrix_1.sum(axis=1) / n
    # long_average_returns = long_final_matrix.mean(axis=1)
    # 创建一个新的DataFrame，包含平均收益率
    long_df = pd.DataFrame(long_average_returns, columns=['Average_Returns'])
    # factor_df_1 =  pd.DataFrame(factor_averge_1, columns=['Average_Factor'])

    #对因子值矩阵2进行排名，并将排名后n的货币对应的值设置为True，排名前面的设置为False
    short_bool_factor = short_ranked_factor <= n
    # 将布尔值因子矩阵转换为浮点数因子矩阵，True设置为-1.00，False设置为0.00
    short_float_factor = short_bool_factor.astype(float) * -1
    # 计算做空部分的最终矩阵，将浮点数因子矩阵和收益率矩阵相乘
    short_final_matrix = short_float_factor * returns_df
    short_float_factor_2 = short_bool_factor.astype(float)
    factor_matrix_2 = short_float_factor_2 * double_factor_df
    # 计算做空部分每行的平均收益率
    # short_average_returns = short_final_matrix.mean(axis=1)
    short_average_returns = short_final_matrix.sum(axis=1) / (2*n)
    factor_averge_2 = factor_matrix_2.sum(axis=1) / n

    # 创建一个新的DataFrame，包含平均收益率（做空部分）
    short_df = pd.DataFrame(short_average_returns, columns=['Average_Returns'])
    factor_df_2 = pd.DataFrame(factor_averge_2, columns=['Average_Factor'])
    # 将做空部分的平均收益率和做多部分的平均收益率相加
    longshort_df = long_df
    # averge_factor_df = factor_df_1 + factor_df_2
    final_matrix = long_float_factor
    final_matrix_20 = long_float_20_factor

    #计算提调仓时间的累计收益率
    cumulative_returns_withoutfee = (1 + longshort_df).cumprod()
    # last_longshort_return = cumulative_returns_withoutfee.iloc[-1]['Average_Returns']
    # print("最终净值：", last_longshort_return)

    return longshort_df,cumulative_returns_withoutfee,final_matrix, final_matrix_20


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
    # print("换手率:", turn_mean)
    print("换手率: %s", turn_mean)
    return turnover, turnover_matrix,turn_matrix

def longshort_withfee(factor_name ,longshort_df, turnover, cost):
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
    # plt.plot(cumulative_returns_withfee.index, cumulative_returns_withfee['Average_Returns'])
    # plt.xlabel('时间')
    # plt.ylabel('累计收益率')
    # plt.title('累计收益率图像')

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns_withfee.index, cumulative_returns_withfee["Average_Returns"],
             linestyle="-", label="strategy")  # 设置折线图颜色为蓝色

    #设置横轴刻度间隔
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=15))
    plt.xlabel("Time")
    plt.ylabel("value")
    plt.title(factor_name)
    plt.grid(True)


    # 生成文件名和保存路径
    folder_path = r'E:\pythonProject\picture'
    file_name = factor_name + '.png'

    # 检查文件夹是否存在，如果不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存图表
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close('all')  # 关闭当前创建的图形
    # plt.show()
    last_longshort_return = cumulative_returns_withfee.iloc[-1]['Average_Returns']
    # print("最终净值:", last_longshort_return)
    print("最终净值: %s", last_longshort_return)
    # input("Enter Please...")
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


#rankic
def calculate_rankic(returns_df, factor_df):
    # 对因子值矩阵进行排名
    ranked_factor_df = factor_df.rank()

    # 计算收益率和因子值之间的相关系数
    factor_df = ranked_factor_df.shift()
    returns_df = returns_df.transpose()
    factor_df = factor_df.transpose()
    ic = returns_df.corrwith(factor_df)

    # 计算 Rank IC 序列的均值
    ic_mean = ic.mean()

    # 计算 Rank IC 序列的方差
    ic_var = ic.var()

    # 计算 Rank IR（Rank Information Ratio）
    rank_ir = ic_mean / ic_var

    # 打印 Rank IC 和 Rank IR
    # print("Rank IC:", ic)
    # print("Rank IC Mean:", ic_mean)
    # print("Rank IC Variance:", ic_var)
    # print("Rank IR:", rank_ir)
    print("Rank IC: %s", ic)
    print("Rank IC Mean: %s", ic_mean)
    print("Rank IC Variance: %s", ic_var)
    print("Rank IR: %s", rank_ir)

    return ic, ic_mean, ic_var
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
    # print('ic 均值:',sample_mean)
    # print('IR:',IR_value)
    # print('IC t值:',ic_t_value)
    print("ic 均值: %s", sample_mean)
    print("IR: %s", IR_value)
    print("IC t值: %s", ic_t_value)



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
    # print("最大回撤:", abs(max_drawdown))
    print("最大回撤: %s", abs(max_drawdown))

    return abs(max_drawdown)
#
# def plot_market_returns(market_returns):
#     # 获取日期索引和市场收益率数据
#     dates = market_returns.index
#     market_returns = market_returns.values
#
#     # 创建折线图
#     # plt.plot(dates, market_returns, linestyle="-", color="yellow",label="market")
#
#     # 设置横轴刻度间隔为每两个月
#     # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
#
#     # plt.xlabel("Time")
#     # plt.ylabel("Market Returns")
#     # plt.title("Market Returns")
#     # plt.grid(False)  # 关闭网格线

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
    # print('年化收益率:',annualized_returns['Average_Returns'])
    print('年化收益率: %s', annualized_returns['Average_Returns'])
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
    # print('年化波动率:',annualized_volatility['Average_Returns'])
    print('年化波动率: %s', annualized_volatility['Average_Returns'])

    return annualized_volatility


#多空夏普率
def calculate_longshort_sharpe_ratio(annualized_returns,annualized_volatility):
    sharpe_ratio = annualized_returns / annualized_volatility
    # print('多空夏普率', sharpe_ratio.item())
    print('多空夏普率: %s', sharpe_ratio.item())

    return sharpe_ratio

# def fama_macbeth(return_series, factor_matrix):
#     n_times, n_assets = return_series.shape
#
#     # 存储每个时间点上的因子贝塔
#     betas = np.zeros(n_times)
#
#     for t in range(n_times):
#         # 取当前时间点的收益率和因子值
#         y = return_series[t, :]
#         X = sm.add_constant(factor_matrix.iloc[t, :])
#
#         # 执行多元OLS回归
#         model = sm.OLS(y, X)
#         results = model.fit()
#
#         # 提取因子暴露（贝塔）
#         beta = results.params[1]
#
#         # 存储贝塔值
#         betas[t] = beta
#
#     # 计算总因子暴露的均值
#     avg_beta = np.mean(betas)
#
#     # 绘制散点图
#     # plt.scatter(range(n_times), betas, alpha=0.5)
#     # plt.axhline(avg_beta, color='red', linestyle='dashed', linewidth=2, label='Avg Beta')
#     # plt.title('Fama-MacBeth Regression Results')
#     # plt.xlabel('Time')
#     # plt.ylabel('Beta')
#     # plt.legend()
#     # plt.show()
#
#     return betas

################################################## 因子研究模块 ######################################

# # 价格流动性
# def calculate_price_liquidity( amt, retn,window):
#     rolling_amt_sum = amt.rolling(window).sum()  # 计算前window个amt之和
#     rolling_retn_std = retn.rolling(window).std()  # 计算retn的标准差
#
#     price_elasticity = rolling_amt_sum / rolling_retn_std  # 计算价格弹性
#
#     return price_elasticity

# # 平滑动量
# def calculate_momentum(retn,close, window):
#     cumulative_returns = calculate_returns(close,window)
#     absolute_sum = retn.abs().rolling(window).sum()  # 计算窗口内所有收益率的绝对值加总
#
#     smooth_momentum = cumulative_returns / absolute_sum  # 计算smooth动量因子
#
#     return smooth_momentum

# def calculate_rolling_retn(returns, window):
#     rolling_std = returns.rolling(window).std()  # 计算收益率的滚动标准差
#     rolling_mean = returns.rolling(window).mean()  # 计算收益率的滚动均值
#     return rolling_std, rolling_mean


#日收益率因子时序性与增强（开源证券）
# def calculate_time_series_enhanced_factors(returns_df,window):
#     # 第一步：统计涨跌幅度和时间信息
#     ret = returns_df
#     ret = ret.reset_index(drop = True)
#     # 计算收益率和对应的数字排序
#     positive_ret = ret[ret > 0]
#     negative_ret = ret[ret < 0]
#     positive_ret_index = positive_ret.index
#     negative_ret_index = negative_ret.index
#     # 计算收益率之和
#     positive_sum = positive_ret.sum()
#     positive_ret= ret[ret > 0].T
#     negative_sum = negative_ret.sum()
#     negative_ret = ret[ret < 0].T
#     # 计算因子值
#
#     positive_factor0 = positive_ret_index*positive_ret
#     positive_factor = positive_factor0.divide(positive_ret.sum(axis=1), axis=0)
#     negative_factor0 = negative_ret_index*negative_ret
#     negative_factor = negative_factor0.divide(positive_ret.sum(axis=1), axis=0)
#     #把positive_factor和negative_factor中的NAN值设置为0
#     positive_factor = positive_factor.fillna(0)
#     negative_factor = negative_factor.fillna(0)
#     # 合并正负因子值
#     # 第四步：计算跌幅时间重心偏离因子
#     residuals = (positive_factor - negative_factor)  # 回归残差
#
#     residuals = residuals.T
#     enhanced_factor = residuals.diff(window) / residuals.shift(window) # 取残差的20日均值作为选股因子
#
#     # 重新定义横轴为资产名称
#     # enhanced_factor.columns = ret.columns
#     enhanced_factor.index = returns_df.index
#     return enhanced_factor

# # 因子中性化
# def factor_neutralization(factor_matrix, industry_matrices_dict):
#     # 初始化中性化因子矩阵的 DataFrame
#     neutralized_factor_matrix = pd.DataFrame(0,index=factor_matrix.index, columns=factor_matrix.columns)
#
#     # 遍历每个行业矩阵
#     for industry, industry_matrix in industry_matrices_dict.items():
#         # 将行业矩阵与因子矩阵相乘
#         industry_factor_product = industry_matrix * factor_matrix
#         # 计算每行的均值
#         row_means = np.mean(industry_factor_product, axis=1)
#
#         # 减去每行的均值
#         neutralized_factor_matrix += industry_factor_product.sub(row_means, axis=0)
#
#     return neutralized_factor_matrix
# def calculate_S_matrix(returns_matrix):
#     median_returns = np.nanmedian(returns_matrix, axis=0)  # 计算每列的中位数，忽略NaN值
#     S_matrix = np.abs(returns_matrix - median_returns)  # 计算差异矩阵
#
#     return S_matrix,median_returns
#
# # 自信序列因子
# # 自信序列因子
# #
# def calculate_CP_Intraday(daily_returns):
#     # 计算收益率的均值和标准差
#     mean_retn = daily_returns.mean(axis=0)
#     std_retn = daily_returns.std(axis=0)
#
#     # 计算快速上涨区间和快速下跌区间的阈值
#     upper_threshold = mean_retn + std_retn
#     lower_threshold = mean_retn - std_retn
#
#     # 获取分钟序号
#     minutes = np.arange(daily_returns.shape[0])
#
#     # 初始化 CP_Intraday 矩阵
#     CP_Intraday = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns)
#
#     # 计算快速上涨区间和快速下跌区间的分钟序号中位数差值
#     for asset in daily_returns.columns:
#         asset_returns = daily_returns[asset].values
#         up_indices = np.where(asset_returns > upper_threshold[asset])[0]
#         down_indices = np.where(asset_returns < lower_threshold[asset])[0]
#         up_median = np.median(minutes[up_indices])
#         down_median = np.median(minutes[down_indices])
#         CP_Intraday[asset] = down_median - up_median
#
#     return CP_Intraday
#
# def calculate_returns_stats(returns_matrix,window_size):
#
#     returns_array = returns_matrix.to_numpy()
#     n, m = returns_array.shape
#     rolling_mean = np.empty((n, m))
#     rolling_std = np.empty((n, m))
#     rolling_mean[:] = np.nan
#     rolling_std[:] = np.nan
#
#     returns_array = returns_matrix.to_numpy()
#
#     for i in range(window_size - 1, n):
#         rolling_mean[i, :] = np.mean(returns_array[i - window_size + 1:i + 1, :], axis=0)
#         rolling_std[i, :] = np.std(returns_array[i - window_size + 1:i + 1, :], axis=0)
#
#     rolling_mean = pd.DataFrame(rolling_mean, index=returns_matrix.index, columns=returns_matrix.columns)
#     rolling_std = pd.DataFrame(rolling_std, index=returns_matrix.index, columns=returns_matrix.columns)
#     upper_threshold = rolling_mean + rolling_std
#     lower_threshold = rolling_mean - rolling_std
#     upper_ret = returns_matrix - upper_threshold
#     lower_ret = returns_matrix - lower_threshold
#     # print(upper_ret)
#     upper_ret = upper_ret.reset_index(drop=True)
#     lower_ret = lower_ret.reset_index(drop=True)
#     positive_ret = upper_ret[upper_ret > 0]
#     negative_ret = upper_ret[upper_ret < 0]
#     positive_ret_index = positive_ret.index
#     negative_ret_index = negative_ret.index
#     # 计算收益率之和
#     positive_sum = positive_ret.sum()
#     positive_ret = upper_ret[upper_ret > 0].T
#     negative_sum = negative_ret.sum()
#     negative_ret = upper_ret[upper_ret < 0].T
#     # 计算因子值
#     # print(positive_ret_index)
#     # print(positive_ret)
#     positive_factor0 = positive_ret_index * positive_ret
#     positive_factor = positive_factor0.divide(positive_ret.sum(axis=1), axis=0)
#     negative_factor0 = negative_ret_index * negative_ret
#     negative_factor = negative_factor0.divide(positive_ret.sum(axis=1), axis=0)
#     # 把positive_factor和negative_factor中的NAN值设置为0
#     positive_factor = positive_factor.fillna(0)
#     negative_factor = negative_factor.fillna(0)
#     # 合并正负因子值
#     # 第四步：计算跌幅时间重心偏离因子
#     residuals = (positive_factor - negative_factor)
#     residuals = residuals.T
#     enhanced_factor = residuals.diff(window_size) / residuals.shift(window_size)
#     enhanced_factor = -enhanced_factor
#     # 重新定义横轴为资产名称
#     # enhanced_factor.columns = ret.columns
#     enhanced_factor.index = returns_matrix.index
#
#
#     return rolling_mean, rolling_std,  enhanced_factor

# # t期最高价距离动量因子
# def calculate_distance_momentum(close, t):
#     max_price = close.rolling(t).max()  # 过去 t 个交易日内的最高价
#     max_price_delayed = max_price.shift(1)  # 将最高价向后延迟一天
#
#     momentum_factor = (close / max_price_delayed)-1  # 计算动量因子
#
#
#     return momentum_factor
#
#
#
# #模糊度
# def get_Blur(data_1,N):#data为收益率
#     Blur=data_1.rolling(N).std()
#     Blur.index = pd.DatetimeIndex(Blur.index)
#     Blur=Blur.rolling(N).std()
#     return Blur
#
# #日模糊关联度因子
# def daily_rel(data_1,data_2,regwindow=24*12):  #data_1为模糊度，data_2为成交金额
#     namelist = data_1.columns.tolist()
#     data_1.index = data_1.index.strftime("%Y-%m-%d %H:%M:%S")
#     NumTime, NumCol = data_1.shape
#     RelMtx = pd.DataFrame(columns=namelist, index=data_2.index)
#     for i in range(NumCol-1):
#         for t in range(regwindow, NumTime - 1):
#             df_1=data_1.iloc[(t + 1 - regwindow):t, i]
#             df_2=data_2.iloc[(t + 1 - regwindow):t, i]
#             q=df_1.corr(df_2)
#             RelMtx.iloc[t+1,i]=q
#     return RelMtx
#
# #模糊关联度因子
# def month_rel(df_1,df_2,regwindow=12*24*30):   #df_1为收益率，df_2为成交金额
#     namelist=df_1.columns.tolist()
#     data=pd.DataFrame(daily_rel(get_Blur(df_1,2*24*10),df_2))
#     data_1=data.rolling(regwindow).mean()
#     data_2=data.rolling(regwindow).std()
#     data = data_1+data_2
#     return data_1,data_2,data
# #量价相关性
# def vol_price_rel(data_1,data_2,regwindow=24*12):
#     namelist=data_1.columns.tolist()
#     NumTime, NumCol = data_1.shape
#     RelMtx = pd.DataFrame(columns=namelist,index=data_1.index)
#     for i in range(NumCol):
#         for t in range(regwindow, NumTime - 1):
#             df_1 = pd.Series(data_1.iloc[(t+1-regwindow):t, i])  # 利用Series将列表转换成新的、pandas可处理的数据
#             df_2 = pd.Series(data_2.iloc[(t+1-regwindow):t, i])
#             q = df_1.corr(df_2)  # 计算相关系数
#             RelMtx.iloc[t+1, i] = q
#     #RelMtx=RelMtx.resample(NumDay).sum()/N
#     return RelMtx
#
# # 硬币动量反转因子
# def calculate_coin_factor(returns_matrix, window):
#     unique_std = returns_matrix.rolling(window).std()  # 计算资产在窗口期内的独特度标准差
#     unique_mean = returns_matrix.rolling(window).mean()  # 计算资产在窗口期内的独特度均值
#     # 计算市场的均值
#     market_std = unique_std.mean(axis=1)
#     # 根据市场均值调整 unique_std 矩阵
#     std_adjusted = unique_std.sub(market_std, axis=0)
#     std_adjusted = -std_adjusted
#     # 将大于0的值设为True，不大于0的设为False
#     bool_matrix = std_adjusted.gt(0)
#     # 将True设为1，False设为-1
#     float_matrix = bool_matrix.astype(int) * 2 - 1
#     # 计算硬币因子矩阵
#     # coin_factor_matrix = float_matrix * returns_matrix
#     # returns = calculate_returns(close,window=10)
#     coin_factor_matrix = float_matrix * retn
#     return coin_factor_matrix
# #高频偏度
# #高频偏度
# def get_skew(ret,gap1,NumberTime=0.5*12):
#     ret.index = pd.to_datetime(ret.index, format='%Y-%m-%d %H:%M:%S')
#     ret_1 = ret ** 3
#     ret_2 = ret ** 2
#     ret_1 = ret_1.resample(gap1).sum()
#     ret_2 = ret_2.resample(gap1).sum()
#     ret_2 = ret_2 ** (1.5)
#     ret = ((NumberTime) ** 0.5 * ret_1) / ret_2
#     # ret.index = retn.index
#     return ret
#
# # 成交量切割收益率
# def calculate_rolling_stats_topn_bottom(returns, volumes, window, n):
#     rolling_mean_topn = pd.DataFrame(index=returns.index, columns=returns.columns)
#     rolling_mean_bottomn = pd.DataFrame(index=returns.index, columns=returns.columns)
#
#     for i in range(window, len(returns) + 1):
#         window_returns = returns.iloc[i - window:i]
#         window_volumes = volumes.iloc[i - window:i]
#
#         # 对窗口期内的成交量进行排名
#         ranked_volumes = window_volumes.rank(axis=0, ascending=False)
#
#         # 找出排名前n的时间对应的收益率
#         topn_returns = window_returns.where(ranked_volumes <= n)
#         # 找出排名后n的时间对应的收益率
#         bottomn_returns = window_returns.where(ranked_volumes > len(ranked_volumes) - n)
#
#         # 计算排名前n的收益率的均值
#         rolling_mean_topn.iloc[i - 1] = topn_returns.mean(axis=0)
#         # 计算排名后n的收益率的均值
#         rolling_mean_bottomn.iloc[i - 1] = bottomn_returns.mean(axis=0)
#
#     return rolling_mean_topn, rolling_mean_bottomn

#
# def calculate_rolling_top_bottom(returns, volumes, window, n):
#     # 将成交量矩阵与收益率矩阵的索引对齐
#     aligned_volumes = volumes.reindex(returns.index)
#     # 对成交量矩阵在窗口期内进行排名isin
#     ranked_volumes = aligned_volumes.rolling(window).apply(lambda x: pd.Series(x).rank(ascending=False)[-1])
#     # 找出排名前n的时间对应的收益率矩阵
#     topn_returns = returns.where(ranked_volumes <= n)
#     # 找出排名后n的时间对应的收益率矩阵
#     bottomn_returns = returns.where(ranked_volumes > len(ranked_volumes) - n)
#     # 计算每个滚动窗口期内排名前n的收益率的均值
#     rolling_mean_topn = topn_returns.rolling(window).mean()
#     # 计算每个滚动窗口期内排名后n的收益率的均值
#     rolling_mean_bottomn = bottomn_returns.rolling(window).mean()
#
#     return rolling_mean_topn, rolling_mean_bottomn
#
# def rolling_zscore(df, n,reverse):
#     if reverse:
#         df_reversed = df.iloc[::-1]
#         zscore_df = []
#         # 遍历每个时间点
#         for i in range(len(df)):
#             # 计算当前窗口大小
#             current_window = min(i + 1, n)
#             # 获取数据子集
#             subset = df_reversed.iloc[max(0, i - current_window + 1):i + 1]
#             # 计算滚动均值和滚动标准差
#             rolling_mean = subset.mean()
#             rolling_std = subset.std()
#             # 计算 z 值
#             zscore = (df_reversed.iloc[i] - rolling_mean) / rolling_std
#             # 将 z 值添加到结果 DataFrame 中
#             zscore_df.append(zscore)
#         zscore_df = pd.DataFrame(zscore_df)
#         zscore_df.index = df_reversed.index
#     else:
#         zscore_df = []
#         # 遍历每个时间点
#         for i in range(len(df)):
#             # 计算当前窗口大小
#             current_window = min(i + 1, n)
#             # 获取数据子集
#             subset = df.iloc[max(0, i - current_window + 1):i + 1]
#             # 计算滚动均值和滚动标准差
#             rolling_mean = subset.mean()
#             rolling_std = subset.std()
#             # 计算 z 值
#             zscore = (df.iloc[i] - rolling_mean) / rolling_std
#             # 将 z 值添加到结果 DataFrame 中
#             zscore_df.append(zscore)
#         zscore_df = pd.DataFrame(zscore_df)
#         zscore_df.index = df.index
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(df.index, df.values, label='原始数据')
#     # plt.plot(zscore_df.index, zscore_df.values, label='z 值')
#     # plt.xlabel('时间')
#     # plt.ylabel('值')
#     # plt.title("z值图像")
#     # # plt.title('原始数据和 z 值的图像')
#     # plt.legend()
#     # plt.show()
#     return zscore_df
#
# def fill_nan_with_row_mean(matrix):
#     matrix_without_nan = matrix.dropna()
#     # print(matrix_without_nan)
#     # 使用每一行的均值填充NaN值
#     filled_matrix = matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
#
#     return filled_matrix
#
# def market_neutralize(factor, mak_cp):
#     # 输入因子矩阵和市值矩阵，输出中性化后的因子矩阵
#     if mak_cp.isna().any().any():
#         print("mak_cp中存在NaN值")
#         return None
#
#     print("DataFrame中没有NaN值")
#     factor.fillna(factor.mean(), inplace=True)
#     # 添加截距项
#     X = sm.add_constant(mak_cp)
#
#     # 使用线性回归拟合市值因子和因子矩阵之间的关系
#     model = sm.OLS(factor.values, X.values)
#     results = model.fit()
#
#     # 提取回归的残差
#     residuals = results.resid
#
#     # 将残差转换回DataFrame
#     residuals_df = pd.DataFrame(residuals, index=factor.index, columns=factor.columns)
#
#     return residuals_df
# def system_kewness(market,retn):
#     market[0] = 1
#     # 创建一个空的DataFrame，索引与`retn`相同
#     market_df = pd.DataFrame(index=retn.index, columns=retn.columns)
#
#     # 填充每一列
#     for column in market_df.columns:
#         market_df[column] = market
#     # market = pd.DataFrame(market).T
#     # # market = fill_nan_with_row_mean(market)
#     # market.iloc[0]=  1
#     # retn = fill_nan_with_row_mean(retn)
#     # market_data = pd.concat([market] * len(retn.columns), axis=1)
#     # market_df = pd.DataFrame(index=retn.index, columns=retn.columns, data=market_data.values)
#     retn_residuals = market_neutralize(retn,market_df)
#     # 计算 market 的均值
#     mean_market = market.mean()
#     # 将 market 减去均值
#     market_residual = market - mean_market
#     # market_residual_data = pd.concat([market_residual] * len(retn.columns), axis=1)
#     market_residuals = pd.DataFrame(index=retn_residuals.index, columns=retn_residuals.columns)
#     for column in market_residuals.columns:
#         market_residuals[column] = market_residual
#     cs1_mean = retn_residuals-market_residuals
#     squared_retn = retn_residuals.applymap(lambda x: x ** 2)
#     squared_marekt = market_residuals.applymap(lambda x: x ** 2)
#     result_df = squared_marekt.mul(squared_retn)
#     sqrt_df = result_df.applymap(np.sqrt)
#     cs = cs1_mean.div(sqrt_df)
#     return cs
#
# def coeffof_variation_non_fluidity(amt,retn):
#     #非流动性变异系数
#     # 识别数据类型不一致的列
#     numeric_columns = amt.select_dtypes(include=['float', 'int']).columns
#     string_columns = amt.select_dtypes(include=['object']).columns
#
#     # 针对数值型数据列，使用平均值填充缺失值
#     amt[numeric_columns] = amt[numeric_columns].fillna(amt[numeric_columns].mean())
#
#     # 针对字符串型数据列，使用众数填充缺失值或者其他适当的方式处理
#
#     # 例如，使用众数填充字符串型数据列的缺失值
#     for col in string_columns:
#         mode_value = amt[col].mode()[0]
#         amt[col].fillna(mode_value, inplace=True)
#
#     # amt.fillna(amt.mean(), inplace=True)
#     retn.fillna(retn.mean(),inplace=True)
#     amt = amt.astype(float)  # 将amt中的数据类型转换为浮点数型
#     retn = retn.astype(float)  # 将retn中的数据类型转换为浮点数型
#
#     illiq = amt/retn
#     # illiq_array = np.array(illiq)
#     # illiq_array = illiq_array/1000000000
#     # 计算每一列的标准差
#     # illiq.fillna(illiq.mean(), inplace=True)
#     # col_std = np.std(illiq_array, axis=1)
#     # 计算a的绝对值矩阵
#     # cv_illq = []
#     # # abs_a = np.abs(illiq_array)
#     # for column in illiq.columns:
#     #     column_data = illiq[column]
#     #     column_data = pd.Series(column_data)
#     #     std = column_data.mean()/1000000000000000000000000000000000000000000000000000
#     #     abs = np.abs(column_data)
#     #
#     #     new_column = std/abs
#     #     cv_illq.append(new_column)
#     # cv_illq = pd.DataFrame(cv_illq)
#     # # 计算b
#     # # cv_illq = col_std / abs_a
#     # cv_illq = pd.DataFrame(cv_illq)
#     return illiq

def adjust_dates_to_season_dates(csv_file_path, open_dates):
    """
    将指定csv文件中的日期调整为最近的开盘日

    :param csv_file_path: csv文件路径
    :param open_dates: 包含所有开盘日期的Pandas DateTimeIndex对象
    :return: 包含调整后开盘日期的DataFrame
    """
    # 读取csv文件到DataFrame，并假设日期列的列名为"dates"
    season_dates_df = pd.read_csv(csv_file_path, header=None, names=['dates'], parse_dates=['dates'])

    # 将非开盘日的日期调整为最近的开盘日
    adjusted_dates = []
    for date in season_dates_df['dates']:
        # 确保date是pd.Timestamp类型
        date = pd.to_datetime(date)
        # 找到小于或等于该日期的最近开盘日
        mask = open_dates <= date
        nearest_open_date = open_dates[mask].max()  # 使用 .max() 获取小于等于该日期的最大值
        adjusted_dates.append(nearest_open_date)

    # 替换原来的日期为调整后的日期
    adjusted_dates_index = pd.DatetimeIndex(adjusted_dates)

    return adjusted_dates_index



def database_reading_and_reshape_to_wide_format_df(engine, table_name, value_column):
    """
        从指定的表中读取数据，删除重复行，将指定的值列转换为宽格式DataFrame。

        参数:
        - engine: SQLAlchemy引擎，用于执行数据库查询。
        - table_name: 字符串，指定从中读取数据的表名。
        - value_column: 字符串，指定要处理成宽格式的值列名。

        返回:
        - DataFrame: 重塑后的宽格式DataFrame。
        """
    # 读取balancesheet表中value_column的数据
    query = f"SELECT ts_code, end_date, {value_column} FROM {table_name}" ##有问题，有的股票指数没有2023.12.31日，所以最终因子矩阵只到9.30，不够balance_season_returns



    df_factor = pd.read_sql(query, engine)
    # 删除完全重复的行，保留第一次出现的行
    df_factor = df_factor.drop_duplicates()

    # end_date转换为datetime
    df_factor['end_date'] = pd.to_datetime(df_factor['end_date'])

    # 删除 'ts_code' 列为 NaN 或 None 的行
    df_factor = df_factor.dropna(subset=['ts_code'])

    # 删除 'end_date' 列为 NaT 的行
    df_factor = df_factor.dropna(subset=['end_date'])

    #有的因子读了出现None
    df_factor = df_factor.fillna(value=np.nan)

    #慎重，使用报错
    df_factor.fillna(0, inplace=True)


    # # 设置索引
    # df_factor.set_index('end_date', inplace=True)

    # 重塑数据框，以ts_code为列，value_column为值
    df_factor_pivoted = df_factor.pivot_table(index='end_date', columns='ts_code', values=value_column, aggfunc='first')
    return df_factor_pivoted


def sql_adjusted_season_factor(engine, table_name, factor_column, open_date):
    # 访问数据库


    # # 定义要产生因子矩阵的因子列表
    # query_first = f"SELECT * FROM {table_name} LIMIT 1"
    # df = pd.read_sql(query_first, engine)
    # # 获取所有列名
    # columns = df.columns.tolist()

    # 找到 "finan_exp" 在列名列表中的位置
    # try:
    #     start_index = columns.index(factor_column)#如'finan_exp'
    #     factors = columns[start_index:]
    # except ValueError:
    #     # 如果 "finan_exp" 不在列名列表中，打印错误信息
    #     factors = []
    #     print(factor_column, " column not found in",table_name, " table.")
    # print(factors)

    # 初始化一个空字典来存储结果DataFrame
    # factor_dfs = {}
    # # 循环遍历因子列表，为每个因子调用函数并将结果存储在字典中，要获取对应因子矩阵即factor_dfs['total_share']
    # for factor in factors:
    #     factor_dfs[factor] = database_reading_and_reshape_to_wide_format_df(engine, 'balance_sheet', factor)

    factor_df = database_reading_and_reshape_to_wide_format_df(engine, table_name, factor_column)

    #定义一个空df
    adjusted_factor_df = pd.DataFrame()

    #这里要添加一个新功能，对时间索引进行截断，因为oepndata只有2023.1.4以后的，同时也要实现3年5年10年的时间跨度
    #问题，把20213-9-30的数据也给去掉了，请debug
    # 遍历 finan_exp 的索引（日期）
    for date in factor_df.index:
        # 如果日期已经是一个开盘日，则直接使用这个日期
        if date in open_date:
            adjusted_date = date
        else:
            # 否则，找到这个日期之前最近的一个开盘日
            mask = open_date < date  # 找到所有在这个日期之前的开盘日
            adjusted_date = open_date[mask].max()  # 获取这些日期中最晚的一个，即最近的一个开盘日
            # 将原始数据加入到新的 DataFrame 中，但使用调整后的日期作为索引
        adjusted_factor_df = pd.concat([adjusted_factor_df, factor_df.loc[[date]].set_index(pd.DatetimeIndex([adjusted_date]))])

    # 假设adjusted_factor_df是你的DataFrame，而adjusted_date是2013年1月4日对应的调整后日期
    # 切片并更新DataFrame
    adjusted_factor_df = adjusted_factor_df[adjusted_factor_df.index >= '2013-01-04']
    # print(open_date)
    return adjusted_factor_df

# #min_max_nomalize归一化
# def min_max_normalize(df):
#     return 2*(df - df.min()) / (df.max() - df.min())-1

def min_max_normalize(df):
    return (df - df.mean()) / df.std()

##因为df_factor里面的股票指数数量少于aligned_score_df里面的，这里设计对齐函数
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

def result_df_to_stock_name(result_df):

    # df = pd.read_excel("/HuaQi/Data/stock_in_market.xls")
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


# def calculate_stock_occurrences(stock_data):
#     # stock_data为[[股票1, 股票2, ...], [股票1, 股票2, ...], ...]
#
#     stock_occurrences = defaultdict(int)
#
#     daily_averages = []
#
#     for i in range(len(stock_data)):
#         current_stocks = set(stock_data[i])
#         for j in range(i + 1, len(stock_data)):
#             common_stocks = current_stocks & set(stock_data[j])
#             for stock in common_stocks:
#                 stock_occurrences[stock] += 1
#
#         total_occurrences = sum(stock_occurrences.values())
#
#         if len(current_stocks) > 0:
#             daily_average = total_occurrences / len(current_stocks)
#             daily_averages.append(daily_average)
#         else:
#             daily_averages.append(0)
#
#         stock_occurrences.clear()
#     #
#     return daily_averages
# from collections import defaultdict
# import pandas as pd

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
#
# def daily_factor_calling(close,turn):
#     BiasFactor10day = daily_factor_script.calculate_10_daily_biass(close)
#     BiasFactor15day = daily_factor_script.calculate_15_daily_biass(close)
#     BiasFactor20day = daily_factor_script.calculate_20_daily_biass(close)
#     BiasFactor30day = daily_factor_script.calculate_30_daily_biass(close)
#     BiasFactor50day = daily_factor_script.calculate_50_daily_biass(close)
#     BiasFactor100day = daily_factor_script.calculate_100_daily_biass(close)
#
#     MovingAverage5day = daily_factor_script.calculate_5_day_moving_average(close)
#     MovingAverage10day = daily_factor_script.calculate_10_day_moving_average(close)
#     MovingAverage15day = daily_factor_script.calculate_15_day_moving_average(close)
#     MovingAverage30day = daily_factor_script.calculate_30_day_moving_average(close)
#     MovingAverage50day = daily_factor_script.calculate_50_day_moving_average(close)
#     MovingAverage100day = daily_factor_script.calculate_100_day_moving_average(close)
#
#     MovingTurnover5day = daily_factor_script.calculate_5_day_average_turnover_factor(turn)
#     MovingTurnover10day = daily_factor_script.calculate_10_day_average_turnover_factor(turn)
#     MovingTurnover15day = daily_factor_script.calculate_15_day_average_turnover_factor(turn)
#     MovingTurnover30day = daily_factor_script.calculate_30_day_average_turnover_factor(turn)
#     MovingTurnover50day = daily_factor_script.calculate_50_day_average_turnover_factor(turn)
#     MovingTurnover100day = daily_factor_script.calculate_100_day_average_turnover_factor(turn)
#
#     BollingerBandUpperLine5day = daily_factor_script.calculate_5_day_upper_bollinger_band(close)
#     BollingerBandUpperLine10day = daily_factor_script.calculate_10_day_upper_bollinger_band(close)
#     BollingerBandUpperLine15day = daily_factor_script.calculate_15_day_upper_bollinger_band(close)
#     BollingerBandUpperLine30day = daily_factor_script.calculate_30_day_upper_bollinger_band(close)
#     BollingerBandUpperLine50day = daily_factor_script.calculate_50_day_upper_bollinger_band(close)
#     BollingerBandUpperLine100day = daily_factor_script.calculate_100_day_upper_bollinger_band(close)
#
#     BollingerBandLowerLine5day = daily_factor_script.calculate_5_day_lower_bollinger_band(close)
#     BollingerBandLowerLine10day = daily_factor_script.calculate_10_day_lower_bollinger_band(close)
#     BollingerBandLowerLine15day = daily_factor_script.calculate_15_day_lower_bollinger_band(close)
#     BollingerBandLowerLine30day = daily_factor_script.calculate_30_day_lower_bollinger_band(close)
#     BollingerBandLowerLine50day = daily_factor_script.calculate_50_day_lower_bollinger_band(close)
#     BollingerBandLowerLine100day = daily_factor_script.calculate_100_day_lower_bollinger_band(close)
#
#     return (BiasFactor10day,BiasFactor15day,BiasFactor20day ,BiasFactor30day ,BiasFactor50day,BiasFactor100day,
#             MovingAverage5day,MovingAverage10day,MovingAverage15day,MovingAverage30day,MovingAverage50day,MovingAverage100day,
#             MovingTurnover5day,MovingTurnover10day,MovingTurnover15day,MovingTurnover30day,MovingTurnover50day,MovingTurnover100day,
#             BollingerBandUpperLine5day,BollingerBandUpperLine10day,BollingerBandUpperLine15day,BollingerBandUpperLine30day,BollingerBandUpperLine50day,BollingerBandUpperLine100day,
#             BollingerBandLowerLine5day,BollingerBandLowerLine10day,BollingerBandLowerLine15day,BollingerBandLowerLine30day,BollingerBandLowerLine50day,BollingerBandLowerLine100day)


if __name__ == "__main__":

    ######################基本准备部分####################
    # print('basic数据导入开始')
    print('basic数据导入开始')

    open1, close, high, low, vol, amt, score_data, ret, retn, market = ReadRawdata()
    turn = pd.read_pickle(r"D:\桌面\HuaQiStudy\Data\全a指数换手率turn.pkl")
    # 开盘日
    open_date = open1.index

    ##股票交集操作
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
    high = factor_correct(high, aligned_score_df)
    low = factor_correct(low, aligned_score_df)
    vol = factor_correct(vol, aligned_score_df)
    amt = factor_correct(amt, aligned_score_df)
    ret = factor_correct(ret, aligned_score_df)
    retn = factor_correct(retn, aligned_score_df)
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

    # 定义起止时间
    start_date = '2013-01-04 00:00:00'
    end_date = '2023-12-31 00:00:00'
    csv_file_path = "D:\\桌面\\HuaQiStudy\\Data\\季度时间.csv"  # 替换为您的CSV文件路径

    #这一行输出格式factor_correct
    table_name = 'balance_sheet'
    first_factor = 'total_share'

    #用于筛选因子的因子筛
    english_factor = [
        'money_cap', 'accounts_receiv', 'oth_receiv', 'inventories',
        'oth_cur_assets', 'total_cur_assets', 'intan_assets', 'goodwill',
        'lt_amor_exp', 'defer_tax_assets', 'oth_nca', 'total_assets',
        'adv_receipts', 'int_payable', 'div_payable', 'defer_inc_non_cur_liab',
        'total_hldr_eqy_exc_min_int', 'accounts_receiv_bill', 'oth_rcv_total',
        'oth_pay_total', 'c_fr_oth_operate_a', 'c_inf_fr_operate_a',
        'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes',
        'oth_cash_pay_oper_act', 'c_prepay_amt_borr', 'c_cash_equ_beg_period',
        'c_cash_equ_end_period'
    ]

    # 定义要产生因子矩阵的因子列表
    query_first = f"SELECT * FROM {table_name} LIMIT 1"
    df = pd.read_sql(query_first, engine)
    # 获取所有列名
    columns = df.columns.tolist()

    #找到 "first_column" 在列名列表中的位置
    try:
        start_index = columns.index(first_factor)
        factors = columns[start_index:]
    except ValueError:
        # 如果 "factor_column" 不在列名列表中，打印错误信息
        factors = []
        print(first_factor, " column not found in", table_name, " table.")

    # print(factors)

    # (BiasFactor10day, BiasFactor15day, BiasFactor20day, BiasFactor30day, BiasFactor50day, BiasFactor100day,
    # MovingAverage5day, MovingAverage10day, MovingAverage15day, MovingAverage30day, MovingAverage50day, MovingAverage100day,
    # MovingTurnover5day, MovingTurnover10day, MovingTurnover15day, MovingTurnover30day, MovingTurnover50day, MovingTurnover100day,
    # BollingerBandUpperLine5day, BollingerBandUpperLine10day, BollingerBandUpperLine15day, BollingerBandUpperLine30day, BollingerBandUpperLine50day, BollingerBandUpperLine100day,
    # BollingerBandLowerLine5day, BollingerBandLowerLine10day, BollingerBandLowerLine15day, BollingerBandLowerLine30day, BollingerBandLowerLine50day, BollingerBandLowerLine100day)=daily_factor_calling(close,turn)
    #
    # daily_factors = [BiasFactor10day, BiasFactor15day, BiasFactor20day, BiasFactor30day, BiasFactor50day, BiasFactor100day,
    # MovingAverage5day, MovingAverage10day, MovingAverage15day, MovingAverage30day, MovingAverage50day, MovingAverage100day,
    # MovingTurnover5day, MovingTurnover10day, MovingTurnover15day, MovingTurnover30day, MovingTurnover50day, MovingTurnover100day,
    # BollingerBandUpperLine5day, BollingerBandUpperLine10day, BollingerBandUpperLine15day, BollingerBandUpperLine30day, BollingerBandUpperLine50day, BollingerBandUpperLine100day,
    # BollingerBandLowerLine5day, BollingerBandLowerLine10day, BollingerBandLowerLine15day, BollingerBandLowerLine30day, BollingerBandLowerLine50day, BollingerBandLowerLine100day]

    # 循环遍历因子列表，为每个因子调用函数并将结果存储在字典中，要获取对应因子矩阵即factor_dfs['total_share']
    for factor_column in factors:
        print(factor_column)
        if factor_column not in english_factor:
            continue
        print("%s", factor_column)


        #这一行，出现了2012.12.31这行变成了NAT，多出来一行了。
        factor_df_season = sql_adjusted_season_factor(engine, table_name, factor_column, open_date)
        factor_df_season = -factor_df_season.astype(float)
        #非常关键，因为calculate_long_short_returns里面double_float = A+B这个算法要保证矩阵一致！去掉BJ结尾的代码
        factor_df_season = factor_correct(factor_df_season, open1)

        # #如果是季度财报数据，window=1是跨一个季度了
        # daily_returns = calculate_returns(close,window=1)#数据是一天级别数据

        # print('basic数据导入结束')

        # ########################因子计算部分###################
        # print('因子计算开始')
        #
        # # 流动性因子与反转 #####
        # liquidity = calculate_price_liquidity(amt,retn,window=10)
        # liquidity = -liquidity
        #
        # # # # 反转八个三十分钟收益率因子
        # fourhours_returns = calculate_returns(close,window=8)
        # # fourhours_returns = -fourhours_returns
        #
        # # # # 收益率时序增强因子(or反转)
        # time_enhanced_factor = calculate_time_series_enhanced_factors(retn,5)
        # # time_enhanced_factor = -time_enhanced_factor
        #
        # # # # 过度自信，重拾自信因子
        # CP_Intraday =  calculate_CP_Intraday(daily_returns)
        # CP_Intraday = -CP_Intraday
        # rolling_mean, rolling_std, cp_intraday = calculate_returns_stats(retn,4)
        # cp_intraday = -cp_intraday
        # # # 反转
        # # # cp_intraday = -cp_intraday
        #
        # # # 平滑动量
        # smooth_momentum = calculate_momentum(retn,close,window=20)
        # # smooth_momentum = -smooth_momentum
        # # # # 最高价距离(bad)
        # # # top_momentum_factor = calculate_distance_momentum(close,20)
        # # # top_momentum_factor = -top_momentum_factor
        # # #
        # # coin_factor_matrix  =calculate_coin_factor(retn,10)
        # # coin_factor_matrix = -1*coin_factor_matrix
        # #
        # # blur_data = get_Blur(retn,2*24*10)
        # # blur_data2 = month_rel(blur_data,amt,regwindow=2*24*10)
        # # print(blur_data)
        # # RelMtx = vol_price_rel(retn,vol,regwindow=24*12)
        # # gap = get_skew(retn,gap1='1D',NumberTime=0.5*12)
        #
        # # 动量
        # # rolling_std, rolling_mean = calculate_rolling_retn(retn,48)
        # # rolling_std = -rolling_std
        # # rolling_mean_top, rolling_mean_bottomn = calculate_rolling_stats_topn_bottom(retn,vol,10,10)
        # # rolling_mean_top, rolling_mean_bottomn = calculate_rolling_top_bottom(retn,vol,10,10)
        # # print('ok')
        #
        # # 协偏度（系统偏度）
        # # cs = system_kewness(market,retn)
        # # cs = -cs
        #
        # # 变异系数
        # cv_illq = coeffof_variation_non_fluidity(amt,retn)
        # # cv_illq = -cv_illq
        # print('因子计算结束')
        # #########################因子中性化部分##################
        # print('因子中性化开始')
        # # time_enhanced_factor = industry_neutralize(cs_indus_code, time_enhanced_factor)
        # # neutralized_factor_matrix = factor_neutralization(cs, industry_matrices_dict)
        # print('因子中性化结束')

        #########################回测运行部分##################
        # print('回测开始')
        print('回测开始')

        # 季调仓

        adjusted_season_dates_df = adjust_dates_to_season_dates(csv_file_path, open_date)
        balance_season_dates = generate_rebalance_dates(start_date, end_date, interval, adjusted_season_dates_df)
        adjusted_close = close.loc[adjusted_season_dates_df]

        daily_season_returns = calculate_returns(adjusted_close, window=1)  # 数据是一季度级别数据

        #balance_season_dates和factor_df_season容易not in index，特别是遇到很疏散的因子
        df_factor_season, df_retn_season, df_ret_season, df_market_season = calculate_previous_factor(balance_season_dates, daily_season_returns, factor_df_season, market)

        #longshort_df_season全nan了，cumulative_returns_withoutfee_season全1，
        if calculate_long_short_returns_select:
            longshort_df_season, cumulative_returns_withoutfee_season, final_matrix_season, final_matrix_season_20 = calculate_long_short_returns(retn, df_factor_season, aligned_score_df, 50)  # 计算多空收益率
        else:
            longshort_df_season, cumulative_returns_withoutfee_season, final_matrix_season, final_matrix_season_20 = calculate_long_returns(retn, df_factor_season, aligned_score_df, 50)  # 计算多空收益率#计算多头收益率

        #这里出问题，换手率全0
        turnover_season, turnover_matrix_season, turn_matrix_season = normalize_holdings_matrix(final_matrix_season, df_ret_season)

        longshort_df_with_fee_season, cumulative_returns_with_fee_season, last_longshort_return = longshort_withfee(factor_column, longshort_df_season, turnover_season, tax)
        ic_season, ic_mean_season, ic_var_season = calculate_ic(df_retn_season, df_factor_season)
        IR_season = perform_t_test(ic_season)
        max_drawdown_season = calculate_max_drawdown(longshort_df_with_fee_season)
        longshort_df_with_fee_season = longshort_df_with_fee_season.squeeze()
        annualized_returns_season = calculate_annualized_returns(cumulative_returns_with_fee_season)
        annualized_volatility_season = calculate_annualized_volatility(cumulative_returns_with_fee_season)
        sharpe_ratio_season = calculate_longshort_sharpe_ratio(annualized_returns_season, annualized_volatility_season)

        result_long_df_season = find_stocks_with_value_one(final_matrix_season_20.T)
        result_short_df_season = find_stocks_with_value_one((-final_matrix_season_20).T)
        result_long_return_season = result_df_to_stock_name(result_long_df_season)
        result_short_return_season = result_df_to_stock_name(result_short_df_season)

        result_df = find_stocks_with_value_one(final_matrix_season.T)
        # result_df = result_df.values.tolist()

        # all_data = []
        # for index, row in result_df.iterrows():
        #     all_data.append(row['stocks'])
        # daily_averages = calculate_stock_occurrences(all_data)

        all_data = result_df['stocks'].tolist()
        dates = result_df.index

        daily_averages = calculate_stock_occurrences(all_data, dates)

        #输入到数据库
        store_results_to_database(engine, factor_column, table_name, last_longshort_return, annualized_returns_season, annualized_volatility_season, sharpe_ratio_season, ic_mean_season, IR_season)

        output_data_season = {
            "turnover": turnover_season,
            "longshort_df_with_fee": longshort_df_with_fee_season.squeeze(),  # 注意这里已经转换为Series
            "cumulative_returns_with_fee": cumulative_returns_with_fee_season,
            "ic": ic_season,
            "ic_mean": ic_mean_season,
            "ic_var": ic_var_season,
            "IR": IR_season,
            "max_drawdown": max_drawdown_season,
            "annualized_returns": annualized_returns_season,
            "annualized_volatility": annualized_volatility_season,
            "sharpe_ratio": sharpe_ratio_season,
            "result_long_return": result_long_return_season,
            "result_short_return": result_short_return_season,
            "daily_averages": daily_averages,
            "market": market
        }


        # #日调仓时点收益率
        # balance_dates = generate_rebalance_dates(start_date,end_date,interval,open_date)
        # df_factor,df_retn,df_ret,df_market = calculate_previous_factor(balance_dates,daily_returns,smooth_momentum,market)
        #
        # ##记得加上score分数矩阵
        # if calculate_long_short_returns_select:
        #     longshort_df,cumulative_returns_withoutfee,final_matrix, final_matrix_20 = calculate_long_short_returns(df_retn,df_factor, aligned_score_df,50)#计算多空收益率
        # else:
        #     longshort_df, cumulative_returns_withoutfee, final_matrix, final_matrix_20 = calculate_long_returns(df_retn, df_factor, aligned_score_df, 50)#计算多头收益率
        #
        # ##回传前端部分
        # turnover, turnover_matrix,turn_matrix = normalize_holdings_matrix(final_matrix,df_ret)
        # longshort_df_with_fee, cumulative_returns_with_fee, last_longshort_return = longshort_withfee(longshort_df, turnover, tax)
        # ic,ic_mean,ic_var = calculate_ic(df_retn, df_factor)
        # IR = perform_t_test(ic)
        # max_drawdown = calculate_max_drawdown(longshort_df_with_fee)
        # longshort_df_with_fee = longshort_df_with_fee.squeeze()
        # annualized_returns = calculate_annualized_returns(cumulative_returns_with_fee)
        # annualized_volatility = calculate_annualized_volatility(cumulative_returns_with_fee)
        # sharpe_ratio = calculate_longshort_sharpe_ratio(annualized_returns,annualized_volatility)
        #
        # result_long_df = find_stocks_with_value_one(final_matrix_20.T)
        # result_short_df = find_stocks_with_value_one((-final_matrix_20).T)
        # result_long_return = result_df_to_stock_name(result_long_df)
        # result_short_return = result_df_to_stock_name(result_short_df)
        #
        # print('回测结束')
        # # print(result_df)
        # # plt.show()
        #
        # output_data = {
        #     "turnover": turnover,
        #     # "turnover_matrix": turnover_matrix,
        #     # "turn_matrix": turn_matrix,
        #     "longshort_df_with_fee": longshort_df_with_fee.squeeze(),  # 注意这里已经转换为Series
        #     "cumulative_returns_with_fee": cumulative_returns_with_fee,
        #     "ic": ic,
        #     "ic_mean": ic_mean,
        #     "ic_var": ic_var,
        #     "IR": IR,
        #     "max_drawdown": max_drawdown,
        #     "annualized_returns": annualized_returns,
        #     "annualized_volatility": annualized_volatility,
        #     "sharpe_ratio": sharpe_ratio,
        #     # "result_df": result_df
        #     "result_long_return": result_long_return,
        #     "result_short_return": result_short_return
        # }

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


        output_folder = "D:/桌面/HuaQiStudy/Output_daily_factor"
        # 确保目标文件夹存在，如果不存在则创建
        os.makedirs(output_folder, exist_ok=True)


        # 指定文件名
        file_name_suffix = '.json'
        file_name = f'{factor_column}{file_name_suffix}'

        # 使用os.path.join来构建文件的完整路径
        file_path = os.path.join(output_folder, file_name)
        # 打开一个文件用于写入，'w'表示写模式
        with open(file_path, 'w') as f_i_l_e:
            # 使用json.dump将数据写入文件，指定CustomEncoder处理复杂对象
            json.dump(output_data_season, f_i_l_e, cls=CustomEncoder, indent=4)
        #
        # # print(f"数据已保存到文件：{file_name}")
        # json_str = json.dumps(output_data_season, cls=CustomEncoder)
        #
        # (json_str)

