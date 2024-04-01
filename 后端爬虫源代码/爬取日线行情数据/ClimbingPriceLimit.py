import time
import tushare as ts
import xlrd
import os
import pandas as pd
from sqlalchemy import create_engine

# 设置Tushare的token,这样之后请求api的时候就不用带token了
my_token = '78997d1a8a700f6e0ae112e14b8907c650463e631603f71092968cfc'
pro = ts.pro_api(my_token)
ts.set_token(my_token)

if __name__ == '__main__':
    current_directory = os.getcwd()
    workbook = xlrd.open_workbook(
        r'/Users/yinliqi/Desktop/CITICUP/ClimbingTitle/ClimbingFinancialData/上市股票一览.xls')
    sheet = workbook.sheet_by_index(0)
    first_column_data = []
    for row in range(1, sheet.nrows):
        first_column_data.append(sheet.cell_value(row, 0))

    result_df = pd.DataFrame()  # 创建一个空的DataFrame

    for stock_code in first_column_data:
        print(stock_code)

        df = pro.daily(ts_code=stock_code, start_date='20231108', end_date='20240307', fields = 'trade_date, change, pct_chg')
        df['stock_code'] = stock_code  # 添加名为'stock_code'的列，并填入股票代码内容
        print(df)
        # 保存当前循环的数据到CSV文件
        df.to_csv('output_partial.csv', mode='a', header=not os.path.exists('output.csv'), index=False)

        # result_df = pd.concat([result_df, df], ignore_index=True)  # 将df数据追加到result_df中
        time.sleep(0.12)

    # result_df.to_csv('output.csv', index=False)  # 将result_df保存为CSV文件
# 合并所有数据到一个CSV文件
    all_data = pd.read_csv('output.csv')
    all_data.to_csv('output.csv', index=False)