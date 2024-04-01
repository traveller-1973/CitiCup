import time
import pandas as pd
from sqlalchemy import create_engine
import tushare as ts
import mysql.connector
import xlrd
import xlwt
import os

# 以爬取利润表为例给出配置说明

# 设置Tushare的token,这样之后请求api的时候就不用带token了
my_token = '78997d1a8a700f6e0ae112e14b8907c650463e631603f71092968cfc'
pro = ts.pro_api(my_token)
ts.set_token(my_token)

if __name__ == '__main__':
    # root为mysql用户名，冒号后面为mysql的密码
    engine = create_engine('mysql+mysqlconnector://root:wjswbyydyh828@localhost/FinancialData')
    # 此处为读取上市股票的文件
    current_directory = os.getcwd()
    workbook = xlrd.open_workbook(r'/Users/yinliqi/Desktop/CITICUP/ClimbingTitle/ClimbingFinancialData/上市股票一览.xls')
    sheet = workbook.sheet_by_index(0)
    first_column_data = []
    for row in range(1, sheet.nrows):
        first_column_data.append(sheet.cell_value(row, 0))
  #  result = [item.split('.')[0]for item in first_column_data]

    for stock_code in first_column_data:
        print(stock_code)
        # pro = ts.pro_api()
        # 此处修改起止日期
        df = pro.income(ts_code=stock_code, start_date='20130101', end_date='20240304')
                        # ,
                        # fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
        print(df)
        df.to_sql(name='test', con=engine, if_exists='append', index=False)
        time.sleep(0.12)

        # 10年



