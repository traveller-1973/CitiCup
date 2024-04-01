import time
from sqlalchemy import create_engine
import tushare as ts
import xlrd
import os

# 设置Tushare的token,这样之后请求api的时候就不用带token了
my_token = '78997d1a8a700f6e0ae112e14b8907c650463e631603f71092968cfc'
pro = ts.pro_api(my_token)
ts.set_token(my_token)

if __name__ == '__main__':
    engine = create_engine('mysql+mysqlconnector://root:wjswbyydyh828@localhost/FinancialData')

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
        df = pro.balancesheet(ts_code=stock_code, start_date='20130101', end_date='20240304')
                        # ,
                        # fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
        print(df)
        df.to_sql(name='Balance_Sheet', con=engine, if_exists='append', index=False)
        time.sleep(0.12)

        # 10年



