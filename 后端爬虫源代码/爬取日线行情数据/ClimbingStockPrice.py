# 导入库
import tushare as ts
import time, os
import datetime
import pandas as pd
from config import config
import logging

# 设置Tushare的token,这样之后请求api的时候就不用带token了
my_token = 'bf682e0a37a42f1e325af13c62199f756130e38c377deca9fc3bfca6'
pro = ts.pro_api(my_token)
ts.set_token(my_token)

# 日志文件
logging.basicConfig(
filename=config.gpjg_logfile,
level=logging.ERROR,
format='%(levelname)s:%(asctime)s\t%(message)s'
)

# 获得A股code
code = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date,list_status')#上市
time.sleep(1) #友好一点,请求一下休息一下
code2 = pro.query('stock_basic', exchange='', list_status='P', fields='ts_code,symbol,name,area,industry,list_date,list_status')#暂停上市
time.sleep(2)
code3 = pro.query('stock_basic', exchange='', list_status='D', fields='ts_code,symbol,name,area,industry,list_date,list_status')#退市
code=pd.concat([code,code2,code3],axis=0,ignore_index=True)#将返回的DataFrame拼接起来

stock_code=code['ts_code'].tolist()#股票代码
stock_name=code['name'].tolist()#股票名称

# 获取股票价格
def get_gpjg(ts_code,ts_name,start_date, end_date,freq='D', adj=None,adjfactor=False):
    info=' %s %s %s ... ' % (ts_code+'_'+ts_name, start_date,end_date)
    for xx in range(5):# 失败后,尝试5次
        try:
            print('try',info,xx)
            logging.critical('try'+info+str(xx))
            # 文档 https://waditu.com/document/2?doc_id=109
            # 原api（start_date，end_date】 因此将前后都减去1天得到【start_datei，end_datei）
            df = ts.pro_bar(ts_code=ts_code, adj=adj,adjfactor=adjfactor, start_date=start_date, end_date=end_date,freq=freq)
            print('get',info,xx)
            print('wait~')
            logging.critical('get'+info+str(xx))#日志
            return df
        except:
            print('failed'+info+str(xx))
            logging.critical('failed'+info+str(xx))
            time.sleep(config.gpjg_wait_time)
    error_img='error! %s'% '\t'.join(map(str,[get_gpjg.__name__, ts_code,ts_name,start_date, end_date,freq, adj,adjfactor]))
    print(error_img)
    logging.error(error_img)
    raise RuntimeError(error_img)#失败抛出异常


if __name__ == '__main__':
    for adj in ['未复权', 'qfq', 'hfq']:  # 不复权 qfq前复权 hfq后复权
        for i in range(len(stock_code)):  # 按股票代码依次获取
            ts_code = stock_code[i]
            ts_name = stock_name[i]
            dir_path = os.path.join(config.gpjg_dir,
                                    '%s_%s_%s' % (adj, config.gpjg_start_date, config.gpjg_end_date))  # 输出文件目录
            file_path = os.path.join(dir_path, '%s.csv' % (ts_code))  # 输出文件的全路径
            if not os.path.exists(config.gpjg_dir):
                os.mkdir(config.gpjg_dir)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            if os.path.exists(file_path):  # 避免多次运行代码时重复获取数据
                continue
            if adj == '未复权':
                df = get_gpjg(ts_code, ts_name, config.gpjg_start_date, config.gpjg_end_date, freq='D', adj=None,
                              adjfactor=False)
            else:
                df = get_gpjg(ts_code, ts_name, config.gpjg_start_date, config.gpjg_end_date, freq='D', adj=adj,
                              adjfactor=True)

            if df is None:  # 获取失败后记录在日志
                error_img = 'error! return is None! %s' % '\t'.join(
                    map(str, [ts_code, ts_name, config.gpjg_start_date, config.gpjg_end_date, adj]))
                print(error_img)
                logging.error(error_img)
                continue
            df.sort_values(by='trade_date', inplace=True)  # 按 时间 顺序 保存
            df.to_csv(file_path, index=False, sep=config.sep, encoding='utf-8')  # 保存
            time.sleep(config.gpjg_wait_time)  # 设置等待时间,避免被服务器当初恶意攻击
