import datetime
import os

# 配置参数类
class Config():
    root='./'#项目根目录
    sep=','#文件分隔符

    gpjg_start_date='20231108' #设置起始日期，包含 （存在数据丢失）
    gpjg_end_date='20240219'  #设置结束日期，包含
    gpjg_logfile='股票价格-%s.log' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    gpjg_dir=os.path.join(root,'股票价格')
    gpjg_iok=200 # 每分钟请求的次数，根据官方每分钟最多调取500次
    gpjg_wait_time=1+60//gpjg_iok #每次请求之间等待的时间，默认1秒加一分钟除以请求次数

config=Config()# 这句别漏了
