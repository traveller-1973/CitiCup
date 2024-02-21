import urllib.request
from bs4 import BeautifulSoup

# 这个文件用于解析一个网页动态渲染后的源码内容
if __name__ == '__main__':
    stock_code = '000998'
    url = 'http://guba.eastmoney.com/list,' + stock_code + ',1,f.html'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req)
        html = resp.read().decode('utf-8')
        with open('news_' + stock_code + '.html', 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception as e:
        print("发生异常：", e)
