import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import  xlrd


if __name__ == '__main__':
    # 打开Excel文件
    workbook = xlrd.open_workbook(r'E:\桌面\日常事务\大学学科竞赛\花旗杯\上市股票一览.xls')
    # 选择第一个工作表
    sheet = workbook.sheet_by_index(0)
    # 获取第一列数据
    first_column_data = []
    for row in range(1, sheet.nrows):
        first_column_data.append(sheet.cell_value(row, 0))
    result = [item.split('.')[0] for item in first_column_data]
    # 打印第一列数据
    for stock_code in result:
       # print(stock_code)
        # 使用requests获取网页内容
        url = 'http://guba.eastmoney.com/list,' + stock_code + ',1,f.html'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 使用Selenium获取网页源代码
        chrome_options = Options()
        chrome_options.add_argument('--disable-popup-blocking')
        # driver = webdriver.Chrome(executable_path='/Users/yinliqi/Downloads/chromedriver_mac_arm64/chromedriver', options=chrome_options)
        driver = webdriver.Chrome('C:/Program Files/Google/Chrome/Application/chromedriver')

        driver.get(url)
        driver.implicitly_wait(10)
        html_content = driver.page_source
        driver.quit()

        # 使用BeautifulSoup解析页面内容
        soup_selenium = BeautifulSoup(html_content, 'html.parser')

        body_tag = soup_selenium.find('body')
        # print(body_tag)
        if body_tag:
            script_tags = body_tag.find_all('script')
            if script_tags:
                script_content = script_tags[1].string
                match_infos = re.findall(r'"post_title":"(.*?)".*?"post_publish_time":"(.*?)"', script_content)
                if match_infos:
                    count = 0
                    for match_info in match_infos:
                        post_title = match_info[0]
                        post_publish_time = match_info[1]
                        print("Post Title:", post_title)
                        print("Post Publish Time:", post_publish_time)
                        count += 1
                        if count == 3:  # 只打印前三个帖子信息
                            break
                else:
                    print("No match found.")
            else:
                print("No <script> tag found inside <body inmaintabuse='1'>.")
        else:
            print("No <body inmaintabuse='1'> tag found.")


