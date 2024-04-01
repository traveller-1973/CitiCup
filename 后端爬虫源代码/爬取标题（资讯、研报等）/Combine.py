import pandas as pd

if __name__ == '__main__':
    # 读取第一个excel文件
    df1 = pd.read_excel('ClimbingTitle(1).xls')

    # 读取第二个文件
    df2 = pd.read_excel('ClimbingTitle(2).xls')

    # 拼接两个 DataFrame
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # 将结果保存到新的 Excel 文件
    df_combined.to_excel(r'E:\ClimbingTitle\ClimbingTitle\Config\ClimbingTitle.xls',  index=False, engine='openpyxl')
