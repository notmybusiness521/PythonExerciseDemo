# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:机器学习实战基于sklearn和tensorflow第二章代码
'''
import os
import pandas as pd
#1数据探索
def data_exploration(filename):
    df = pd.read_csv(filename)
    print(df.head())
    print(df.info())
    print(df.describe())










def main():
    PROJECT_ROOT_DIR = "."
    DATA_PATH = "data"
    filename = os.path.join(PROJECT_ROOT_DIR, DATA_PATH, "housing.csv")
    data_exploration(filename)
    pass

if __name__ == "__main__":
    main()



























