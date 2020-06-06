# --*--coding=utf8--*--
'''
@Author:wjp
@Time:20190202
@Describe:常用函数功能
'''
import os
import re
import csv
import numpy as np
import pandas as pd


def read_csv(filename, header=False):
    res = []
    with open(filename) as f:
        f_csv = csv.reader(f)
        if header:
            headers = next(f_csv)
            header = False
        for row in f_csv:
            res.append(row)
    return res

def write_csv(data, filename):
    with open(filename, "wb") as f:
        f_csv = csv.writer(f)
        for item in data:
            f_csv.writerow(item)

def read_text(filename, columns, delimeter):
    res = []
    with open(filename, "rb") as f:
        while True:
            line = f.readline()
            if line:
                line = re.sub("[\r\n]", "", line)
                lines = line.split(delimeter)
                if len(lines) != columns:
                    continue
                res.append(lines)
            else:
                break
    return res

def get_dic(filename, colidx1, colidx2):
    res_dic = dict()
    data = read_csv(filename)
    for item in data:
        res_dic[item[colidx1]] = item[colidx2]
    return res_dic
def main():
    filename = "./input/npload.csv"
    # res = np.loadtxt(filename,dtype=str, delimiter="\t")
    # print type(res)
    # write_csv(res, "./input/npload.csv")
    df = pd.read_csv(filename, header=None, index_col=0, usecols=(1,2,3), skiprows=0)
    print(df.head(5))
    pd.read_excel()
    pd.pivot()
    # res = read_text(filename, 4, "\t")
    # for idx, item in enumerate(res):
    #     print idx, item

if __name__ == '__main__':
    main()

















