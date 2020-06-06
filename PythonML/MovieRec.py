# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:电影推荐的数据探索
'''
import matplotlib.pyplot as plt
import numpy as np
from CommonFunction.common import read_csv,read_text,write_csv
from collections import Counter


def get_user_data(filename, colnum):
    #获取用户相关数据
    ages = []
    occupation = []
    temp_occupation = []
    user_data = read_text(filename, 5, "|")
    for item in user_data:
        ages.append(int(item[colnum]))
        temp_occupation.append(item[3])
    occupation_x = np.array([e[0] for e in Counter(temp_occupation).items()])
    occupation_y = np.array([e[1] for e in Counter(temp_occupation).items()])
    x_axis = occupation_x[np.argsort(occupation_y)]
    y_axis = occupation_y[np.argsort(occupation_y)]
    for i in range(len(x_axis)):
        occupation.append((x_axis[i], y_axis[i]))
    return ages,occupation
def get_item_data(filename, colum):
    #获取电影数据
    item_data = read_text(filename, colum, "|")
    years = []
    for item in item_data:
        year = convent_year(item[2])
        years.append(year)
    return years
def get_rate_data(filename, colmun):
    #评分分布
    rates = []
    rate_data = read_text(filename, colmun, "\t")
    for item in rate_data:
        rate = item[2]
        if rate:
            rates.append(rate)
    return rates

def convent_year(y):
    try:
        return int(y[-4:])
    except:
        return 1900
def plt_hist_distribution(ages):
    #年龄分布图
    plt.hist(ages, bins=20, color='lightblue', density=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.show()

def user_rate_distrution():
    # 用户评分分布探索
    filename = "E:\\data\\ml-100k\\u.data"
    rates = get_rate_data(filename, 4)
    total = len(rates)
    rates = [(ele[0], round(ele[1] / float(total), 2)) for ele in sorted(Counter(rates).items())]
    print(rates)
    pos = np.arange(len(rates))
    width = 1.0
    x_label = [e[0] for e in rates]
    y = [e[1] for e in rates]
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(x_label)
    plt.bar(pos, y, width, color='blue')
    # plt.xticks(rotation=90)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()
def plt_occupations_distrution(occupations):
    #职业分布图
    x, y = [], []
    for item in occupations:
        x.append(item[0])
        y.append(item[1])
    pos = np.arange(len(x))
    width = 1.0
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(x)
    plt.bar(pos, y, width, color='blue')
    plt.xticks(rotation=30)
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.show()
def user_rate_count():
    filename = "E:\\data\\ml-100k\\u.data"
    data = read_text(filename, 4, "\t")
    user_ids = []
    for item in data:
        user_ids.append(item[0])
    user_rate_times = [e[1] for e in Counter(user_ids).items()]
    plt_hist_distribution(user_rate_times)

def main():
    #用户分布探索
    # filename = "E:\\data\\ml-100k\\u.user"
    # ages, occupations = get_user_data(filename, 1)
    # plt_hist_distribution(ages)
    # plt_occupations_distrution(occupations)
    #电影年龄分布探索
    # filename = "E:\\data\\ml-100k\\u.item"
    # years = get_item_data(filename, 24)
    # years = [1998-year for year in years if year!=1900]
    # plt_hist_distribution(years)
    user_rate_count()
    # plt_hist_distribution(rates)
    pass


if __name__ == '__main__':
    main()