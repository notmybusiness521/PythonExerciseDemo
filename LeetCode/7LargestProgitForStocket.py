# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法
来计算你所能获取的最大利润。注意你不能在买入股票前卖出股票。
输入: [7,1,5,3,6,4]
输出: 7
输入: [1,2,3,4,5]
输出: 4
'''


def max_profit(array):
    if len(array) < 2:
        return 0
    profit = 0
    n = len(array)
    for i in range(1, n):
        profit += max(array[i] - array[i - 1], 0)
    return profit


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    print(max_profit(a))