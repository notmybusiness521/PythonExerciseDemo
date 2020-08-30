# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import time


def fib(n):
    """
    递归自顶向下
    """
    if n <= 0:
        print("num must big than 0")
    if n == 1 or n == 2:
        return 1
    return fib(n - 1) + fib(n - 2)


def dp_fib(n):
    """
    利用备忘录记录临时结果
    """
    if n < 1:
        return 0
    temp = [0] * (n + 1)
    return helper(temp, n)


def helper(l, n):
    if n == 1 or n == 2:
        return 1
    if l[n] != 0:
        return l[n]
    l[n] = helper(l, n - 1) + helper(l, n - 2)
    return l[n]


def fib_s2(n):
    """
    自底向上
    """
    res = [0] * (n + 1)
    res[1], res[2] = 1, 1
    for i in range(3, n + 1):
        res[i] = res[i - 1] + res[i - 2]
    return res[n]


if __name__ == '__main__':
    t1 = time.time()
    print(fib_s2(30))
    t2 = time.time()
    print(t2 - t1)
