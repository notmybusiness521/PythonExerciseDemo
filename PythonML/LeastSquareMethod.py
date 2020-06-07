# --*--coding=utf8--*--


'''
@Author:
@Time:
@Describe:
'''


"""
举例：我们用目标函数 y=sin2πxy=sin2πx , 加上一个正态分布的噪音干扰，用多项式去拟合【例1.1 11页】
"""

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

regularization = 0.0001

def real_func(x):
    """
    真实函数
    :param x:
    :return:
    """
    return np.sin(2 * np.pi * x)


def fit_func(p, x):
    """
    拟合多项式函数ps: numpy.poly1d([1,2,3]) 生成  1x^2+2x^1+3x^0
    :param p: 多项式参数
    :param x: 变量
    :return:
    """
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, x, y):
    """
    残差
    :param p: 多项式参数
    :param x: 变量
    :param y: 函数真实值
    :return: ret
    """
    ret = fit_func(p, x) - y
    return ret


def fitting(x, y, M=0):
    """
    :param M: 多项式的次数
    :return:
    """
    p_init = np.random.rand(M + 1)
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    # print('Fitting Parameters:', p_lsq[0])
    # 可视化
    # plt.plot(x_points, real_func(x_points), label='real')
    # plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    # plt.plot(x, y, 'bo', label='noise')
    # plt.legend()
    # plt.show()
    return p_lsq


def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret


def reguarlizer_fitting(x, y, p_lsq_9, M=0):
    """
    L1: regularization*abs(p)
    L2: 0.5 * regularization * np.square(p)
    :param M: 多项式的次数
    :return:
    """
    p_init = np.random.rand(M + 1)
    p_lsq = leastsq(residuals_func_regularization, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
    plt.plot(
        x_points,
        fit_func(p_lsq[0], x_points),
        label='regularization')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq


if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)
    # 加上正态分布噪音的目标函数的值
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
    # print(x)
    p_lsq_9 = fitting(x, y, M=9)
    reguarlizer_fitting(x, y, p_lsq_9, M=9)
    pass