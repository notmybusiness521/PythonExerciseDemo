# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
from sklearn.datasets import load_iris

import numpy as np


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open("../input/testSet.txt")
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data, labels):
    """
    梯度上升算法求权重
    :param dataMatIn: 拟合数据:[[1,2,3,], [1,3,4]]
    :param classLabels: 标签 eg:[0,0,0,1,1,1]
    :return:
    """
    data_matrix = np.mat(data)
    label_mat = np.mat(labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = label_mat - h
        weights = weights + alpha*data_matrix.transpose()*error
    return weights


def plot_best_fit(weights):
    """
    画出决策边界
    :param weights: 权重矩阵
    :return: None
    """
    import matplotlib.pyplot as plt
    data, label = load_data_set()
    data_arr = np.array(data)
    n = np.shape(data)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, color="red", marker="s")
    ax.scatter(xcord2, ycord2, s = 30, color="green")
    x = np.arange(3.0, 8.0, 0.1)
    y = (-weights[0, 0] - weights[1, 0]*x) / weights[2, 0]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == '__main__':
    data, label = load_data_set()
    weights = grad_ascent(data, label)
    plot_best_fit(weights)

