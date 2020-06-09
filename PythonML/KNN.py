# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:K紧邻算法案例
'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import combinations


def L(x, y, p=2):
    """
    距离度量：p=1  曼哈顿距离 p=2 欧氏距离 p=∞ 切比雪夫距离
    x1 = [1, 1], x2 = [5, 1]
    :param x: 点1
    :param y: 点2
    :param p: 幂
    :return: 距离
    """
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # print(df.head())
    # plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    # plt.xlabel('sepal length')
    # plt.ylabel('sepal width')
    # plt.legend()
    # plt.show()
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


class KNN:
    def __init__(self, X_train, y_train, k = 3, p=2):
        """
        :param X_train: 训练样本
        :param y_train: 样本标签
        :param k: 紧邻个数
        :param p: 距离度量
        """
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.p = p

    def predict(self, X):
        k_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            k_list.append((dist, self.y_train[i]))
            # print(dist, y_train[i])
        for i in range(self.k, len(self.X_train)):
            max_index = k_list.index(max(k_list, key=lambda x:x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if k_list[max_index][0] > dist:
                k_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in k_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x:x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    # print(X_test)
    clf = KNN(X_train, y_train)
    print(clf.score(X_test, y_test))

    df = pd.DataFrame(load_iris().data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    test_point = [6.0, 3.0]
    print(clf.predict(test_point))
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
