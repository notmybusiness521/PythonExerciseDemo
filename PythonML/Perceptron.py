# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe: 感知机练习
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self, shape):
        self.w = np.ones(shape, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def fit(self, X_train, y_train):
        flag = False
        while not flag:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                flag = True
        return "Perceptron Model Done!"


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # print(df.label.value_counts())
    # plot picture
    # plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    # plt.xlabel("sepal length")
    # plt.ylabel("sepal width")
    # plt.legend()
    # plt.show()

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    perceptron = Model(len(X[0]))
    perceptron.fit(X, y)
    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
    plt.plot(X[50:100, 0], X[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()