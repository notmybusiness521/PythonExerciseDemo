# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


def plot():
    nums = np.arange(-10, 10, step=1)
    plt.plot(nums, sigmoid(nums), 'r')
    plt.show()


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1-y, np.log(1 - model(X, theta)))
    return np.sum((left - right) / len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad

def shuffleData(data):
    pass


def stopCriterion(type, value, threshold):
    if type == STOP_ITER: return value > threshold
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold


def descent(data, theta, batchSize, stopType, threshold, alpha):
    init_time = time.time
    i = 0
    k = 0
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize
        if k >= len(y):
            k = 0
            X, y = shuffleData(data)
        theta = theta - alpha * grad
        costs.append(cost(X, y, theta))
        i += 1
        if stopType == STOP_ITER:value = i
        elif stopType == STOP_COST:value=costs
        elif stopType == STOP_GRAD:value=grad
        if stopCriterion(stopType, value, threshold):break
    return theta, i-1, costs, grad, time.time()-init_time


def predict(X, theta):
    return [1 if x >=0.5 else 0 for x in model(X, theta)]


def main():
    theta = np.zeros([1, 3])
    pass

if __name__ == '__main__':
    main()