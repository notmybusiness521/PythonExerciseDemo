# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import numpy as np
if __name__ == '__main__':
    left = np.array([13, 14, 15, 35])
    right = np.array([25, 49, 68, 71, 73])
    l_mean = left.mean()
    r_mean = right.mean()
    print(l_mean, r_mean)
    print(np.sum(np.square(left - l_mean)) + np.sum(np.square(right - r_mean)))

