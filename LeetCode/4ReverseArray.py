# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
'''
import math


def reverse_array(array, k):
    n = len(array)
    for i in range(math.floor(n/2)):
        array[i], array[n - i - 1] = array[n - i - 1], array[i]

    for j in range(math.floor(k / 2)):
        array[j], array[k - j - 1] = array[k - j - 1], array[j]

    for m in range(math.floor((n-k) / 2)):
        array[k + m], array[n - m - 1] = array[n - m - 1], array[k + m]

    return array


if __name__ == '__main__':
    a = [1, 3, 5, 7, 9]
    reverse_array(a, 2)
    print(a)
