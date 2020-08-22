# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:给定两个数组，编写一个函数来计算它们的交集。
@Example:
输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2,2]
输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [4,9]
'''


def solution1(arr1, arr2):
    res = []
    dict1 = dict()

    for e in arr1:
        if dict1.get(e) is not None:
            dict1[e] = dict1[e] + 1
        else:
            dict1[e] = 1

    for num in arr2:
        if dict1.get(num) is not None:
            dict1[num] = dict1[num] - 1
            if dict1[num]>=0:
                res.append(num)
    return res


def solution2(arr1, arr2):
    dict1 = dict()

    for e in arr1:
        if dict1.get(e) is not None:
            dict1[e] += 1
        else:
            dict1[e] = 1

    i = 0
    for num in arr2:
        if dict1.get(num) is not None:
            dict1[num] -= 1
            if dict1[num]>=0:
                arr2[i] = dict1[num]
                i += 1
    return arr2[0:i]


# 排好序的数组如何找到交集？
def solutionSort(arr1, arr2):
    arr1.sort()
    arr2.sort()
    i, j, k = 0, 0, 0
    # 如果a，b是数值变量， 则&， |表示位运算， and，or则依据是否非0来决定输出，
    while i<len(arr1) and j<len(arr2):
        if arr1[i] > arr2[j]:
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            arr1[k] = arr1[i]
            i += 1
            j += 1
            k += 1
    return arr1[0:k]


if __name__ == '__main__':
    arr1 = [1,2,2,1,2,3,4,5,12,34,45,564,567,67]
    arr2 = [2,2,5,7,45,567,3456,234,9]
    print(solution1(arr1, arr2))
    print(solutionSort(arr1, arr2))







