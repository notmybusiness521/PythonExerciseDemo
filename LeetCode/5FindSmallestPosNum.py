# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:有一个随机序列的数组，找到其中缺失的最小正整数
举例如下，在[1,  2,  0] 中，该最小正整数应为3
在[3,  4,  -1,  1]中，该最小正整数应该为2
'''


def smallest_integer(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


if __name__ == '__main__':
    a = [3, 4, -1, 1]
    print(smallest_integer(a))
