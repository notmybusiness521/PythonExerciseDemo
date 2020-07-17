# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        temp = {}
        for idx, ele in enumerate(nums):
            mod = target - ele
            if temp.get(mod) != None:
                return [idx, temp.get(mod)]
            else:
                temp[ele] = idx


if __name__ == '__main__':
    nums = [-1,-2,-3,-4,-5]
    target = -8
    s = Solution()
    print(s.twoSum(nums, target))
    # nums = [3, 2, 4, 5, 6]
    # t = nums[2:].index(4)
    # print(-8>-1)