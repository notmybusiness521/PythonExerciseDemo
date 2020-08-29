# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
'''


def delete_repeat_num(nums, val):
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return nums[:k]


def remove_duplicates(nums):
    """
    给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
    不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) == 0:
        return 0
    k = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[k]:
            k += 1
            nums[k] = nums[i]
    return k + 1 # nums[:k + 1]


if __name__ == '__main__':
    nums = [3, 2, 2, 3]
    val = 3
    print(delete_repeat_num(nums, val))
    # print(remove_duplicates(nums))
