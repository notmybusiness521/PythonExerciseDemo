# --*--coding=utf8--*--

'''
@Author:
@Time:
@Describe:编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，则返回""
@Example:
输入: ["flower","flow","flight"]
输出: "fl"
输入: ["dog","racecar","car"]
输出: ""
'''


def solution(array):
    if len(array) < 1:
        return ""
    prefix = array[0]
    for ele in array:
        idx = 0
        while idx < len(prefix):
            if ele.__contains__(prefix) and ele.startswith(prefix[0]):
                break
            else:
                prefix = prefix[:len(prefix)-1]
            idx += 1
    return prefix


if __name__ == '__main__':
    s = ["dog", "racecar", "car"]
    print(solution(s))
