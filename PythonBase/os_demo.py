# --*--coding=utf8--*--
'''
@Author:wjp
@Time:20190403
@Describe:将spark中的examples目录下的所有例子的开头部分的注释去掉。方便再windows下学习代码
'''

import os


def read_pyfile(filename):
    '''
    读取py文件并处理,将#开头略过，其他保存为一个list
    :param filename: py文件目录
    :return: 一个列表，每一个元素是文件里的一行
    '''
    res = []
    if filename.endswith(".py"):
        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                if line:
                    if line.startswith("#"):
                        continue
                    res.append(line)
                else:
                    break
    print("Read % s success!" % filename)
    return res


def write_pyfile(lines, filename):
    '''
    将去除#开头的py文件内容列表，重新写到一个py文件里
    :param lines: read_pyfile函数的返回结果
    :param filename: 写入文件目录
    :return: 无
    '''
    with open(filename, "wb") as f:
        for line in lines:
            f.write(line)


def batch_deal_file(file_path):
    '''
    批量读写py文件
    :param file_path: 文件路径
    :return: 无
    '''
    for path, directory, files in os.walk(file_path):
        # 路径，路径下的目录名，路径下的文件名
        for f in files:
            filename = path + "\\" + f
            res = read_pyfile(filename)
            write_pyfile(res, filename)

def main():
    path = "C:\\Users\\asus\\Desktop\\Desk\\python\\"
    batch_deal_file(path)


if __name__ == '__main__':
    main()











