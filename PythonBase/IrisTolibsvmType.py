# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
from CommonFunction.common import read_csv,write_csv
if __name__ == '__main__':
    filename = "C:\\Users\\asus\\Desktop\\Desk\\iris.csv"
    iris_data = read_csv(filename, header=True)
    res = []
    for item in iris_data:
        if len(item)!=5:
            continue
        # temp = []
        # if item[-1]=="2":
        #     label = "1"
        # else:
        #     label = item[-1]
        label = item[-1]
        first = "1" + ":" + item[0]
        second = "2" + ":" + item[1]
        third = "3" + ":" + item[2]
        four = "4" + ":" + item[3]
        # temp.extend([label, first, second, third, four])
        # res.append(" ".join(temp) + "\n")
        res.append(" ".join([label, first, second, third, four]) + "\n")
    with open("C:\\Users\\asus\\Desktop\\Desk\\iris_calss3.txt", "wb") as f:
        for ele in res:
            f.write(ele)