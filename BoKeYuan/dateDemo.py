# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import time
import calendar
from datetime import date
from functools import wraps
def timeStampToDate(timeStamp):
    """
    将时间戳转换为日期格式
    :param timeStamp: Long：1556454836
    :return:日期："%Y-%m-%d %H:%M:%S"
    """
    local_time = time.localtime(timeStamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", local_time)
def dateDiff(date1, date2):
    """
    计算两个日期之间的天数
    :param date1:最近的日期
    :param date2:远一点的日期
    :return:间隔天数
    """
    d = date1 - date2
    return d.days

def test_time():
    test = []
    for i in range(10000000):
        test.append(i)

def time_fun(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "cost:", end - start, "seconds!")
        return result
    return wrapper

@time_fun
def test_timefun():
    test = []
    for i in range(10000000):
        test.append(i)

def main():
    date1 = date.today()
    date2 = date(1991, 5, 21)
    print(dateDiff(date1, date2))


if __name__=="__main__":
    main()
    print(time.time()) #1556456582.96
    dates = time.localtime()
    print("Date: ", dates)#time.struct_time(tm_year=2019, tm_mon=4, tm_mday=28, tm_hour=21, tm_min=3, tm_sec=2, tm_wday=6, tm_yday=118, tm_isdst=0)
    print(str(dates.tm_year)+"年",str(dates.tm_mon)+"月",str(dates.tm_mday)+"日",str(dates.tm_hour)+"时",str(dates.tm_min)+"分",str(dates.tm_sec)+"秒")
    #2019年 4月 28日 21时 3分 2秒
    print("一年中的第"+str(dates.tm_yday)+"天","星期"+str(dates.tm_wday+1))#一年中的第118天 星期7
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1556454836)))#2019-04-28 20:33:56

    start = time.time()
    test_time()
    end = time.time()
    print("Program cost:",end - start," seconds")

    test_timefun()
    cal = calendar.month(2019, 4)
    print("以下输出2019年4月份的日历:")
    print(cal)





