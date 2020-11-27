# Create your tests here.

"""
链接:https://pan.baidu.com/s/1LKwHgxTR-LtA8YEepAMb5A 提取码:2d63
"""


# 测试多进程
from multiprocessing import Pool, Process, cpu_count


def Aa(a):
    print(a)


def Bb(b):
    print(b)

if __name__ == '__main__':
    pool = Pool(cpu_count())

