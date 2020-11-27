# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : save_word.py
# Datetime : 2020/11/20 下午2:45


# -*- coding: utf-8 -*-
# import sys

import zipfile

# import os
# import time
in_path = '/home/zheng/Documents/TestFile/qweqwe.docx'
out_path = '/home/zheng/Documents/TestFile/1.docx'


def docx_replace(old_file, new_file, rep):
    zin = zipfile.ZipFile(old_file, 'r')
    zout = zipfile.ZipFile(new_file, 'w')
    for item in zin.infolist():
        buffer = zin.read(item.filename)

        if (item.filename == 'word/document.xml' or 'header' or 'footer' in item.filename):

            res = buffer.decode('UTF-8')

            for r in rep:
                res = res.replace(r, rep[r])
            buffer = res.encode('UTF-8')
        zout.writestr(item, buffer)
    zout.close()
    zin.close()


def cecap_120(score_120):
    global coefficient
    coefficient = 0
    if score_120 < 65.0:
        coefficient = 1.0
    elif score_120 >= 65.0 and score_120 < 66.0:
        coefficient = 0.9
    elif score_120 >= 66.0 and score_120 < 67.0:
        coefficient = 0.8
    elif score_120 >= 67.0 and score_120 < 68.0:
        coefficient = 0.7
    elif score_120 >= 68.0 and score_120 < 69.0:
        coefficient = 0.6
    elif score_120 >= 69.0 and score_120 < 70.0:
        coefficient = 0.5
    elif score_120 >= 70.0 and score_120 < 71.0:
        coefficient = 0.4
    elif score_120 >= 71.0 and score_120 < 72.0:
        coefficient = 0.3
    elif score_120 >= 72.0 and score_120 < 73.0:
        coefficient = 0.2
    elif score_120 >= 73.0 and score_120 < 74.0:
        coefficient = 0.1
    else:
        coefficient = 0
    return coefficient


def cecap_60(score_60):
    global coefficient
    coefficient = 0
    if score_60 < 55.0:
        coefficient = 1.0
    elif score_60 >= 55.0 and score_60 < 56.0:
        coefficient = 0.9
    elif score_60 >= 56.0 and score_60 < 57.0:
        coefficient = 0.8
    elif score_60 >= 57.0 and score_60 < 58.0:
        coefficient = 0.7
    elif score_60 >= 58.0 and score_60 < 59.0:
        coefficient = 0.6
    elif score_60 >= 59.0 and score_60 < 60.0:
        coefficient = 0.5
    elif score_60 >= 60.0 and score_60 < 61.0:
        coefficient = 0.4
    elif score_60 >= 61.0 and score_60 < 62.0:
        coefficient = 0.3
    elif score_60 >= 62.0 and score_60 < 63.0:
        coefficient = 0.2
    elif score_60 >= 63.0 and score_60 < 65.0:
        coefficient = 0.1
    else:
        coefficient = 0
    return coefficient


def Condition_Score(coefficient):
    return coefficient * 8


def Acoustic_Score_Coefficient(Total_Score):
    return (Total_Score * 100) / 16


def akustik_ergebnis(Total_val):
    if Total_val >= 90:
        return '是'
    else:
        return '否'


# messung_code = input('试验编号：')
# fahrzeug_inner_code = input('VW车型代码：')
# fahrzeug_info = input('内部车型代号：')
# fahrzeug_code = input('车型编号：')
# leitung = input('发动机功率kw：')
# vorbeifahrt_geraeusch = input('通过噪声值dB：')
# score_120 = input('120km/h的CECAP值：')
# score_120 = eval(score_120)
# score_60 = input('60km/h的CECAP值：')
# score_60 = eval(score_60)

messung_code = '试验编号：'
fahrzeug_inner_code = 'VW车型代码：'
fahrzeug_info = '内部车型代号：'
fahrzeug_code = '车型编号：'
leitung = '发动机功率kw：'
vorbeifahrt_geraeusch = "qwewqe"
score_120 = eval("123")
score_60 = eval("23")

rep = {
    '*1000*': messung_code,
    '*1001*': fahrzeug_inner_code,
    '*1002*': fahrzeug_info,
    '*1003*': leitung,
    '*1004*': fahrzeug_code,
    '*1005*': vorbeifahrt_geraeusch,
    '*1006*': str(score_60),
    '*1007*': str(score_120),
    '*1008*': str(cecap_60(score_60)),
    '*1009*': str(cecap_120(score_120)),
    '*1010*': str(Condition_Score(cecap_60(score_60))),
    '*1011*': str(Condition_Score(cecap_120(score_120))),
    '*1012*': str(Condition_Score(cecap_120(score_120)) + Condition_Score(cecap_60(score_60))),
    '*1013*': str(Acoustic_Score_Coefficient(Condition_Score(cecap_120(score_120)) + Condition_Score(cecap_60(score_60)))),
    '*1014*': akustik_ergebnis(
        Acoustic_Score_Coefficient(Condition_Score(cecap_120(score_120)) + Condition_Score(cecap_60(score_60)))
    )
}
from pprint import pprint

# pprint(rep)
docx_replace(in_path, out_path, rep)
