# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : utils.py
# Datetime : 2020/11/17 上午9:29


# 基于机器学习的汽车噪音智能识别


import math
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from keras.models import load_model
from matplotlib.ticker import MaxNLocator
from numba import njit
from numpy.fft._pocketfft import fft
from scipy.signal.windows import hann

from settings.dev import model_path, wav_path, audio_path


def read_wav(wav_test):
    '''
        input:
        wav_test(str): 待测音频文件

        output:
        y: 归一化后音频数据
        t: 时间序列
    '''
    sr, y = scipy.io.wavfile.read(wav_test)
    y = y.astype(np.float64)

    if len(y) <= sr * 2:
        raise Exception("WAV Time is too short!")

    y = y / y.max()  # 加入归一化
    t_serie = np.arange(len(y)) / 44100

    return y, t_serie


def readAscFile(file_path):  # head_content needs to be completed
    '''
    :param file_path: 文件路径
    :return: data_matrix: 读取到的数据矩阵，第一列为时间数据，最后一列为转速数据，中间列为各通道声压级
    '''
    head_content = ""
    datas = []
    with open(wav_path + file_path.name, 'r', encoding='unicode_escape') as f:
        # 循环读取文件的每一行
        for file_content in f:
            if not file_content:  # 如果读到文件最后一行则跳出循环
                break
            if not file_content[0].isdigit():  # 如果读取到的是字符串
                head_content += file_content  # 则将字符串内容添加到head_content
            elif file_content[0].isdigit():  # 如果读取到的是数字
                file_content = file_content.split()  # 将改行按" "分割组成列表
                datas.append(file_content)  # 将读取到的数据存入datas
        data_matrix = np.array(datas, dtype=np.float64)  # 将列表转为np.array数据结构
    return data_matrix


def detectFs(raw_time):  # seems to be completed
    """
    :param raw_time: 时间数据
    :return: 数据的采样频率
    """
    fs_type = [4096, 8192, 16384, 32768, 65536, 22050, 44100, 48000]  # 能识别的采样频率列表
    fs_type = np.array(fs_type)
    fsd = len(raw_time) / raw_time[-1]  # fsd = raw_time长度/时间 例：fsd = 44099.98631
    fs = fs_type[abs(fs_type - fsd) < 1]  # 识别fs_type 中哪一个采样率与fsd的差值小于1，则该采样率即为数据采样率
    if len(fs) == 1:
        return float(fs)
    else:
        return fsd


def dataA(raw_time, raw_data):  # need to be improved
    '''
    :param raw_time:时间数据
    :param raw_data: 声压级数据
    :return: A计权后的数据
    思路：对声压级数据fft后进行频率A计权，后逆fft得到时域上的A计权数据
    '''
    fs = 44100  # 得到数据采样率
    n = len(raw_time)  # raw_time数据长度
    if n % 2 == 0:
        f = np.arange(n // 2) / n * fs  #####  np.arange(n//2 +1) / n --- NA
        fa = librosa.A_weighting(f)
        temp1 = np.fft.fft(raw_data)
        # temp2 = 20*np.log10(temp1/2e-5)
        faf = 10 ** (fa / 20)  # *2e-5
        faf2 = faf[::-1]
        fafc = np.hstack((faf, faf2))
    else:
        f = np.arange(n // 2 + 1) / (n - 1) * fs  ## f 为 0到22050的列表 len(f) = n//2 + 1
        fa = librosa.A_weighting(f)  # fa:根据频率f得到的频率A计权特性
        temp1 = np.fft.fft(raw_data)
        faf = 10 ** (fa / 20)  # *2e-5    ##### 将分贝的加减转化为声压能量的乘法 temp3 = temp1*fafc
        faf2 = faf[::-1]  # 翻转faf
        fafc = np.hstack((faf, faf2[1:]))  # 拼接faf与faf2  len(fafc) = n
    temp3 = temp1 * fafc
    temp4 = np.fft.ifft(temp3)  # 逆fft得到temp4
    dataA = temp4.real  # 取实部得到dataA
    return dataA


@njit
def get_first_index(A, k, delta):
    for i in range(len(A)):
        if A[i] > k - delta and A[i] < k + delta:
            return i
    return -1


def rms(sig, window_size=None):
    fun = lambda a, size: np.sqrt(np.sum([a[size - i - 1:len(a) - i] ** 2 for i in range(size - 1)]) / size)
    if len(sig.shape) == 2:
        return np.array([rms(each, window_size) for each in sig])
    else:
        if not window_size:
            window_size = len(sig)
        return fun(sig, window_size)


def rpmSelect2(raw_rpm, rpm_step, rpmtype='rising'):
    '''
    :param raw_time:时间数据
    :param raw_rpm:转速数据
    :param rpm_step: 步长
    :param rpmtype:'falling'；
    :return: rpml: 从最低转速到最高转速的列表或从最高转速到最低转速的列表（步长为rpm_step）
              rpmf:对应raw_time中的采样点位置（相差不超过deltaR）
    '''
    if rpmtype == 'falling':  # 如果是falling，将raw_rpm翻转
        raw_rpm = raw_rpm[::-1]
    r_minp = np.argmin(raw_rpm)  # raw_rpm最小值对应位置
    r_maxp = np.argmax(raw_rpm)  # raw_rpm最大值对应位置
    s_rpm = raw_rpm[r_minp:r_maxp + 1]
    rpm_ini = np.ceil(s_rpm[0] / rpm_step) * rpm_step  # rpm_ini：1110
    rpm_end = np.floor(s_rpm[-1] / rpm_step) * rpm_step  # rpm_end：5980
    rpml = np.arange(rpm_ini, rpm_end + rpm_step, rpm_step)  # rpml: 从最低转速到最高转速的列表（步长为rpm_step）
    rpmf = np.zeros(len(rpml))

    '''
    寻找rpml中每个转速对应的raw_time中的采样点位置（相差不超过deltaR）
    '''

    tag = 0
    for i in range(len(rpml)):
        deltaR = 0.05
        not_find = True
        while not_find:
            j = get_first_index(s_rpm, rpml[i], deltaR)
            if j != -1:
                not_find = False
                rpmf[i] = j
            else:
                deltaR = deltaR + 0.05
    rpmf = rpmf + r_minp
    if rpmtype == 'falling':
        rpml = rpml[::-1]
        t01 = len(raw_rpm) - rpmf
        rpmf = t01[::-1]
    return rpmf, rpml


def fft_time_1(raw_time, raw_data, spectrum_size=16384, overlap=50, weighting=1):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param spectrum_size: 谱线数
    :param overlap: 重叠率
    :param weighting: 等于1时计算A计权声级
    :return: f:每条谱线对应的频率
              time_array: fft vs. time中的时间轴序列
              pm:每一时间每一频率下对应的fft幅值（rms)
    '''
    fs = 44100
    stepping = np.floor(spectrum_size * (100 - overlap) / 100)
    window_data = int((len(raw_time) - spectrum_size) // stepping + 1)  # 计算窗口数量
    time_array = np.arange(spectrum_size / 2, spectrum_size / 2 + stepping * (window_data), stepping) / fs
    #    time_array = np.arange(raw_time[0],raw_time[0]+spectrum_size/2*(window_data),spectrum_size/2)/fs
    pp = np.zeros((spectrum_size // 2 + 1, window_data))  # 预分配fft_vs_time 矩阵
    # 计算每一time_array时间对应的FFT值
    for i in range(window_data):
        x = raw_data[int(i * stepping):int(i * stepping + spectrum_size)]  # 获得每一窗口的raw_data数据
        n = spectrum_size  # Length for FFT
        wn = hann(n)
        xx = x * wn
        y = fft(xx)
        f = fs * np.arange(n // 2 + 1) / n
        p = np.abs(y / n)
        p1 = p[:n // 2 + 1]
        p1[1: -1] *= 2
        p2 = 2 * p1  # recover from window functions
        p2 = p2 * 2 ** -0.5  # peak to rms
        pp[0:, i] = p2  # 生成所有分块数据的FFT矩阵
    pm = 20 * np.log10(pp / 2e-5)
    #    pm = pm.transpose()
    if weighting == 1:
        for i in range(window_data):
            pm[:, i] = pm[:, i] + librosa.A_weighting(f)
        return f, time_array, pm
    else:
        return f, time_array, pm


def fft_time(raw_data, fs=44100, spectrum_size=8192, overlap=75):
    '''
    :param raw_data: 声压数据
    :param spectrum_size: 谱线数
    :param overlap: 重叠率
    :param weighting: 等于1时计算A计权声级
    :return: f:每条谱线对应的频率
              time_array: fft vs time中的时间轴序列
              pm:每一时间每一频率下对应的fft幅值（rms)
    '''

    stepping = np.floor(spectrum_size * (100 - overlap) / 100)
    window_data = int((len(raw_data) - spectrum_size) // stepping + 1)  # 计算窗口数量
    time_array = np.arange(spectrum_size / 2, spectrum_size / 2 + stepping * (window_data), stepping) / fs
    pp = np.zeros((spectrum_size // 2 + 1, window_data))  # 预分配fft_vs_time 矩阵
    # 计算每一time_array时间对应的FFT值
    for i in range(window_data):
        x = raw_data[int(i * stepping):int(i * stepping + spectrum_size)]  # 获得每一窗口的raw_data数据
        n = spectrum_size  # Length for FFT
        wn = hann(n)
        xx = x * wn
        y = fft(xx)
        f = fs * np.arange(n // 2 + 1) / n
        p = np.abs(y / n)
        p1 = p[:n // 2 + 1]
        p1[1: -1] *= 2
        p2 = 2 * p1  # recover from window functions
        p2 = p2 * 2 ** -0.5  # peak to rms
        pp[0:, i] = p2  # 生成所有分块数据的FFT矩阵
    pm = 20 * np.log10(pp / 2e-5)
    #    pm = pm.transpose()
    for i in range(window_data):
        pm[:, i] = pm[:, i] + librosa.A_weighting(f)

    return f, pm


def fft_rpm(raw_time, raw_data, raw_rpm, rpm_step, rpmtype, spectrum_size, fs, weighting):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param raw_rpm: 转速数据
    :param rpm_step: 转速步长
    :param rpmtype: rpmtype: 'falling'
    :param spectrum_size: 谱线数
    :param fs: 采样率
    :param weighting: 等于1时计算A计权声级
    :return:f:每条谱线对应的频率
             rpml:fft vs rpm中的时间轴序列
             pm:每一转速每一频率下对应的fft幅值（rms)
    '''
    rpmf, rpml = rpmSelect2(raw_rpm, rpm_step, rpmtype)  # rpml: 从最高转速到最低转速的列表（步长为rpm_step）rpmf:对应raw_time中的采样点位置（相差不超过deltaR）
    pp = np.zeros((spectrum_size // 2 + 1, len(rpml)))
    for i in range(len(rpml)):
        try:
            #### 提取每一转速对应的声压数据（包含spectrum_size个采样点）
            x = raw_data[int(rpmf[i] - spectrum_size / 2 + 1):int(rpmf[i] + spectrum_size / 2 + 1)]
            n = spectrum_size  # Length for FFT
            wn = hann(n)
            xx = x * wn
            y = fft(xx)
            f = fs * np.arange(n // 2 + 1) / n
            p = np.abs(y / n)
            p1 = p[:n // 2 + 1]
            p1[1: -1] *= 2
            p2 = 2 * p1  # recover from window functions
            p2 = p2 * 2 ** -0.5  # peak to rms
            pp[0:, i] = p2  # 生成所有分块数据的FFT矩阵
        except:
            pp[0:, i] = pp[0:, i]

    # 如果pp中有存在全为0的列则去除，并去除对应的rpml值
    t01 = []
    for i in range(len(rpml)):
        if pp[:, i].all() == 0:
            t01.append(i)
    rpml = np.delete(rpml, t01)
    pp = np.delete(pp, t01, axis=1)
    pm = 20 * np.log10(pp / 2e-5)
    if weighting == 1:
        for i in range(len(rpml)):
            pm[:, i] = pm[:, i] + librosa.A_weighting(f)
        return f, rpml, pm
    else:
        return f, rpml, pm


def fft_average(raw_data, spectrum_size=16384, fs=44100, overlap=50, weighting=1):
    '''
    :param raw_data: 声压数据
    :param spectrum_size: 谱线数
    :param fs: 采样频率
    :param overlap: 重叠率
    :param weighting: 等于1时计算A计权声级
    :return: f:每条谱线对应的频率
              db:每条谱线对应的声压级
    '''
    stepping = np.floor(spectrum_size * (100 - overlap) / 100)  # 根据spectrum_size和overlap计算每次窗口移动的步长
    window_data = int((len(raw_data) - spectrum_size) // stepping + 1)  # 计算含有多少个窗
    pp = np.zeros((window_data, spectrum_size // 2 + 1))
    n = spectrum_size  # Length for FFT
    wn = hann(n)  # 汉宁窗
    # 计算每个窗对应的频谱

    for i in range(window_data):
        x = raw_data[int(i * stepping):int(i * stepping + spectrum_size)]  # 获得分块数据
        xx = x * wn  # 分块数据加窗
        y = fft(xx)
        f = fs * np.arange(n // 2 + 1) / n  # 频域取FFT结果的前一半
        p = np.abs(y / n)  # FFT结果取模值
        p1 = p[:n // 2 + 1]  # FFT结果取前半部分
        p1[1: -1] *= 2  # FFT后半部分幅值叠加到前半部分
        p2 = 2 * p1  # 恢复窗函数幅值
        p2 = p2 * 2 ** -0.5  # peak to rms
        pp[i, 0:] = p2  # 生成所有分块数据的FFT矩阵

    pp_avr = rms(pp.transpose())  # 对分块数据求均方根
    db = 20 * np.log10(pp_avr / 2.0e-5)  # 将幅值转为声压级
    # 如果weighting==1则用计算频谱上的A计权声级
    if weighting == 1:
        dba = db + librosa.A_weighting(f)

        return f, dba
    else:
        return f, db


def fun_sum_db(raw_data, WindowType='Hanning'):
    '''
    raw_data:谱线dB值
    WindowType:FFT时加窗类型
    '''
    n = len(raw_data)
    raw_data_recovery = 10 ** (raw_data / 20) * 2e-5
    if WindowType == 'Hanning':
        energy_sum = sum((raw_data_recovery / 2 * 1.663) ** 2)
    else:
        energy_sum = sum(raw_data_recovery ** 2)
    sum_db = 20 * math.log(energy_sum ** 0.5 / 2e-5, 10)
    return sum_db


def Octave_Time(raw_time, raw_data, fs=44100, spectrum_size=16384, overlap=50, weighting=1):
    #     Octave_band = [22.4,45,90,180,355,710,1400]
    #     f, time_array, pm = fft_time_1(t_serie, y, fs, spectrum_size, overlap, weighting)

    #     octave_output = []

    #     for i in range(len(Octave_band)-1):
    #         f_band = f[np.where((f>=Octave_band[i]) & (f<Octave_band[i+1]))]
    #         octave_serie = []

    #         for j in range(pm.shape[1]):
    #             sum_db = fun_sum_db(pm[:,j])
    #             octave_serie.append(sum_db)

    #         octave_output.append(octave_serie)

    #     octave_output = np.array(octave_output)
    Octave_band = [22.4, 45, 90, 180, 355, 710, 1400]
    f, time_array, pm = fft_time_1(raw_time, raw_data)

    octave_output = []

    for i in range(len(Octave_band) - 1):
        f_band = np.where((f >= Octave_band[i]) & (f < Octave_band[i + 1]))
        #     print(f_band.shape)
        octave_serie = []

        for j in range(pm.shape[1]):
            sum_db = fun_sum_db(pm[:, j][f_band])
            octave_serie.append(sum_db)

        octave_output.append(octave_serie)

    octave_output = np.array(octave_output)

    return octave_output, time_array


## 添加A计权
def order_vfft(rpmf, rpml, raw_time, raw_data, rpm_step, rpmtype, order, orderResolution, orderWidth, weighting):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param raw_rpm: 转速数据
    :param rpm_step: 转速步长
    :param rpmtype: 'falling'
    :param order: 需要提取的阶次
    :param orderResolution: 决定每个转速提取的blockSize长度，用于fft
    :param orderWidth: 每个转速对应的阶次频率不一定刚好为fft中对应的频率，故加入orderWidth在一定的误差范围内寻找阶次频率。
    :param smoothFrac: 局部加权回归中的参数值，smoothFrac：截取数据比例
    :return: rpml: 从最低转速到最高转速的列表或从最高转速到最低转速的列表（步长为rpm_step）
             dbo：对应转速下阶次频率的分贝值
    '''
    rpmf = rpmf[np.where(rpml > 1000)]
    rpml = rpml[np.where(rpml > 1000)]

    rpml_selected = []
    dbo = []
    fs = detectFs(raw_time)
    #     dbo = np.zeros(np.size(rpml))
    for i in range(np.size(rpml)):
        ####  根据rpml和orderResolution决定blockSize（即从rpml对应的时间点，往前往后截取的数据长度。）
        fsResolution = rpml[i] / 60 * orderResolution
        blockSize = fs / fsResolution

        try:
            fft_start_point = rpmf[i] - blockSize // 2 + 1
            fft_end_point = rpmf[i] + blockSize // 2 + 1
            if fft_start_point < 0 or fft_end_point > len(raw_data) + 1:
                continue
            rpml_selected.append(rpml[i])
            x = raw_data[int(fft_start_point): int(rpmf[i] + blockSize // 2 + 1)]
            n = len(x)
            wn = hann(n)
            xx = x * wn  # 加汉宁窗
            y = fft(xx)
            f = fs * np.arange(n // 2 + 1) / n
            ####  得到xx对应的fft值后，进一步后处理获得每个频率上对应的声压级
            p = np.abs(y / n)
            p1 = p[:n // 2 + 1]
            p1[1: -1] *= 2
            p2 = 2 * p1
            p2 = p2 * 2 ** - 0.5  ###  *0.707得到RMS值
            db = 20 * np.log10(p2 / 2e-5)
            if weighting == 1:
                db = db + librosa.A_weighting(f)
            # 寻找order对应的频率，及该频率对应的db值，order对应的频率不可能与f中的频率刚好一致，故通过orderWidth寻找该频带中的频率
            fsFloor = rpml[i] / 60 * (order - orderWidth / 2)
            fsTop = rpml[i] / 60 * (order + orderWidth / 2)
            fsSelected = np.where((f > fsFloor) & (f < fsTop))
            if len(fsSelected[0]) == 1:
                dbo_i = db[fsSelected[0][0]]
            else:
                #### 如果两个频率都处于2阶转频的范围，则求两条谱线总的db值
                dbs = db[fsSelected[0]]
                ppre = 10 ** (dbs / 20) * 2e-5
                pef = sum(ppre ** 2) ** 0.5
                #                 pef = sum((ppre/2*1.663)**2) ** 0.5
                dbo_i = 20 * np.log10(pef / 2e-5)

        # 如果上段Tay中的代码有错误则该转速对应的dbo值为0
        except:
            print(rpml[i], '转速出错')
            dbo_i = 0
        dbo.append(dbo_i)

    rpml_selected = np.array(rpml_selected)
    dbo = np.array(dbo)
    t01 = np.where(dbo != 0)  # 寻找dbo不等于0的位置
    ##### -------------------------------
    rpml_selected = rpml_selected[t01[0]]
    dbo = dbo[t01[0]]  # 舍弃dbo为0的值

    return rpml_selected, dbo


def level_time(raw_time, raw_data, timeWeighting=0.25, A=1):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param timeWeighting: 积分时间，0.125则为fast 1则为slow
    :param A:如果A =1 则利用dataA进行A计权
    :return: level_time（fsat or slow)后的数据
    思路：将时域的卷积转化为频域的乘积
    期间对时间数据与声压数据进行拼接，使fft与ifft过程更加准确
    '''
    fs = 44100
    if A == 1:
        raw_data = dataA(raw_time, raw_data)

    # 对时间数据与声压数据进行拼接，变成3倍长度
    timep1 = raw_time[:-1] - raw_time[-1]
    timep2 = raw_time[1:] + raw_time[-1]
    timep = np.hstack((timep1, raw_time, timep2))
    datap1 = raw_data[::-1]
    datap = np.hstack((datap1[:-1], raw_data, datap1[1:]))
    timepe = timep - timep[0]

    # 将时域的卷积转化为频域的乘积
    raw_e = np.exp(-timepe / timeWeighting)
    ff = np.fft.fft(datap ** 2) * np.fft.fft(raw_e)
    ff2 = np.fft.ifft(ff)
    ff2r = ff2.real
    ff2rs = ff2r[len(raw_time) - 1:len(raw_time) * 2 - 1]

    pa = (ff2rs / fs / timeWeighting) ** 0.5
    lpa = 20 * np.log10(pa / 2e-5)

    return lpa


def level_rpm(raw_time, raw_data, raw_rpm, rpm_step=25, rpmtype='rising', timeWeighting=0.25, A=1):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param raw_rpm: 转速数据
    :param rpm_step: 转速步长
    :param rpmtype: 'falling'
    :param timeWeighting: 积分时间长度
    :param A: 如果A =1 则利用dataA进行A计权
    :return: lpr转速为rpml时对应的声压分贝值
    '''
    lpa = level_time(raw_time, raw_data, timeWeighting, A)  # 计算level_time
    rpmf, rpml = rpmSelect2(raw_rpm, rpm_step, rpmtype)  # rpml: 从最高转速到最低转速的列表（步长为rpm_step）rpmf:对应raw_time中的采样点位置（相差不超过deltaR）
    lpr = np.zeros(len(rpmf))

    for i in range(len(rpmf)):
        lpr[i] = lpa[int(rpmf[i])]  # 根据rpmf中采样点的位置找寻对应了level_time值

    return rpml, lpr


def Order_Spectrum(raw_rpm, raw_time, raw_data, orderRange=100, rpm_step=25, rpmtype='rising', orderResolution=0.5, orderWidth=0.5, weighting=1):
    '''
    :param raw_time: 时间数据
    :param raw_data: 声压数据
    :param raw_rpm: 转速数据
    :param rpm_step: 转速步长
    :param rpmtype: 'rising'
    :param order: 需要提取的阶次
    :param orderResolution: 决定每个转速提取的blockSize长度，用于fft
    :param orderWidth: 每个转速对应的阶次频率不一定刚好为fft中对应的频率，故加入orderWidth在一定的误差范围内寻找阶次频率。
    :param smoothFrac: 局部加权回归中的参数值，smoothFrac：截取数据比例
    :param weighting: weighting=1时进行A计权

    :return: rpml: 从最低转速到最高转速的列表或从最高转速到最低转速的列表（步长为rpm_step）
             dbo：对应转速下阶次频率的分贝值
    '''

    rpmf, rpml = rpmSelect2(raw_rpm, rpm_step, rpmtype)
    order_db = []
    for i in np.arange(1, orderRange / orderResolution + 1):
        order = i * orderResolution

        # TODO: 并行计算
        max_rpm, dbo = order_vfft(rpmf, rpml, raw_time, raw_data, rpm_step, rpmtype, order, orderResolution, orderWidth, weighting)
        max_db = max(dbo)
        max_rpm = max_rpm[np.argmax(dbo)]
        # sum_db = fun_sum_db(dbo)
        order_db.append([order, max_db, max_rpm])
    order_db = np.array(order_db)

    return order_db


def Order_detect(order_db, window_size=8, overlap=50):
    """

    :param order_db: 阶次谱
    :param window_size: 窗包含的阶次线数
    :param overlap: 重叠率
    :return:     order_disatance:问题阶次及对应凸显度, order_important:前八各问题阶次及对应问题转速
    """
    # order_db = order_db[]
    order = order_db[:, 0]
    db = order_db[:, 1]
    rpm = order_db[:, 2]
    order_distance = []
    order_important = []

    stepping = np.floor(window_size * (100 - overlap) / 100)
    window_data = int((order_db.shape[0] - window_size) // stepping + 1)  # 计算窗口数量

    for i in range(window_data):
        x = db[int(i * stepping): int(i * stepping + window_size)]  # 获得每一窗口的db数据
        x_max = max(x)
        x_min = min(x)
        distance = x_max - x_min
        order_max = order[np.where(x == x_max)[0][0] + i * int(stepping)]
        rpm_max = rpm[np.where(x == x_max)[0][0] + i * int(stepping)]
        order_distance.append([order_max, rpm_max, distance])

    order_distance = np.array(order_distance)
    order_distance = order_distance[np.where(order_distance[:, 0] >= 10)]
    order_distance = order_distance[np.argsort(order_distance[:, 2])]

    for i in range(len(order_distance) - 1, -1, -1):
        tag = 0
        order_problem = order_distance[i, 0]
        rpm_problem = order_distance[i, 1]
        if len(order_important) == 0:
            order_important.append([order_problem, rpm_problem])
        else:
            for j in order_important:
                if order_problem >= j[0] - 1 and order_problem <= j[0] + 1:
                    tag = 1
            if tag == 0:
                order_important.append([order_problem, rpm_problem])

    # return order_distance, order_important, '问题最大的前3阶次，及对应问题转速为：\n', order_important[:3]
    return '问题最大的前3阶次，及对应问题转速为：\n', order_important[:3]


def figure_level_time(raw_time, raw_data, timeWeighting=0.25, A=1):
    lpa = level_time(raw_time, raw_data, timeWeighting, A)
    lpa = lpa[20: -20]
    raw_data = raw_data[20: -20]
    raw_time = raw_time[20: -20]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].plot(raw_time, raw_data)
    axes[0].grid('on')
    axes[0].set_xlabel('time/s')
    axes[0].set_ylabel('Pa')
    axes[0].set_title('Raw Signal')

    axes[1].plot(raw_time, lpa)
    axes[1].grid('on')
    axes[1].set_xlabel('time/s')
    axes[1].set_ylabel('dB')
    axes[1].set_title('Level vs. Time')
    fig.tight_layout()

    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "level_vs_time.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_fft_time(raw_time, raw_data):
    f, ta, pm = fft_time_1(raw_time, raw_data)
    # pm = pm.transpose()
    #     plt.figure(figsize=(12,12))
    x2, y2 = np.meshgrid(ta, f)
    levels = MaxNLocator(nbins=50).tick_values(pm.min(), pm.max())
    fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(x2, y2, pm, levels=levels)
    ax.set_yscale('log')
    ax.set_ylim(100, 10000)
    fig.colorbar(cf, ax=ax)
    ax.set_title('contourf for fft versus time')
    fig.tight_layout()
    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "fft_vs_time.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_fft_average(raw_time, raw_data, fs=44100, spectrum_size=16384, overlap=50, weighting=1):
    f, db = fft_average(raw_data, spectrum_size, fs, overlap, weighting)
    plt.figure(figsize=(8, 8))
    plt.plot(f, db)
    plt.xscale('log')
    plt.xlim(10, 20000)
    plt.xlabel('fs/Hz')
    plt.ylabel('dB(A)')
    plt.grid(b=bool, which='both')
    plt.title('fft average')
    plt.tight_layout()
    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "fft_vs_average.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_level_rpm(raw_time, raw_data, raw_rpm, rpm_step=25, rpmtype='rising', timeWeighting=0.25, A=1):
    rpml, lpr = level_rpm(raw_time, raw_data, raw_rpm)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].plot(raw_time, raw_data)
    axes[0].grid('on')
    axes[0].set_xlabel('time/s')
    axes[0].set_ylabel('Pa')
    axes[0].set_title('Raw Signal')

    axes[1].plot(rpml, lpr)
    axes[1].grid('on')
    axes[1].set_xlabel('rpm/s')
    axes[1].set_ylabel('dB')
    axes[1].set_title('Level vs. RPM')

    fig.tight_layout()

    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "level_vs_rpm.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_fft_rpm(raw_time, raw_data, raw_rpm, rpm_step=25, rpmtype='rising', spectrum_size=16384, weighting=1):
    #     fs = detectFs(raw_time)
    fs = 44100
    weighting = 1
    f, rpml, pm = fft_rpm(raw_time, raw_data, raw_rpm, rpm_step, rpmtype, spectrum_size, fs, weighting)
    # pm = pm.transpose()
    x2, y2 = np.meshgrid(rpml, f)
    levels = MaxNLocator(nbins=50).tick_values(pm.mean() - 30, pm.mean() + 30)
    #    levels = MaxNLocator(nbins=50).tick_values(10, 70)
    fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(x2, y2, pm, levels=levels)
    #     ax.set_yscale('log')
    ax.set_ylim(100, 10000)
    fig.colorbar(cf, ax=ax)
    ax.set_title('contourf for fft versus rpm')
    fig.tight_layout()

    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "fft_vs_rpm.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_Order_Spectrum(order_db):
    plt.figure(figsize=(8, 8))
    plt.plot(order_db[:, 0], order_db[:, 1])
    # plt.plot(output[:,0], output[:,2])

    plt.xlabel('order')
    plt.ylabel('db')
    plt.title('Order spectrum (peak hold)  (25.0rpm,0.0-100.0,r=0.5<0.5ord>,HAN)')
    plt.grid(b=bool, which='both')
    plt.tight_layout()

    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "order_spectrum.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def figure_Octave_Time(raw_time, raw_data, fs=44100, spectrum_size=16384, overlap=50, weighting=1):
    octave_db, time_array = Octave_Time(raw_time, raw_data,
                                        fs, spectrum_size,
                                        overlap, weighting)
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    figname = ['31.5Hz', '63Hz', '125Hz', '250Hz', '500Hz', '1000Hz']

    for i in range(2):
        for j in range(3):
            axes[i][j].plot(time_array, octave_db[i * 3 + j, :])
            axes[i][j].grid('on')
            axes[i][j].set_xlabel('time/s')
            axes[i][j].set_ylabel('dB')
            axes[i][j].set_title(figname[i * 3 + j])

    fig.tight_layout()

    absolute_dir = os.getcwd() + '/static/img/result/'
    # figurepath = 'f' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + str(np.random.randint(1000, 9999)) + ".png"
    figurepath = "octave_vs_time.png"
    plt.savefig(absolute_dir + figurepath)

    return '/static/img/result/' + figurepath


def compare(y1, model, fre_band, wav_standard):
    """

    :param y1:
    :param model:
    :param fre_band:
    :param wav_standard:
    :return:
    """
    sr = 44100
    model_predict = load_model(model)
    wav_time = np.floor(len(y1) / sr)
    y1 = y1[int(sr * (wav_time / 2 - 1)): int(sr * (wav_time / 2 + 1))]

    if model == model_path + 'Tickern.h5':
        f1, pm1 = fft_time(y1, sr, spectrum_size=1024, overlap=50)
    else:
        f1, pm1 = fft_time(y1, sr)

    f_select1 = np.where((f1 >= fre_band[0]) & (f1 < fre_band[1]))
    pm_select1 = pm1[f_select1[0]]

    sr, y2 = scipy.io.wavfile.read(wav_standard)
    if len(y2) > sr * 2:
        y2 = y2[0:sr * 2]

    y2 = y2 / y2.max()  # 试验加入归一化

    if model == model_path + 'Tickern.h5':
        f2, pm2 = fft_time(y2, sr, spectrum_size=1024, overlap=50)
    else:
        f2, pm2 = fft_time(y2, sr)

    f_select2 = np.where((f2 >= fre_band[0]) & (f2 < fre_band[1]))
    pm_select2 = pm2[f_select2[0]]

    mfcc_distance = abs(pm_select1 - pm_select2).reshape(1, pm_select1.shape[0], pm_select1.shape[1])
    mfcc_distance = np.swapaxes(mfcc_distance, axis1=1, axis2=2)

    similarity = model_predict.predict(mfcc_distance)
    similarity = similarity[0][1] * 100

    if int(similarity) < 1:
        return 0
    return "%.2f" % similarity


# 怠速音频
def lazy_run(file_object):
    # Step1:波形输出
    y, t_serie = read_wav(file_object)

    # Step2:绘制原始波形与level vs. time
    level_vs_time = figure_level_time(t_serie, y)

    # Step3:输出FFT(average)
    fft_vs_average = figure_fft_average(t_serie, y)

    # Step4:输出Octave VS. Time
    octave_vs_time = figure_Octave_Time(t_serie, y)

    # Step5:输出colormap
    fft_vs_time = figure_fft_time(t_serie, y)

    # 计算各类别噪声概率]
    noise_category = [
        ('Mahlendes.h5', (500, 750), 'Mahlendes.wav'),
        ('300Hz.h5', (250, 450), '300Hz.wav'),
        ('Tockern.h5', (500, 750), 'Tockern.wav'),
        ('Tickern.h5', (1000, 8000), 'Tickern.wav'),
        ('Heulen.h5', (550, 850), 'Heulen.wav')
    ]
    key = []
    value = []
    for i in range(len(noise_category)):
        model = model_path + noise_category[i][0]
        fre_band = noise_category[i][1]
        wav_standard = audio_path + noise_category[i][2]  # 文件路径
        similarity = compare(y, model, fre_band, wav_standard)  # 显示该类别噪声概率
        key.append("{}".format(noise_category[i][0]))
        value.append("{}".format(similarity))
    return {
        "level_vs_time": level_vs_time,
        "fft_vs_average": fft_vs_average,
        "octave_vs_time": octave_vs_time,
        "fft_vs_time": fft_vs_time,
        "key_": key,
        "value_": value
    }


# 加速音频
def speed_run(file_object):
    # Step1:波形输出
    datam = readAscFile(file_object)
    t_serie = datam[:, 0]
    y = datam[:, 1]
    rpm_serie = datam[:, 2]

    # Step2:绘制原始波形与level vs. time
    level_vs_rpm = figure_level_rpm(t_serie, y, rpm_serie)

    # Step3:输出FFT(average)
    fft_vs_average = figure_fft_average(t_serie, y)

    # Step4:输出阶次谱
    order_db = Order_Spectrum(rpm_serie, t_serie, y, orderRange=100, rpm_step=25, rpmtype='rising', orderResolution=0.5, orderWidth=0.5, weighting=1)
    order_spectrum = figure_Order_Spectrum(order_db)
    info = Order_detect(order_db)
    # Step5:输出colormap
    fft_vs_rpm = figure_fft_rpm(t_serie, y, rpm_serie)

    return {
        "level_vs_time": level_vs_rpm,
        "fft_vs_average": fft_vs_average,
        "octave_vs_time": order_spectrum,
        "fft_vs_time": fft_vs_rpm,
        # "order_distance": order_distance.tolist(),
        # "order_important": order_important,
        "info_data": info
    }
