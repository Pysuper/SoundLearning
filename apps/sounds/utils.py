# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : utils.py
# Datetime : 2020/11/17 上午9:29

import librosa
import numpy as np
import scipy.io.wavfile
from keras.models import load_model
from numpy.fft._pocketfft import fft
from scipy.signal.windows import hann


def read_wav(wav_test):
    """
    :param wav_test: 待测音频文件, 路径
    :return: 归一化后音频数据
    """
    sr, y = scipy.io.wavfile.read(wav_test)
    y = y.astype(np.float64)

    if len(y) > sr * 2:
        raise Exception("WAV Time is too short!")
    y = y / y.max()  # 加入归一化
    return y


def fft_time(raw_data, fs=44100, spectrum_size=8192, overlap=75):
    """
    :param raw_data: 声压数据
    :param fs:
    :param spectrum_size: 谱线数
    :param overlap: 重叠率
    :return: f: 每条谱线对应的频率
             time_array: fft vs time中的时间轴序列
             pm: 每一时间每一频率下对应的fft幅值(rms)
    """
    stepping = np.floor(spectrum_size * (100 - overlap) / 100)
    window_data = int((len(raw_data) - spectrum_size) // stepping + 1)  # 计算窗口数量
    pp = np.zeros((spectrum_size // 2 + 1, window_data))  # 预分配fft_vs_time 矩阵

    f = fs * np.arange(spectrum_size // 2 + 1) / spectrum_size  # TODO： Change
    pm = ""

    # 计算每一time_array时间对应的FFT值
    for i in range(window_data):
        x = raw_data[int(i * stepping):int(i * stepping + spectrum_size)]  # 获得每一窗口的raw_data数据
        y = fft(x * hann(spectrum_size))
        p = np.abs(y / spectrum_size)
        p1 = p[:spectrum_size // 2 + 1]
        p1[1: -1] *= 2
        p2 = 2 * p1 * 2 ** -0.5  # peak to rms
        pp[0:, i] = p2  # 生成所有分块数据的FFT矩阵

        pm = 20 * np.log10(pp / 2e-5)
        pm[:, i] = pm[:, i] + librosa.A_weighting(f)

    return f, pm


def compare(y1, model, fre_band, wav_standard):
    """
    :param y1: 待测音频数据(y)
    :param model: 训练好的模型(h5文件), 路径
    :param fre_band: 提取的频带范围, 自定义的元祖
    :param wav_standard: 标准音频, 路径
    :return: 相似度
    """
    sr = 44100
    model_predict = load_model(model)
    wav_time = np.floor(len(y1) / sr)
    y1 = y1[int(sr * (wav_time / 2 - 1)): int(sr * (wav_time / 2 + 1))]

    f1, pm1 = fft_time(y1, fs=sr)

    f_select1 = np.where((f1 >= fre_band[0]) & (f1 < fre_band[1]))
    pm_select1 = pm1[f_select1[0]]

    sr, y2 = scipy.io.wavfile.read(wav_standard)
    if len(y2) > sr * 2:
        y2 = y2[0:sr * 2]

    y2 = y2 / y2.max()  # 试验加入归一化
    f2, pm2 = fft_time(y2, sr)
    f_select2 = np.where((f2 >= fre_band[0]) & (f2 < fre_band[1]))
    pm_select2 = pm2[f_select2[0]]

    mfcc_distance = abs(pm_select1 - pm_select2).reshape(1, pm_select1.shape[0], pm_select1.shape[1])
    mfcc_distance = np.swapaxes(mfcc_distance, axis1=1, axis2=2)

    similarity = model_predict.predict(mfcc_distance)
    similarity = similarity[0][1] * 100

    return similarity


def run(wav_file, model_file, sr, fre_band, wav_standard):
    wav_result = read_wav(wav_file)  # Step1:波形输出
    f, pm = fft_time(wav_result, sr)  # Step2:输出colormap
    similarity = compare(wav_result, model_file, fre_band, wav_standard)  # 显示该类别噪声概率
    return {
        "wav_result": np.ndarray.tolist(wav_result),
        "colormap": (np.ndarray.tolist(f), np.ndarray.tolist(pm)),
        "similarity": '{}概率：{}%'.format(model_file[:-3], similarity)
    }
