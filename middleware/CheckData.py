# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : CheckData.py
# Datetime : 2020/11/16 下午1:57


from django.http import HttpResponse
from django.shortcuts import redirect
from django.utils.deprecation import MiddlewareMixin


class CheckDataMiddleware(MiddlewareMixin):
    @staticmethod  # 在视图之前执行
    def process_request(request):
        print("process_request_______")
        return redirect('/')

    @staticmethod  # 基于请求响应
    def process_response(request, response):
        print("md1  process_response 方法！", id(request))
        return response

    @staticmethod  # 在视图之前执行 顺序执行
    def process_view(request, view, args, kwargs):
        # 通过了就直接return，不通过就用redirect做跳转
        print("md1  process_view 方法！")
        return

    @staticmethod  # 引发错误 才会触发这个方法
    def process_exception(request, exception):
        return HttpResponse(exception)  # 返回错误信息
