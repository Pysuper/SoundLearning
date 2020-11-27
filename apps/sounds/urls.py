# !/usr/bin/env
# python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : urls.py
# Datetime : 2020/11/16 下午1:00


from django.conf.urls import url

from sounds.views import sound

urlpatterns = [
    url(r'', sound),
]
