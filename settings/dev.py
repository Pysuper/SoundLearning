# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author   : Zheng Xingtao
# File     : dev.py
# Datetime : 2020/11/16 下午12:47

import os
import sys

from utils import theme

############################################################  SQL配置  ############################################################
DOCKER_SERVER = "192.168.50.230"
REDIS_PORT = 16379
MYSQL_PORT = 13306

############################################################ Django默认配置 ############################################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'apps'))

SECRET_KEY = '-*d9m2^xe-p)2kbmwr@bb13n33m9mfr@h#b0krb2v838n$ra-+'
DEBUG = True
ALLOWED_HOSTS = ['*', ]
# DEBUG = False

# 中间件自定义白名单
WHITE_REGEX_URL_LIST = ["/", ]

# 子应用
INSTALLED_APPS = [
    'simpleui',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    # 'dwebsocket',
    'sounds.apps.SoundsConfig',
]

# 中间件
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

# 模板文件
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join('templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# 配置数据库
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'HOST': DOCKER_SERVER,
        'PORT': MYSQL_PORT,
        'USER': 'root',
        'PASSWORD': 'root',
        'NAME': 'Sound',
        'OPTIONS': {
            'read_default_file': os.path.dirname(os.path.abspath(__file__)) + '/my.cnf',
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES,"
                            "NO_ZERO_IN_DATE,NO_ZERO_DATE,"
                            "ERROR_FOR_DIVISION_BY_ZERO,"
                            "NO_AUTO_CREATE_USER'",
        },
    }
}

# 日志文件
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(lineno)d %(message)s'
        },
        'simple': {
            'format': '[%(levelname)s] %(message)s'
        },
    },
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs/sound.log"),
            'maxBytes': 300 * 1024 * 1024,
            'backupCount': 10,
            'formatter': 'verbose'
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'propagate': True,
            'level': 'INFO',
        },
    }
}

# 配置缓存
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://{}:{}/0".format(DOCKER_SERVER, REDIS_PORT),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
    "session": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://{}:{}/1".format(DOCKER_SERVER, REDIS_PORT),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
}

# 密码验证
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', },
]

SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "session"

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_L10N = True
USE_TZ = False
ROOT_URLCONF = 'sound.urls'
WSGI_APPLICATION = 'sound.wsgi.application'

# DRF配置
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': ('rest_framework_jwt.authentication.JSONWebTokenAuthentication',),
    'DEFAULT_PAGINATION_CLASS': 'scripts.pagination.StandardResultsSetPagination',
}

# CORS允许访问的域名
CORS_ORIGIN_WHITELIST = (
    'https://127.0.0.1:5666',
    'https://localhost:5666',
    'http://localhost:5666',
    'http://192.168.43.230:5666',
    'http://192.168.50.230:5666',
)
CORS_ALLOW_CREDENTIALS = True

# 静态文件目录
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'), ]

########################################################## simpleui 使用 ##########################################################
SIMPLEUI_HOME_INFO = False  # 服务器信息
SIMPLEUI_ANALYSIS = False  # 不收集分析信息
SIMPLEUI_STATIC_OFFLINE = True  # 离线模式

##########################################################  配置长连接  ##########################################################
# WEBSOCKET_ACCEPT_ALL = True


##########################################################  文件路径配置  ##########################################################
audio_path = os.path.join(BASE_DIR, 'apps/sounds/audio/')
model_path = os.path.join(BASE_DIR, 'apps/sounds/model_/')
wav_path = os.path.join(BASE_DIR, 'apps/sounds/wav_file/')

