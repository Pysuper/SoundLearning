# 编写.dockerignore文件
# 容器只运行单个应用
# 将多个RUN指令合并为一个
# 基础镜像的标签不要用latest
# 每个RUN指令后删除多余文件
# 选择合适的基础镜像(alpine版本最好)
# 设置WORKDIR和CMD
# 使用ENTRYPOINT (可选)
# 在entrypoint脚本中使用exec
# COPY与ADD优先使用前者
# 合理调整COPY与RUN的顺序
# 设置默认的环境变量，映射端口和数据卷
# 使用LABEL设置镜像元数据
# 添加HEALTHCHECK
FROM kpavlovsky/python3.7:latest

# 创建文件夹
RUN mkdir /root/config /root/static /root/pligins && apt-get update && apt-get upgrade -y && apt-get install -y nginx
COPY ./SoundLearning/utils/nginx.conf /etc/nginx/nginx.conf
COPY ./SoundLearning/utils/sound.conf /etc/nginx/sites-available/sound.conf
COPY ./SoundLearning/utils/tensorflow-2.3.1-cp37-cp37m-manylinux2010_x86_64.whl /root/pligins/tensorflow-2.3.1-cp37-cp37m-manylinux2010_x86_64.whl
RUN ln -s /etc/nginx/sites-available/sound.conf /etc/nginx/sites-enabled/sound.conf && groupadd -r nginx && useradd -r -g nginx nginx && touch /run/nginx.pid && nginx -c /etc/nginx/nginx.conf && nginx -s reload && service nginx restart && pip3 install /root/pligins/tensorflow-2.3.1-cp37-cp37m-manylinux2010_x86_64.whl && apt-get update && apt-get install libsndfile1 && apt-get install -y nginx && apt-get install -y uwsgi && apt-get install -y vim && pip3 install django_redis && pip3 install django && pip3 install uwsgi && pip3 install pymysql && pip3 install librosa && pip3 install keras && pip3 install scipy numpy && pip3 install django-simpleui && pip install django-cors-headers && pip install django-restframework

# 配置
EXPOSE 80
WORKDIR /root/home/SoundLearning
# ENTRYPOINT "" 用来执行脚本
CMD uwsgi --ini sound.ini

# 运行
RUN cd /root/home/SoundLearning/ && uwsgi --ini sound.ini
