# pull the nvidia image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# prohibit interaction during this build
ARG DEBIAN_FRONTEND=noninteractive

# replace apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-dev

# install python libs
RUN pip install opencv-python timm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install cython -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install basicsr -i https://pypi.tuna.tsinghua.edu.cn/simple