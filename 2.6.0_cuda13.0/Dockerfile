ARG PYTORCH="2.14.1"
ARG CUDA="13.0"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1 libglx-mesa0 python3-dev python3-pip python3-venv git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/PaddleClas.venv && \
    /opt/PaddleClas.venv/bin/python -m ensurepip --upgrade && \
    /opt/PaddleClas.venv/bin/python -m pip install --upgrade setuptools

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN /opt/PaddleClas.venv/bin/pip install -U --no-cache-dir \
        "paddlepaddle-gpu==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

RUN /opt/PaddleClas.venv/bin/pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

WORKDIR /opt

RUN git clone https://github.com/PaddlePaddle/PaddleClas.git && \
    cd PaddleClas && \
    git reset --hard 202ac79a99aa368f2c95cb1871bbc11574f685da
# we need to fix the numpy version to make it work for Python 3.12
COPY requirements_312.txt /opt/PaddleClas/requirements.txt
RUN cd /opt/PaddleClas && \
    /opt/PaddleClas.venv/bin/pip install --no-cache-dir -v -e . && \
    /opt/PaddleClas.venv/bin/pip install --no-cache-dir -U "numpy<2.0.0" "scipy==1.15.3" && \
    /opt/PaddleClas.venv/bin/pip install --no-cache-dir -U https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleClas/tools/
COPY custom_engine.py /opt/PaddleClas/ppcls/engine/
COPY predict*.py /opt/PaddleClas/tools/
COPY paddleclas_* /usr/bin/
RUN touch /opt/PaddleClas/__init__.py

ENV PYTHONPATH=/opt/PaddleClas

