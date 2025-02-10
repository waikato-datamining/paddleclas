ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.11.0+cpu torchvision==0.12.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /opt

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN pip install -U --no-cache-dir \
        "paddlepaddle==2.6.2" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

RUN git clone https://github.com/PaddlePaddle/PaddleClas.git && \
    cd PaddleClas && \
    git reset --hard 1cde9bce0c1c0ed8dce87d11fcb039c6e3502d03 && \
    pip install --no-cache-dir -v -e .

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY export_config.py /opt/PaddleClas/tools/
COPY predict*.py /opt/PaddleClas/tools/
COPY paddleclas_* /usr/bin/

ENV PYTHONPATH=/opt/PaddleClas
