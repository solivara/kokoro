FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y wget espeak-ng python3.10 python3-pip

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install kokoro misaki[zh]  soundfile flask flask_cors --no-cache-dir

ENV HF_ENDPOINT=https://hf-mirror.com

RUN huggingface-cli download hexgrad/Kokoro-82M-v1.1-zh

WORKDIR /workspace

COPY api.py /workspace

CMD ["python3", "api.py"]