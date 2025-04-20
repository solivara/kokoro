FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

COPY api.py /workspace

COPY whl/ /workspace/whl/

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y wget espeak-ng python3.10 python3-pip

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install kokoro misaki[zh]  soundfile flask flask_cors --no-cache-dir

RUN pip3 install /workspace/whl/en_core_web_sm-3.8.0-py3-none-any.whl

ENV HF_ENDPOINT=https://hf-mirror.com

RUN huggingface-cli download hexgrad/Kokoro-82M-v1.1-zh

EXPOSE 50032

CMD ["python3", "api.py"]