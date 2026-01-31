# https://github.com/Kaggle/docker-python/releases
FROM gcr.io/kaggle-gpu-images/python:v163

RUN pip install --no-cache-dir \
    hydra-core schedulefree colorednoise
    
