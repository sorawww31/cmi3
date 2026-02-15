# https://github.com/Kaggle/docker-python/releases
FROM gcr.io/kaggle-gpu-images/python:latest

RUN pip install --no-cache-dir \
    hydra-core schedulefree colorednoise
    
