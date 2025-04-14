FROM python:3.10.12
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1

# FROM ubuntu:22.04

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-tk \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]

