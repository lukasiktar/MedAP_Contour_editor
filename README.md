# MedAP Contour Editor

Annotation editor.

---

## Requirements

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- X11 (on Linux) to forward GUI from container
- Download 'networks' folder (run setup.sh script)
- Download model from [link](https://www.kaggle.com/models/lukaiktar/mucsnet_prostate) and name the file 'MUCSNet.pth'

---

## Build Docker Image

```bash
docker build -t docker-image:tag .

xhost +local:docker

docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --gpus all \
  docker-image:tag
```
