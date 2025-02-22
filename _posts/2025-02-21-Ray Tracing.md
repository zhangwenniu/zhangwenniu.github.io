---
layout: mypost
title: Ray Tracing, Intersection of a Ray and a Mesh
categories: [Ray Tracing]
---


# 相比于NeuS额外安装的库

- optix
- cuda-python
- cupy

optix安装：

```bash
# 下载optix 7.6
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# https://developer.nvidia.com/optix/downloads/7.6.0/linux64-x86_64




git clone https://github.com/78ij/python-optix
cd python-optix
export OPTIX_PATH=/home/zhangwenniu/github/optix/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64
export CUDA_PATH=/usr/local/cuda
export OPTIX_EMBED_HEADERS=1
pip install .
```

cuda-python安装：

```bash
pip install cuda-python
```

cupy安装：

```bash
pip install cupy
```

