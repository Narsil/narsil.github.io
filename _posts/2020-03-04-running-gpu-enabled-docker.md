---
layout: post
title: 'Running a docker with GPU enabled (for pytorch and tensorflow)'
author: nicolas
categories: [ml, docker]
---

Sometimes if you want to contain dependencies you might want to use docker
to containerize your projects. You can also use it for GPU
In order to run docker images with GPU enabled, you are going to need:

# Install docker

```bash
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

[source](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

# Install nvidia-container-toolkit

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

[source](https://github.com/NVIDIA/nvidia-docker)

# Launch your docker

``bash
docker run python:3-slim --gpu all -it ipython

```

Caveat: This is the only option for now, docker-compose _CANNOT_ run the --gpu option.
To check updates, look at this [issue](https://github.com/docker/compose/issues/6691)
```
