---
layout: post
title: 'Running a docker with GPU enabled (for pytorch and tensorflow)'
author: nicolas
tags: [ml, docker]
---

Sometimes if you want to contain dependencies you might want to use docker
to containerize your projects. You can also use it for GPU
In order to run docker images with GPU enabled, you are going to need:

# Install docker

```
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

```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

[source](https://github.com/NVIDIA/nvidia-docker)

# Launch the docker for PyTorch

In order to use cuda you need a nvidia enabled image, that will make everything simpler.
You could of course link your own cuda library via volume mounting but it's cumbersome (and I didn't check that it works)

1. Create an account on [https://ngc.nvidia.com/](https://ngc.nvidia.com/)
2. Go to the create an API key page [https://ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key)
3. Generate the key and copy it

```
docker login nvcr.io
Username: $oauthtoken
Password: <Your Key>
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:20.02-py3 bash
python -c "import torch; print(torch.cuda.is_available())"
# True
```

If you fail to login the `docker run` command will fail with `unauthenticated` error.

Caveat: This is the only option for now, docker-compose _CANNOT_ run the --gpu option.
To check updates for docker compose, look at this [issue](https://github.com/docker/compose/issues/6691)

Bonus: Nvidia put up _a lot_ of containers with various libraries enabled check it out in their [catalog](https://ngc.nvidia.com/catalog/)

## Enjoy !
