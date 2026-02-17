# GPU Server Deployment Guide

To fix the error `could not select device driver "nvidia"`, you must install the NVIDIA drivers and the NVIDIA Container Toolkit on your server.

This guide assumes you are using **Ubuntu 20.04/22.04 LTS**.

## 1. Install NVIDIA Drivers

First, ensure your GPU is detected and drivers are installed.

```bash
# Check for GPU
lspci | grep -i nvidia or lshw -C display

# Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver (e.g., 535)
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

Verify installation with `nvidia-smi`. You should see your GPU listed.

## 2. Install Docker

If Docker is not installed:

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  \"$(. /etc/os-release && echo "$VERSION_CODENAME")\" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker packages:
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 3. Install NVIDIA Container Toolkit ⚠️ (CRITICAL STEP)

This connects Docker to your GPU. Without this, Docker cannot see the GPU even if drivers are installed.

```bash
# Configure the production repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package listing:
sudo apt-get update

# Install the toolkit:
sudo apt-get install -y nvidia-container-toolkit

# Configure the Docker daemon to recognize the NVIDIA runtime:
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes:
sudo systemctl restart docker
```

## 4. Verify GPU Access in Docker

Run a test container to confirm GPU access:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If this command shows the `nvidia-smi` output inside the container, you are ready to deploy!

## 5. Deploy FASHN VTON

Now you can build and run your project:

```bash
docker compose up --build -d
```
