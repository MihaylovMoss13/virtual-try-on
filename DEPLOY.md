# Руководство по развертыванию на GPU сервере

Чтобы исправить ошибку `could not select device driver "nvidia"`, необходимо установить драйверы NVIDIA и NVIDIA Container Toolkit на ваш сервер.

Это руководство рассчитано на **Ubuntu 20.04/22.04 LTS**.

## 1. Установка драйверов NVIDIA

Сначала убедитесь, что ваш GPU обнаружен и драйверы установлены.

```bash
# Проверка наличия GPU
lspci | grep -i nvidia
# или
lshw -C display

# Добавление репозитория драйверов
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Установка рекомендуемого драйвера (например, 535)
sudo apt install nvidia-driver-535

# Перезагрузка
sudo reboot
```

Проверьте установку командой `nvidia-smi`. Вы должны увидеть список ваших видеокарт.

## 2. Установка Docker

Если Docker еще не установлен:

```bash
# Добавление официального GPG ключа Docker:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Добавление репозитория в источники Apt:
echo \
  "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  \"$(. /etc/os-release && echo "$VERSION_CODENAME")\" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Установка пакетов Docker:
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 3. Установка NVIDIA Container Toolkit ⚠️ (КРИТИЧЕСКИЙ ШАГ)

Этот инструмент связывает Docker с вашим GPU. Без него Docker не увидит видеокарту, даже если драйверы установлены.

```bash
# Настройка репозитория:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновление списка пакетов:
sudo apt-get update

# Установка инструментария:
sudo apt-get install -y nvidia-container-toolkit

# Настройка демона Docker для использования NVIDIA runtime:
sudo nvidia-ctk runtime configure --runtime=docker

# Перезапуск Docker для применения изменений:
sudo systemctl restart docker
```

## 4. Проверка доступа к GPU в Docker

Запустите тестовый контейнер, чтобы подтвердить доступ к GPU:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Если эта команда покажет вывод `nvidia-smi` внутри контейнера — вы готовы к развертыванию!

## 5. Развертывание FASHN VTON

Теперь вы можете собрать и запустить проект:

```bash
docker compose up --build -d
```
