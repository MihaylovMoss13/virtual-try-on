# Тестирование развертывания FASHN VTON

Следуйте этим шагам, чтобы проверить установку на GPU-сервере.

## 1. Запуск контейнера

Убедитесь, что Docker-контейнер собран и запущен:

```bash
docker compose up -d --build
```

Проверьте логи, чтобы убедиться в отсутствии ошибок при запуске:

```bash
docker compose logs -f
```

## 2. Скачивание весов модели (Только один раз)

Перед запуском инференса необходимо скачать веса модели (~2GB) в примонтированный том.

```bash
docker compose run --rm fashn-vton python scripts/download_weights.py --weights-dir /app/weights
```

*Примечание: Веса сохраняются в папку `./public/weights` на вашем хост-сервере, поэтому они сохраняются между перезапусками.*

## 3. Запуск тестового инференса (Встроенные изображения)

Запустите скрипт инференса, используя примеры изображений, включенные в контейнер:

```bash
docker compose run --rm fashn-vton python examples/basic_inference.py \
    --weights-dir /app/weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops \
    --output-dir /app/outputs
```

## 4. Тестирование с вашими изображениями

1. Поместите ваши изображения в директорию `public/inputs` на вашем сервере.
   - Пример: `public/inputs/me.jpg` и `public/inputs/tshirt.jpg`

2. Запустите инференс, указывая пути к этим файлам (обратите внимание на путь `/app/inputs/` внутри контейнера):

```bash
docker compose run --rm fashn-vton python examples/basic_inference.py \
    --weights-dir /app/weights \
    --person-image /app/inputs/me.jpg \
    --garment-image /app/inputs/tshirt.jpg \
    --category tops \
    --output-dir /app/outputs
```

## Устранение неполадок с GPU

Если инференс идет медленно (~минуты) или падает с ошибкой, проверьте, видит ли контейнер GPU:

```bash
docker compose run --rm fashn-vton nvidia-smi
```

- **Успех**: Вы видите таблицу с GPU (например, Tesla T4, RTX 3090 и т.д.).
- **Ошибка**: "command not found" или "no devices found" -> Проверьте шаги в `DEPLOY.md` по установке NVIDIA Container Toolkit.
