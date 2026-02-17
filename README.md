# FASHN VTON v1.5: Эффективная виртуальная примерка без масок в пиксельном пространстве

<div align="center">
  <a href="https://fashn.ai/research/vton-1-5"><img src='https://img.shields.io/badge/Project-Page-1A1A1A?style=flat' alt='Project Page'></a>&ensp;
  <a href='https://huggingface.co/fashn-ai/fashn-vton-1.5'><img src='https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=flat&logo=HuggingFace&logoColor=FFD21E' alt='Hugging Face Model'></a>&ensp;
  <a href="https://huggingface.co/spaces/fashn-ai/fashn-vton-1.5"><img src='https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat&logo=HuggingFace&logoColor=FFD21E' alt='Hugging Face Spaces'></a>&ensp;
  <a href=""><img src='https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=flat&logo=arXiv&logoColor=b31b1b' alt='arXiv'></a>&ensp;
  <a href="LICENSE"><img src='https://img.shields.io/badge/License-Apache--2.0-gray?style=flat' alt='License'></a>
</div>

от [FASHN AI](https://fashn.ai)

Модель виртуальной примерки, которая генерирует фотореалистичные изображения непосредственно в пиксельном пространстве, не требуя масок сегментации.

<p align="center">
  <img src="https://static.fashn.ai/repositories/fashn-vton-v15/results/hero_collage.webp" alt="FASHN VTON v1.5 примеры" width="900">
</p>

Этот репозиторий содержит минимальный код для запуска инференса виртуальной примерки с весами модели FASHN VTON v1.5. На вход подается изображение человека и изображение одежды, модель генерирует фотореалистичное изображение человека в этой одежде. Поддерживаются как фотографии на моделях, так и фото одежды в раскладке (flat-lay).

---

## Локальная установка

Мы рекомендуем использовать виртуальное окружение:

```bash
git clone https://github.com/fashn-AI/fashn-vton-1.5.git
cd fashn-vton-1.5
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Примечание:** Установка включает `onnxruntime-gpu` для определения позы с ускорением на GPU. Убедитесь, что CUDA правильно настроена в вашей системе. Для окружений только с CPU (например, macOS), замените на версию для CPU:

```bash
pip uninstall onnxruntime-gpu && pip install onnxruntime
```

---

## Развертывание через Docker (Рекомендуется)

Для быстрого запуска на сервере с GPU:

1. Настройте сервер по инструкции в [DEPLOY.md](DEPLOY.md).
2. Соберите и запустите контейнер:
   ```bash
   docker compose up -d --build
   ```
3. Протестируйте работу по инструкции в [TESTING.md](TESTING.md).

---

## Веса модели

Скачайте необходимые веса модели (~2 GB всего):

```bash
python scripts/download_weights.py --weights-dir ./weights
```

Это загрузит:
- `model.safetensors` — веса TryOnModel с [HuggingFace](https://huggingface.co/fashn-ai/fashn-vton-1.5)
- `dwpose/` — ONNX модели DWPose для определения позы

**Примечание:** Веса парсера человека (~244 MB) загружаются автоматически при первом использовании в кэш HuggingFace. Установите переменную `HF_HOME`, чтобы изменить местоположение.

---

## Использование

```python
from fashn_vton import TryOnPipeline
from PIL import Image

# Инициализация пайплайна (автоматически использует GPU, если доступен)
pipeline = TryOnPipeline(weights_dir="./weights")

# Загрузка изображений
person = Image.open("examples/data/model.webp").convert("RGB")
garment = Image.open("examples/data/garment.webp").convert("RGB")

# Запуск инференса
result = pipeline(
    person_image=person,
    garment_image=garment,
    category="tops",  # "tops" (верх) | "bottoms" (низ) | "one-pieces" (платья/комбинезоны)
)

# Сохранение результата
result.images[0].save("output.png")
```

### CLI (Командная строка)

```bash
python examples/basic_inference.py \
    --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops
```

**Примечание:** Пайплайн автоматически использует GPU, если оно доступно. Веса модели примерки хранятся в bfloat16 и будут работать с точностью bf16 на GPU Ampere+ (RTX 30xx/40xx, A100, H100). На более старом оборудовании или CPU веса конвертируются в float32.

Смотрите [`examples/basic_inference.py`](examples/basic_inference.py) для дополнительных опций.

---

## Категории

| Категория | Описание |
|----------|-------------|
| `tops` | Верхняя одежда: футболки, блузки, куртки |
| `bottoms` | Нижняя одежда: брюки, юбки, шорты |
| `one-pieces` | Одежда в полный рост: платья, комбинезоны |

---

## API

FASHN предоставляет набор [API для Fashion AI](https://fashn.ai/products/api), включая виртуальную примерку, генерацию моделей, image-to-video и многое другое. Смотрите [документацию](https://docs.fashn.ai/), чтобы начать.

---

## Цитирование

Если вы используете FASHN VTON v1.5 в своих исследованиях, пожалуйста, цитируйте:

```bibtex
@article{bochman2026fashnvton,
  title={FASHN VTON v1.5: Efficient Maskless Virtual Try-On in Pixel Space},
  author={Bochman, Dan and Bochman, Aya},
  journal={arXiv preprint},
  year={2026},
  note={Paper coming soon}
}
```

---

## Лицензия

Apache-2.0. Смотрите [LICENSE](LICENSE) для подробностей.

**Сторонние компоненты:**
- [DWPose](https://github.com/IDEA-Research/DWPose) (Apache-2.0)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0)
- [fashn-human-parser](https://github.com/fashn-AI/fashn-human-parser) ([License](https://github.com/fashn-AI/fashn-human-parser?tab=readme-ov-file#license))
