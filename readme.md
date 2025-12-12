# README.md

# YOLOv8 Setup Guide

Цей файл описує, як підготувати Python віртуальне середовище та встановити всі необхідні залежності для тренування YOLOv8 на локальному датасеті.

---

## 1. Створення віртуального середовища (venv)

1. **Перевірка версії Python**  
   Рекомендовано Python 3.10–3.12:
   ```bash
   python3 --version
   ```

2. **Створення venv**
   ```bash
   python3 -m venv .venv
   ```
   - `.venv` — папка для віртуального середовища (можна змінити на будь-яку назву).

3. **Активація venv**
   - macOS / Linux:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (CMD):
     ```cmd
     .venv\Scripts\activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

4. **Перевірка активації**
   ```bash
   which python   # macOS/Linux
   where python   # Windows
   ```
   Повинен показувати Python всередині `.venv`.

5. **Оновлення pip**
   ```bash
   pip install --upgrade pip
   ```

---

## 2. Встановлення необхідних бібліотек

1. Створіть файл `requirements.txt` з наступним вмістом:

```
ultralytics>=8.2.0
numpy==1.26.4
opencv-python
matplotlib
pillow
pyyaml
tqdm
requests
torch
torchvision
```

2. Встановіть всі залежності:
```bash
pip install -r requirements.txt
```

---

## 3. Перевірка встановлення

```bash
python -c "import torch; import ultralytics; import numpy; print('OK')"
```

Якщо виводить `OK` — середовище готове.

---

## 4. Використання пристрою GPU/CPU

Код автоматично вибирає доступний пристрій:

```python
import torch

if torch.cuda.is_available():
    device = "cuda"  # Nvidia GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon GPU (M1/M2/M3)
else:
    device = "cpu"   # CPU fallback

print(f"Using device: {device}")
```

---

## 5. Додаткові поради

- Використовуйте локальні датасети, щоб не залежати від Roboflow API.
- Для тренування YOLOv8 можна використовувати `yolov8n.pt` як pretrained модель або `yolov8n.yaml` для тренування з нуля.
- На Mac Intel модель тренуватиметься на CPU.



## 6.Квантизація

Hailo підтримує:

- PTQ (post-training quantization)

- QAT (quantization aware training)

- Найпростіший варіант — PTQ у Hailo Model Zoo:

```
hailomz quantize \
    --ckpt model.onnx \
    --calib-dataset <path to images> \
    --output model_quantized.onnx
```

## 7.Компіляція у HEF (Hailo Executable File)

HEF — це файл, який виконується на Hailo-8.


```
hailo_compiler \
    model_quantized.onnx \
    --hw-arch hailo8 \
    --output-file model.hef
```

## 8.Запуск inference на Hailo-8

Python runtime:

```
from hailo_platform import HEF, VDevice, HailoStream

hef = HEF("model.hef")
vdev = VDevice()
network_groups = hef.configure(vdev)
input_stream = network_groups.input_streams[0]
output_stream = network_groups.output_streams[0]

with HailoStream(input_stream, output_stream) as stream:
    result = stream.infer(my_input_np_array)
    print(result)
```

CLI:
```
hailortcli run --hef model.hef --input my_input.npy
```