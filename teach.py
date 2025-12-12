# =====================================================================
# 1. Встановлення бібліотек
# =====================================================================
#pip install ultralytics roboflow

# =====================================================================
# 2. Імпорт
# =====================================================================
# from roboflow import Roboflow
from ultralytics import YOLO
import torch
import platform

# =====================================================================
# 3. Перевірка GPU, Визначення пристрою
# =====================================================================
#print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Немає GPU")
if torch.cuda.is_available():
    device = "cuda"  # Nvidia GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon GPU (M1/M2/M3)
else:
    device = "cpu"   # CPU fallback

print(f"Platform: {platform.system()} ({platform.machine()})")
print(f"Using device: {device}")

# # Шлях до YAML (потрібен YOLO для тренування)
data_yaml = "./BALLS/data.yaml"
print("DATA YAML:", data_yaml)

# =====================================================================
# 5. Завантаження моделі YOLOv8
# =====================================================================
model = YOLO("yolov8n.pt")   # маленька модель, швидко навчається

# =====================================================================
# 6. Запуск тренування
# =====================================================================
results = model.train(
    data=data_yaml, 
    epochs=20,
    imgsz=640,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=2,
)

# =====================================================================
# 7. Перевірка результатів
# =====================================================================
print("Training completed. Best weights saved here:")
print(results.save_dir)

