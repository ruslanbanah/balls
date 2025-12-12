from ultralytics import YOLO

# Завантажуємо готову модель
model = YOLO("./train11/weights/best.pt") #run/train11/weights/best.pt

# Експорт у ONNX
model.export(format="onnx", opset=11, dynamic=False)  # створить model.onnx
