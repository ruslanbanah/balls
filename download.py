from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY") # ⟵ замініть YOUR_API_KEY на ваш реальний ключ API
project = rf.workspace("object-detection-examples").project("balls-dataset")
dataset = project.version(1).download("yolov8")  # ⟵ це завантажує ZIP та розпаковує
