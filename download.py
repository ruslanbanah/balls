from roboflow import Roboflow

rf = Roboflow(api_key="OX5pDCI2h3yXHo91bFMb")
project = rf.workspace("object-detection-examples").project("balls-dataset")
dataset = project.version(1).download("yolov8")  # ⟵ це завантажує ZIP та розпаковує
