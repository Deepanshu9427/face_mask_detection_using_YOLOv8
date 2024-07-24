pip install ultralytics

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="w3VlaVyNSYjfavsws1Hu")
project = rf.workspace("computer-vision-projects-hfyvb").project("face_mask_detection-gpiqp")
version = project.version(2)
dataset = version.download("yolov8-obb")

from ultralytics import YOLO

# Define your dataset path (replace with your actual path)
data_path = "/content/face_mask_detection-2/data.yaml"

# Choose a pre-trained YOLOv8 model (recommended for faster training)
model_name = "yolov8s.pt"  # You can choose other sizes like yolov8m, yolov8l, etc.

# Train the model
model = YOLO(model_name)
results = model.train(
    data=data_path,  # Path to your dataset YAML file
    imgsz=720,  # Image size for training (adjust based on your needs)
    epochs=50,  # Number of training epochs (adjust based on dataset size)
    batch=16,  # Batch size for training (adjust based on GPU memory)
)