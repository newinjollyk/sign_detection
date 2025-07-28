
from ultralytics import YOLO

# === Config ===
MODEL_PATH = '/home/newin/Projects/sign detection/runs/train/yolov8n_custom/weights/best.pt'
DATA_PATH = '/home/newin/Projects/sign detection/Dataset/data.yaml'
IMG_SIZE = 640

# === Run test evaluation ===
model = YOLO(MODEL_PATH)
results = model.val(data=DATA_PATH, imgsz=IMG_SIZE, split='test')

# === Print key metrics ===
metrics = results.metrics
print(f"\n📊 Test Results:")
print(f"Precision:     {metrics.precision:.3f}")
print(f"Recall:        {metrics.recall:.3f}")
print(f"mAP@0.5:       {metrics.map50:.3f}")
print(f"mAP@0.5:0.95:  {metrics.map:.3f}")
