from ultralytics import YOLO

# Load a YOLOv8n model architecture (nano size)
model = YOLO('yolov8n.yaml')  # You can also use 'yolov8n.pt' for transfer learning

# Train the model on your custom dataset
model.train(
    data='/home/newin/Projects/sign detection/Dataset/data.yaml',  # 👈 replace with path to your dataset config file
    epochs=50,                      # Number of training epochs
    imgsz=640,                      # Image size for training
    batch=16,                       # Batch size (adjust based on GPU memory)
    name='yolov8n_custom',         # Name of the training run
    project='runs/train',          # Folder to save results
    workers=4,                      # Number of dataloader workers
    device=0                        # GPU (use device=0) or CPU (device='cpu')
)