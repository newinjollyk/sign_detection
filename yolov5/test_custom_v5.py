import subprocess

# Set paths
YOLOV5_PATH = '/home/newin/Projects/sign detection/yolov5'
WEIGHTS_PATH = '/home/newin/Projects/sign detection/yolov5/runs/train/yolov5n_sign_50epochs3/weights/best.pt'
DATA_YAML_PATH = '/home/newin/Projects/sign detection/yolov5/data.yaml'
IMG_SIZE = 640

# Format paths with quotes in case of spaces
cmd = [
    'python', f'{YOLOV5_PATH}/val.py',
    '--weights', WEIGHTS_PATH,
    '--data', DATA_YAML_PATH,
    '--img', str(IMG_SIZE),
    '--task', 'test',
    '--conf', '0.25',
    '--iou', '0.6',
    '--save-json'
]

# Run and capture output
print("🔍 Evaluating YOLOv5 model on test set...\n")
result = subprocess.run(cmd, capture_output=True, text=True)

# Print output
print(result.stdout)
print("✅ Done. Check `runs/val/exp/` for full results.")
