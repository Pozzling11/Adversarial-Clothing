import torch
from ultralytics import YOLO
import cv2
from pathlib import Path

yolo = YOLO("yolov8n.pt")
dev = torch.device("mps")
model = yolo.model
model.eval().to(dev)
for p in model.parameters():
    p.requires_grad_(False)

img_path = next(Path("data/clean").glob("*.jpg"), None) or next(Path("data/clean").glob("*.png"), None)
print(f"Using: {img_path}")
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
IMG_SIZE = 640
img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_t = torch.from_numpy(img_rgb).float() / 255.0
img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dev)

raw = model(img_t)
if isinstance(raw, (list, tuple)):
    raw = raw[0]

print(f"raw shape: {raw.shape}")
print(f"ch0: min={raw[0,0,:].min().item():.4f}  max={raw[0,0,:].max().item():.4f}  (x1 if decoded box)")
print(f"ch1: min={raw[0,1,:].min().item():.4f}  max={raw[0,1,:].max().item():.4f}  (y1 if decoded box)")
print(f"ch4: min={raw[0,4,:].min().item():.6f}  max={raw[0,4,:].max().item():.6f}  (person if layout is [box,cls])")
print(f"ch4 top-50 mean: {torch.topk(raw[0,4,:], 50).values.mean().item():.6f}")
print(f"ch4 top-5 vals:  {[round(v,4) for v in torch.topk(raw[0,4,:], 5).values.tolist()]}")

# Compare with yolo.predict
res = yolo.predict(source=img_bgr, conf=0.01, classes=[0], verbose=False)
boxes = res[0].boxes
if boxes is not None and len(boxes):
    print(f"\nyolo.predict best conf: {boxes.conf.max().item():.4f}")
else:
    print("\nyolo.predict: no detections")

# Also check what the max across ALL class channels is for this image
print(f"\nMax across all class channels (ch4-83): {raw[0, 4:, :].max().item():.6f}")
print(f"Which class channel has the max: {raw[0, 4:, :].max(dim=0).indices.mode().values.item() + 4}")
