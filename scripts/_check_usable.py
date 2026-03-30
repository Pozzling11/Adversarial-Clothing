"""Count usable images in data/clean with 20px shoulder-width floor."""
import numpy as np, cv2
from ultralytics import YOLO
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

model = YOLO("yolov8n-pose.pt")
img_dir = Path("data/clean")
imgs = sorted(p for p in img_dir.iterdir()
              if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._"))

print(f"Total files with supported extensions: {len(imgs)}\n")
print(f"{'image':60s}  {'sw':>6s}  {'torso_w':>7s}  {'torso_h':>7s}  {'status':>10s}")
print("-" * 100)

usable = 0
skipped_sw = []
skipped_torso = []
no_person = []
all_sws = []

for p in imgs:
    img = cv2.imread(str(p))
    if img is None:
        print(f"{p.name:60s}  {'':>6s}  {'':>7s}  {'':>7s}  UNREADABLE")
        continue
    h0, w0 = img.shape[:2]
    scale = 640 / max(h0, w0)
    results = model(img, imgsz=640, verbose=False)
    found_person = False
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.data.cpu().numpy()
        if len(kpts) == 0:
            continue
        k = kpts[0]
        found_person = True
        ls_x, ls_y = k[5][:2] * scale
        rs_x, rs_y = k[6][:2] * scale
        lh_x, lh_y = k[11][:2] * scale
        rh_x, rh_y = k[12][:2] * scale
        sw = abs(ls_x - rs_x)
        margin = sw * 0.20
        torso_w = int(min(640, max(ls_x, rs_x) + margin)) - int(max(0, min(ls_x, rs_x) - margin))
        torso_h = int(min(640, max(lh_y, rh_y))) - int(max(0, min(ls_y, rs_y)))
        all_sws.append(sw)

        if sw < 20:
            status = "SKIP(sw)"
            skipped_sw.append((p.name, sw))
        elif torso_w < 20 or torso_h < 20:
            status = "SKIP(dim)"
            skipped_torso.append((p.name, torso_w, torso_h))
        else:
            status = "OK"
            usable += 1

        print(f"{p.name:60s}  {sw:6.1f}  {torso_w:7d}  {torso_h:7d}  {status:>10s}")
        break

    if not found_person:
        no_person.append(p.name)
        print(f"{p.name:60s}  {'':>6s}  {'':>7s}  {'':>7s}  NO_PERSON")

print("-" * 100)
print(f"\nSummary:")
print(f"  Total loadable images:      {len(imgs)}")
print(f"  Usable (sw>=20, dims>=20):  {usable}")
print(f"  Skipped (sw < 20px):        {len(skipped_sw)}")
for name, sw in skipped_sw:
    print(f"    {name}: sw={sw:.1f}px")
print(f"  Skipped (torso dims < 20):  {len(skipped_torso)}")
for name, tw, th in skipped_torso:
    print(f"    {name}: torso_w={tw}, torso_h={th}")
print(f"  No person detected:         {len(no_person)}")
for name in no_person:
    print(f"    {name}")

if all_sws:
    print(f"\nShoulder width distribution (all detected):")
    print(f"  min={min(all_sws):.1f}  p10={np.percentile(all_sws,10):.1f}  median={np.median(all_sws):.1f}  mean={np.mean(all_sws):.1f}  max={max(all_sws):.1f}")
    print(f"\nAt 20px floor: {sum(1 for s in all_sws if s < 20)} would be skipped")
    print(f"At 30px floor: {sum(1 for s in all_sws if s < 30)} would be skipped")
    print(f"At 40px floor: {sum(1 for s in all_sws if s < 40)} would be skipped")
