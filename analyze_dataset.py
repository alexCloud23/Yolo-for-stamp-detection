import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# -----------------------------
# MODEL_PATH = "models/best.pt"
# IMAGE_DIR = "stamps.yolo26/test/images"
# LABEL_DIR = "stamps.yolo26/test/labels"
# METRICS_PATH = "metrics/metrics.json"
# -----------------------------

MODEL_PATH = "models/best.pt"

IMAGE_DIR = "stamps.yolo26/test/images"
LABEL_DIR = "stamps.yolo26/test/labels"

OUTPUT_DIR = "metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_analysis.json")

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / (union + 1e-6)

def load_labels(label_path, w, h):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            boxes.append((int(cls), x1, y1, x2, y2))

    return boxes

model = YOLO(MODEL_PATH)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

results = []

for img_name in tqdm(images):

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gt = load_labels(label_path, w, h)

    pred = model.predict(img, verbose=False)[0]

    preds = []

    for b in pred.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        cls = int(b.cls.item())
        conf = float(b.conf.item())

        preds.append((cls, x1, y1, x2, y2, conf))

    matched_gt = set()
    matched_pred = set()

    tp = 0
    fp = 0
    fn = 0

    ious = []

    for i, p in enumerate(preds):

        best_iou = 0
        best_j = -1

        for j, g in enumerate(gt):

            if g[0] != p[0]:
                continue

            score = iou(p[1:5], g[1:5])

            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou > 0.5:

            tp += 1
            matched_pred.add(i)
            matched_gt.add(best_j)
            ious.append(best_iou)

        else:
            fp += 1

    fn = len(gt) - len(matched_gt)

    mean_iou = np.mean(ious) if ious else 0

    results.append({
        "image": img_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_iou": mean_iou
    })

sorted_by_iou = sorted(
    results,
    key=lambda x: x["mean_iou"]
)

best_5 = sorted_by_iou[-5:]
worst_5 = sorted_by_iou[:5]

output = {
    "best_5": best_5,
    "worst_5": worst_5,
    "all": results
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print("Saved to:", OUTPUT_FILE)