import streamlit as st
import os
import json
import random
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.visualization import (
    draw_gt,
    draw_pred,
    draw_overlay
)

st.set_page_config(
    page_title="YOLO26 Stamp Detection",
    layout="wide"
)

st.markdown(
"""
<style>
[data-testid="metric-container"] {
    background-color: #102A22;
    border: 1px solid #22C55E;
    padding: 15px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #22C55E;
    font-weight: bold;
}
.stButton button {
    background-color: #22C55E;
    color: black;
    border-radius: 10px;
    border: none;
}
.stButton button:hover {
    background-color: #16A34A;
    color: white;
}
</style>
""",
unsafe_allow_html=True
)

MODEL_PATH = "models/best.pt"
IMAGE_DIR = "stamps.yolo26/test/images"
LABEL_DIR = "stamps.yolo26/test/labels"
METRICS_PATH = "metrics/metrics.json"

EXPECTED_CLASS_NAMES = {
    0: "stamp",
    1: "signature"
}


def is_stamp_model(names):
    if len(names) != 2:
        return False
    values = {str(v).lower() for v in names.values()}
    return values in ({"stamp", "signature"}, {"0", "1"})


def get_class_name(cls_id):
    if len(model.names) == 2 and cls_id in EXPECTED_CLASS_NAMES:
        return EXPECTED_CLASS_NAMES[cls_id]
    return model.names.get(cls_id, model.names.get(str(cls_id), f"class_{cls_id}"))


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
model_is_valid = is_stamp_model(model.names)


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


metrics = load_metrics()


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def apply_nms(predictions, iou_threshold):
    if not predictions:
        return predictions

    kept = []
    by_class = {}

    for pred in predictions:
        by_class.setdefault(pred[0], []).append(pred)

    for cls_preds in by_class.values():
        cls_preds = sorted(cls_preds, key=lambda p: p[5], reverse=True)
        selected = []

        for pred in cls_preds:
            box = pred[1:5]
            if all(box_iou(box, other[1:5]) <= iou_threshold for other in selected):
                selected.append(pred)

        kept.extend(selected)

    return sorted(kept, key=lambda p: p[5], reverse=True)


def predict(image_path, conf, imgsz, max_det):
    result = model.predict(
        image_path,
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False
    )[0]
    return result


def extract_predictions(result):
    predictions = []
    if result.boxes is None:
        return predictions

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        predictions.append((cls, x1, y1, x2, y2, conf))

    return predictions


st.title("Детекция штампов и подписей")
st.write("Демонстрация обученной модели YOLO26n")

st.sidebar.header("Настройки модели")
confidence = st.sidebar.slider("Порог уверенности (Confidence)", 0.01, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("Порог NMS IoU", 0.1, 1.0, 0.7, 0.05)
image_size = st.sidebar.selectbox("Размер изображения", [320, 416, 512, 640, 800], index=3)
max_det = st.sidebar.slider("Максимальное количество объектов", 1, 20, 20)

st.subheader("Метрики модели")
if metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", round(metrics["precision"], 4))
    c2.metric("Recall", round(metrics["recall"], 4))
    c3.metric("F1", round(metrics["f1"], 4))
    c4.metric("mAP@0.5", round(metrics["map50"], 4))
    c5.metric("mAP@0.5:0.95", round(metrics["map5095"], 4))

st.divider()

st.subheader("Выбор изображения")
images = [x for x in os.listdir(IMAGE_DIR) if x.lower().endswith((".jpg", ".jpeg", ".png"))]

col1, col2 = st.columns([4, 1])
with col1:
    selected_image = st.selectbox("Выберите изображение", images)

with col2:
    random_button = st.button("Random")

if random_button:
    selected_image = random.choice(images)

image_path = os.path.join(IMAGE_DIR, selected_image)
label_path = os.path.join(LABEL_DIR, selected_image.rsplit(".", 1)[0] + ".txt")

image = load_image(image_path)
start = time.time()
result = predict(image_path, confidence, image_size, max_det)
inference_time = time.time() - start

raw_predictions = extract_predictions(result)
predictions = apply_nms(raw_predictions, iou_threshold)
prediction_img = draw_pred(image, predictions)

st.subheader("Текущие параметры")
a, b, c, d, e = st.columns(5)
a.metric("Confidence", confidence)
b.metric("IoU", iou_threshold)
c.metric("Image size", image_size)
d.metric("Detections", len(predictions))
e.metric("Inference time", f"{inference_time*1000:.1f} ms")

st.divider()

st.subheader("Результат обнаружения")
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Original", use_container_width=True)

with col2:
    st.image(prediction_img, caption="Prediction", use_container_width=True)

st.subheader("Обнаруженные объекты")
if predictions:
    for p in predictions:
        st.write(f"**Class:** {get_class_name(p[0])}  \n**Confidence:** {p[5]:.3f}")
else:
    st.warning("Объекты не найдены")

if os.path.exists(label_path):
    st.divider()
    st.subheader("Ground Truth")
    gt_img = draw_gt(image, label_path)
    overlay_img = draw_overlay(image, label_path, predictions)

    c1, c2 = st.columns(2)
    with c1:
        st.image(gt_img, caption="Ground Truth", use_container_width=True)
    with c2:
        st.image(overlay_img, caption="GT + Prediction Overlay", use_container_width=True)