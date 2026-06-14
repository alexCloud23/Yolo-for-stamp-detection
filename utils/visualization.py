import cv2
import os


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            boxes.append((int(cls), x1, y1, x2, y2))

    return boxes


def draw_gt(img, label_path):
    img = img.copy()
    h, w = img.shape[:2]

    gt = load_yolo_labels(label_path, w, h)

    for cls, x1, y1, x2, y2 in gt:
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

    return img


def draw_pred(img, boxes):
    img = img.copy()

    for cls, x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2
        )

    return img


def draw_overlay(img, label_path, boxes):
    gt_img = draw_gt(img, label_path)
    pred_img = draw_pred(gt_img, boxes)
    return pred_img