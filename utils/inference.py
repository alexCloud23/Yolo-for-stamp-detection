from ultralytics import YOLO
import time


class Detector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(
        self,
        image,
        conf=0.25,
        iou=0.7,
        imgsz=640,
        max_det=100
    ):

        start = time.time()

        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False
        )

        latency = (time.time() - start) * 1000

        return results[0], latency