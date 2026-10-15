import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """YOLO11 detector with the existing (N, 6) pipeline contract.

    Returns rows: x1, y1, x2, y2, confidence, COCO class ID.
    Coordinates are in original-frame pixels.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        input_size=640,
        score_thr=0.15,
        nms_thr=0.45,
        classes=None,
        threads=2,
    ):
        self.size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.allowed = sorted(
            (classes or {
                2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
            }).keys()
        )
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, img):
        result = self.model.predict(
            source=img,
            imgsz=self.size,
            conf=self.score_thr,
            iou=self.nms_thr,
            classes=self.allowed,
            verbose=False,
            device="cpu",
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)

        return np.concatenate((xyxy, conf, cls), axis=1).astype(np.float32)
