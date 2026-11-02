from linecache import cache

from torchvision.datasets import coco

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt")
    model.train(
        data = "coco8.yaml",
        epochs = 10,
        imgsz=640,
        batch=2,
        cache = False,
        workers = 0,
    )