# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model  = YOLO(r"/root/autodl-tmp/Deep_Learning/ultralytics-8.3.163/runs/detect/train13/weights/best.pt")
#     model.val(split="val")

from ultralytics import YOLO

MODEL = r"D:\Deep_Learning\ultralytics-8.3.163\runs\detect\mushroom\yolo11s_mushroom_21cls_960\weights\best.pt"

# 修改成你Windows路径，用原始字符串 r"" 避免转义问题
DATA = r"D:\Deep_Learning\ultralytics-8.3.163\datasets\datasets_of_gugu\datasets1.yaml"

model = YOLO(MODEL)

conf_list = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10
]

for conf in conf_list:
    metrics = model.val(
        data=DATA,
        imgsz=960,
        batch=4,
        conf=conf,
        iou=0.7,
        plots=False,
        workers=0
    )

    print(
        f"conf={conf:.3f} | "
        f"Precision={metrics.box.mp:.4f} | "
        f"Recall={metrics.box.mr:.4f} | "
        f"mAP50={metrics.box.map50:.4f} | "
        f"mAP50-95={metrics.box.map:.4f}"
    )


