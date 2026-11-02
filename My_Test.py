from ultralytics import YOLO

model = YOLO(r"yolo11x.pt")
print(model.task)
# print(model.anchors)
# print(model.classes)
print(model.names)
print(sum(p.numel() for p in model.parameters()))