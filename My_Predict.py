# 图片检测
from ultralytics import YOLO

# 加载模型
yolo = YOLO('yolo26n.pt', task='detect')

# ---------------------- 图片检测 ----------------------
# result = yolo.predict(source='ultralytics/assets/bus.jpg', show=True)

# ---------------------- 视频检测 ----------------------
result = yolo.predict(source='./palace.mp4', show=True)

# ---------------------- 屏幕检测 ----------------------
# result = yolo.predict(source='screen', show=True)

# ---------------------- 摄像头检测 ----------------------
# result = yolo.predict(source=0, show=True)
