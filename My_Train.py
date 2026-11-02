# # from torch.backends.cuda import flash_sdp_enabled
# #
# # from ultralytics import YOLO
# #
# # if __name__ == '__main__':
# #     model = YOLO(r"/root/autodl-tmp/Deep_Learning/ultralytics-8.3.163/yolo11n.pt")
# #     model.train(
# #         data = r"/root/autodl-tmp/Deep_Learning/ultralytics-8.3.163/datasets/datasets_of_gugu/datasets1.yaml",
# #         epochs = 200,
# #         imgsz=640,
# #         batch=8,
# #         cache = False,
# #         workers = 4,
# #         device = 0,
# #         lr0 = 0.001
# #     )
#
# from ultralytics import YOLO
# import torch
# import os
#
#
# if __name__ == '__main__':
#
#     # ==========================================
#     # 1. 数据集和模型路径
#     # ==========================================
#
#     data_path = (
#         r"/root/autodl-tmp/Deep_Learning/"
#         r"ultralytics-8.3.163/datasets/"
#         r"datasets_of_gugu/datasets1.yaml"
#     )
#
#     # 使用 YOLO11s 预训练权重
#     # 如果需要和原始 YOLO11n 对照，
#     # 将这里修改为 yolo11n.pt
#     model_path = "yolo11s.pt"
#
#     # ==========================================
#     # 2. GPU 检查
#     # ==========================================
#
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA GPU 不可用")
#
#     print("当前 GPU：", torch.cuda.get_device_name(0))
#
#     print(
#         "GPU 显存：",
#         round(
#             torch.cuda.get_device_properties(0).total_memory
#             / 1024**3, 2
#         ),
#         "GB"
#     )
#
#     # ==========================================
#     # 3. 加载预训练模型
#     # ==========================================
#
#     model = YOLO(model_path)
#
#     # ==========================================
#     # 4. 21 类蘑菇检测训练
#     # ==========================================
#
#     results = model.train(
#
#         # ---------- 数据集 ----------
#         data=data_path,
#
#         # ---------- 基础训练 ----------
#         epochs=200,
#         imgsz=960,
#         batch=8,
#         device=0,
#         workers=8,
#
#         pretrained=True,
#         amp=True,
#         cache='disk',
#
#         # ---------- 优化器 ----------
#         optimizer='AdamW',
#         lr0=0.001,
#         lrf=0.01,
#         momentum=0.9,
#         weight_decay=0.0005,
#
#         # ---------- 学习率调度 ----------
#         cos_lr=True,
#         warmup_epochs=3.0,
#         warmup_momentum=0.8,
#         warmup_bias_lr=0.1,
#
#         # ---------- 损失函数 ----------
#         # 暂时使用 YOLO11 默认损失权重
#         # 先通过实验确定是否需要进一步调整
#         box=7.5,
#         cls=0.5,
#         dfl=1.5,
#
#         # ---------- 蘑菇颜色增强 ----------
#         # 保护菌盖、菌褶等颜色特征
#         hsv_h=0.008,
#         hsv_s=0.35,
#         hsv_v=0.25,
#
#         # ---------- 几何增强 ----------
#         degrees=10.0,
#         translate=0.05,
#         scale=0.3,
#         shear=0.0,
#         perspective=0.0,
#
#         # ---------- 翻转增强 ----------
#         fliplr=0.5,
#         flipud=0.0,
#
#         # ---------- 混合增强 ----------
#         mosaic=0.5,
#         mixup=0.0,
#
#         # ---------- 后期收敛 ----------
#         close_mosaic=25,
#         patience=60,
#
#         # ---------- 多尺度 ----------
#         multi_scale=False,
#
#         # ---------- 实验控制 ----------
#         seed=42,
#         deterministic=True,
#
#         # ---------- 保存设置 ----------
#         save=True,
#         save_period=20,
#         plots=True,
#         val=True,
#
#         project='runs/mushroom',
#         name='yolo11s_mushroom_21cls_960',
#         exist_ok=True,
#         verbose=True
#     )
#
#     # ==========================================
#     # 5. 加载最佳权重
#     # ==========================================
#
#     best_path = model.trainer.best
#
#     print("\n训练完成！")
#     print("最佳权重：", best_path)
#
#     best_model = YOLO(best_path)
#
#     # ==========================================
#     # 6. 最终验证
#     # ==========================================
#
#     metrics = best_model.val(
#         data=data_path,
#         imgsz=960,
#         batch=8,
#         device=0,
#         split='val',
#         plots=True
#     )
#
#     # ==========================================
#     # 7. 输出评价指标
#     # ==========================================
#
#     print("\n========== 21类蘑菇检测结果 ==========")
#
#     print("Precision:", metrics.box.mp)
#     print("Recall:", metrics.box.mr)
#     print("mAP50:", metrics.box.map50)
#     print("mAP50-95:", metrics.box.map)
#
#     print("\n========== 各类别 mAP50-95 ==========")
#
#     for class_id, class_map in enumerate(metrics.box.maps):
#
#         class_name = best_model.names[class_id]
#
#         print(
#             f"类别 {class_id:02d} | "
#             f"{class_name} | "
#             f"mAP50-95: {class_map:.4f}"
#         )
#
#     print("\n======================================")
#



