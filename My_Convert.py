import torch
import onnx
import onnxruntime as ort
import numpy as np
import onnxsim
import os
from ultralytics import YOLO

# ===================== 配置区，修改这里 =====================
pt_weight_path = "yolo11s.pt"
# 输出onnx文件，使用绝对路径
out_onnx = r"D:\Deep_Learning\ultralytics-8.3.163\yolo11s.onnx"
out_onnx_simplified = r"D:\Deep_Learning\ultralytics-8.3.163\yolo11s_simplified.onnx"
input_size = (1, 3, 640, 640)
# =========================================================

model = YOLO(pt_weight_path)
pt_model = model.model
pt_model.eval()

dummy_input = torch.randn(*input_size, dtype=torch.float32)

# 导出onnx
torch.onnx.export(
    pt_model,
    dummy_input,
    out_onnx,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    dynamo=False, # 关闭dynamo，解决Split num_outputs报错
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "output0": {0: "batch_size"}
    }
)
print(f"✅ 原始ONNX导出完成：{out_onnx}")

# onnxsim模型简化
onnx_model = onnx.load(out_onnx)
onnx_model_simplified, ok = onnxsim.simplify(onnx_model)
assert ok, "onnxsim简化失败！"
onnx.save(onnx_model_simplified, out_onnx_simplified)
print(f"✅ 简化ONNX保存成功：{out_onnx_simplified}")
print("文件存在？", os.path.exists(out_onnx_simplified))

# 校验模型
onnx.checker.check_model(onnx_model_simplified)
print("✅ ONNX模型校验通过")

# ONNXRuntime推理
ort_session = ort.InferenceSession(out_onnx_simplified, providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: dummy_input.numpy()}
ort_output = ort_session.run(None, ort_inputs)[0]

# PyTorch推理
with torch.no_grad():
    torch_out = pt_model(dummy_input)[0].cpu().numpy()

# 结果对比
np.testing.assert_allclose(torch_out, ort_output, rtol=1e-02, atol=1e-03)
print("🎉 PT与ONNX Runtime推理结果对齐！")
