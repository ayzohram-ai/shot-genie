import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import clip
import os
import cv2
import pandas as pd

# -------------------------
# 设备
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# 美学模型定义
# -------------------------
class SimpleAestheticModel(nn.Module):
    def __init__(self, clip_model, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.clip_model = clip_model
        self.regression_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        scores = self.regression_head(features.float())
        return scores.squeeze(-1)

# -------------------------
# Grad-CAM
# -------------------------
class GradCAM:
    def __init__(self, target_module):
        self.gradients = None
        self.activations = None
        target_module.register_forward_hook(self._forward_hook)
        try:
            target_module.register_full_backward_hook(self._backward_hook)
        except Exception:
            target_module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, upsample_size):
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are None. Did you call backward()?")

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1,1,1,1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_upsampled = F.interpolate(cam, size=upsample_size, mode='bilinear', align_corners=False)
        return cam_upsampled.squeeze().cpu().numpy()

# -------------------------
# 基础美学特征
# -------------------------
def compute_basic_features(img: Image.Image):
    img_np = np.array(img)/255.0
    brightness = img_np.mean()
    contrast = img_np.std()
    saturation = np.std(img_np, axis=2).mean()
    return {"brightness": round(float(brightness),2),
            "contrast": round(float(contrast),2),
            "saturation": round(float(saturation),2)}

# -------------------------
# CLIP prompt解释
# -------------------------
def explain_with_prompts(image, clip_model, preprocess):
    prompts = ["Good lighting", "Poor lighting",
               "Balanced composition", "Unbalanced composition",
               "Clear subject", "Distracting background"]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocess(image).unsqueeze(0).to(device))
        text_features = clip_model.encode_text(text_tokens)
        logits = (image_features @ text_features.T).softmax(dim=-1)
    topk = logits[0].topk(3)
    return [(prompts[i], float(logits[0][i])) for i in topk.indices]

# -------------------------
# 分析单张图片
# -------------------------
def analyze_image(image_path, model, clip_model, preprocess, use_gradcam=True):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        print(f"⚠️ 跳过 {image_path}: 无法打开")
        return None

    image_tensor = preprocess(img).unsqueeze(0).to(device)

    # 美学分数
    model.eval()
    with torch.no_grad():
        score = round(model(image_tensor).item(),2)

    # Grad-CAM
    cam_path = None
    if use_gradcam:
        target_layer = model.clip_model.visual.conv1
        gradcam = GradCAM(target_layer)

        with torch.enable_grad():
            image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
            features = model.clip_model.encode_image(image_tensor_grad)
            score_tensor = model.regression_head(features.float()).squeeze()
            model.zero_grad()
            score_tensor.backward()
            orig_w, orig_h = img.size
            cam_map = gradcam.generate_cam((orig_h, orig_w))
            heatmap = np.uint8(255*cam_map)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            img_np = np.array(img.resize((orig_w, orig_h)))
            overlay = np.uint8(0.5*img_np + 0.5*heatmap)
            cam_path = os.path.splitext(image_path)[0] + "_gradcam.jpg"
            Image.fromarray(overlay).save(cam_path)

    # 基础特征
    features = compute_basic_features(img)
    # CLIP解释
    clip_exp = explain_with_prompts(img, clip_model, preprocess)
    return {
        "image": os.path.basename(image_path),
        "score": score,
        "brightness": features["brightness"],
        "contrast": features["contrast"],
        "saturation": features["saturation"],
        "clip1": f"{clip_exp[0][0]} ({clip_exp[0][1]:.2f})",
        "clip2": f"{clip_exp[1][0]} ({clip_exp[1][1]:.2f})",
        "clip3": f"{clip_exp[2][0]} ({clip_exp[2][1]:.2f})",
        "gradcam_path": cam_path
    }

# -------------------------
# 批量处理函数
# -------------------------
def analyze_folder(folder_path, model, clip_model, preprocess, use_gradcam=True, output_csv="results.csv"):
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png",".jpeg"))]
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        print(f"🔹 处理 {img_file} ...")
        try:
            res = analyze_image(img_path, model, clip_model, preprocess, use_gradcam=use_gradcam)
            if res:
                results.append(res)
        except Exception as e:
            print(f"⚠️ 跳过 {img_file}: {e}")

    if results:
        df = pd.DataFrame(results)
        # 保存 CSV
        df.to_csv(output_csv, index=False)
        print(f"✅ 批量分析完成，结果保存到 {output_csv}")

# -------------------------
# 主函数
# -------------------------
def main():
    model_path = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\best_aesthetic_model.pth"
    folder_path = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\shooting\photo"  # 对已拍照片直接打分
    output_csv = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\results_photo.csv"

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # 美学模型
    aesthetic_model = SimpleAestheticModel(clip_model=clip_model).to(device)
    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        aesthetic_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        aesthetic_model.load_state_dict(checkpoint)
    aesthetic_model.eval()
    print("✅ 模型加载完成")

    analyze_folder(folder_path, aesthetic_model, clip_model, preprocess, use_gradcam=True, output_csv=output_csv)

if __name__ == "__main__":
    main()
