# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
aesthetic_runtime.py — Win/Py3.8 安全离线版
- 彻底绕开 clip.load() 的哈希校验，用 torch.jit.load(BytesIO) 直接加载 ViT-B/32 .pt
- 在 CPU 完成所有反序列化，再迁移到目标 device（默认 CPU）
- 兼容 {state_dict | model_state_dict | 裸 state_dict}
- 提供基于梯度的 saliency 热力图（可叠加到帧上）
"""
import os, sys, io, json, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# --- numpy._core 垫片（最顶部，避免旧 NumPy 结构差异） ---
try:
    import numpy._core as ncore
except ModuleNotFoundError:
    import numpy.core as ncore
    np._core = ncore
    sys.modules["numpy._core"] = ncore
# ----------------------------------------------------

# ==== 路径配置（请按需调整） ====
LOCAL_CLIP_DIR   = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\CLIP\models"
LOCAL_CLIP_PATH  = os.path.join(LOCAL_CLIP_DIR, "ViT-B-32.pt")  # 官方JIT权重
AESTHETIC_MODEL_PATH = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\best_aesthetic_model.pth"

# ==== 懒加载缓存 ====
_MODEL = None
_CLIP  = None
_PREPROCESS = None
_DEVICE = None

# ==== 预处理（与 openai/CLIP 一致） ====
def _build_preprocess(n_px=224):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

# ==== 离线加载 CLIP（不走 clip.load） ====
def _load_clip_offline_nohash():
    if not os.path.exists(LOCAL_CLIP_PATH):
        raise FileNotFoundError(f"❌ 缺少 CLIP 权重: {LOCAL_CLIP_PATH}")
    print(f"🧠 正在离线加载 CLIP (jit) : {LOCAL_CLIP_PATH}")
    with open(LOCAL_CLIP_PATH, "rb") as f:
        blob = f.read()
    bio = io.BytesIO(blob)
    model = torch.jit.load(bio, map_location="cpu")
    model.eval()
    preprocess = _build_preprocess(224)
    print("✅ CLIP (jit) 加载成功（离线/CPU）")
    return model, preprocess

# ==== 简单美学回归头 ====
class SimpleAestheticModel(nn.Module):
    def __init__(self, clip_model, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.clip_model = clip_model  # JIT model，含 encode_image()
        self.regression_head = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
        )

    def forward(self, images):
        feats = self.clip_model.encode_image(images)  # [B,512]
        return self.regression_head(feats.float()).squeeze(-1)

# ==== 安全加载主入口 ====
def get_model_and_preprocess(model_path=AESTHETIC_MODEL_PATH):
    """
    返回 (model, clip_model, preprocess, device)
    - CLIP JIT 在 CPU 上加载（BytesIO）
    - 评分头权重先在 CPU 反序列化，再迁移到 device（默认 CPU；可设 AESTHETIC_DEVICE=cuda:0）
    """
    global _MODEL, _CLIP, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _CLIP, _PREPROCESS, _DEVICE

    # 设备选择
    env_dev = os.getenv("AESTHETIC_DEVICE", "").strip()
    target_device = torch.device(env_dev) if env_dev else torch.device("cpu")

    # 1) 加载 CLIP（CPU/JIT/BytesIO）
    clip_model, preprocess = _load_clip_offline_nohash()

    # 2) 构建评分模型（先 CPU）
    model = SimpleAestheticModel(clip_model=clip_model).to("cpu").eval()

    # 3) 安全加载权重（先读内存，再 load 到 CPU）
    weights_ok = False
    if not os.path.exists(model_path):
        print(f"[AESTHETIC][WARN] 未找到评分权重：{model_path}，将以未加载权重运行。")
    else:
        try:
            with open(model_path, "rb") as f:
                blob = f.read()
            bio = io.BytesIO(blob)
            try:
                ckpt = torch.load(bio, map_location="cpu", weights_only=True)  # 新版 torch
            except TypeError:
                bio.seek(0)
                ckpt = torch.load(bio, map_location="cpu")
            # 解析 state_dict
            if isinstance(ckpt, dict):
                sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                sd = ckpt
            # 清理 DataParallel 前缀
            from collections import OrderedDict
            clean = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
            model.load_state_dict(clean, strict=True)
            weights_ok = True
            print("✅ Aesthetic 回归权重加载完成。")
        except Exception as e:
            import traceback; print("[AESTHETIC][WARN] 权重加载失败：", repr(e)); traceback.print_exc()

    # 4) 迁移到目标设备（CLIP+评分头）
    try:
        model = model.to(target_device)
        clip_model = clip_model.to(target_device)
    except Exception as e:
        print(f"[AESTHETIC][WARN] 无法迁移到 {target_device}，改用 CPU：{e}")
        target_device = torch.device("cpu")
        model = model.to(target_device)
        clip_model = clip_model.to(target_device)

    _MODEL, _CLIP, _PREPROCESS, _DEVICE = model, clip_model, preprocess, target_device
    tag = f"(device={_DEVICE}, weights={'OK' if weights_ok else 'MISSING'})"
    print(f"✅ Aesthetic 模型就绪 {tag}")
    return _MODEL, _CLIP, _PREPROCESS, _DEVICE

# ==== 基础特征 ====
def _basic_feats(img: Image.Image):
    arr = np.asarray(img).astype(np.float32) / 255.0
    return {
        "brightness": float(arr.mean()),
        "contrast":   float(arr.std()),
        "saturation": float(np.std(arr, axis=2).mean()) if arr.ndim == 3 else 0.0,
    }

# ==== Saliency 热力图（基于输入梯度）====
def saliency_heatmap(pil_img: Image.Image, model, preprocess, device):
    """
    返回 HxW 的归一化热力图（numpy float32）
    """
    img = preprocess(pil_img).unsqueeze(0).to(device)           # [1,3,224,224]
    img.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(img)                                        # [1]
        score = out.squeeze()
        score.backward(retain_graph=False)
        grad = img.grad.detach().abs().mean(dim=1)[0]           # [224,224]
        g = grad / (grad.max() + 1e-8)
        return g.clamp(0,1).float().cpu().numpy()

# ==== 叠加热力图到 BGR 帧 ====
def overlay_heatmap_bgr(frame_bgr: np.ndarray, heatmap_224: np.ndarray, alpha=0.45):
    import cv2
    H, W = frame_bgr.shape[:2]
    hm = cv2.resize((heatmap_224*255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = (alpha*hm_color + (1-alpha)*frame_bgr).astype(np.uint8)
    return out

# ==== 文件打分 ====
def score_image(image_path, use_gradcam=False, save_json=True, save_csv_path=None, save_heat_path=None):
    model, clip_model, preprocess, device = get_model_and_preprocess()
    img = Image.open(image_path).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        s = float(model(x).item())

    feats = _basic_feats(img)
    result = {
        "image": os.path.basename(image_path),
        "score": round(s,2),
        "brightness": round(feats["brightness"],2),
        "contrast":   round(feats["contrast"],2),
        "saturation": round(feats["saturation"],2),
        "ts": time.time(),
    }

    if save_heat_path:
        hm = saliency_heatmap(img, model, preprocess, device)
        import cv2
        bgr = cv2.cvtColor(np.asarray(img)[:,:,::-1], cv2.COLOR_RGB2BGR)
        over = overlay_heatmap_bgr(bgr, hm)
        cv2.imwrite(save_heat_path, over)

    if save_json:
        try:
            with open(image_path + ".score.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[AESTHETIC][WARN] 写 JSON 失败：", e)

    if save_csv_path:
        try:
            os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
            header = ["image","score","brightness","contrast","saturation","ts"]
            need = not os.path.exists(save_csv_path)
            with open(save_csv_path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if need: w.writerow(header)
                w.writerow([result[k] for k in header])
        except Exception as e:
            print("[AESTHETIC][WARN] 写 CSV 失败：", e)

    return result

# ==== 内存打分（PIL）====
def score_image_from_pil(image: Image.Image, use_gradcam=False, want_heat=False):
    model, clip_model, preprocess, device = get_model_and_preprocess()
    img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        s = float(model(x).item())
    out = {
        "score": round(s,2),
        "brightness": round(_basic_feats(img)["brightness"],2),
        "contrast": round(_basic_feats(img)["contrast"],2),
        "saturation": round(_basic_feats(img)["saturation"],2),
    }
    if want_heat:
        hm = saliency_heatmap(img, model, preprocess, device)  # 224x224
        out["heatmap224"] = hm  # numpy float32
    return out
