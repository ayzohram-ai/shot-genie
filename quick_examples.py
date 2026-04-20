# -*- coding: utf-8 -*-
"""
quick_examples.py  (稳定版)
---------------------------------------------------
- 顶部 numpy._core 兼容垫片（修复 torch.load 旧环境保存的权重）
- safe_torch_load(): 安全加载权重，兼容多种保存格式
- 统一一次性加载 CLIP+preprocess（避免重复加载/句柄冲突）
- 去除 CPU 情况下的 autocast；只在 CUDA 可用时才用 autocast
- SimpleAestheticModel 不再让 CLIP/JIT 偷转 CUDA
- 提供 SmoothGrad 热力图生成（适配 ViT 输入，实时可用）
---------------------------------------------------
"""
import sys, os, io, json, base64, time
from io import BytesIO
from pathlib import Path
from collections import OrderedDict


import torch
import torch.nn as nn
from PIL import Image

# 仅在需要可视化时再用
import matplotlib.pyplot as plt
import cv2

# 统一选择设备：环境变量 AESTHETIC_DEVICE 优先
def get_device():
    dev = os.getenv("AESTHETIC_DEVICE", "").strip()
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ============ 一次性加载 CLIP 与 preprocess ============
# 使用 openai/CLIP 的官方包
import clip
_CLIP_MODEL = None
_CLIP_PREPROC = None

def get_clip_once(model_name="ViT-B/32", device=DEVICE):
    global _CLIP_MODEL, _CLIP_PREPROC
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROC
    # 强制在目标 device 上加载；避免内部偷偷切到 cuda:0
    clip_model, preproc = clip.load(model_name, device=device, jit=False)
    _CLIP_MODEL, _CLIP_PREPROC = clip_model, preproc
    return _CLIP_MODEL, _CLIP_PREPROC


# ============ 安全权重加载工具 ============
def safe_torch_load(path, map_location="cpu"):
    """
    安全加载 checkpoint：
    1) 尝试 weights_only=True（新 torch）
    2) 不行则退回传统方式
    3) 统一通过 BytesIO 避免句柄问题
    """
    with open(path, "rb") as f:
        blob = f.read()
    bio = io.BytesIO(blob)
    try:
        return torch.load(bio, map_location=map_location, weights_only=False)
    except TypeError:
        bio.seek(0)
        return torch.load(bio, map_location=map_location)


def extract_state_dict(ckpt):
    """
    从各种格式里抽出真正的 state_dict，
    并去掉 DataParallel 的 'module.' 前缀。
    """
    if hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    clean = OrderedDict()
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v
    return clean


# ============ 模型定义 ============
class SimpleAestheticModel(nn.Module):
    """美学评分模型（使用外部已加载的 CLIP 模型）"""
    def __init__(self, clip_model, clip_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.clip = clip_model
        self.regression_head = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor):
        # 强制保证输入与 CLIP 在同一设备
        images = images.to(next(self.clip.parameters()).device, non_blocking=True)
        with torch.no_grad():  # 提取特征不反传
            feats = self.clip.encode_image(images)        # [B, clip_dim]
            feats = feats.float()
        return self.regression_head(feats).squeeze(-1)


# ============ 预测器 ============
class AestheticPredictor:
    """稳健的美学评分预测器"""
    def __init__(self, model_path: str, device: torch.device = DEVICE, clip_name: str = "ViT-B/32"):
        self.device = device
        self.model_path = model_path

        # 统一加载 CLIP 和 preprocess
        self.clip, self.preprocess = get_clip_once(clip_name, device=self.device)
        clip_dim = 512 if "ViT-B" in clip_name else (768 if "ViT-L" in clip_name else 512)

        # 构建回归头并加载权重
        self.model = SimpleAestheticModel(self.clip, clip_dim=clip_dim).to(self.device)
        self.model.eval()

        print(f"正在加载权重: {model_path}")
        ckpt = safe_torch_load(model_path, map_location="cpu")
        sd = extract_state_dict(ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] missing keys: {missing}")
        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected}")
        print(f"✅ 模型加载完成，运行在: {self.device}")

    @torch.no_grad()
    def predict_single(self, image_path_or_pil):
        """对单张图片打分，返回 float[0,1]"""
        if isinstance(image_path_or_pil, (str, os.PathLike)):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        x = self.preprocess(img).unsqueeze(0).to(self.device)
        # 仅在 CUDA 时才用 autocast
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                s = self.model(x).item()
        else:
            s = self.model(x).item()
        return float(s)

    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=32):
        """批量预测，返回与 image_paths 对齐的分数列表（失败为 None）"""
        results = []
        batch = []
        metas = []

        def flush():
            nonlocal results, batch, metas
            if not batch:
                return
            x = torch.stack(batch).to(self.device)
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    scores = self.model(x).detach().cpu().numpy().tolist()
            else:
                scores = self.model(x).detach().cpu().numpy().tolist()
            for (idx, _path), sc in zip(metas, scores):
                results[idx] = float(sc)
            batch.clear()
            metas.clear()

        results = [None] * len(image_paths)
        for i, p in enumerate(image_paths):
            try:
                img = Image.open(p).convert("RGB")
                t = self.preprocess(img)
                batch.append(t)
                metas.append((i, p))
                if len(batch) >= batch_size:
                    flush()
            except Exception as e:
                print(f"[WARN] 跳过 {p}: {e}")
                results[i] = None
        flush()
        return results

    def grad_heatmap(self, image_path_or_pil, smooth=25, noise=0.15):
        """
        SmoothGrad 风格的输入梯度热力图（适用于 ViT，实时可用）
        返回: (score, heatmap_bgr_np)
        """
        # 前向拿分
        if isinstance(image_path_or_pil, (str, os.PathLike)):
            raw = Image.open(image_path_or_pil).convert("RGB")
        else:
            raw = image_path_or_pil.convert("RGB")

        x = self.preprocess(raw).unsqueeze(0).to(self.device).requires_grad_(True)
        # 关闭回归头的 dropout 影响
        self.model.eval()

        # 仅对回归头开启梯度；CLIP 编码固定（上面 forward 已经 no_grad）
        # 这里做一个“近似”的输入梯度：直接对输出 w.r.t. x 求梯度
        score = self.model(x)
        s = float(score.item())

        # SmoothGrad
        grads = []
        for _ in range(max(1, smooth)):
            noise_t = (torch.randn_like(x) * noise) if noise > 0 else 0
            x_noisy = (x + noise_t).clamp(0, 1).detach().requires_grad_(True)
            y = self.model(x_noisy)
            self.model.zero_grad(set_to_none=True)
            y.backward(torch.ones_like(y))
            g = x_noisy.grad.detach()
            grads.append(g)
        g = torch.stack(grads, dim=0).mean(0)  # [1,3,H,W]
        g = g.abs().mean(1, keepdim=True)      # [1,1,H,W] channel-avg
        g = g[0,0]
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        g_np = (g.detach().cpu().numpy() * 255).astype(np.uint8)

        # 放到原图大小
        h, w = raw.size[1], raw.size[0]
        heat = cv2.resize(g_np, (w, h), interpolation=cv2.INTER_CUBIC)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # 混合显示
        raw_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(raw_bgr, 0.5, heat_color, 0.5, 0)
        return s, overlay


# =========================== 2. 文件夹批量处理 ===========================
import pandas as pd

class FolderProcessor:
    def __init__(self, model_path):
        self.predictor = AestheticPredictor(model_path)
        self.supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def process_folder(self, folder_path, output_csv=None, min_score=0.0, max_score=1.0):
        files = []
        for root, _, fs in os.walk(folder_path):
            for f in fs:
                if Path(f).suffix.lower() in self.supported:
                    files.append(os.path.join(root, f))
        if not files:
            print("❌ 未找到图片")
            return None

        scores = self.predictor.predict_batch(files)
        rows = []
        for p, s in zip(files, scores):
            if s is None:
                continue
            if min_score <= s <= max_score:
                rows.append({
                    "file_path": p,
                    "filename": os.path.basename(p),
                    "aesthetic_score": float(s),
                    "score_category": self._bucket(s)
                })
        df = pd.DataFrame(rows).sort_values("aesthetic_score", ascending=False).reset_index(drop=True)
        if output_csv and not df.empty:
            df.to_csv(output_csv, index=False, encoding="utf-8")
            print("✅ 已保存:", output_csv)
        return df

    @staticmethod
    def _bucket(s):
        return "优秀" if s >= 0.8 else ("良好" if s >= 0.6 else ("一般" if s >= 0.4 else "较差"))


# =========================== 3. Web API（可选） ===========================
from flask import Flask, request, jsonify

class AestheticAPI:
    def __init__(self, model_path):
        self.app = Flask(__name__)
        self.predictor = AestheticPredictor(model_path)

        @self.app.route("/predict", methods=["POST"])
        def predict():
            if "image" not in request.files:
                return jsonify({"error": "缺少 image 文件"}), 400
            img = Image.open(request.files["image"].stream).convert("RGB")
            s = self.predictor.predict_single(img)
            return jsonify({
                "aesthetic_score": s,
                "score_category": FolderProcessor._bucket(s)
            })

        @self.app.route("/predict_heat", methods=["POST"])
        def predict_heat():
            if "image" not in request.files:
                return jsonify({"error": "缺少 image 文件"}), 400
            img = Image.open(request.files["image"].stream).convert("RGB")
            s, overlay = self.predictor.grad_heatmap(img)
            _, buf = cv2.imencode(".jpg", overlay)
            b64 = base64.b64encode(buf).decode("ascii")
            return jsonify({
                "aesthetic_score": s,
                "score_category": FolderProcessor._bucket(s),
                "heatmap_jpg_base64": b64
            })

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "device": str(self.predictor.device)})

    def run(self, host="0.0.0.0", port=5000, debug=False):
        print(f"🚀 API: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# =========================== 4. 实时摄像头评分（可选） ===========================
class RealTimeAesthetic:
    def __init__(self, model_path):
        self.predictor = AestheticPredictor(model_path)
        self.cap = None

    def start(self, cam_id=0, snap_thresh=0.5, out_dir="snapshots"):
        os.makedirs(out_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        n = 0
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            s = self.predictor.predict_single(pil)
            cv2.putText(frame, f"Score:{s:.3f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            bar = int(min(max(s,0.0),1.0) * 300)
            cv2.rectangle(frame, (10,40), (10+bar,60), (0,255,0), -1)
            cv2.rectangle(frame, (10,40), (310,60), (255,255,255), 1)

            if s >= snap_thresh:
                ts = int(time.time())
                path = os.path.join(out_dir, f"{ts}_score{s:.2f}.jpg")
                cv2.imwrite(path, frame)

            cv2.imshow("Aesthetic RT", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('h'):
                _, overlay = self.predictor.grad_heatmap(pil)
                cv2.imshow("heat", overlay)
            n += 1
        self.cap.release()
        cv2.destroyAllWindows()


# =========================== 5. CLI 演示 ===========================
def main():
    model_path = r"D:\plug\AirSim\Scripts\shot genie\ue_bridge\bridge\1best_aesthetic_model.pth"
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在:", model_path)
        return
    print("选择功能: 1单图 2文件夹 3API 4实时摄像头")
    c = input("输入 1/2/3/4: ").strip()
    if c == "1":
        pred = AestheticPredictor(model_path)
        path = input("图片路径: ").strip().strip('"')
        s = pred.predict_single(path)
        print("Score:", s)
        want_heat = input("输出热力图? y/n: ").lower().startswith("y")
        if want_heat:
            s, overlay = pred.grad_heatmap(path)
            print("Score:", s)
            cv2.imwrite("heat_overlay.jpg", overlay)
            print("已保存 heat_overlay.jpg")
    elif c == "2":
        proc = FolderProcessor(model_path)
        folder = input("文件夹路径: ").strip().strip('"')
        outcsv = os.path.join(folder, "aesthetic_scores.csv")
        df = proc.process_folder(folder, outcsv)
        if df is not None:
            print(df.head())
    elif c == "3":
        api = AestheticAPI(model_path)
        api.run(debug=True)
    elif c == "4":
        rt = RealTimeAesthetic(model_path)
        rt.start()
    else:
        print("无效选择")

if __name__ == "__main__":
    # 友情提示：部分功能需要
    # pip install flask requests opencv-python
    main()
