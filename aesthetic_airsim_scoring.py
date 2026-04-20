# -*- coding: utf-8 -*-
"""
aesthetic_airsim_scoring.py — 稳健评分/热力图/快照
- 延迟导入 AirSim，连接兜底
- 实时取帧→打分；≥阈值自动快照 + 叠加热力图
- CSV 记录
"""
import os, sys, time, argparse
import numpy as np
import cv2
from typing import Optional
from collections import deque
from PIL import Image

# ---- quiet noisy torch dispatch / jit logs on Windows ----
for _k in [
    "PYTORCH_SHOW_DISPATCH_TRACE",
    "TORCH_SHOW_DISPATCH_TRACE",
    "TORCH_LOGS",
    "PYTORCH_JIT_LOG_LEVEL",
    "PYTORCH_JIT_ENABLE_PROFILER",
    "TORCH_SHOW_CPP_STACKTRACES",
]:
    if os.environ.get(_k):
        os.environ.pop(_k, None)
try:
    import torch
    # 关闭内部 dispatch 打印（如存在）
    if hasattr(torch._C, "_dispatch_set_log_level"):
        torch._C._dispatch_set_log_level(0)
except Exception:
    pass


# 强制先用 CPU，稳定后可显式设：set AESTHETIC_DEVICE=cuda:0
os.environ.setdefault("AESTHETIC_DEVICE", "cpu")


# 延迟导入 runtime（避免导入期崩溃）
import importlib
_rt = importlib.import_module("aesthetic_runtime")
score_image_from_pil = _rt.score_image_from_pil
get_model_and_preprocess = _rt.get_model_and_preprocess
overlay_heatmap_bgr = _rt.overlay_heatmap_bgr

PRINT_EVERY = 0.5


class AestheticAirSimScorer:
    def __init__(self,
                 z_agl: float = -1.5,
                 fps: float = 3.0,
                 avg: int = 10,
                 show: bool = False,
                 csv: Optional[str] = None,
                 duration: Optional[float] = None,
                 takeoff: bool = True,
                 thresh: float = 0.5,
                 snap: bool = True,
                 snap_dir: Optional[str] = "snapshots",
                 cooldown: float = 2.0,
                 cam_name: str = "0"):
        self.z_agl = float(z_agl)
        self.period = 1.0 / max(0.1, float(fps))
        self.q = deque(maxlen=max(1, int(avg)))
        self.show = bool(show)
        self.csv = csv
        self.duration = duration
        self.takeoff = takeoff
        self.cam_name = str(cam_name)
        self.thresh = float(thresh)
        self.snap = bool(snap)
        self.snap_dir = snap_dir
        self.cooldown = float(cooldown)

        self._stop = False
        self._last_score_t = 0.0
        self._last_log_t = 0.0
        self._last_snap_t = 0.0

        # 加载模型（无 warmup）
        print("🧠 Loading aesthetic model (safe, CPU first)...")
        get_model_and_preprocess()
        print("✅ Aesthetic model ready.")

        # 延迟导入 AirSim
        self._airsim_ok, self.client, self._airsim = False, None, None
        try:
            import airsim
            self._airsim = airsim
            self.client = airsim.MultirotorClient()
            try:
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self._airsim_ok = True
                print("✅ AirSim connected.")
            except Exception as e:
                print(f"[AESTHETIC][WARN] AirSim confirmConnection failed: {e}")
        except Exception as e:
            print(f"[AESTHETIC][WARN] AirSim import/init failed: {e}")

        # CSV 初始化
        if self.csv:
            need_header = (not os.path.exists(self.csv)) or (os.path.getsize(self.csv) == 0)
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.csv)) or ".", exist_ok=True)
                if need_header:
                    with open(self.csv, "w", encoding="utf-8", newline="") as f:
                        f.write("ts,score,avg,abs_path\n")
            except Exception as e:
                print(f"[AESTHETIC][CSV][WARN] init failed: {e}")
                self.csv = None

    def _append_csv(self, score: float, avg: float, abs_path: Optional[str] = None):
        if not self.csv: return
        try:
            with open(self.csv, "a", encoding="utf-8", newline="") as f:
                f.write(f"{time.time():.3f},{score:.4f},{avg:.4f},{abs_path or ''}\n")
        except Exception as e:
            print(f"[AESTHETIC][CSV][WARN] {e}")

    def request_stop(self):
        print("🛑 stop requested"); self._stop = True

    def run(self):
        if not self._airsim_ok:
            print("[AESTHETIC][ERR] AirSim not connected; abort.")
            return

        t0 = time.time()
        try:
            if self.takeoff:
                print("🛫 Takeoff..."); self.client.takeoffAsync().join()
                print(f"↕️ MoveToZ={self.z_agl:.2f}"); self.client.moveToZAsync(self.z_agl, 1.0).join()
            else:
                print("🚁 Skip takeoff.")

            print("✅ Start aesthetic scoring loop...")
            while True:
                if self._stop: print("⛔ stop flag"); break
                if self.duration and (time.time() - t0) >= self.duration:
                    print("⏱ duration reached"); break

                now = time.time()
                if (now - self._last_score_t) < self.period:
                    time.sleep(0.01); continue
                self._last_score_t = now

                # 取图
                try:
                    rs = self.client.simGetImages([
                        self._airsim.ImageRequest(self.cam_name, self._airsim.ImageType.Scene, False, False)
                    ])
                except Exception as e:
                    print(f"[AESTHETIC][WARN] RPC: {e}"); time.sleep(0.2); continue
                if not rs or rs[0] is None:
                    time.sleep(0.03); continue

                scene = rs[0]
                img1d = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
                expected = scene.height * scene.width * 3
                if img1d.size != expected:
                    time.sleep(0.03); continue

                image_rgb = img1d.reshape(scene.height, scene.width, 3)
                pil_img = Image.fromarray(np.ascontiguousarray(image_rgb))
                frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # 打分 & 热力图（只在需要保存/显示时计算 heat）
                res = score_image_from_pil(pil_img, use_gradcam=False, want_heat=True)
                score = float(res.get("score", 0.0))
                heat = res.get("heatmap224", None)  # numpy [224,224] or None

                self.q.append(score)
                avg = sum(self.q) / len(self.q)

                if (now - self._last_log_t) >= PRINT_EVERY:
                    self._last_log_t = now
                    print(f"[AESTHETIC] score={score:.2f}  avg@{len(self.q)}={avg:.2f}")

                self._append_csv(score, avg)

                # 快照（含热力图）
                if score >= self.thresh and self.snap and (now - self._last_snap_t > self.cooldown):
                    os.makedirs(self.snap_dir, exist_ok=True)
                    base = os.path.join(self.snap_dir, f"{int(time.time())}_score{score:.2f}")
                    img_path = base + ".jpg"
                    cv2.imwrite(img_path, frame_bgr)
                    if heat is not None:
                        over = overlay_heatmap_bgr(frame_bgr, heat)
                        cv2.imwrite(base + "_heat.jpg", over)
                    # 写一份旁路 JSON
                    try:
                        import json
                        with open(base + ".json", "w", encoding="utf-8") as f:
                            json.dump({
                                "score": score, "avg": avg, "ts": time.time(),
                                "image": os.path.abspath(img_path)
                            }, f, ensure_ascii=False, indent=2)
                    except Exception: pass
                    print(f"📸 snapshot -> {img_path}")
                    self._last_snap_t = now

                # 叠字显示
                if self.show:
                    overlay = frame_bgr.copy()
                    text = f"score={score:.2f} avg={avg:.2f}"
                    cv2.rectangle(overlay, (0,0), (overlay.shape[1], 40), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, dst=frame_bgr)
                    cv2.putText(frame_bgr, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    if heat is not None:
                        frame_show = overlay_heatmap_bgr(frame_bgr, heat, alpha=0.35)
                    else:
                        frame_show = frame_bgr
                    cv2.imshow("aesthetic_score", frame_show)
                    try:
                        if (cv2.waitKey(1) & 0xFF) == 27 or \
                           cv2.getWindowProperty("aesthetic_score", cv2.WND_PROP_VISIBLE) < 1:
                            print("🧹 ESC/close"); break
                    except cv2.error:
                        self.show = False

            # 收尾
            try:
                self.client.moveByVelocityAsync(0,0,0,0.3).join()
                self.client.hoverAsync().join()
            except Exception: pass

        except KeyboardInterrupt:
            print("🧹 KeyboardInterrupt")
        finally:
            try:
                if self.show: cv2.destroyAllWindows()
            except Exception: pass
            print("🏁 Scoring session end.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=-1.5)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--avg", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--no-takeoff", action="store_true")
    ap.add_argument("--cam", type=str, default="0")
    ap.add_argument("--thresh", type=float, default=0.5)
    return ap.parse_args()


if __name__ == "__main__":
    import traceback
    try:
        args = parse_args()
        scorer = AestheticAirSimScorer(
            z_agl=args.z, fps=args.fps, avg=args.avg, show=args.show,
            csv=args.csv, duration=args.duration, takeoff=(not args.no_takeoff),
            cam_name=args.cam, thresh=args.thresh,
        )
        scorer.run()
    except SystemExit:
        raise
    except Exception as e:
        print("[FATAL]", repr(e)); traceback.print_exc(); raise
