# -*- coding: utf-8 -*-
"""
路径B：不依赖外网与大模型的占位评分器版本
流程：AirSim 连接 -> 起飞 -> YOLO找人并居中 -> 抓拍 -> 启发式评分（亮度+对比度+人脸面积）
依赖：airsim, ultralytics, opencv-python, numpy
"""

import os
import time
import math
import json
from datetime import datetime

import numpy as np
import cv2

# ---- 如果你的 AirSim 安装在非默认路径，这里按需调整 ----
import airsim

# ========== 可调参数 ==========
ALTITUDE = -2.0                 # 起飞目标高度（NED坐标：向下为正，所以-2米是离地2米）
CENTER_BAND = 0.03              # 构图达标阈值（归一化像素）
CTRL_DT = 0.10                  # 控制周期(s)
MAX_VX = 1.0                    # 前后最大速度（本脚本仅横移与抓拍，vx只用于微调）
MAX_VY = 1.0                    # 侧向最大速度
K_EX_TO_VY = 2.0                # ex -> vy 比例 (人越靠右，vy越大向右飞)
CONF_MIN = 0.40                 # 最低置信度（低于仅控制不抓拍）
TIMEOUT_FIND = 15.0             # 寻人超时(s)
TIMEOUT_ALIGN = 12.0            # 构图对齐超时(s)

# YOLO 权重（本地路径）。如果 ultralytics 自带权重已缓存，可留空让其默认下载/读取（建议填本地避免联网）
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "").strip()  # 例如: r"D:\models\yolo\yolov8n.pt"

# 输出文件
OUT_DIR = os.path.join(os.getcwd(), "shooting", "photo")
OUT_IMG = os.path.join(OUT_DIR, "center_composition.jpg")
OUT_CSV = os.path.join(os.getcwd(), "results_fallback.csv")


# ========== 简单的离线“美学”占位评分器 ==========
def simple_aesthetic_score(img_path: str) -> float:
    """
    一个不依赖网络的启发式评分：亮度(45%) + 对比度(35%) + 人脸面积(20%)
    返回 0~100
    """
    img = cv2.imread(img_path)
    if img is None:
        return 50.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bright = np.clip(gray.mean() / 255.0, 0, 1)      # 平均亮度
    contrast = np.clip(gray.std() / 128.0, 0, 1)     # 亮度标准差近似对比度

    # 粗糙人脸面积比重（用于人像构图奖励）
    face_score = 0.0
    try:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = casc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 0:
            h, w = gray.shape[:2]
            areas = [(fw * fh) / (w * h) for (fx, fy, fw, fh) in faces]
            face_score = np.clip(max(areas) * 4.0, 0, 1)  # 面积放大一些
    except Exception:
        pass

    score = 100.0 * (0.45 * bright + 0.35 * contrast + 0.20 * face_score)
    return float(np.clip(score, 0, 100))


# ========== YOLO 人体检测 ==========
class YoloPerson:
    def __init__(self, weights=None, conf=0.25, device="cpu"):
        from ultralytics import YOLO
        if weights and os.path.isfile(weights):
            self.model = YOLO(weights)
        else:
            # 无权重路径则用默认（可能尝试联网缓存，建议提供本地路径）
            self.model = YOLO("yolov8n.pt")
        self.conf = float(conf)
        self.device = device

    def detect_best_person(self, image_bgr: np.ndarray):
        rs = self.model.predict(image_bgr, conf=self.conf, device=self.device, verbose=False)
        if not rs:
            return None
        r = rs[0]
        if r.boxes is None or r.boxes.xyxy is None or len(r.boxes.xyxy) == 0:
            return None
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        # coco: person 类别 id=0
        mask = (clss == 0) & (confs >= self.conf)
        if not np.any(mask):
            return None
        boxes, confs = boxes[mask], confs[mask]
        i = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[i].tolist()
        return {"bbox": [x1, y1, x2, y2], "conf": float(confs[i])}


def bbox_center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


# ========== AirSim 实用函数 ==========
def connect_airsim():
    cli = airsim.MultirotorClient()
    cli.confirmConnection()
    cli.enableApiControl(True)
    cli.armDisarm(True)
    return cli

def takeoff_and_hold(cli: airsim.MultirotorClient, z_hold=ALTITUDE, timeout_sec=8.0):
    # 起飞到近地，随后移动到目标高度
    cli.takeoffAsync(timeout_sec=timeout_sec).join()
    # 使用 moveToZ 控制高度（NED：向下为正，所以负数是抬升）
    cli.moveToZAsync(z_hold, velocity=1.5).join()
    # 悬停一下
    cli.hoverAsync().join()

def get_rgb_frame(cli: airsim.MultirotorClient):
    """
    返回 (image_bgr, W, H)，获取失败返回 (None, 0, 0)
    """
    rs = cli.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    if not rs or rs[0].width == 0:
        return None, 0, 0
    scene = rs[0]
    img1d = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
    if img1d.size != scene.height * scene.width * 3:
        return None, 0, 0
    image_rgb = img1d.reshape(scene.height, scene.width, 3)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, scene.width, scene.height

def save_current_frame(cli: airsim.MultirotorClient, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rs = cli.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    if not rs or rs[0].width == 0:
        raise RuntimeError("simGetImages failed")
    scene = rs[0]
    img1d = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
    if img1d.size != scene.height * scene.width * 3:
        raise RuntimeError("image buffer shape mismatch")
    image_rgb = img1d.reshape(scene.height, scene.width, 3)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(out_path, image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out_path}")
    return out_path


# ========== 居中控制 ==========
def center_person_and_capture(cli: airsim.MultirotorClient,
                              det: YoloPerson,
                              save_path: str,
                              center_band=CENTER_BAND,
                              ctrl_dt=CTRL_DT,
                              timeout_find=TIMEOUT_FIND,
                              timeout_align=TIMEOUT_ALIGN,
                              conf_min=CONF_MIN):
    """
    循环：先寻人（timeout_find），再对齐（timeout_align），对齐成功抓拍图片保存到 save_path
    """
    t0 = time.time()
    print("[CENTER] 寻找人物中 ...")
    while True:
        if (time.time() - t0) > timeout_find:
            raise TimeoutError("找人超时")

        frame, W, H = get_rgb_frame(cli)
        if frame is None:
            cli.hoverAsync().join()
            time.sleep(ctrl_dt)
            continue

        hit = det.detect_best_person(frame)
        if hit:
            print(f"[CENTER] 人物检测到，置信度={hit['conf']:.2f}，开始对齐 ...")
            break

        cli.hoverAsync().join()
        time.sleep(ctrl_dt)

    # 对齐阶段
    t1 = time.time()
    while True:
        if (time.time() - t1) > timeout_align:
            print("[CENTER][WARN] 对齐超时，执行兜底抓拍")
            cli.hoverAsync().join()
            time.sleep(0.2)
            save_current_frame(cli, save_path)
            print(f"[CAPTURE] 已保存 -> {save_path}")
            return save_path

        frame, W, H = get_rgb_frame(cli)
        if frame is None:
            cli.hoverAsync().join()
            time.sleep(ctrl_dt)
            continue

        hit = det.detect_best_person(frame)
        if not hit:
            print("[CENTER] 本帧未检出，保持悬停")
            cli.hoverAsync().join()
            time.sleep(ctrl_dt)
            continue

        (x1, y1, x2, y2) = hit["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ex = (cx - W / 2.0) / W            # 右为正, 约[-0.5,0.5]
        ey = (cy - H / 2.0) / H            # 下为正, 约[-0.5,0.5]

        # 侧向控制（vy）：把人推到中心
        err_x = -ex   # 人在右( ex>0 ) -> vy>0，向右飞，使人回到中心
        vy = float(np.clip(K_EX_TO_VY * err_x, -MAX_VY, MAX_VY))

        # 这里简单化：仅横移；高度和朝向保持不变（如果需要可加 ey->dz/yaw 控制）
        vx = 0.0
        yaw_rate = 0.0

        # 达标判定：x方向进入阈值
        ok_x = (abs(ex) < center_band)

        if ok_x and hit["conf"] >= conf_min:
            cli.hoverAsync().join()
            time.sleep(0.25)  # 防抖
            save_current_frame(cli, save_path)
            print(f"[CAPTURE] 居中达标并已保存 -> {save_path}")
            return save_path

        # 发送一次速度控制（固定高度）
        # 获取当前姿态高度
        state = cli.getMultirotorState()
        z_hold = state.kinematics_estimated.position.z_val
        cli.moveByVelocityZAsync(vx=vx, vy=vy, z=z_hold, duration=ctrl_dt, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()
        time.sleep(ctrl_dt)


def main():
    print("[MAIN] 连接 AirSim ...")
    cli = connect_airsim()
    print("[MAIN] 起飞 ...")
    takeoff_and_hold(cli, ALTITUDE)

    print("[MAIN] 初始化 YOLO ...")
    det = YoloPerson(weights=YOLO_WEIGHTS or None, conf=0.25, device="cpu")

    print("[MAIN] 寻人并居中抓拍 ...")
    img_path = center_person_and_capture(cli, det, save_path=OUT_IMG)

    print("[MAIN] 启发式离线评分 ...")
    score = simple_aesthetic_score(img_path)
    print(f"[AESTHETIC-FALLBACK] {img_path} -> score={score:.2f}")

    # 追加记录到 CSV（方便后续比对人类模型/占位模型）
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": img_path,
        "score": f"{score:.2f}",
        "note": "fallback_heuristic_brightness_contrast_face"
    }
    write_header = not os.path.exists(OUT_CSV)
    import csv
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[CSV] 记录写入：{OUT_CSV}")

    print("[MAIN] 完成。悬停中。")
    cli.hoverAsync().join()
    # 如需降落：
    # cli.landAsync().join()
    # cli.armDisarm(False); cli.enableApiControl(False)


if __name__ == "__main__":
    main()
