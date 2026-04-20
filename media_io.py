# -*- coding: utf-8 -*-
"""
media_io.py —— 照片/视频 I/O 工具
统一：
- AirSim 图像抓取为 bytes（内部已加 rpc_lock）
- 保存 bytes 到文件
- OpenCV 视频写入器
- 从 AirSim 捕获帧并写入视频（内部已加 rpc_lock）
"""

import os
import cv2
import numpy as np
import airsim

# 统一 RPC 互斥锁（和 handlers/拍照模板等共用同一把锁）
from client import rpc_lock


def ensure_dirs(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _get_scene_image(cli: airsim.MultirotorClient, camera="0", compress=True):
    """
    抓取一帧 Scene 图像。
    - compress=True: 返回编码后的 PNG/JPEG 字节（image_data_uint8 可直接 imdecode）
    - compress=False: 返回解码后的 BGR ndarray
    """
    req = [airsim.ImageRequest(str(camera), airsim.ImageType.Scene, pixels_as_float=False, compress=compress)]
    with rpc_lock:
        responses = cli.simGetImages(req)

    if not responses or len(responses) == 0:
        return None

    img = responses[0]
    if compress:
        # 压缩：直接返回编码后的 PNG/JPEG 字节
        data = img.image_data_uint8
        if data is None:
            return None
        return bytes(data)
    else:
        # 非压缩：需要根据宽高重塑为 (H, W, 3) 再返回 BGR
        if img.width <= 0 or img.height <= 0:
            return None
        arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        if arr.size != img.width * img.height * 3:
            return None
        rgb = arr.reshape((img.height, img.width, 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr


def capture_photo_bytes_from_sim(cli: airsim.MultirotorClient, camera="0") -> bytes:
    """
    返回可直接保存到 jpg/png 的字节串。
    - 优先用 compress=True 拿到 PNG/JPEG bytes。
    - 若失败再回退到非压缩帧并本地编码为 JPEG。
    """
    # 尝试直接拿压缩字节
    data = _get_scene_image(cli, camera=camera, compress=True)
    if data:
        return data

    # 回退：拿非压缩帧并编码为 jpg
    frame = _get_scene_image(cli, camera=camera, compress=False)
    if frame is None:
        print("[PHOTO][ERROR] 无法从 AirSim 获取图像数据")
        return b""
    ok, enc = cv2.imencode(".jpg", frame)
    if not ok:
        print("[PHOTO][ERROR] 本地编码 JPEG 失败")
        return b""
    return enc.tobytes()


def save_bytes(data: bytes, path: str):
    if not data:
        print(f"[PHOTO][WARN] 空数据，跳过保存: {path}")
        return
    ensure_dirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)
    print(f"[PHOTO] 已保存: {path}")


def start_video_recording(video_path: str, fps: int = 20, frame_size=(640, 480)) -> cv2.VideoWriter:
    """
    创建一个 OpenCV VideoWriter。
    注意：frame_size 需要和写入帧尺寸匹配（capture_and_write_frame 会自动 resize）
    """
    ensure_dirs(os.path.dirname(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    print(f"[VIDEO] 开始录制 -> {video_path}")
    return vw


def stop_video_recording(vw: cv2.VideoWriter):
    if vw is not None:
        vw.release()
    print("[VIDEO] 停止录制")


def capture_and_write_frame(cli: airsim.MultirotorClient, vw: cv2.VideoWriter, camera="0"):
    """
    从 AirSim 抓取一帧（压缩 PNG/JPEG 字节），解码为 BGR，再写入 VideoWriter。
    内部 RPC 已加锁。若帧尺寸和 writer 不一致则进行 resize。
    """
    data = _get_scene_image(cli, camera=camera, compress=True)
    if not data:
        print("[VIDEO][WARN] 空帧/抓帧失败")
        return
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("[VIDEO][WARN] 解码失败，丢弃该帧")
        return

    # 若尺寸不匹配，按 writer 目标尺寸缩放
    w = int(vw.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w > 0 and h > 0 and (frame.shape[1] != w or frame.shape[0] != h):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    vw.write(frame)
