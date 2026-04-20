# -*- coding: utf-8 -*-
"""
video_stream.py
负责视频采集、JPEG 编码、自适应降载（质量/降采样/帧率），并通过 ZMQ PUSH 发送。
"""

import time
import zmq
import numpy as np
import cv2
from client import rpc_lock

def video_push_loop(stop_evt, cli, push_addr, cam, fps_init, downscale_init, jpg_quality_init):
    import airsim  # 延迟导入
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(push_addr)
    print("[UE/Video] PUSH ->", push_addr)

    fps = max(8, int(fps_init))
    downscale = float(downscale_init)
    jpg_quality = int(jpg_quality_init)

    MIN_FPS = 8; MIN_Q = 30; MIN_DS = 0.4
    LARGE_MB_WARN, LARGE_MB_HARD = 5.0, 8.0
    period = 1.0/max(1,fps)
    t_stat = time.time(); sent=dropped=0; consec_fail=0

    while not stop_evt.is_set():
        t0 = time.time()
        try:
            with rpc_lock:
                buf = cli.simGetImage(cam, airsim.ImageType.Scene)
            if not buf:
                time.sleep(0.01); continue
            arr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                consec_fail += 1
                if consec_fail>=6:
                    print("⚠️ [UE/Video] decode fail ×6; consider lowering camera resolution in settings.json")
                    consec_fail=0
                time.sleep(0.02); continue
            consec_fail = 0

            if 0.0 < downscale < 1.0:
                img = cv2.resize(img, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

            h,w = img.shape[:2]
            if max(h,w) > 1920:
                scale = 1920/max(h,w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            if not ok:
                time.sleep(0.01); continue
            jb = jpg.tobytes(); sz = len(jb)/(1024*1024)

            if sz > LARGE_MB_WARN:
                if sz > LARGE_MB_HARD and jpg_quality > MIN_Q:
                    new_q = max(MIN_Q, jpg_quality-20)
                    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(new_q)])
                    jb = jpg.tobytes(); sz = len(jb)/(1024*1024)
                    jpg_quality = new_q
                    print(f"[UE/Video] HARD q->{new_q}, size={sz:.2f}MB")
                elif jpg_quality > MIN_Q:
                    new_q = max(MIN_Q, jpg_quality-10)
                    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(new_q)])
                    jb = jpg.tobytes(); sz = len(jb)/(1024*1024)
                    jpg_quality = new_q
                    print(f"[UE/Video] q->{new_q}, size={sz:.2f}MB")
                elif downscale > MIN_DS:
                    downscale = max(MIN_DS, downscale-0.1)
                    print(f"[UE/Video] ds->{downscale:.2f}")
                elif fps > MIN_FPS:
                    fps = max(MIN_FPS, int(fps*0.8))
                    period = 1.0/fps
                    print(f"[UE/Video] fps->{fps}")

            try:
                sock.send(jb, flags=zmq.NOBLOCK); sent+=1
            except zmq.Again:
                dropped+=1

            if time.time()-t_stat >= 2.0:
                print(f"[UE/Video] sent={sent} dropped={dropped} q={jpg_quality} ds={downscale:.2f} fps={fps}")
                sent=dropped=0; t_stat=time.time()

        except Exception:
            time.sleep(0.03)

        dt = time.time()-t0
        if dt < period:
            time.sleep(period-dt)

    print("[UE/Video] Stopped.")
