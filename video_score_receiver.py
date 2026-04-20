# -*- coding: utf-8 -*-
"""
video_score_receiver.py
绑定 ZMQ PULL，接收 UE 侧发送的 JPEG 帧，抽帧做美学评分并平滑展示。

示例：
  python video_score_receiver.py --bind tcp://0.0.0.0:5555 --fps 3 --avg 10
"""

import argparse
import time
from collections import deque

import zmq
import numpy as np
import cv2
from PIL import Image

from aesthetic_runtime import score_image_from_pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://0.0.0.0:5555", help="PULL.bind address")
    ap.add_argument("--fps", type=float, default=3.0, help="scoring fps (抽帧频率)")
    ap.add_argument("--avg", type=int, default=10, help="moving average window")
    ap.add_argument("--show", action="store_true", help="show window with score overlay")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 200)
    pull.bind(args.bind)
    print(f"[RECV] PULL.bind <- {args.bind}")

    last_score_time = 0.0
    period = 1.0 / max(0.1, args.fps)
    q = deque(maxlen=max(1, args.avg))

    try:
        while True:
            try:
                jb = pull.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue

            now = time.time()
            if (now - last_score_time) < period:
                continue
            last_score_time = now

            arr = np.frombuffer(jb, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            try:
                res = score_image_from_pil(pil_img, use_gradcam=False)
                q.append(res["score"])  # type: ignore
                avg = sum(q) / len(q)
                print(f"[AESTHETIC][video] score={res['score']:.2f} avg@{len(q)}={avg:.2f}")

                if args.show:
                    h, w = frame_bgr.shape[:2]
                    overlay = frame_bgr.copy()
                    text = f"score={res['score']:.2f} avg={avg:.2f}"
                    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
                    cv2.putText(frame_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("video_score", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            except Exception as e:
                print(f"[AESTHETIC][video][ERR] {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        pull.close(0)
        print("[RECV] exit")


if __name__ == "__main__":
    main()


