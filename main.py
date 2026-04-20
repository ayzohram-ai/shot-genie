# -*- coding: utf-8 -*-
"""
main.py
顶层入口：应用 msgpack 补丁 -> 连接 AirSim -> 启动视频/指令线程 -> 阻塞等待退出。
"""

import argparse
import threading
import time

import patches_msgpack
patches_msgpack.apply()  # 必须在 import airsim 之前

from client import connect_multirotor
from video_stream import video_push_loop
from cmd_loop import cmd_pull_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recv-ip", default="10.4.168.145")
    ap.add_argument("--vport", type=int, default=5555)
    ap.add_argument("--cport", type=int, default=5556)
    ap.add_argument("--cam", default="front_center")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--downscale", type=float, default=0.75)
    ap.add_argument("--jpgq", type=int, default=70)
    ap.add_argument("--z-hold", type=float, default=-2.0)
    ap.add_argument("--deadman", type=float, default=1.2)
    ap.add_argument("--poll-hz", type=float, default=20.0)
    ap.add_argument("--sticky", action="store_true", help="一次 nav 后持续执行，直到 hover/land/takeoff")
    ap.add_argument("--sticky-hold", action="store_true", help="结合“X秒”持续时间，自动定时结束")
    ap.add_argument("--debug", action="store_true", help="打印 RAW / PARSED / EXEC")
    args = ap.parse_args()

    push_addr = f"tcp://{args.recv_ip}:{args.vport}"   # UE PUSH -> receiver PULL.bind
    pull_bind = f"tcp://0.0.0.0:{args.cport}"          # UE PULL.bind <- receiver PUSH.connect

    cli = connect_multirotor()

    stop_evt = threading.Event()
    th_video = threading.Thread(
        target=video_push_loop,
        args=(stop_evt, cli, push_addr, args.cam, args.fps, args.downscale, args.jpgq),
        daemon=True
    )
    th_cmd = threading.Thread(
        target=cmd_pull_loop,
        args=(stop_evt, cli, pull_bind, args.z_hold, args.deadman, args.poll_hz,
              args.sticky, args.sticky_hold, args.debug),
        daemon=True
    )
    th_video.start(); th_cmd.start()
    print(f"[UE] Running. Video->{push_addr} | Cmd<-{pull_bind} | sticky={args.sticky} sticky_hold={args.sticky_hold} debug={args.debug}")

    try:
        while th_video.is_alive() and th_cmd.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        th_video.join(1.0); th_cmd.join(1.0)
        print("[UE] Bye.")

if __name__ == "__main__":
    main()
