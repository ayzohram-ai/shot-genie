# -*- coding: utf-8 -*-
"""
cmd_loop.py —— 精简主循环（薄壳）
依赖：
  - handlers.py（路由、执行器、上下文、ZMQ“取最新”）
  - media_io.py（照片/视频I/O）
保持原有 AirSim / ZMQ 行为与消息协议不变。
"""

import time
import airsim
from handlers import (
    Context,
    ZmqPullServer,
    dispatch_json,
    tick_sticky_and_deadman,
)


def cmd_pull_loop(
    stop_evt,
    cli,
    bind_addr: str,
    z_init: float = -2.0,
    deadman_sec: float = 1.2,
    poll_hz: float = 20.0,
    sticky: bool = True,
    sticky_hold: bool = True,
    debug: bool = False
):
    server = ZmqPullServer(bind_addr)
    print("[UE/Cmd] PULL bind at", bind_addr)

    ctx = Context(
        cli=cli,
        z_hold=z_init,
        debug=debug,
        deadman_sec=deadman_sec,
        sticky_enabled=sticky,
        sticky_hold=sticky_hold,
    )

    dt = 1.0 / max(1.0, float(poll_hz))
    last_cmd_ts = time.time()

    print("[UE/Cmd] loop start (debug=%s, sticky=%s, sticky_hold=%s)" %
          ("on" if debug else "off", "on" if sticky else "off", "on" if sticky_hold else "off"))

    while not stop_evt.is_set():
        t0 = time.time()

        s = server.recv_latest()  # 只取最新一条
        if s:
            if dispatch_json(s, ctx):
                last_cmd_ts = time.time()

        # 粘滞 + deadman
        tick_sticky_and_deadman(ctx, last_cmd_ts)

        # 节拍对齐
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("[UE/Cmd] loop stop")
