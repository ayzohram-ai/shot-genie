# -*- coding: utf-8 -*-
"""
states.py
定义三种运行态：
- TrackState：基于人像检测的视觉追踪 PD
- CalibState：原地自转的人像校准
- StickyState：粘滞/低通执行器，支持“X秒”持续与平滑过渡
"""

from dataclasses import dataclass, field
import time
import numpy as np
from typing import Optional

@dataclass
class TrackState:
    enabled: bool = False
    last_ts: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    h:  float = 0.0
    h_ref: float = 0.35
    # 增益
    k_yaw: float = 90.0
    k_vx:  float = 2.0
    k_vz:  float = 1.5
    # 限幅
    vx_max: float = 2.5
    vz_max: float = 1.5
    yaw_max: float = 60.0

    def update_vision(self, cx, cy, h, ts=None):
        self.cx, self.cy, self.h = float(cx), float(cy), float(h)
        self.last_ts = ts if ts is not None else time.time()

    def compute_cmd(self):
        # 误差：ex>0 人在右；ez>0 人在下方→上升
        ex = (self.cx - 0.5)
        ez = (0.5 - self.cy)
        ed = (self.h_ref - self.h)

        yaw_rate = float(np.clip(self.k_yaw * ex, -self.yaw_max, self.yaw_max))
        vx = float(np.clip(self.k_vx * ed, -self.vx_max, self.vx_max))
        vz = float(np.clip(self.k_vz * ez, -self.vz_max, self.vz_max))  # 正号先用 ENU 直觉
        # 返回 NED：vz 上负下正
        return vx, 0.0, -vz, yaw_rate

@dataclass
class CalibState:
    active: bool = False
    yaw_rate_deg: float = 45.0
    turns: int = 1
    start_ts: float = 0.0
    duration: float = 0.0

    def start(self, yaw_rate_deg=45.0, turns=1):
        self.active = True
        self.yaw_rate_deg = float(max(10.0, min(120.0, yaw_rate_deg)))
        self.turns = int(max(1, min(3, turns)))
        self.duration = (360.0 * self.turns) / self.yaw_rate_deg
        self.start_ts = time.time()
        print(f"[CALIB] start: yaw_rate={self.yaw_rate_deg} deg/s, turns={self.turns}, T={self.duration:.1f}s")

    def step(self):
        if not self.active:
            return 0.0, False
        elapsed = time.time() - self.start_ts
        if elapsed >= self.duration:
            self.active = False
            print("[CALIB] done")
            return 0.0, True
        return self.yaw_rate_deg, False

@dataclass
class StickyState:
    active: bool = False
    vx_tgt: float = 0.0
    vy_tgt: float = 0.0
    vz_tgt: float = 0.0
    yaw_tgt: float = 0.0
    vx_f: float = 0.0
    vy_f: float = 0.0
    vz_f: float = 0.0
    yaw_f: float = 0.0
    dur: float = 0.1
    end_ts: Optional[float] = None

    def clear(self):
        self.active = False
        self.vx_tgt = self.vy_tgt = self.vz_tgt = self.yaw_tgt = 0.0
        self.vx_f = self.vy_f = self.vz_f = self.yaw_f = 0.0
        self.dur = 0.1
        self.end_ts = None

    def set_nav(self, vx, vy, vz, yaw, dur, hold_sec=None):
        self.active = True
        self.vx_tgt, self.vy_tgt, self.vz_tgt, self.yaw_tgt = vx, vy, vz, yaw
        self.dur = max(0.05, float(dur))
        self.end_ts = (time.time() + float(hold_sec)) if (hold_sec and hold_sec > 0) else None
        self.vx_f, self.vy_f, self.vz_f, self.yaw_f = vx, vy, vz, yaw
