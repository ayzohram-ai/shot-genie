# -*- coding: utf-8 -*-
"""
handlers.py —— 路由 + 执行器 + 业务处理（集中）
保留原有接口与行为，包括：
- calib / rc / asr / photo / video 的消息协议
- NED 高度 z_hold 的管理
- 粘滞（StickyState）与 deadman
- 线程安全：统一走 rpc_lock

依赖：
- media_io.py：照片/视频 I/O
- shooting/photo_templates.py：拍照模板
- shooting/video_templates.py：视频模板
- portrait/portrait_detector.py：人像校准（可选）
- multimodal/intent_server.py：ASR 语义执行（可选）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import json
import time
import zmq
import airsim
import threading

# === 线程安全锁（与视频线程共享） ===
from client import rpc_lock  # 复用你的锁

# === 状态（你现有的 CalibState / StickyState） ===
from states import CalibState, StickyState

# === 可选的 llm_exec（兜底） ===
try:
    from multimodal.intent_server import llm_exec
except Exception:
    def llm_exec(text: str) -> str:
        return "[llm_exec not available] " + text

# === 人像检测与飞到目标（可选） ===
try:
    from portrait.portrait_detector import PortraitDetector
except Exception:
    PortraitDetector = None

# === 媒体 I/O（照片/视频） ===
from media_io import (
    ensure_dirs,
    capture_photo_bytes_from_sim,
    save_bytes,
    start_video_recording,
    stop_video_recording,
    capture_and_write_frame,
)

# === 拍照/视频模板：从 shooting/ 下引用（与你现在的名字保持一致） ===
from shooting.photo_templates import (
    photo_center, photo_left_rule_of_thirds, photo_right_rule_of_thirds, photo_negative_space_composition
)
from shooting.video_templates import (
    video_orbit_shot, video_zoom_in, video_zoom_out, video_tilt_up
)

# 额外：导入整个模块以便调用你新增的 replicate_shot/execute_template（若你也实现了该入口）
# 不与上面的函数导入冲突
from shooting import photo_templates as photo_templates


# ---------------- ZMQ：只取“最新”一条 ----------------
class ZmqPullServer:
    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.setsockopt(zmq.RCVHWM, 200)
        self.pull.bind(bind_addr)

    def recv_latest(self) -> Optional[str]:
        latest = None
        while True:
            try:
                latest = self.pull.recv(flags=zmq.NOBLOCK).decode("utf-8", "ignore").strip()
            except zmq.Again:
                break
        return latest


# ---------------- 运行上下文 ----------------
@dataclass
class PageState:
    active_page: Optional[str] = None
    last_recommend: Optional[dict] = None

@dataclass
class Context:
    cli: airsim.MultirotorClient
    z_hold: float
    debug: bool
    deadman_sec: float
    sticky_enabled: bool
    sticky_hold: bool
    calib: CalibState = field(default_factory=CalibState)
    sticky: Optional[StickyState] = field(default_factory=StickyState)
    photo_state: PageState = field(default_factory=PageState)
    video_state: PageState = field(default_factory=PageState)


# ---------------- 动作中断（方案A新增） ----------------
def _interrupt_current_motion(ctx: Context):
    """硬/软打断当前动作，使新指令立刻生效。"""
    with rpc_lock:
        try:
            ctx.cli.cancelLastTask()  # 若可用则硬取消
        except Exception:
            pass
        ctx.cli.hoverAsync()  # 软打断，清理残余速度


# ---------------- 统一执行器（非阻塞，移除 .join()） ----------------
def exec_move(ctx: Context, z_target: float, vx: float, vy: float, seconds: float, yaw_rate: float = 0.0):
    seconds = max(0.05, float(seconds))
    if ctx.debug:
        print(f"[MOVE] z={z_target:.2f}, vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}, T={seconds:.2f}")
    with rpc_lock:
        ctx.cli.moveByVelocityZAsync(
            vx, vy, z_target, seconds,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )  # 非阻塞：不再 .join()

def exec_move_to_z(ctx: Context, z_target: float, z_speed: float):
    z_speed = max(0.2, float(z_speed))
    timeout = max(3.0, abs(z_speed) + 2.0)
    if ctx.debug:
        print(f"[MOVEZ] z-> {z_target:.2f} at {z_speed:.2f} m/s, timeout={timeout:.1f}s")
    with rpc_lock:
        ctx.cli.moveToZAsync(z_target, velocity=z_speed, timeout_sec=timeout)  # 非阻塞：不再 .join()

def hover(ctx: Context):
    if ctx.debug:
        print("[HOVER]")
    with rpc_lock:
        ctx.cli.hoverAsync()


# ---------------- 业务处理：calib / rc / asr ----------------
import threading
from typing import Dict, Any

# 在 Context 上挂几个属性（第一次用时动态加就行）
def _ensure_calib_attrs(ctx):
    if not hasattr(ctx, "calib_thread"):
        ctx.calib_thread = None
    if not hasattr(ctx, "calib_stop_event"):
        ctx.calib_stop_event = None
    if not hasattr(ctx, "calib_detector"):
        ctx.calib_detector = None

def _calib_worker(ctx):
    """线程函数：运行 PortraitDetector.fly_to_target(stop_event=...)"""
    try:
        det = PortraitDetector()
        ctx.calib_detector = det
        det.fly_to_target(stop_event=ctx.calib_stop_event)
    except Exception as e:
        print(f"[calib] worker error: {type(e).__name__}: {e}")
    finally:
        # 清理现场
        ctx.calib_detector = None
        ctx.calib_thread = None
        ctx.calib_stop_event = None
        print("[calib] worker exit")


def handle_calib(obj: Dict[str, Any], ctx: Context):
    action = str(obj.get("action", "start")).lower()

    if action == "start":
        if PortraitDetector is None:
            print("[calib] PortraitDetector not available")
            return

        # 若已在运行，避免重复启动
        if getattr(ctx.calib, "active", False):
            if ctx.debug: print("[calib] already active; ignore start")
            return

        det = PortraitDetector()
        ctx.calib.det = det                         # 动态挂载引用（无需改 states 类）
        ctx.calib.active = True

        th = threading.Thread(target=det.fly_to_target, daemon=True)
        ctx.calib.thread = th
        th.start()
        if ctx.debug:
            print("[calib] fly_to_target thread started")
        return

    if action == "stop":
        # 若没有在运行，忽略
        if not getattr(ctx.calib, "active", False):
            if ctx.debug: print("[calib] not active; ignore stop")
            return

        # 安全停止：通知检测器退出循环
        det = getattr(ctx.calib, "det", None)
        if det is not None:
            try:
                det.request_stop()                 # <<< 关键：外部停止
            except Exception as e:
                if ctx.debug: print(f"[calib][WARN] request_stop failed: {e}")

        # 等待线程退出（给一点超时，避免卡死）
        th = getattr(ctx.calib, "thread", None)
        if th and th.is_alive():
            th.join(timeout=2.0)

        # 清理与解锁：打断残余动作 -> hover -> 标记非 active
        _interrupt_current_motion(ctx)
        hover(ctx)

        # 清理标志与引用
        ctx.calib.active = False
        for k in ("det", "thread"):
            if hasattr(ctx.calib, k):
                setattr(ctx.calib, k, None)

        print("[calib] stopped and released control")
        return


def handle_rc(obj: Dict[str, Any], ctx: Context):
    """
    RC：up/down/left/right/forward/back/backward/yaw_left/yaw_right
      - 平面/转向：在 z_hold 执行；若 sticky 开启则记录为粘滞目标
      - 上/下：vz*seconds 积分更新 z_hold，再执行一拍
      - 方案A：新指令先打断旧动作，然后立即执行
    """
    _interrupt_current_motion(ctx)  # 允许 RC 指令立刻抢占

    cmd = str(obj.get("cmd", "")).lower()
    mag = float(obj.get("mag", 1.0))
    seconds = float(obj.get("seconds", 0.25))

    vx = vy = vz = yaw = 0.0
    if   cmd == "left":       vy =  -1.0 * mag
    elif cmd == "right":      vy =   1.0 * mag
    elif cmd == "forward":    vx =   1.2 * mag
    elif cmd in ("back","backward"): vx = -1.2 * mag
    elif cmd == "yaw_left":   yaw = -25.0 * mag
    elif cmd == "yaw_right":  yaw =  25.0 * mag
    elif cmd == "up":         vz =  -0.8 * mag  # NED：上负
    elif cmd == "down":       vz =   0.8 * mag  # NED：下正
    else:
        if ctx.debug: print(f"[RC] unknown cmd: {cmd}")
        return

    # 粘滞目标
    if ctx.sticky:
        ctx.sticky.set_nav(vx, vy, vz, yaw, dur=max(0.1, seconds),
                           hold_sec=(seconds if ctx.sticky_hold and seconds > 0 else None))

    # 立即执行一拍（非阻塞）
    if vz != 0.0:
        z_target = ctx.z_hold + vz * seconds
        exec_move(ctx, z_target, 0.0, 0.0, seconds, yaw_rate=0.0)
        ctx.z_hold = z_target
    else:
        exec_move(ctx, ctx.z_hold, vx, vy, seconds, yaw_rate=yaw)

    if ctx.debug:
        print(f"[RC] {cmd} -> vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yaw={yaw:.1f} "
              f"T={seconds:.2f} z_hold={ctx.z_hold:.2f}")

def handle_asr(obj: Dict[str, Any], ctx: Context):
    """
    ASR 文本 → intent_server.llm_exec(text)
    不在本端做飞控；只透传获取计划/回复。
    """
    text = str(obj.get("text", "")).strip()
    if not text:
        if ctx.debug: print("[ASR] empty text")
        return
    try:
        reply = llm_exec(text)
        if ctx.debug:
            print(f"[ASR] text={text}\n[ASR][reply]={reply}")
    except Exception as e:
        if ctx.debug:
            print(f"[ASR][ERR] llm_exec failed: {e}")


# ---------------- OPS 适配层：传给模板，避免循环依赖 ----------------
class _Ops:
    def __init__(self):
        # 飞控
        self.move = exec_move
        self.move_to_z = exec_move_to_z
        self.hover = hover
        # 媒体 I/O
        self.ensure_dirs = ensure_dirs
        self.capture_photo_bytes_from_sim = capture_photo_bytes_from_sim
        self.save_bytes = save_bytes
        self.start_video_recording = start_video_recording
        self.stop_video_recording = stop_video_recording
        self.capture_and_write_frame = capture_and_write_frame

OPS = _Ops()


# ---------------- PHOTO：模板动作（委托给 shooting/photo_templates） ----------------
PHOTO_ACTIONS = {
    "center_composition":          lambda ctx: photo_center(ctx, OPS),
    "left_rule_of_thirds":         lambda ctx: photo_left_rule_of_thirds(ctx, OPS),
    "right_rule_of_thirds":        lambda ctx: photo_right_rule_of_thirds(ctx, OPS),
    "negative_space_composition":  lambda ctx: photo_negative_space_composition(ctx, OPS),
}

def _coerce_bbox_norm(b, tol=1e-3, eps=1e-6):
    # 接受 dict/list，并转 [cx,cy,w,h]，略
    if isinstance(b, dict):
        cx, cy, w, h = float(b['cx']), float(b['cy']), float(b['w']), float(b['h'])
    else:
        cx, cy, w, h = [float(v) for v in b]

    # 基本裁剪
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w  = min(max(w,  0.0), 1.0)
    h  = min(max(h,  0.0), 1.0)

    # 先算边界
    x0, x1 = cx - w/2.0, cx + w/2.0
    y0, y1 = cy - h/2.0, cy + h/2.0

    # 若仅微弱越界（≤ tol），直接clip到[0,1]并“回推”宽高以保持中心
    if x0 < 0.0 - tol or x1 > 1.0 + tol or y0 < 0.0 - tol or y1 > 1.0 + tol:
        # 明显越界，直接报错
        raise ValueError(f"bbox out of bounds: {(x0,y0,x1,y1)}")

    # 轻微越界，做柔性修复
    x0 = max(x0, 0.0); x1 = min(x1, 1.0)
    y0 = max(y0, 0.0); y1 = min(y1, 1.0)

    # 按裁剪后的边界回算 w,h（中心保持不变）
    w_fit = max(x1 - x0, eps)
    h_fit = max(y1 - y0, eps)

    # 由于中心保持不变，cx,cy 不动；只更新 w,h
    if abs(w_fit - w) > 1e-12 or abs(h_fit - h) > 1e-12:
        print(f"[PHOTO] bbox auto-fit: w {w:.6f}->{w_fit:.6f}, h {h:.6f}->{h_fit:.6f}")

    return [cx, cy, w_fit, h_fit]



def handle_photo(obj: Dict[str, Any], ctx: Context) -> bool:
    pl = obj.get("payload") or {}
    action = (obj.get("action") or "").lower()

    if action == "enter_page":
        page = pl.get("page")
        ctx.photo_state.active_page = page
        print(f"[PHOTO] enter_page -> {page}")
        return True

    if action == "template_select":
        _interrupt_current_motion(ctx)  # 新模板前打断
        tpl = pl.get("template")
        human = pl.get("human_name")
        print(f"[PHOTO] template_select -> {tpl} ({human})")
        fn = PHOTO_ACTIONS.get(tpl)
        if fn:
            fn(ctx)
        else:
            if ctx.debug: print(f"[PHOTO][WARN] 未知构图模板: {tpl}")
        return True

    if action == "recommend_detect_start":
        print("[PHOTO] recommend_detect_start")
        # TODO: 异步场景分析，写入 ctx.photo_state.last_recommend
        return True

    if action == "recommend_apply":
        _interrupt_current_motion(ctx)  # 应用推荐前打断
        tpl = pl.get("template") or ((ctx.photo_state.last_recommend or {}).get("template"))
        if tpl and tpl in PHOTO_ACTIONS:
            PHOTO_ACTIONS[tpl](ctx)
        else:
            print("[PHOTO][WARN] no template to apply; ignore")
        return True

    # ===== ★ 新增：接收上传模板的复刻参数，并调用 photo_templates.replicate_shot =====
    if action == "upload_template_exec":
        bbox_norm_raw  = pl.get("bbox_norm")
        source_path = pl.get("source_path")
        try:
             bbox_norm = _coerce_bbox_norm(bbox_norm_raw)  # ← 统一为 [cx,cy,w,h]
        except Exception as e:
              print(f"[PHOTO] upload_template_exec: invalid bbox_norm -> {bbox_norm_raw} | {type(e).__name__}: {e}")
              return True

        if not source_path or not isinstance(source_path, str):
            print(f"[PHOTO] upload_template_exec: invalid source_path -> {source_path}")
            return True

        print(f"[PHOTO] upload_template_exec: bbox_norm={bbox_norm}, source_path={source_path}")

        try:
                photo_templates.replicate_shot(
                    bbox_norm=bbox_norm,
                    source_path=source_path,
                    ctx=ctx,
                    ops=OPS,  # ← 关键：把 OPS 传给复刻函数
                    options={
                             "use_area": False,            # 用宽度匹配人物大小
                             "save_name": "replicated_from_ref.jpg",
                             "center_band": 0.03,
                             "area_band": 0.03,            # 这里沿用字段名，作为“大小阈值”
                             "control_y": True,
                             "conf_min": 0.55              # 低置信度不抓拍
                    }
                )
        except Exception as e:
            print(f"[PHOTO] upload_template_exec error: {type(e).__name__}: {e}")
        return True

    if ctx.debug:
        print(f"[PHOTO][WARN] unknown action: {action} | obj={obj}")
    return True


# ---------------- VIDEO：模板动作（委托给 shooting/video_templates） ----------------
VIDEO_TEMPLATES = {
    "orbit_shot":   lambda ctx: video_orbit_shot(ctx, OPS, speed=1.0, seconds=4.0),
    "zoom_in":     lambda ctx: video_zoom_in(ctx, OPS, speed=0.5, seconds=2.0),
    "zoom_out":    lambda ctx: video_zoom_out(ctx, OPS, speed=0.5, seconds=2.0),
    "tilt_up":  lambda ctx: video_tilt_up(ctx, OPS, speed=0.5, seconds=4.0),
}

def handle_video(obj: Dict[str, Any], ctx: Context) -> bool:
    pl = obj.get("payload") or {}
    action = (obj.get("action") or "").lower()

    if action == "enter_page":
        page = pl.get("page")
        ctx.video_state.active_page = page
        print(f"[VIDEO] enter_page -> {page}")
        return True

    if action == "template_select":
        _interrupt_current_motion(ctx)  # 新模板前打断
        tpl = pl.get("template")
        human = pl.get("human_name")
        print(f"[VIDEO] template_select -> {tpl} ({human})")
        fn = VIDEO_TEMPLATES.get(tpl)
        if fn:
            fn(ctx)
        else:
            if ctx.debug: print(f"[VIDEO][WARN] 未知视频模板: {tpl}")
        return True

    if action == "recommend_detect_start":
        print("[VIDEO] recommend_detect_start")
        # TODO: 异步分析
        return True


    if ctx.debug:
        print(f"[VIDEO][WARN] unknown action: {action} | obj={obj}")
    return True


# ---------------- 路由：统一入口 ----------------
def dispatch_json(s: str, ctx: Context) -> bool:
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        obj = json.loads(s)
    except Exception as e:
        if ctx.debug: print("[ROUTER][json_error]", e)
        return False

    t = str(obj.get("type", "")).lower()
    if t == "calib":
        handle_calib(obj, ctx);  return True
    if t == "rc":
        handle_rc(obj, ctx);     return True
    if t == "asr":
        handle_asr(obj, ctx);    return True
    if t == "photo":
        return handle_photo(obj, ctx)
    if t == "video":
        return handle_video(obj, ctx)

    if ctx.debug: print(f"[ROUTER] unknown type: {t} | obj={obj}")
    return False


# ---------------- 粘滞 + deadman 的每拍更新 ----------------
def tick_sticky_and_deadman(ctx: Context, last_cmd_ts: float):
    now = time.time()

    # 粘滞
    if ctx.sticky_enabled and ctx.sticky and ctx.sticky.active and not ctx.calib.active:
        st = ctx.sticky
        if st.end_ts and now >= st.end_ts:
            st.clear()
            hover(ctx)
            if ctx.debug: print("[STICKY] timed stop -> hover")
        else:
            alpha = 0.35
            st.vx_f = (1-alpha)*st.vx_f + alpha*st.vx_tgt
            st.vy_f = (1-alpha)*st.vy_f + alpha*st.vy_tgt
            st.vz_f = (1-alpha)*st.vz_f + alpha*st.vz_tgt
            st.yaw_f= (1-alpha)*st.yaw_f+ alpha*st.yaw_tgt
            if abs(st.vz_f) > 1e-6:
                ctx.z_hold += st.vz_f * st.dur
            exec_move(ctx, ctx.z_hold, st.vx_f, st.vy_f, st.dur, yaw_rate=st.yaw_f)

    # deadman
    if (now - last_cmd_ts) > ctx.deadman_sec and not ctx.calib.active and not (ctx.sticky and ctx.sticky.active):
        hover(ctx)
        if ctx.debug: print("[DEADMAN] -> hover")
