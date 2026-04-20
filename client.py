# -*- coding: utf-8 -*-
"""
client.py
负责 AirSim 客户端连接与全局 RPC 锁。
"""

import threading

# 全局 RPC 互斥锁（跨线程共享）
rpc_lock = threading.Lock()

def connect_multirotor():
    """
    创建并初始化 AirSim MultirotorClient。
    注意：必须在 patches_msgpack.apply() 之后再导入 airsim。
    """
    import airsim  # 延迟导入，确保补丁已应用
    cli = airsim.MultirotorClient()
    with rpc_lock:
        cli.confirmConnection()
        cli.enableApiControl(True)
    print("[UE] AirSim connected & API control enabled.")
    return cli
