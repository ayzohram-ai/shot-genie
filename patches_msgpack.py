# -*- coding: utf-8 -*-
"""
patches_msgpack.py
仅负责在 import airsim 之前提升 msgpack 限制，避免大图像/大数组报错。
"""

def apply():
    try:
        import msgpack
        _orig_unpackb = msgpack.unpackb

        def _patched_unpackb(data, **kwargs):
            kwargs.setdefault('max_bin_len', 20 * 1024 * 1024)
            kwargs.setdefault('max_array_len', 10_000_000)
            kwargs.setdefault('max_map_len',   10_000_000)
            kwargs.setdefault('strict_map_key', False)
            return _orig_unpackb(data, **kwargs)

        msgpack.unpackb = _patched_unpackb

        _OrigUnpacker = msgpack.Unpacker
        class _PatchedUnpacker(_OrigUnpacker):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault('max_bin_len', 20 * 1024 * 1024)
                kwargs.setdefault('max_array_len', 10_000_000)
                kwargs.setdefault('max_map_len',   10_000_000)
                kwargs.setdefault('strict_map_key', False)
                super().__init__(*args, **kwargs)

        msgpack.Unpacker = _PatchedUnpacker
        print("✅ [PATCH] msgpack limits raised")
    except Exception as e:
        print(f"⚠️ [PATCH] msgpack patch failed: {e}")
