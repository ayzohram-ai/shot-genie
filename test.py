import sys
import numpy as np


import numpy.core as ncore     # 旧结构 (<=1.24)
source = "numpy.core"

print(f"[OK] Imported {source}")
print(f"NumPy version: {np.__version__}")
print(f"Module: {ncore.__name__}")
print(f"File:   {getattr(ncore, '__file__', 'built-in')}")

# 做一些基本健康检查
checks = {
    "has multiarray": hasattr(ncore, "multiarray"),
    "has umath": hasattr(ncore, "umath"),
    "can make ndarray": hasattr(np, "ndarray") and isinstance(np.array([1,2,3]), np.ndarray),
}
failed = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(f"{k:>18}: {v}")

# 若有关键能力缺失，返回非零退出码方便脚本/CI判断
if failed:
    print(f"[WARN] Missing: {', '.join(failed)}")
    sys.exit(1)
else:
    print("[PASS] ncore import looks good.")

