import numpy as np
from functools import lru_cache

# ---------- 1. RR (vectorised) ----------
def rr_encode(adj: np.ndarray, eps: float, rng: np.random.Generator):
    """
    adj: (n,n) uint8     -> returns encoded (n,n) uint8
    """
    p = np.exp(eps) / (np.exp(eps) + 1)
    q = 1 - p
    mask_flip = rng.random(adj.shape) < q          # 需要翻转的位置
    out = adj.copy()
    out[mask_flip] ^= 1                            # bitwise flip
    return out

def rr_decode(reports: np.ndarray, eps: float):
    p = np.exp(eps) / (np.exp(eps) + 1)
    q = 1 - p
    return np.clip((reports.astype(float) - q) / (p - q), 0.0, 1.0)

# ---------- 2. OUE (vectorised) ----------
def oue_encode(adj: np.ndarray, eps: float, rng: np.random.Generator):
    p = 0.5
    q = 1.0 / (np.exp(eps) + 1.0)
    keep_1   = rng.random(adj.shape) < p           # 对 1 保留
    flip_0   = rng.random(adj.shape) < q           # 对 0 误报
    out = (adj & keep_1) | ((1 - adj) & flip_0)
    return out.astype(np.uint8)

def oue_decode(reports: np.ndarray, eps: float):
    # 输入 shape (n,n) → 返回列概率 (n,)
    p = 0.5
    q = 1.0 / (np.exp(eps) + 1.0)
    mean = reports.mean(axis=0)
    return np.clip((mean - q) / (p - q), 0.0, 1.0)

# ---------- 3. HR (vectorised Hadamard) ----------
@lru_cache(maxsize=None)
def hadamard(m: int):
    """
    返回 m×m Hadamard (m 为 2 的幂)。懒加载 + 缓存。
    """
    H = np.array([[1]])
    while H.shape[0] < m:
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H

def hr_encode(adj: np.ndarray, eps: float, rng: np.random.Generator):
    """
    每行邻接向量 → Hadamard 投影 → 只发送 (index, sign)
    返回: reports_index (n,) int  和  reports_sign (n,) int8
    """
    n = adj.shape[1]
    m = 1 << int(np.ceil(np.log2(n)))          # 最近的 2^k
    pad = m - n
    if pad:                                    # 零填充
        adj = np.pad(adj, ((0, 0), (0, pad)))
    H = hadamard(m)
    s = 2 * adj - 1                            # 0/1 → -1/+1
    proj = H @ s.T                             # shape (m,n)
    idx = rng.integers(0, m, size=n)           # 随机选一维
    sign = np.sign(proj[idx, range(n)]).astype(np.int8)
    flip_mask = rng.random(n) < 1.0 / (np.exp(eps) + 1)
    sign = sign ^ flip_mask.astype(np.int8)    # 局部 RR 噪声
    return idx.astype(np.uint16), sign         # 索引+符号

def hr_aggregate(indices, signs, n, eps):
    m = 1 << int(np.ceil(np.log2(n)))
    H = hadamard(m)
    count = np.zeros((m,))
    for idx, s in zip(indices, signs):
        count[idx] += s
    est = (H.T @ count) / len(indices)         # 估计 ±1 向量值
    est = est[:n]
    prob = (est + 1) / 2                       # 映射回 [0,1]
    prob = np.clip(prob, 0.0, 1.0)
    return np.tile(prob, (n, 1))               # (n,n) 概率矩阵

# ------------------- tests -------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(100, 100), dtype=np.uint8)
    rr = rr_encode(A, 1.0, rng); assert rr.shape == A.shape
    oue = oue_encode(A, 1.0, rng); assert oue.shape == A.shape
    idx, sig = hr_encode(A, 1.0, rng)
    est_hr = hr_decode(idx, sig, 100, 1.0); assert est_hr.shape == A.shape
