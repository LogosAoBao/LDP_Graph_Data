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


def srr_encode(adj: np.ndarray, eps: float, rng: np.random.Generator, q: float = 0.1):
    """
    SRR 机制：先以概率 q 抽样，再对抽中的坐标使用 RR 编码
    返回：位置数组 (u,v)，值数组 y
    """
    n = adj.shape[1]
    sampled = rng.random(adj.shape) < q
    reports = []
    positions = []

    p = np.exp(eps) / (np.exp(eps) + 1)
    for u in range(adj.shape[0]):
        for v in range(n):
            if sampled[u, v]:
                x = adj[u, v]
                flip = rng.random() < (1 - p)
                y = x if not flip else 1 - x
                reports.append(y)
                positions.append((u, v))
    return np.array(positions), np.array(reports, dtype=np.uint8)

def srr_decode(positions, reports, shape, eps: float):
    """
    将 SRR 上报重构为 (n,n) 概率矩阵
    """
    p = np.exp(eps) / (np.exp(eps) + 1)
    q = 1 - p
    A = np.zeros(shape, dtype=float)
    C = np.zeros(shape, dtype=int)
    for (u, v), y in zip(positions, reports):
        A[u, v] += (y - q) / (p - q)
        C[u, v] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        A /= C
        A = np.nan_to_num(A, nan=0.0)
    return A


def boue_encode(adj: np.ndarray, eps: float, rng: np.random.Generator, B: int = 32):
    """
    分桶 OUE 编码：每个桶最多上传一条边，用 OUE 对桶内位置编码
    返回：(n, B, m+1) 的三维编码张量
    """
    n = adj.shape[1]
    m = int(np.ceil(n / B))
    buckets = [[] for _ in range(B)]
    for v in range(n):
        h = v % B
        buckets[h].append(v)

    encoded = np.zeros((adj.shape[0], B, m + 1), dtype=np.uint8)
    eps_per_bucket = eps
    p = np.exp(eps_per_bucket) / (np.exp(eps_per_bucket) + m)
    q = 1.0 / (np.exp(eps_per_bucket) + m)

    for u in range(adj.shape[0]):
        for b in range(B):
            choices = [v for v in buckets[b] if adj[u, v] == 1]
            if choices:
                selected = rng.choice(choices)
                idx = buckets[b].index(selected) + 1  # 1 ~ m
            else:
                idx = 0  # sentinel
            vec = np.zeros(m + 1)
            for i in range(m + 1):
                vec[i] = rng.random() < (p if i == idx else q)
            encoded[u, b] = vec
    return encoded

def boue_decode(encoded: np.ndarray, eps: float, n: int, B: int = 32):
    """
    解码 B-OUE，返回列概率向量 (n,)
    """
    m = int(np.ceil(n / B))
    eps_per_bucket = eps
    p = np.exp(eps_per_bucket) / (np.exp(eps_per_bucket) + m)
    q = 1.0 / (np.exp(eps_per_bucket) + m)

    est = np.zeros(n)
    count = np.zeros(n)

    for b in range(B):
        start = b * m
        for i in range(m):
            v = start + i
            if v >= n: continue
            col = encoded[:, b, i + 1]
            mean = col.mean()
            est[v] += (mean - q) / (p - q)
            count[v] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        est /= count
        est = np.nan_to_num(est, nan=0.0)
    return np.clip(est, 0.0, 1.0)


# ------------------- tests -------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(100, 100), dtype=np.uint8)
    rr = rr_encode(A, 1.0, rng); assert rr.shape == A.shape
    oue = oue_encode(A, 1.0, rng); assert oue.shape == A.shape
    idx, sig = hr_encode(A, 1.0, rng)
    est_hr = hr_decode(idx, sig, 100, 1.0); assert est_hr.shape == A.shape
