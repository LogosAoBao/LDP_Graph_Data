import numpy as np
from scipy.linalg import hadamard

# ==========================================================
# 0.  shared helpers
# ==========================================================
_rng_global = np.random.default_rng(0)

# ==========================================================
# 1.  Randomised Response  (same as before)
# ==========================================================

def rr_encode(bit: int, eps: float, rng=_rng_global):
    p = np.exp(eps) / (np.exp(eps) + 1)
    return bit if rng.random() < p else 1 - bit

def rr_decode(count_one: int, n: int, eps: float):
    p = np.exp(eps) / (np.exp(eps) + 1)
    return np.clip((count_one / n - (1 - p)) / (p - (1 - p)), 0, 1)

# ==========================================================
# 2.  OUE (unchanged)
# ==========================================================

def oue_encode(vec: np.ndarray, eps: float, rng=_rng_global):
    p = 0.5
    q = 1 / (np.exp(eps) + 1)
    noisy = vec.copy()
    flip = rng.random(vec.shape) < (1 - p)
    noisy[flip] ^= 1
    noisy ^= (rng.random(vec.shape) < q)
    return noisy.astype(np.uint8)

def oue_decode(reports: np.ndarray, eps: float):
    p = 0.5
    q = 1 / (np.exp(eps) + 1)
    s = reports.mean(0)
    return np.clip((s - q) / (p - q), 0, 1)

# ==========================================================
# 3.  **Hadamard Response (real implementation)**
# ==========================================================

def _get_hadamard(k: int):
    """ Small cache for Hadamard matrices of size power‑of‑two >= k """
    m = 1 << (k - 1).bit_length()
    return hadamard(m)


def hr_encode(vec: np.ndarray, eps: float, rng=_rng_global):
    k = vec.size
    m = 1 << int(np.ceil(np.log2(k)))  # 1024
    if m != k:  # ← 新增：零填充到 m
        vec = np.pad(vec, (0, m - k))
    H = _get_hadamard(m)
    # project vec onto Hadamard basis → sign pattern length m
    proj = H[:k] @ (2 * vec - 1)   # ±1 projection
    # choose *one* coordinate uniformly & privatise its sign with RR
    idx = rng.integers(0, proj.size)
    sign_bit = 1 if proj[idx] > 0 else 0
    priv = rr_encode(sign_bit, eps, rng)
    # upload (index, priv_bit)  ← uses log2(m)+1   bits
    return idx, priv


def hr_aggregate(reports, k: int, eps: float):
    """Aggregate list of (idx, bit) and return prob‑vector length k"""
    m = 1 << (k - 1).bit_length()
    counts = np.zeros(m, dtype=int)
    for idx, bit in reports:
        counts[idx] += 1 if bit else 0
    total = len(reports)
    est_sign = rr_decode(counts, total, eps) * 2 - 1  # → in (‑1,1)
    H = _get_hadamard(k)
    est = (H[:k, :m] @ est_sign) / m
    return np.clip((est + 1) / 2, 0, 1)