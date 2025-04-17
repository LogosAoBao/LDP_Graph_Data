import numpy as np, random
from mechanisms import rr_encode, rr_decode, oue_encode, oue_decode

def test_rr_unbiased():
    rng = random.Random(0)
    p, q = 0.75, 0.25
    reports = [rr_encode(1, p, q, rng) for _ in range(10000)]
    est = rr_decode(reports, p, q)
    assert abs(est - 1.0) < 0.05  # high prob keep 1s

def test_oue_shapes():
    vec = np.zeros(8, dtype=np.uint8); vec[2] = 1
    rep = oue_encode(vec, 1.0, np.random.default_rng(0))
    assert rep.shape == vec.shape
    est = oue_decode(np.vstack([rep, rep]), 1.0)
    assert est.size == vec.size
