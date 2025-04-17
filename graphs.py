import networkx as nx
import numpy as np
import pickle, os, random as rd

# -----------------------------------
# ❶ 生成三类合成图
# -----------------------------------
def gen_er(n, p, seed=None):
    rd.seed(seed)
    return nx.fast_gnp_random_graph(n, p, seed=seed)

def gen_ba(n, m, seed=None):
    rd.seed(seed)
    return nx.barabasi_albert_graph(n, m, seed=seed)

def gen_ws(n, k, beta, seed=None):
    rd.seed(seed)
    return nx.watts_strogatz_graph(n, k, beta, seed=seed)

# ❷ 读取 Facebook 图（pickle）
def load_facebook(path: str = "data/graphs/facebook.pkl") -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 未找到，请先运行 download_facebook.py"
        )
    with open(path, "rb") as fp:
        return pickle.load(fp)

# ---------- 3 × assert 单元测试 ----------
assert gen_er(20, 0.2).number_of_nodes() == 20
tmp = "data/graphs/_tmp_fb.pkl"
G = nx.Graph(); G.add_edge(1, 2)
pickle.dump(G, open(tmp, "wb"))
assert load_facebook(tmp).number_of_edges() == 1
os.remove(tmp)
