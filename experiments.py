import time, itertools, pandas as pd
from graphs import gen_er, gen_ba, gen_ws, load_facebook
from simulation import simulate

# -------- 实验配置 --------
#EPS_LIST   = [0.1, 0.5, 1, 2]
EPS_LIST   = [0.1, 1]
MECH_LIST  = ["RR", "OUE", "HR"]
SEED       = 42
N_SYN      = 1_000               # 合成图节点数
OUT_CSV    = "results/metrics.csv"

# Facebook 子图
from graphs import load_facebook
fb_full = load_facebook()
fb_graph = fb_full.subgraph(list(fb_full)[:1000]).copy()  # sample 1k nodes

# ❶ 准备所有图
GRAPH_LIST = [
    ("ER", gen_er(N_SYN, 0.01, SEED)),
    ("BA", gen_ba(N_SYN, 3, SEED)),
    ("WS", gen_ws(N_SYN, 10, 0.1, SEED)),
    # ("FB", load_facebook())      # ← 新增：真实 Facebook 图
]

# ❷ 运行实验
def run():
    results = []
    for eps, mech, (gname, G) in itertools.product(EPS_LIST, MECH_LIST, GRAPH_LIST):
        t0 = time.time()
        res = simulate(G, mech, eps, seed=SEED)
        res.update(dict(graph=gname, mech=mech, eps=eps))
        results.append(res)
        print(f"{gname:<3}  {mech}  ε={eps:<3}  ✓  {time.time()-t0:.1f}s")
    # 保存
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n全部完成，结果写入 {OUT_CSV}")

if __name__ == "__main__":
    run()
