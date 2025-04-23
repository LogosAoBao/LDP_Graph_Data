"""
plotting.py
--------------------------------------------
Reads results/metrics.csv produced by experiments.py‑v2 and
generates figures under figs/ :
    Graph_metric.png    for metric in {l1,l2,clust}
    comm_cost.png       (avg bits per user, log‑scale)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV     = "results/metrics.csv"
FIG_DIR = "figs"
METRICS = ["l1", "l2", "clust"]   # clust = clustering‑coef error, may be absent
def _ensure_file():
    if not os.path.exists(CSV):
        raise FileNotFoundError(
            f"{CSV} not found — run experiments.py first.")


def _plot_metric_curves(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        print(f"[skip] column '{metric}' not in CSV; "
              "did you enable that metric in experiments?")
        return

    for graph in sorted(df.graph.unique()):
        sub = df[df.graph == graph]
        plt.figure()
        for mech in sorted(sub.mech.unique()):
            tmp = sub[sub.mech == mech].sort_values("eps")
            plt.plot(tmp.eps, tmp[metric],
                     marker="o", label=mech, linewidth=2)
        plt.xscale("log")
        plt.xlabel("Privacy ε")
        plt.ylabel(f"{metric.upper()} error")
        plt.title(f"{graph}: {metric.upper()} vs ε")
        plt.grid(alpha=.3, linestyle="--")
        plt.legend()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = f"{FIG_DIR}/{graph}_{metric}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print("saved", path)


def _plot_comm_cost(df: pd.DataFrame):
    if "bits_per_user" not in df.columns:
        # fallback to bits / n_graphs if old CSV
        if "bits" in df.columns:
            df["bits_per_user"] = df.bits / df.graph.map(
                df.groupby("graph").first()["n_nodes"])
        else:
            print("[skip] bits info not found; cannot plot comm_cost.")
            return
    plt.figure()
    avg_bits = df.groupby("mech").bits_per_user.mean().sort_index()
    avg_bits.plot(kind="bar", logy=True)
    plt.ylabel("bits per user (log‐scale)")
    plt.title("Average communication cost")
    plt.grid(axis="y", which="both", linestyle="--", alpha=.4)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f"{FIG_DIR}/comm_cost.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", path)


def main():
    _ensure_file()
    df = pd.read_csv(CSV)
    for m in METRICS:
        _plot_metric_curves(df, m)
    _plot_comm_cost(df)
    print("all figures are under figs/")


# ---------------- simple sanity tests ----------------
assert callable(main)
assert os.path.splitext(CSV)[1] == ".csv"
assert "matplotlib" in str(plt.__doc__).lower()

if __name__ == "__main__":
    main()
