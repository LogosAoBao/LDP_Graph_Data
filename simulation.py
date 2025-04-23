import numpy as np, networkx as nx
from mechanisms import rr_encode, rr_decode, oue_encode, oue_decode, hr_encode, hr_aggregate
from evaluation import degree_error, clustering_error, assortativity_error
from evaluation import avg_clustering, edge_precision_recall



def simulate(graph, mechanism: str, eps: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(graph)
    A = nx.to_numpy_array(graph, dtype=np.uint8)

    if mechanism == "RR":
        reports = rr_encode(A, eps, rng)
        col_prob = rr_decode(reports, eps).mean(axis=0)

    elif mechanism == "OUE":
        reports = oue_encode(A, eps, rng)
        col_prob = oue_decode(reports, eps)

    elif mechanism == "HR":
        idxs, signs = hr_encode(A, eps, rng)
        col_prob = hr_aggregate(idxs, signs, n, eps)[0]  # any row, all the same

    else:
        raise ValueError(f"unknown mechanism {mechanism}")

    est_mat = np.tile(col_prob, (n, 1))  # simplified decode (row = col prob)
    true_deg = A.sum(axis=1)
    est_deg = est_mat.sum(axis=1)

    l1 = np.mean(np.abs(true_deg - est_deg))
    l2 = np.sqrt(np.mean((true_deg - est_deg) ** 2))
    clust_err = abs(nx.average_clustering(graph) - avg_clustering(est_mat))
    prec, rec = edge_precision_recall(graph, est_mat)

    return {
        "l1": l1,
        "l2": l2,
        "clust": clust_err,
        "prec": prec,
        "rec": rec,
        "bits": reports.size if mechanism != "HR" else len(idxs) * (np.log2(n).astype(int) + 1),
        "n_nodes": n
    }
