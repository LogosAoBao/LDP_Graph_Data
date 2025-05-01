import numpy as np, networkx as nx
from mechanisms import rr_encode, rr_decode, oue_encode, oue_decode, hr_encode, hr_aggregate, srr_encode,srr_decode,boue_encode,boue_decode
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
        col_prob = hr_aggregate(idxs, signs, n, eps)[0]

    elif mechanism == "SRR":
        pos, rep = srr_encode(A, eps, rng)
        est_mat = srr_decode(pos, rep, A.shape, eps)

    elif mechanism == "BOUE":
        rep = boue_encode(A, eps, rng)
        col_prob = boue_decode(rep, eps, A.shape[1])

    else:
        raise ValueError(f"unknown mechanism {mechanism}")

    # ✅ 如果不是 SRR，才需要生成 est_mat from col_prob
    if mechanism not in ["SRR"]:
        est_mat = np.tile(col_prob, (n, 1))

    if mechanism == "RR" or mechanism == "OUE":
        bits = reports.size
    elif mechanism == "HR":
        bits = len(idxs) * (int(np.log2(n)) + 1)
    elif mechanism == "SRR":
        bits = len(rep) * (int(np.log2(n ** 2)) + 1)  # 位置 + 比特
    elif mechanism == "BOUE":
        bits = rep.size  # rep 是三维张量
    else:
        bits = -1  # fallback

    #est_mat = np.tile(col_prob, (n, 1))  # simplified decode (row = col prob)
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
        #"bits": reports.size if mechanism != "HR" else len(idxs) * (np.log2(n).astype(int) + 1),
        "bits": bits,
        "n_nodes": n
    }
