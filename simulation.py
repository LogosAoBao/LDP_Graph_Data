import numpy as np, networkx as nx
from mechanisms import rr_encode, rr_decode, oue_encode, oue_decode, hr_encode, hr_aggregate
from evaluation import degree_error, clustering_error, assortativity_error


def simulate(G: nx.Graph, mech: str, eps: float, seed=0):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=np.uint8)

    # --- encode ---
    reports = []
    for u in range(n):
        vec = A[u]
        if mech == "RR":
            reports.append([rr_encode(int(b), eps, rng) for b in vec])
        elif mech == "OUE":
            reports.append(oue_encode(vec, eps, rng))
        else:  # HR
            reports.append(hr_encode(vec, eps, rng))

    # --- decode ---
    if mech == "RR":
        col_prob = [rr_decode(sum(col), n, eps) for col in zip(*reports)]
        row_prob = [rr_decode(sum(row), n, eps) for row in reports]
    elif mech == "OUE":
        col_prob = oue_decode(np.array(reports), eps)
        row_prob = oue_decode(np.array(reports).T, eps)
    else:  # HR
        col_prob = hr_aggregate(reports, n, eps)
        # for symmetry just reuse col → row; HR already unbiased
        row_prob = col_prob

    P = (np.tile(col_prob, (n, 1)) + np.tile(row_prob, (n, 1)).T) / 2

    # --- metrics ---
    true_deg = A.sum(1)
    est_deg = P.sum(1)
    l1, l2 = degree_error(true_deg, est_deg)
    clus_err = clustering_error(G, P)
    assort_err = assortativity_error(G, P)

    if mech == "HR":  # HR: index + bit  → log2(n)+1 bits
        bits_per_user = int(np.ceil(np.log2(n))) + 1
    else:
        bits_per_user = n

    return dict(l1=l1, l2=l2, c=clus_err, a=assort_err,
                bits=bits_per_user, graph="FB", mech=mech, eps=eps)