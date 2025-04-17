import numpy as np, networkx as nx


def degree_error(true_deg, est_deg):
    l1 = np.mean(np.abs(true_deg - est_deg))
    l2 = np.sqrt(np.mean((true_deg - est_deg) ** 2))
    return l1, l2


def clustering_error(G: nx.Graph, P):
    true_c = nx.average_clustering(G)
    est_adj = (P > 0.5).astype(int)
    est_c  = nx.average_clustering(nx.from_numpy_array(est_adj))
    return abs(true_c - est_c)


def assortativity_error(G: nx.Graph, P):
    true_a = nx.degree_assortativity_coefficient(G)
    est_adj = (P > 0.5).astype(int)
    H = nx.from_numpy_array(est_adj)
    try:
        est_a = nx.degree_assortativity_coefficient(H)
    except ZeroDivisionError:
        est_a = 0.0
    return abs(true_a - est_a)