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

def avg_clustering(P):
    """
    P: 估计邻接矩阵（概率矩阵）→ 二值化后计算平均聚类系数
    """
    est_adj = (P > 0.5).astype(int)
    G = nx.from_numpy_array(est_adj)
    return nx.average_clustering(G)


def edge_precision_recall(G: nx.Graph, P):
    """
    输入：真实图 G，估计概率矩阵 P
    输出：边的 precision 和 recall（0.5 二值化为边）
    """
    n = len(G)
    true_adj = nx.to_numpy_array(G, dtype=int)
    est_adj = (P > 0.5).astype(int)

    TP = np.logical_and(true_adj == 1, est_adj == 1).sum() // 2
    FP = np.logical_and(true_adj == 0, est_adj == 1).sum() // 2
    FN = np.logical_and(true_adj == 1, est_adj == 0).sum() // 2

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return precision, recall
