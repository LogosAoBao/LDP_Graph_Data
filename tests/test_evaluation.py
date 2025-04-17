import numpy as np, networkx as nx
from evaluation import degree_error, edge_precision_recall

def test_degree_error_zero():
    d = np.array([0, 1, 2])
    l1, l2 = degree_error(d, d)
    assert l1 == 0 and l2 == 0

def test_edge_pr_recall_one():
    G = nx.complete_graph(4)
    n = G.number_of_nodes()
    pr, rc = edge_precision_recall(G, np.ones(n*n))
    assert abs(rc - 1.0) < 1e-6
