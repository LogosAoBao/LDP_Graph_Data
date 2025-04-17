from graphs import gen_er, gen_ba, gen_ws
import networkx as nx

def test_synthetic_sizes():
    er = gen_er(50, 0.1, 0)
    ba = gen_ba(50, 2, 0)
    ws = gen_ws(50, 4, 0.2, 0)
    assert er.number_of_nodes() == ba.number_of_nodes() == 50
    assert nx.is_connected(er) or nx.number_connected_components(er) > 0
    assert nx.utils.is_networkx_graph(ws)

def test_degree_bounds():
    g = gen_er(30, 0.2, 1)
    degs = [d for _, d in g.degree()]
    assert all(0 <= d < 30 for d in degs)
