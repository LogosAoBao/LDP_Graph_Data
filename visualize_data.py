import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(G, title, pos=None):
    plt.figure(figsize=(4.5, 4.5))
    pos = pos or nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=30, edge_color="gray", node_color="skyblue", with_labels=False)
    plt.title(title)
    plt.axis("off")

# 生成图
n = 100  # 节点数量，可以设为 100 左右看结构更明显

# ER 图
p = 0.05
G_er = nx.erdos_renyi_graph(n, p)
plot_graph(G_er, f"ER Graph (p={p})")

# BA 图
m = 3
G_ba = nx.barabasi_albert_graph(n, m)
plot_graph(G_ba, f"BA Graph (m={m})")

# WS 图
k = 4  # 每个节点初始连接 k 个邻居
rewire_p = 0.2
G_ws = nx.watts_strogatz_graph(n, k, rewire_p)
plot_graph(G_ws, f"WS Graph (k={k}, p={rewire_p})")

plt.tight_layout()
plt.show()
