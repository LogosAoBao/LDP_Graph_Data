import os, gzip, urllib.request, pickle, networkx as nx

URL  = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
RAW  = "data/raw/facebook_combined.txt.gz"
OUT  = "data/graphs/facebook.pkl"        # ← 改成 .pkl

def fetch_facebook():
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists(RAW):
        print("Downloading edge list …")
        urllib.request.urlretrieve(URL, RAW)
    else:
        print("Edge list already present ✔")

def build_graph():
    with gzip.open(RAW, "rt") as f:
        edges = (tuple(map(int, line.split())) for line in f)
        G = nx.Graph()
        G.add_edges_from(edges)
    os.makedirs("data/graphs", exist_ok=True)
    with open(OUT, "wb") as fp:          # ← 用标准 pickle 保存
        pickle.dump(G, fp)
    print(f"Graph saved to {OUT}  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")

if __name__ == "__main__":
    fetch_facebook()
    build_graph()
