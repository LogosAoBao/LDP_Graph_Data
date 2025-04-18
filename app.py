# app.py
from flask import Flask, send_from_directory, jsonify, request
import pandas as pd
import os
import time, itertools
from graphs import gen_er, gen_ba, gen_ws, load_facebook
from simulation import simulate

BASE     = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, 'results', 'metrics.csv')

app = Flask(__name__, static_folder='static', static_url_path='')

def regenerate_metrics(n_syn: int):
    SEED      = 42
    EPS_LIST  = [0.1, 1]
    MECH_LIST = ["RR", "OUE", "HR"]
    OUT_CSV   = CSV_PATH

    # Facebook 子图采样
    fb_full  = load_facebook()
    fb_graph = fb_full.subgraph(list(fb_full)[:n_syn]).copy()

    GRAPH_LIST = [
        ("ER", gen_er(n_syn, 0.01, SEED)),
        ("BA", gen_ba(n_syn, 3, SEED)),
        ("WS", gen_ws(n_syn, 10, 0.1, SEED))
        # ("FB", fb_graph)
    ]

    results = []
    for eps, mech, (gname, G) in itertools.product(EPS_LIST, MECH_LIST, GRAPH_LIST):
        t0  = time.time()
        res = simulate(G, mech, eps, seed=SEED)
        res.update(dict(graph=gname, mech=mech, eps=eps))
        results.append(res)
        print(f"{gname:<3}  {mech}  ε={eps:<3}  ✓  {time.time()-t0:.1f}s")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n全部完成，结果写入 {OUT_CSV}")

@app.route('/api/data')
def get_data():
    df = pd.read_csv(CSV_PATH)
    if 'a' in df.columns:
        df = df.drop(columns=['a'])
    df = df.where(pd.notnull(df), None)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/run')
def run_simulation():
    try:
        n_syn = int(request.args.get('value', 1000))
    except (TypeError, ValueError):
        return jsonify({"error": "无效的value参数"}), 400

    regenerate_metrics(n_syn)
    return jsonify({"status": "ok"}), 200

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)