# Local Differential Privacy for Graph Data

This repo contains a **minimal but extensible reference implementation** for our course-project:
*Comparing RR / OUE / HR under Local Differential Privacy on Graph-structured data.*
It simulates how every user perturbs her *adjacency vector* locally (no trusted curator), then evaluates the error of recovered graph statistics, together with the communication cost.

## Repository Structure

```
ldp_graph_privacy/
├── data/
│   ├── graphs/   # facebook.pkl will be saved here
│   └── raw/      # raw edge list (txt.gz) is cached here
├── download_facebook.py   # download + pickle the SNAP FB graph
├── graphs.py              # ER / BA / WS generator + facebook loader
├── mechanisms.py          # RR · OUE · HR (Hadamard Response)
├── simulation.py          # encode ▸ aggregate ▸ decode ▸ metrics
├── evaluation.py          # L1 · L2 · clustering-coef error, P/R ...
├── experiments.py         # loop over eps × mech × graph
├── plotting.py            # draw ε-error curves + comm-cost bar
├── requirements.txt
└── tests/                 # pytest sanity checks
```

## Quick Start

### 0. Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Win: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Grab the Facebook Graph (≈ 2 MB)

```bash
python download_facebook.py  # → data/graphs/facebook.pkl (4039 nodes, 88k edges)
```

*Tip:* the default **FB** experiment only uses the *first 1000 nodes* to keep runtime small. Change `SUBSET = 1000` in `experiments.py` if you want the full graph.

### 2. Run All Experiments (4 graphs × 3 mechs × 2 ε ≈ < 4 min on M-series CPU)

```bash
python experiments.py  # ↳ results/metrics.csv (+ console timer for each run)
```

### 3. Plot

```bash
python plotting.py  # ↳ figs/*.png : ER_l1.png ... FB_clust.png + comm_cost.png
```

Figures are log-scaled where helpful; clustering-coef error (`clust`) and communication cost (`bits_per_user`, log-y) clearly separate RR / OUE / HR.

### 4. Run the Web App
```
python app.py
```

### 4. Tests (Optional)

```bash
pytest -q
```

All ≈ 30 asserts should pass.

## What's Implemented?

| Mechanism | Message length / user | Code status |
|-----------|----------------------|-------------|
| **RR**    | *n* bits             | pure-Python, fully vectorised |
| **OUE**   | *n* bits             | NumPy vectorised |
| **HR**    | ⌈log₂ *n*⌉ bits      | True Hadamard projection + unbiased decode |

## Metrics

* **l1, l2** — degree-distribution errors
* **clust** — absolute error of average clustering coefficient
* **prec / rec** — edge existence (frequent-edge threshold 0.5)
* **bits_per_user** — `reports.size // n`

## Extending

* **More ε values** – edit `EPS_LIST` in `experiments.py`.
* **Full 4k-node Facebook graph** – set `SUBSET = None`.
* **New graph family** – add generator to `graphs.py` and extend `GRAPH_LIST`.
* **Extra metrics** – just append to `evaluation.py`; `experiments.py` will save any numeric output.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError results/metrics.csv` | run **experiments.py** first |
| `ValueError size k != 2^m` in HR | make sure `vec` is zero-padded to the next power-of-two (already handled) |
| Experiments too slow | ① reduce FB subset; ② use fewer ε; ③ check NumPy vectorisation |

## License

MIT — feel free to tweak / extend for academic use. Credits to SNAP for the Facebook social-circles dataset.
