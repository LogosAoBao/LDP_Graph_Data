from flask import Flask, send_from_directory, jsonify
import pandas as pd
import os

BASE     = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, 'results', 'metrics.csv')

df = pd.read_csv(CSV_PATH)
if 'a' in df.columns:
    df = df.drop(columns=['a'])
df = df.where(pd.notnull(df), None)

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/api/data')
def get_data():
    # to_dict 后每个 None 变成 JSON 的 null
    records = df.to_dict(orient='records')
    return jsonify(records)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)