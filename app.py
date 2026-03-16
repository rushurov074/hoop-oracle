"""
March Madness Predictor — Flask Backend
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import math
import random
import json
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ── Algorithm ─────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {"em": 0.40, "seed": 0.25, "win": 0.18, "sos": 0.10, "form": 0.07}
LOGISTIC_SCALE  = 0.35
ROUND_NAMES     = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def adj_em(t): return t["adjo"] - t["adjd"]


def win_prob(a, b, weights=None, noise=0.0):
    w = weights or DEFAULT_WEIGHTS
    raw = (
        w["em"]   * (adj_em(a) - adj_em(b)) +
        w["seed"] * (b["seed"] - a["seed"]) * 2 +
        w["win"]  * (a["win_pct"] - b["win_pct"]) +
        w["sos"]  * (b["sos"] - a["sos"]) / 40 +
        w["form"] * (a["form"] - b["form"]) * 2
    )
    p = 1 / (1 + math.exp(-raw * LOGISTIC_SCALE))
    if noise > 0:
        p = max(0.05, min(0.95, p + random.gauss(0, noise)))
    return p


def predict_game(a, b, weights=None):
    p = win_prob(a, b, weights)
    winner = a if p >= 0.5 else b
    wp = max(p, 1 - p)
    conf = "HIGH" if wp >= 0.70 else ("MODERATE" if wp >= 0.57 else "TOSS-UP")
    lower = a if a["seed"] > b["seed"] else b
    lower_p = p if a["seed"] > b["seed"] else 1 - p
    upset = a["seed"] != b["seed"] and lower_p >= 0.45
    return {
        "winner": winner,
        "loser": b if winner == a else a,
        "prob_a": round(p, 4),
        "prob_b": round(1 - p, 4),
        "confidence": conf,
        "upset_watch": upset,
        "upset_team": lower["name"] if upset else None,
        "upset_prob": round(lower_p, 4) if upset else None,
    }


def sim_bracket_once(teams, weights=None, noise=0.0):
    current = list(teams)
    rounds = []
    while len(current) > 1:
        winners, games = [], []
        for i in range(0, len(current), 2):
            a = current[i]
            b = current[i+1] if i+1 < len(current) else None
            if not b:
                winners.append(a)
                continue
            p = win_prob(a, b, weights, noise)
            w = a if random.random() < p else b
            games.append({"a": a["name"], "b": b["name"],
                          "winner": w["name"], "prob_a": round(p,3), "prob_b": round(1-p,3)})
            winners.append(w)
        rounds.append(games)
        current = winners
    return rounds, current[0] if current else None


def run_monte_carlo(teams, n_sims=5000, noise=0.08, weights=None):
    n_rounds = int(math.log2(len(teams)))
    counts = {t["name"]: [0]*n_rounds for t in teams}
    champ_counts = defaultdict(int)
    total_seed = 0

    for _ in range(n_sims):
        current = list(teams)
        ri = 0
        while len(current) > 1:
            next_r = []
            for i in range(0, len(current), 2):
                a = current[i]
                b = current[i+1] if i+1 < len(current) else None
                if not b:
                    next_r.append(a)
                    continue
                p = win_prob(a, b, weights, noise)
                w = a if random.random() < p else b
                counts[w["name"]][ri] += 1
                next_r.append(w)
            current = next_r
            ri += 1
        if current:
            champ_counts[current[0]["name"]] += 1
            total_seed += current[0]["seed"]

    return {
        "counts": counts,
        "champ_counts": dict(champ_counts),
        "n_sims": n_sims,
        "n_rounds": n_rounds,
        "avg_champ_seed": round(total_seed / n_sims, 2),
        "round_names": ROUND_NAMES[:n_rounds],
    }


# ── Bracket data — loads from teams.json if present ───────────────────────────
from pathlib import Path

def load_teams():
    teams_file = Path(__file__).parent / "teams.json"
    if teams_file.exists():
        with open(teams_file) as f:
            return json.load(f)
    return []

BRACKET_TEAMS = load_teams()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    a, b = data["team_a"], data["team_b"]
    result = predict_game(a, b)
    return jsonify({
        "winner":      result["winner"]["name"],
        "loser":       result["loser"]["name"],
        "prob_a":      result["prob_a"],
        "prob_b":      result["prob_b"],
        "confidence":  result["confidence"],
        "upset_watch": result["upset_watch"],
        "upset_team":  result["upset_team"],
        "upset_prob":  result["upset_prob"],
        "breakdown": {
            "adj_em_a": round(adj_em(a), 2),
            "adj_em_b": round(adj_em(b), 2),
        }
    })


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data    = request.json
    teams   = data.get("teams", BRACKET_TEAMS)
    weights = data.get("weights")
    rounds, champion = sim_bracket_once(teams, weights)
    return jsonify({"rounds": rounds, "champion": champion["name"] if champion else None})


@app.route("/api/montecarlo", methods=["POST"])
def api_montecarlo():
    data    = request.json
    teams   = data.get("teams", BRACKET_TEAMS)
    n_sims  = min(int(data.get("n_sims", 5000)), 25000)
    noise   = float(data.get("noise", 0.08))
    weights = data.get("weights")
    results = run_monte_carlo(teams, n_sims, noise, weights)

    # Format for frontend
    team_rows = []
    for t in teams:
        c = results["counts"].get(t["name"], [0]*results["n_rounds"])
        team_rows.append({
            "name":   t["name"],
            "seed":   t["seed"],
            "record": t.get("record",""),
            "conf":   t.get("conf",""),
            "adj_em": round(adj_em(t), 1),
            "probs":  [round(v/n_sims, 4) for v in c],
        })
    team_rows.sort(key=lambda r: r["probs"][-1], reverse=True)

    champ_list = sorted(results["champ_counts"].items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "teams":          team_rows,
        "champion":       champ_list[0][0] if champ_list else None,
        "champ_prob":     round(champ_list[0][1]/n_sims, 4) if champ_list else 0,
        "n_sims":         n_sims,
        "avg_champ_seed": results["avg_champ_seed"],
        "round_names":    results["round_names"],
        "upset_watch": [
            {"name": t["name"], "seed": t["seed"], "round": ri, "round_name": results["round_names"][ri], "prob": round(c/n_sims,4)}
            for t in teams
            for ri, c in enumerate(results["counts"].get(t["name"],[]))
            if t["seed"] >= 5 and ri >= 2 and c/n_sims >= 0.15
        ]
    })


@app.route("/api/teams/sample")
def api_sample_teams():
    return jsonify(BRACKET_TEAMS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
