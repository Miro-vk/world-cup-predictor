# ⚽ 2026 FIFA World Cup Predictor

A machine learning tournament simulator for the 2026 FIFA World Cup. Trains a match outcome model on 5 years of international results and FIFA rankings, then runs a full 48-team bracket simulation thousands of times to estimate each team's probability of winning the tournament.

---

## Results (5,000 simulations)

| # | Team | Champion % | Final % | Semi-final % |
|---|------|-----------|---------|-------------|
| 1 | Spain | 13.44% | 22.76% | 35.74% |
| 2 | Argentina | 11.40% | 18.60% | 31.00% |
| 3 | Portugal | 9.88% | 18.42% | 31.42% |
| 4 | England | 8.66% | 15.70% | 26.56% |
| 5 | France | 7.98% | 14.70% | 26.78% |
| 6 | Morocco | 7.88% | 14.10% | 24.70% |
| 7 | Belgium | 7.82% | 15.70% | 27.86% |
| 8 | Brazil | 5.82% | 10.66% | 20.54% |

Full results in [`data/output/monte_carlo_results.csv`](data/output/monte_carlo_results.csv). Or create your own set of results using the webapp!

---

## How It Works

```
Raw Data                  Feature Engineering         Model
────────────────          ──────────────────────      ──────────────────────
results.csv       ──►     Rolling 5-game form    ──►  Logistic Regression
rankings.csv              FIFA rank merge              3-class: H / D / A
                          Tournament importance         Temperature scaling

Tournament Simulation
─────────────────────────────────────────────────────────────────────────────
Group stage (12 groups × round-robin)
  → 8 best 3rd-place teams qualified
  → Knockout bracket (official FIFA 2026 structure)
  → Monte Carlo × N: tracks champion / finalist / semi / quarter probabilities
```

**Key design choices:**

- **Temperature scaling (T=0.6)** sharpens the model's match probabilities — without it, Spain vs Iran reads as 55/45 when it should be closer to 70/30.
  > The model trains on noisy real-world data where upsets happen all the time, so it learns to be cautious and hedges toward 50/50. Temperature scaling is a single number we divide the raw scores by before converting to probabilities — anything below 1.0 pushes the confident predictions more extreme, so a strong team's 55% becomes 70% without retraining the model at all.

- **FastPredictor** bypasses pandas per-match, using pre-allocated numpy buffers — 54× faster than naive inference (40ms/sim vs 2.2s/sim).
  > Each simulation plays 104 matches. The naïve approach wraps every single match in a pandas DataFrame just to feed it into the model — a lot of packaging overhead for a tiny table. FastPredictor extracts the maths (imputer values, scaler mean/std) from the trained pipeline once at startup, then does the same calculation directly with numpy arrays that are already sitting in memory. Same answer, a fraction of the work.

- **Official 2026 bracket JSON** drives slot resolution — the 3rd-place qualification logic, group crossovers, and match numbering all match FIFA's published bracket exactly.
  > Rather than hardcoding "Group A winner plays Group B runner-up", the bracket rules live in a JSON file that mirrors what FIFA actually published. The simulator reads slot labels like `"1A"` (Group A winner) or `"W73"` (winner of match 73) and resolves them as the tournament progresses — so if FIFA updates the format, only the JSON needs to change.

---

## Model Performance

Evaluated on a held-out 20% time split (most recent matches):

| Metric | Score |
|--------|-------|
| Accuracy | **63.8%** |
| Macro F1 | **53.0%** |
| Log Loss | **0.824** |

Baseline (always predict home win): ~45%. Football is unpredictable by nature — a 63.8% accuracy means the model gets roughly 2 out of 3 match outcomes right, which is genuinely useful given how often upsets happen.

---

## Features

- `home_rank`, `away_rank`, `rank_diff` — FIFA world rankings at match date
- `home/away_recent_{wins,draws,losses,goals_for,goals_against}_5` — rolling last-5-game form
- `recent_opp_rank_avg_5` — average quality of recent opposition
- `recent_form_points` — points earned in last 5 games
- `tournament_importance` — weighted by competition (World Cup=5, friendly=1)
- `neutral` — venue neutrality flag

---

## Quickstart

```bash
git clone https://github.com/Miro-vk/world-cup-predictor
cd world-cup-predictor
uv sync
source .venv/bin/activate
```

**Run the web app:**
```bash
wc2026 app
# or: streamlit run app.py
```

**Run Monte Carlo simulations:**
```bash
wc2026 simulate --n 5000 --seed 42
```

**Simulate a single tournament:**
```bash
wc2026 once
```

**Retrain the model:**
```bash
python -m world_cup_predictor.train
```

---

## Project Structure

```
world-cup-predictor/
├── app.py                          # Streamlit web app (4 pages)
├── data/
│   ├── raw/
│   │   ├── results.csv             # International match results (2021–2026)
│   │   ├── rankings.csv            # Monthly FIFA rankings
│   │   ├── wc2026_groups.json      # 48 teams in 12 groups
│   │   └── fifa_world_cup_2026_bracket_format.json
│   ├── processed/
│   │   └── matches_with_rankings.csv
│   └── output/
│       └── monte_carlo_results.csv
├── models/
│   └── match_outcome_model.joblib
└── src/world_cup_predictor/
    ├── load_data.py                # Data ingestion
    ├── clean_data.py               # Name normalisation, deduplication
    ├── features.py                 # Rolling form features, team state lookup
    ├── train.py                    # sklearn pipeline, time-split training
    ├── evaluate.py                 # Accuracy / F1 / log-loss on test set
    ├── simulate.py                 # Group stage, bracket, Monte Carlo, FastPredictor
    └── cli.py                      # wc2026 command (simulate / once / app)
```

---

## Web App Pages

| Page | Description |
|------|-------------|
| **Tournament Odds** | Full 48-team probability table with progress bar columns |
| **Match Predictor** | Head-to-head win/draw/loss breakdown for any two teams |
| **Simulate Tournament** | One-click full bracket with expandable round-by-round results |
| **Group Explorer** | Per-group team stats, simulate the group, see every scoreline |

---

## Stack

- **Python 3.12** · **scikit-learn** · **pandas** · **numpy**
- **Streamlit** — web UI
- **joblib** — model serialisation
- **uv** — package management

## Final Message
Football and coding are some of my biggest passions so I hope you enjoy just messing around with the model and seeing what outcomes come from it!