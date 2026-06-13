import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from world_cup_predictor.features import get_current_team_states
from world_cup_predictor.simulate import (
    BRACKET_PATH,
    GROUPS_PATH,
    PROCESSED_PATH,
    FastPredictor,
    load_pipeline,
    monte_carlo,
    predict_match,
    simulate_bracket,
    simulate_group,
)
from world_cup_predictor.train import MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cached resources — loaded once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_pipeline():
    return load_pipeline()

@st.cache_resource
def get_team_states():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    return get_current_team_states(df)

@st.cache_data
def get_groups() -> dict:
    with open(GROUPS_PATH) as f:
        return json.load(f)["groups"]

@st.cache_data
def get_bracket() -> dict:
    with open(BRACKET_PATH) as f:
        return json.load(f)

@st.cache_data
def get_all_teams() -> list[str]:
    groups = get_groups()
    return sorted(t for teams in groups.values() for t in teams)


# ---------------------------------------------------------------------------
# Helper: run one full tournament using cached resources
# ---------------------------------------------------------------------------

def _run_one_tournament() -> dict:
    pipeline = get_pipeline()
    team_states = get_team_states()
    groups = get_groups()
    bracket = get_bracket()

    all_standings = {}
    for letter, teams in groups.items():
        standings, log = simulate_group(teams, team_states, pipeline)
        all_standings[letter] = (standings, log)

    result = simulate_bracket(
        {k: v[0] for k, v in all_standings.items()},
        team_states, pipeline, _bracket_cache=bracket,
    )
    result["group_data"] = all_standings
    return result


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

st.sidebar.title("⚽ WC 2026 Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Tournament Odds", "Match Predictor", "Simulate Tournament", "Group Explorer"],
)

st.sidebar.divider()
st.sidebar.subheader("Monte Carlo Settings")
n_sims = st.sidebar.select_slider(
    "Simulations", options=[500, 1_000, 2_000, 5_000, 10_000], value=5_000
)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
run_btn = st.sidebar.button("▶ Run Simulation", use_container_width=True, type="primary")

if run_btn:
    with st.sidebar:
        with st.spinner(f"Running {n_sims:,} simulations…"):
            t0 = time.time()
            df_mc = monte_carlo(
                n=n_sims,
                seed=int(seed),
                output_path=PROJECT_ROOT / "data" / "output" / "monte_carlo_results.csv",
            )
            elapsed = time.time() - t0
        st.success(f"Done in {elapsed:.1f}s")
    st.session_state["mc_results"] = df_mc
    st.session_state["mc_n"] = n_sims

mc: pd.DataFrame | None = st.session_state.get("mc_results")


# ---------------------------------------------------------------------------
# Page 1 — Tournament Odds
# ---------------------------------------------------------------------------

if page == "Tournament Odds":
    st.title("2026 World Cup — Tournament Odds")

    if mc is None:
        st.info("Press **▶ Run Simulation** in the sidebar to generate odds.")
    else:
        n_label = f"{st.session_state['mc_n']:,}"
        st.caption(f"Based on {n_label} Monte Carlo simulations · seed={seed}")

        display = mc.copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "Rank"

        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "team": st.column_config.TextColumn("Team"),
                "champion_pct": st.column_config.ProgressColumn(
                    "Champion %", min_value=0, max_value=100, format="%.2f%%"
                ),
                "final_pct": st.column_config.ProgressColumn(
                    "Final %", min_value=0, max_value=100, format="%.2f%%"
                ),
                "semi_final_pct": st.column_config.ProgressColumn(
                    "Semi-final %", min_value=0, max_value=100, format="%.2f%%"
                ),
                "quarter_final_pct": st.column_config.ProgressColumn(
                    "Quarter-final %", min_value=0, max_value=100, format="%.2f%%"
                ),
            },
            height=1500,
        )


# ---------------------------------------------------------------------------
# Page 2 — Match Predictor
# ---------------------------------------------------------------------------

elif page == "Match Predictor":
    st.title("Match Predictor")
    st.caption("Select any two teams to see head-to-head win probabilities.")

    teams = get_all_teams()

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home / Team A", teams, index=teams.index("Spain"))
    with col2:
        away = st.selectbox("Away / Team B", teams, index=teams.index("England"))

    if home == away:
        st.warning("Select two different teams.")
    else:
        pipeline = get_pipeline()
        team_states = get_team_states()
        p = predict_match(home, away, team_states, pipeline)
        ph, pd_, pa = p["p_home_win"], p["p_draw"], p["p_away_win"]

        st.divider()

        # Probability bar
        bar_cols = st.columns([ph, pd_, pa])
        with bar_cols[0]:
            st.markdown(
                f"<div style='background:#1f77b4;border-radius:6px;padding:16px;text-align:center;"
                f"color:white;font-size:1.3rem;font-weight:bold'>{home}<br>{ph:.1%}</div>",
                unsafe_allow_html=True,
            )
        with bar_cols[1]:
            st.markdown(
                f"<div style='background:#888;border-radius:6px;padding:16px;text-align:center;"
                f"color:white;font-size:1.3rem;font-weight:bold'>Draw<br>{pd_:.1%}</div>",
                unsafe_allow_html=True,
            )
        with bar_cols[2]:
            st.markdown(
                f"<div style='background:#d62728;border-radius:6px;padding:16px;text-align:center;"
                f"color:white;font-size:1.3rem;font-weight:bold'>{away}<br>{pa:.1%}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Team stat comparison
        ts = get_team_states()
        h_state = ts.get(home, {})
        a_state = ts.get(away, {})

        stat_rows = [
            ("FIFA Rank",        h_state.get("rank"),                       a_state.get("rank")),
            ("Ranking Points",   h_state.get("ranking_points"),             a_state.get("ranking_points")),
            ("Recent Wins (L5)", h_state.get("recent_wins_5"),              a_state.get("recent_wins_5")),
            ("Recent Draws (L5)",h_state.get("recent_draws_5"),             a_state.get("recent_draws_5")),
            ("Recent Losses (L5)",h_state.get("recent_losses_5"),           a_state.get("recent_losses_5")),
            ("Goals For (L5)",   h_state.get("recent_goals_for_5"),         a_state.get("recent_goals_for_5")),
            ("Goals Against (L5)",h_state.get("recent_goals_against_5"),    a_state.get("recent_goals_against_5")),
            ("Form Points (L5)", h_state.get("recent_form_points"),         a_state.get("recent_form_points")),
        ]

        stat_df = pd.DataFrame(stat_rows, columns=["Stat", home, away])
        st.dataframe(stat_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 3 — Simulate Tournament
# ---------------------------------------------------------------------------

elif page == "Simulate Tournament":
    st.title("Simulate a Tournament")

    if st.button("▶ Run One Tournament", type="primary"):
        with st.spinner("Simulating…"):
            result = _run_one_tournament()
        st.session_state["tournament_result"] = result

    tr = st.session_state.get("tournament_result")

    if tr is None:
        st.info("Press **Run One Tournament** to simulate a full bracket.")
    else:
        # Champion banner
        st.success(f"🏆 Champion: **{tr['champion']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🥇 Champion",   tr["champion"])
        c2.metric("🥈 Runner-up",  tr["runner_up"])
        c3.metric("🥉 3rd Place",  tr["third_place"])
        c4.metric("4th Place",     tr["fourth_place"])

        st.divider()

        # Knockout rounds
        bracket = get_bracket()
        ko = bracket["knockout_stage"]
        mr = tr["match_results"]

        stage_labels = {
            "round_of_32":       "Round of 32",
            "round_of_16":       "Round of 16",
            "quarter_finals":    "Quarter-finals",
            "semi_finals":       "Semi-finals",
            "third_place_match": "3rd Place Match",
            "final":             "Final",
        }

        for stage_key, label in stage_labels.items():
            matches = ko[stage_key]
            with st.expander(label, expanded=(stage_key in ("semi_finals", "final"))):
                rows = []
                for m in matches:
                    num = m["match"]
                    if num not in mr:
                        continue
                    res = mr[num]
                    winner = res["winner"]
                    rows.append({
                        "Match": num,
                        "Home": res["home"],
                        "Away": res["away"],
                        "Winner": f"✅ {winner}",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Group standings
        st.subheader("Group Stage Standings")
        group_cols = st.columns(3)
        groups = get_groups()
        for i, (letter, _) in enumerate(sorted(groups.items())):
            standings, log = tr["group_data"][letter]
            df_s = pd.DataFrame(standings)[["team", "pts", "gf", "ga", "gd", "w", "d", "l"]]
            df_s.index = range(1, len(df_s) + 1)
            with group_cols[i % 3]:
                st.markdown(f"**Group {letter}**")
                st.dataframe(df_s, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Group Explorer
# ---------------------------------------------------------------------------

elif page == "Group Explorer":
    st.title("Group Stage Explorer")

    groups = get_groups()
    group_letter = st.selectbox("Select Group", sorted(groups.keys()))
    group_teams = groups[group_letter]

    # Team info
    ts = get_team_states()
    info_rows = []
    for team in group_teams:
        state = ts.get(team, {})
        info_rows.append({
            "Team":           team,
            "FIFA Rank":      state.get("rank"),
            "Rating Points":  state.get("ranking_points"),
            "Form Pts (L5)":  state.get("recent_form_points"),
            "W (L5)":         state.get("recent_wins_5"),
            "D (L5)":         state.get("recent_draws_5"),
            "L (L5)":         state.get("recent_losses_5"),
            "GF (L5)":        state.get("recent_goals_for_5"),
            "GA (L5)":        state.get("recent_goals_against_5"),
        })

    st.subheader(f"Group {group_letter} — Team Stats")
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    st.divider()

    if st.button("▶ Simulate This Group", type="primary"):
        pipeline = get_pipeline()
        team_states = get_team_states()
        with st.spinner("Simulating group…"):
            standings, log = simulate_group(group_teams, team_states, pipeline)
        st.session_state[f"group_{group_letter}"] = (standings, log)

    sim = st.session_state.get(f"group_{group_letter}")
    if sim:
        standings, log = sim

        st.subheader("Final Standings")
        df_s = pd.DataFrame(standings)[["team", "pts", "gf", "ga", "gd", "w", "d", "l"]]
        df_s.index = range(1, len(df_s) + 1)
        df_s.index.name = "Pos"
        st.dataframe(df_s, use_container_width=True)

        st.subheader("Match Results")
        log_df = pd.DataFrame(log)[[
            "home_team", "home_goals", "away_goals", "away_team", "outcome",
            "p_home_win", "p_draw", "p_away_win",
        ]]
        log_df.columns = ["Home", "HG", "AG", "Away", "Result", "p(H)", "p(D)", "p(A)"]
        log_df["p(H)"] = log_df["p(H)"].map("{:.1%}".format)
        log_df["p(D)"] = log_df["p(D)"].map("{:.1%}".format)
        log_df["p(A)"] = log_df["p(A)"].map("{:.1%}".format)
        st.dataframe(log_df, use_container_width=True, hide_index=True)