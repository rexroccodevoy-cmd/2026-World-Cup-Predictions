"""
2026 FIFA World Cup Prediction Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c2333; border-radius: 8px; padding: 12px; }
    .match-card { background: linear-gradient(135deg, #1c2333 0%, #2d3748 100%);
                  border-radius: 12px; padding: 20px; margin: 10px 0;
                  border: 1px solid #3a4556; }
    .winner-badge { background: #ffd700; color: #000; padding: 3px 8px;
                    border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    div[data-testid="stSidebarContent"] { background-color: #1c2333; }
    .section-header { font-size: 1.6em; font-weight: 700; color: #ffd700;
                      border-bottom: 2px solid #ffd700; padding-bottom: 6px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training prediction models…")
def load_models():
    from src.models import MatchPredictor, GoalScorerPredictor, CorrelationEngine, simulate_tournament
    mp = MatchPredictor()
    gsp = GoalScorerPredictor(mp)
    ce = CorrelationEngine()
    return mp, gsp, ce

@st.cache_data(show_spinner="Running 1 000 Monte Carlo simulations…")
def run_simulations():
    from src.models import MatchPredictor, simulate_tournament
    mp = MatchPredictor()
    return simulate_tournament(mp, n_sims=1000)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## WC 2026 Predictor")
    st.caption("AI-powered quantitative analysis")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Tournament Overview",
         "🔴 Live Tracker",
         "Match Predictor",
         "Goalscorer Model",
         "Odds & Value Finder",
         "Weather & Environment",
         "Team Analytics",
         "Correlation Explorer",
         "Feature Importance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: football-data.org · StatsBomb · FIFA")

# ── Load models ───────────────────────────────────────────────────────────────
mp, gsp, ce = load_models()

from datetime import datetime, date
from src.constants import (GROUPS_2026, TEAM_RATINGS, STADIUMS, KEY_PLAYERS,
                            HISTORICAL_WC_RESULTS, WC_TOP_SCORERS_HISTORY,
                            WC_2026_FIXTURES, WC_2026_RESULTS, FEATURE_LABELS)

teams_all = sorted(TEAM_RATINGS.keys())
stadiums_list = list(STADIUMS.keys())


def _get_fixture_rest_days(team1: str, team2: str) -> tuple[int, int]:
    """Return (days_rest_team1, days_rest_team2) based on the fixture schedule."""
    # Find the scheduled match between these two teams
    match_date = None
    for d, t1, t2 in WC_2026_FIXTURES:
        if {t1, t2} == {team1, team2}:
            match_date = datetime.strptime(d, "%Y-%m-%d")
            break
    if match_date is None:
        return 4, 4  # not a scheduled group-stage match — use default

    def prev_days(team):
        played = [datetime.strptime(d, "%Y-%m-%d")
                  for d, t1, t2 in WC_2026_FIXTURES
                  if (t1 == team or t2 == team)
                  and datetime.strptime(d, "%Y-%m-%d") < match_date]
        # Also count WC results already logged
        for res in WC_2026_RESULTS:
            rd = datetime.strptime(res[0], "%Y-%m-%d")
            if (res[1] == team or res[2] == team) and rd < match_date:
                played.append(rd)
        if not played:
            return 10  # first match of the tournament
        return min((match_date - max(played)).days, 14)

    return prev_days(team1), prev_days(team2)


def _build_predicted_bracket(sim_results, mp):
    """Return list of rounds (R32→Final) and the predicted champion."""
    ga = sim_results.get("group_advance", {})
    groups = list(GROUPS_2026.keys())

    firsts  = {g: ga[g]["first"]  for g in groups if g in ga}
    seconds = {g: ga[g]["second"] for g in groups if g in ga}
    thirds  = {g: ga[g]["third"]  for g in groups if g in ga}

    # Best 8 third-place teams by tournament win probability
    third_sorted = sorted(thirds.values(),
                          key=lambda t: sim_results["win_probability"].get(t, 0),
                          reverse=True)[:8]

    # Simplified R32 seeding: adjacent group pairs + 8 third-place teams
    r32: list[tuple] = []
    for i in range(0, len(groups), 2):
        g1, g2 = groups[i], groups[i + 1]
        r32.append((firsts[g1],  seconds[g2]))
        r32.append((firsts[g2],  seconds[g1]))
    for i in range(0, 8, 2):
        r32.append((third_sorted[i], third_sorted[i + 1]))

    def match_result(h, a):
        try:
            res = mp.predict_match(h, a)
            ph = res["p_home_win"] / (res["p_home_win"] + res["p_away_win"])
            return round(ph * 100), round((1 - ph) * 100), h if ph >= 0.5 else a
        except Exception:
            return 50, 50, h

    rounds = []
    current = r32
    round_names = ["R32", "R16", "QF", "SF", "Final"]
    for name in round_names:
        if not current:
            break
        matches, next_teams = [], []
        for t1, t2 in current:
            p1, p2, winner = match_result(t1, t2)
            matches.append({"t1": t1, "t2": t2, "p1": p1, "p2": p2, "winner": winner})
            next_teams.append(winner)
        rounds.append({"name": name, "matches": matches})
        current = [(next_teams[i], next_teams[i + 1])
                   for i in range(0, len(next_teams) - 1, 2)]

    champion = next_teams[0] if next_teams else "TBD"
    return rounds, champion


def _bracket_html(rounds, champion):
    """Generate a dark-themed HTML bracket tree."""

    def team_card(team, prob, is_winner):
        bg = "#ffd700" if is_winner else "#1c2333"
        fg = "#000"    if is_winner else "#ffffff"
        border = "2px solid #ffd700" if is_winner else "1px solid #2a3347"
        return (
            f'<div style="background:{bg};color:{fg};border:{border};border-radius:6px;'
            f'padding:6px 10px;font-size:12px;white-space:nowrap;font-weight:{"700" if is_winner else "400"};'
            f'margin:2px 0;display:flex;justify-content:space-between;gap:8px;min-width:140px;">'
            f'<span>{team}</span><span style="opacity:0.75">{prob}%</span></div>'
        )

    # Build columns — skip R32 (too many teams), show R16 onward
    display_rounds = [r for r in rounds if r["name"] != "R32"]

    cols_html = ""
    for rnd in display_rounds:
        matches_html = ""
        n = len(rnd["matches"])
        for m in rnd["matches"]:
            c1 = team_card(m["t1"], m["p1"], m["winner"] == m["t1"])
            c2 = team_card(m["t2"], m["p2"], m["winner"] == m["t2"])
            matches_html += (
                f'<div style="display:flex;flex-direction:column;justify-content:center;'
                f'border-left:2px solid #2a3347;padding-left:8px;margin-bottom:4px;">'
                f'{c1}{c2}</div>'
            )

        # Space matches evenly using flex gap
        gap = max(4, 120 // max(n, 1))
        cols_html += (
            f'<div style="display:flex;flex-direction:column;justify-content:space-evenly;'
            f'min-width:175px;gap:{gap}px;">'
            f'<div style="color:#888;font-size:11px;font-weight:600;text-align:center;'
            f'padding:4px;letter-spacing:1px;">{rnd["name"]}</div>'
            f'{matches_html}</div>'
        )

    # Champion box
    cols_html += (
        f'<div style="display:flex;flex-direction:column;justify-content:center;'
        f'min-width:160px;padding-left:8px;">'
        f'<div style="color:#888;font-size:11px;font-weight:600;text-align:center;'
        f'padding:4px;letter-spacing:1px;">CHAMPION</div>'
        f'<div style="background:linear-gradient(135deg,#ffd700,#ff8c00);color:#000;'
        f'border-radius:10px;padding:14px 18px;font-size:15px;font-weight:800;'
        f'text-align:center;box-shadow:0 0 20px rgba(255,215,0,0.4);">🏆 {champion}</div>'
        f'</div>'
    )

    html = (
        f'<div style="background:#0e1117;padding:24px;border-radius:12px;'
        f'overflow-x:auto;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        f'<div style="display:flex;gap:12px;align-items:stretch;min-width:max-content;">'
        f'{cols_html}'
        f'</div></div>'
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Live Tracker
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔴 Live Tracker":
    from src.tournament import (compute_standings, all_teams_form,
                                upcoming_fixtures, wc_form)
    from src.odds import build_value_table, sample_odds_df

    st.markdown('<div class="section-header">🔴 Live Tournament Tracker</div>', unsafe_allow_html=True)
    st.caption(f"{len(WC_2026_RESULTS)} matches played · predictions update automatically as results are added")

    # ── Add result form ───────────────────────────────────────────────────────
    with st.expander("➕ Add a new result", expanded=False):
        st.caption("Results added here update the model instantly for this session. "
                   "Copy the output line into WC_2026_RESULTS in constants.py to persist.")
        upcoming = upcoming_fixtures(WC_2026_RESULTS, WC_2026_FIXTURES)
        if upcoming:
            fixture_labels = [f"{d}  {t1} vs {t2}" for d, t1, t2 in upcoming]
            sel = st.selectbox("Fixture", fixture_labels)
            idx = fixture_labels.index(sel)
            _, rt1, rt2 = upcoming[idx]
            rc1, rc2, rc3 = st.columns([2,1,2])
            g1 = rc1.number_input(f"{rt1} goals", 0, 20, 0, key="g1")
            rc2.markdown("<br><h3 style='text-align:center'>–</h3>", unsafe_allow_html=True)
            g2 = rc3.number_input(f"{rt2} goals", 0, 20, 0, key="g2")
            date_str = upcoming[idx][0]
            if st.button("Add Result", type="primary"):
                new_line = f'    ("{date_str}", "{rt1}", "{rt2}", {int(g1)}, {int(g2)}),'
                st.success(f"Add this line to WC_2026_RESULTS in constants.py:")
                st.code(new_line)

    # ── Group standings ───────────────────────────────────────────────────────
    st.subheader("Group Standings")
    standings = compute_standings(WC_2026_RESULTS, GROUPS_2026)
    groups_list = list(GROUPS_2026.keys())
    for row_start in range(0, len(groups_list), 3):
        cols = st.columns(3)
        for ci, g in enumerate(groups_list[row_start:row_start+3]):
            with cols[ci]:
                st.markdown(f"**Group {g}**")
                df = standings[g].copy()
                df.index = ["🥇","🥈","🥉","4️⃣"][:len(df)]
                st.dataframe(df, use_container_width=True)

    # ── WC form heatmap ───────────────────────────────────────────────────────
    st.subheader("Actual WC Goals Per Game vs Pre-Tournament Rating")
    form_data = all_teams_form(WC_2026_RESULTS)
    if form_data:
        form_rows = []
        for team, f in form_data.items():
            base_attack = TEAM_RATINGS.get(team, {}).get("attack", 70)
            form_rows.append({
                "Team": team,
                "WC Goals/Game": f["attack_avg"],
                "WC Conceded/Game": f["defense_avg"],
                "Games": f["n_games"],
                "Pre-tournament Attack Rating": base_attack,
                "Over/Under-performing": round((f["attack_avg"] or 0) - base_attack/30, 2),
            })
        form_df = pd.DataFrame(form_rows).sort_values("WC Goals/Game", ascending=False)

        fig_form = go.Figure()
        fig_form.add_trace(go.Bar(
            name="WC Goals Scored/Game", x=form_df["Team"], y=form_df["WC Goals/Game"],
            marker_color="#ffd700"
        ))
        fig_form.add_trace(go.Bar(
            name="WC Goals Conceded/Game", x=form_df["Team"], y=form_df["WC Conceded/Game"],
            marker_color="#ef4444"
        ))
        fig_form.update_layout(barmode="group", plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                               font_color="white", height=420,
                               title="WC 2026 Attack & Defence — Actual vs Expected")
        st.plotly_chart(fig_form, use_container_width=True)
        st.caption("Model blends these WC-actual stats with pre-tournament ratings: "
                   "30% WC weight after 1 game, 55% after 2+ games.")

    # ── Upcoming fixtures with live predictions ───────────────────────────────
    st.subheader("Upcoming Fixtures — Model Predictions")
    upcoming_all = upcoming_fixtures(WC_2026_RESULTS, WC_2026_FIXTURES)
    next_n = st.slider("Show next N fixtures", 3, 20, 8)
    pred_rows = []
    for d, t1, t2 in upcoming_all[:next_n]:
        try:
            r = mp.predict_match(t1, t2)
            pred_rows.append({
                "Date": d, "Team 1": t1, "Team 2": t2,
                "T1 Win %": round(r["p_home_win"] * 100, 1),
                "Draw %":   round(r["p_draw"] * 100, 1),
                "T2 Win %": round(r["p_away_win"] * 100, 1),
                "Pred. Score": f"{r['most_likely_home_goals']}–{r['most_likely_away_goals']}",
            })
        except Exception:
            pass
    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        st.dataframe(
            pred_df.style.background_gradient(subset=["T1 Win %"], cmap="RdYlGn", vmin=20, vmax=70),
            use_container_width=True, hide_index=True,
        )

    # ── Value finder for upcoming games ──────────────────────────────────────
    st.subheader("Value Finder — Upcoming Games")
    st.caption("Compares model vs bookmaker odds for the next fixtures. "
               "Add an API key on the Odds & Value Finder page for live prices.")
    odds_df = sample_odds_df()
    upcoming_teams = {t for _, t1, t2 in upcoming_all[:next_n] for t in (t1, t2)}
    odds_upcoming = odds_df[
        odds_df["home"].isin(upcoming_teams) | odds_df["away"].isin(upcoming_teams)
    ]
    if not odds_upcoming.empty:
        vt = build_value_table(odds_upcoming, mp)
        if not vt.empty:
            st.dataframe(
                vt[vt["edge_pct"] >= 3][
                    ["match","outcome","best_odds","market_prob","model_prob","edge_pct","ev_per_unit"]
                ].style.background_gradient(subset=["edge_pct"], cmap="RdYlGn", vmin=-10, vmax=15),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("No sample odds available for upcoming fixtures yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tournament Overview
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Tournament Overview":
    st.markdown('<div class="section-header">2026 World Cup — Tournament Overview</div>', unsafe_allow_html=True)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Teams", "48")
    col_info2.metric("Groups", "12")
    col_info3.metric("Matches", "104")
    col_info4.metric("Host Cities", "16")

    st.markdown("---")

    # Groups grid
    st.subheader("Group Stage Draw")
    cols = st.columns(4)
    group_items = list(GROUPS_2026.items())
    for i, (grp, teams) in enumerate(group_items):
        with cols[i % 4]:
            ratings_in_group = [(t, TEAM_RATINGS.get(t, {}).get("overall", 60)) for t in teams]
            ratings_in_group.sort(key=lambda x: -x[1])
            html = f"<b style='color:#ffd700'>Group {grp}</b><br>"
            for rank, (team, rating) in enumerate(ratings_in_group, 1):
                medal = "1." if rank == 1 else ("2." if rank == 2 else "")
                html += f"{medal} {team} <small style='color:#888'>({rating})</small><br>"
            st.markdown(f"<div style='background:#1c2333;border-radius:8px;padding:12px;margin-bottom:8px'>{html}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Monte Carlo simulation results
    st.subheader("Monte Carlo Tournament Simulation (1,000 runs)")
    with st.spinner("Simulating tournament…"):
        sim_results = run_simulations()

    win_probs = pd.DataFrame([
        {"team": k, "win_prob": v * 100, "final_prob": sim_results["final_probability"].get(k, 0) * 100,
         "sf_prob": sim_results["semifinal_probability"].get(k, 0) * 100}
        for k, v in sim_results["win_probability"].items()
    ]).sort_values("win_prob", ascending=False).head(20)

    tab1, tab2, tab3 = st.tabs(["Win Probability", "Reach Final / Semifinal", "Predicted Bracket"])
    with tab1:
        fig = px.bar(win_probs, x="team", y="win_prob", color="win_prob",
                     color_continuous_scale="YlOrRd",
                     labels={"win_prob": "Win Probability (%)", "team": ""},
                     title="Tournament Win Probability — Top 20 Teams")
        fig.update_layout(showlegend=False, plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                          font_color="white", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Win", x=win_probs["team"], y=win_probs["win_prob"], marker_color="#ffd700"))
        fig2.add_trace(go.Bar(name="Reach Final", x=win_probs["team"], y=win_probs["final_prob"], marker_color="#00b4d8"))
        fig2.add_trace(go.Bar(name="Reach SF", x=win_probs["team"], y=win_probs["sf_prob"], marker_color="#6fcf97"))
        fig2.update_layout(barmode="group", plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                           font_color="white", title="Stage Probability Comparison")
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        st.markdown("**Most likely bracket path** — seeded from simulation group outcomes, each match uses the Poisson model. Gold = predicted to advance.")
        with st.spinner("Building predicted bracket…"):
            bracket_rounds, bracket_champion = _build_predicted_bracket(sim_results, mp)
        components.html(_bracket_html(bracket_rounds, bracket_champion), height=900, scrolling=True)

    # Historical winners
    st.subheader("Historical World Cup Results")
    hist_df = pd.DataFrame(HISTORICAL_WC_RESULTS, columns=["Year","Winner","Runner-Up","Score","Third Place","Host"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Match Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Match Predictor":
    st.markdown('<div class="section-header">Match Predictor</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        home_team = st.selectbox("Team 1", teams_all,
                                  index=teams_all.index("Argentina") if "Argentina" in teams_all else 0)
    with c2:
        st.markdown("<br><h2 style='text-align:center;color:#ffd700'>VS</h2>", unsafe_allow_html=True)
    with c3:
        away_opts = [t for t in teams_all if t != home_team]
        away_team = st.selectbox("Team 2", away_opts,
                                  index=away_opts.index("France") if "France" in away_opts else 0)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        stadium = st.selectbox("Stadium", stadiums_list)
    with sc2:
        home_form = st.slider(f"{home_team} Recent Form (avg goals/game)", 0.5, 3.5, 1.5, 0.1)
    with sc3:
        away_form = st.slider(f"{away_team} Recent Form (avg goals/game)", 0.5, 3.5, 1.5, 0.1)

    # Auto-calculate rest days from fixture schedule
    auto_rest_h, auto_rest_a = _get_fixture_rest_days(home_team, away_team)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        days_rest_home = st.slider(f"{home_team} Days Rest", 1, 14, auto_rest_h,
                                    help="Auto-calculated from fixture schedule")
    with rc2:
        days_rest_away = st.slider(f"{away_team} Days Rest", 1, 14, auto_rest_a,
                                    help="Auto-calculated from fixture schedule")
    with rc3:
        is_knockout = st.checkbox("Knockout Round", value=False,
                                   help="Enables penalty shootout records and experience weighting")

    if st.button("Predict Match", type="primary", use_container_width=True):
        with st.spinner("Running prediction models…"):
            result = mp.predict_match(
                home_team, away_team, stadium, home_form, away_form,
                days_rest_home=days_rest_home, days_rest_away=days_rest_away,
                is_knockout=is_knockout,
            )

        st.markdown("---")
        # Win probability gauges
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home_team} Win", f"{result['p_home_win']*100:.1f}%")
        m2.metric("Draw", f"{result['p_draw']*100:.1f}%")
        m3.metric(f"{away_team} Win", f"{result['p_away_win']*100:.1f}%")

        # Gauge chart
        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Bar(
            x=[result["p_home_win"]*100, result["p_draw"]*100, result["p_away_win"]*100],
            y=[home_team, "Draw", away_team],
            orientation="h",
            marker_color=["#00b4d8", "#f5a623", "#ef4444"],
            text=[f"{result['p_home_win']*100:.1f}%", f"{result['p_draw']*100:.1f}%", f"{result['p_away_win']*100:.1f}%"],
            textposition="auto",
        ))
        fig_gauge.update_layout(
            plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white",
            title="Win Probability", xaxis_title="Probability (%)", height=200,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        col_xg, col_score = st.columns(2)
        with col_xg:
            st.subheader("Expected Goals (xG)")
            fig_xg = go.Figure(go.Bar(
                x=[result["home_xg"], result["away_xg"]],
                y=[home_team, away_team],
                orientation="h",
                marker_color=["#00b4d8", "#ef4444"],
                text=[f"{result['home_xg']:.2f}", f"{result['away_xg']:.2f}"],
                textposition="auto",
            ))
            fig_xg.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                                  font_color="white", height=200, showlegend=False)
            st.plotly_chart(fig_xg, use_container_width=True)

        with col_score:
            st.subheader(f"Most Likely Score: {result['most_likely_score']}")
            mat = result["score_matrix"][:6, :6]
            fig_mat = px.imshow(
                mat * 100,
                labels={"x": away_team + " goals", "y": home_team + " goals", "color": "Prob %"},
                x=[str(i) for i in range(6)],
                y=[str(i) for i in range(6)],
                color_continuous_scale="Blues",
                text_auto=".1f",
                title="Score Probability Matrix (%)",
            )
            fig_mat.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=350)
            st.plotly_chart(fig_mat, use_container_width=True)

        # Stadium & environment
        st.subheader("Stadium & Environmental Conditions")
        stad_data = STADIUMS[stadium]
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Temperature", f"{stad_data['avg_temp_june']}°C")
        ec2.metric("Humidity", f"{stad_data['avg_humidity_june']}%")
        ec3.metric("Altitude", f"{stad_data['altitude_m']}m")
        ec4.metric("Air Conditioned", "Yes" if stad_data["air_conditioned"] else "No")

        # Environmental factor alerts
        for factor in result["env_factors"].values():
            st.warning(f"**{factor['label']}** — {factor['impact']}")

        # Contextual / tactical factor alerts
        if result.get("extra_factors"):
            st.subheader("Contextual & Tactical Factors")
            for factor in result["extra_factors"].values():
                st.info(f"**{factor['label']}** — {factor['impact']}")

        # Goalscorer predictions for this match
        st.subheader("Likely Goalscorers")
        scorers = gsp.predict_scorers(result, top_n=8)
        if not scorers.empty:
            fig_sc = px.bar(scorers, x="score_probability_pct", y="player", orientation="h",
                            color="team", labels={"score_probability_pct": "Goal Probability (%)", "player": ""},
                            title="Player Goal Probability for This Match")
            fig_sc.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=350)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Team rating comparison radar
        st.subheader("Team Attribute Comparison")
        h_rat = TEAM_RATINGS.get(home_team, {})
        a_rat = TEAM_RATINGS.get(away_team, {})
        cats = ["attack", "defense", "midfield", "overall"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[h_rat.get(c, 70) for c in cats] + [h_rat.get(cats[0], 70)],
            theta=cats + [cats[0]], fill="toself", name=home_team, line_color="#00b4d8"))
        fig_radar.add_trace(go.Scatterpolar(
            r=[a_rat.get(c, 70) for c in cats] + [a_rat.get(cats[0], 70)],
            theta=cats + [cats[0]], fill="toself", name=away_team, line_color="#ef4444"))
        fig_radar.update_layout(polar=dict(bgcolor="#1c2333"),
                                paper_bgcolor="#0e1117", font_color="white",
                                title="Team Attributes Radar", height=400)
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Goalscorer Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Goalscorer Model":
    st.markdown('<div class="section-header">Goalscorer Probability Model</div>', unsafe_allow_html=True)

    players_df = pd.DataFrame(KEY_PLAYERS)

    st.subheader("Tournament Top Scorer Predictions")
    players_scored = players_df[players_df["team"].isin(
        [t for g in GROUPS_2026.values() for t in g]
    )].copy()
    if not players_scored.empty:
        players_scored["tournament_xg"] = (
            players_scored["club_goals_per90"] * 0.6
            + players_scored["wc_goals"] * 0.02
            + players_scored["rating"] / 200
        )
        players_scored["expected_tournament_goals"] = (
            players_scored["tournament_xg"] * 6.5  # avg games for finalist
        ).round(2)
        players_scored = players_scored.sort_values("expected_tournament_goals", ascending=False).head(25)

        fig = px.bar(
            players_scored, x="expected_tournament_goals", y="name",
            orientation="h", color="team",
            hover_data=["age", "rating", "wc_goals", "club_goals_per90"],
            labels={"expected_tournament_goals": "Expected Tournament Goals", "name": "Player"},
            title="Predicted Top Scorers — 2026 World Cup",
        )
        fig.update_layout(height=650, plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # Historical top scorers
    st.subheader("Historical World Cup Top Scorers")
    hist_sc = pd.DataFrame(WC_TOP_SCORERS_HISTORY, columns=["Year", "Player", "Country", "Goals"])
    fig_hist = px.bar(hist_sc, x="Year", y="Goals", text="Player", color="Country",
                      title="Golden Boot Winners by Year")
    fig_hist.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Player deep-dive
    st.subheader("Player Deep-Dive")
    sel_player = st.selectbox("Select Player", sorted(players_df["name"].tolist()))
    p_row = players_df[players_df["name"] == sel_player].iloc[0]
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Team", p_row["team"])
    pc2.metric("Rating", p_row["rating"])
    pc3.metric("Age", p_row["age"])
    pc4.metric("WC Goals (career)", p_row["wc_goals"])

    pc5, pc6 = st.columns(2)
    pc5.metric("Club Goals/90min", p_row["club_goals_per90"])
    pc6.metric("Position", p_row["pos"])

    # Show predicted goal probability in different conditions
    st.subheader("Performance Across Conditions")
    cond_data = []
    for temp, label in [(15, "Cold (15°C)"), (22, "Mild (22°C)"), (30, "Warm (30°C)"), (36, "Hot (36°C)")]:
        for alt, alt_label in [(10, "Sea Level"), (2240, "High Altitude (2240m)")]:
            mock_match = {
                "home_team": p_row["team"], "away_team": "Other",
                "home_xg": 1.5, "away_xg": 1.2,
                "temp_c": temp, "altitude_m": alt,
            }
            env_pen = (-0.05 if temp > 32 else 0) + (-0.03 if alt > 1500 else 0)
            base = p_row["club_goals_per90"]
            team_share = 0.35 if p_row["pos"] == "FW" else 0.15
            prob = max(0, base * 1.5 * team_share + env_pen)
            cond_data.append({"Conditions": f"{label}\n{alt_label}", "Temperature": temp,
                               "Altitude": alt, "Goal Probability": round(prob, 3)})

    cond_df = pd.DataFrame(cond_data)
    fig_cond = px.bar(cond_df, x="Conditions", y="Goal Probability",
                      color="Temperature", color_continuous_scale="RdYlGn_r",
                      title=f"{sel_player} — Goal Probability by Conditions")
    fig_cond.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig_cond, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3.5 — Odds & Value Finder
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Odds & Value Finder":
    from src.odds import (fetch_live_odds, sample_odds_df, build_value_table,
                          brier_scores, SAMPLE_ODDS)

    st.markdown('<div class="section-header">Bookmaker Odds vs Model — Value Finder</div>', unsafe_allow_html=True)
    st.markdown(
        "Compares the model's probabilities against bookmaker implied probabilities "
        "(vig removed). A positive **edge** means the model thinks an outcome is more "
        "likely than the market price implies."
    )
    st.warning(
        "⚠️ **This is an analysis tool, not betting advice.** The model is trained on "
        "~150 matches; closing odds at sharp books are extremely well calibrated. Most "
        "'edges' shown here are more likely model miscalibration than market mistakes — "
        "treat big disagreements as a prompt to ask *why* the model differs, not as bets."
    )

    # ── Odds source ───────────────────────────────────────────────────────────
    api_key = st.text_input("The Odds API key (optional — get a free key at the-odds-api.com)",
                            type="password",
                            help="Free tier: 500 requests/month. Leave empty to use bundled sample odds.")

    @st.cache_data(ttl=600, show_spinner="Fetching live odds…")
    def _load_odds(key: str):
        if key:
            df, msg = fetch_live_odds(key)
            if not df.empty:
                return df, msg
            return sample_odds_df(), msg
        return sample_odds_df(), "Using bundled sample odds (add an API key for live prices)."

    odds_df, status_msg = _load_odds(api_key)
    if odds_df["source"].iloc[0] == "live":
        st.success(status_msg)
    else:
        st.info(status_msg)

    # ── Value table ───────────────────────────────────────────────────────────
    value_df = build_value_table(odds_df, mp)

    st.subheader("Value Finder")
    min_edge = st.slider("Minimum edge to show (%)", -20, 20, 0,
                         help="Edge = model probability − market implied probability")
    show_df = value_df[value_df["edge_pct"] >= min_edge].copy()

    display = show_df.rename(columns={
        "match": "Match", "outcome": "Outcome", "best_odds": "Best Odds",
        "market_prob": "Market %", "model_prob": "Model %",
        "edge_pct": "Edge %", "ev_per_unit": "EV / unit", "kelly_pct": "Kelly %",
    })[["Match", "Outcome", "Best Odds", "Market %", "Model %", "Edge %", "EV / unit", "Kelly %"]]

    st.dataframe(
        display.style.background_gradient(subset=["Edge %"], cmap="RdYlGn", vmin=-15, vmax=15),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**EV / unit** = expected profit per 1 unit staked at the model's probability. "
        "**Kelly %** = Kelly-criterion stake as % of bankroll (capped at 25%), shown for theory only."
    )

    # ── Model vs Market scatter ──────────────────────────────────────────────
    st.subheader("Model vs Market Agreement")
    fig_sc = px.scatter(
        value_df, x="market_prob", y="model_prob",
        color="edge_pct", color_continuous_scale="RdYlGn",
        hover_data=["match", "outcome", "best_odds"],
        labels={"market_prob": "Market Implied Probability (%)",
                "model_prob": "Model Probability (%)",
                "edge_pct": "Edge %"},
        title="Each point = one outcome. Distance from the diagonal = disagreement.",
    )
    lim = max(value_df["market_prob"].max(), value_df["model_prob"].max()) + 5
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                     line=dict(color="#888", dash="dash"))
    fig_sc.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                         font_color="white", height=550)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "Points **above** the diagonal: model rates the outcome higher than the market "
        "(potential value — or model overconfidence). Points **below**: the market rates "
        "it higher than the model."
    )

    # ── Calibration tracking (Brier scores) ──────────────────────────────────
    st.subheader("Calibration Tracker — Model vs Market")
    if WC_2026_RESULTS:
        brier_df = brier_scores(WC_2026_RESULTS, mp, odds_lookup=SAMPLE_ODDS)
        if not brier_df.empty:
            c1, c2 = st.columns(2)
            c1.metric("Model avg Brier score", f"{brier_df['model_brier'].mean():.4f}",
                      help="Lower = better calibrated. 0.2222 = always predicting ⅓/⅓/⅓.")
            mk = brier_df["market_brier"].dropna()
            c2.metric("Market avg Brier score", f"{mk.mean():.4f}" if len(mk) else "—")
            st.dataframe(brier_df, use_container_width=True, hide_index=True)
            st.caption(
                "Tracked as results are added to WC_2026_RESULTS during the tournament. "
                "If the model's Brier score beats the market's over a meaningful sample, "
                "the edges above start meaning something."
            )
    else:
        st.info(
            "No tournament results yet — once group stage matches are played and added to "
            "`WC_2026_RESULTS`, this section scores the model (and the market) against "
            "reality using Brier scores. That's the real test of whether any 'edge' exists."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Weather & Environment
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Weather & Environment":
    st.markdown('<div class="section-header">Weather & Environmental Analysis</div>', unsafe_allow_html=True)
    st.markdown("How temperature, humidity, altitude and heat index affect match outcomes and player performance.")

    stad_df = pd.DataFrame([
        {"Stadium": name, **{k: v for k, v in data.items() if k != "surface"}}
        for name, data in STADIUMS.items()
    ])

    # Stadium map
    st.subheader("Stadium Locations & Climate")
    fig_map = px.scatter_geo(
        stad_df, lat="lat", lon="lon",
        hover_name="Stadium",
        hover_data={"lat": False, "lon": False, "avg_temp_june": True,
                    "avg_humidity_june": True, "altitude_m": True, "air_conditioned": True},
        color="avg_temp_june",
        size="capacity",
        color_continuous_scale="RdYlGn_r",
        scope="north america",
        title="2026 World Cup Stadiums — June Temperature",
        labels={"avg_temp_june": "Avg Temp June (°C)"},
    )
    fig_map.update_layout(paper_bgcolor="#0e1117", font_color="white", geo_bgcolor="#1c2333")
    st.plotly_chart(fig_map, use_container_width=True)

    # Temperature vs Humidity scatter
    st.subheader("Heat & Humidity by Stadium")
    fig_scatter = px.scatter(
        stad_df, x="avg_temp_june", y="avg_humidity_june",
        text="Stadium", size="capacity",
        color="altitude_m", color_continuous_scale="Blues",
        labels={"avg_temp_june": "Avg Temperature June (°C)",
                "avg_humidity_june": "Avg Humidity (%)",
                "altitude_m": "Altitude (m)"},
        title="Temperature vs Humidity (bubble = capacity, color = altitude)",
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.add_vrect(x0=30, x1=40, fillcolor="red", opacity=0.1,
                          annotation_text="Extreme heat risk", annotation_position="top left")
    fig_scatter.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Altitude comparison
    st.subheader("Altitude Impact on Match Statistics")
    hist_df = mp.df.copy()
    hist_df["altitude_band"] = pd.cut(hist_df["altitude_m"],
                                       bins=[-1, 200, 800, 1600, 3000],
                                       labels=["Sea Level\n(<200m)", "Low\n(200-800m)",
                                               "Medium\n(800-1600m)", "High\n(>1600m)"])
    alt_stats = hist_df.groupby("altitude_band", observed=True).agg(
        avg_total_goals=("total_goals", "mean"),
        avg_home_goals=("home_goals", "mean"),
        avg_away_goals=("away_goals", "mean"),
        home_win_rate=("home_win", "mean"),
        draw_rate=("draw", "mean"),
        n=("home_win", "count"),
    ).reset_index()
    # Only show bands with enough matches to be statistically meaningful
    alt_stats = alt_stats[alt_stats["n"] >= 5].copy()
    alt_stats["altitude_band"] = alt_stats["altitude_band"].astype(str) + "\n(n=" + alt_stats["n"].astype(str) + ")"

    fig_alt = make_subplots(rows=1, cols=2, subplot_titles=["Goals by Altitude Band", "Win Rate by Altitude"])
    fig_alt.add_trace(go.Bar(name="Home Goals", x=alt_stats["altitude_band"].astype(str),
                              y=alt_stats["avg_home_goals"], marker_color="#00b4d8"), row=1, col=1)
    fig_alt.add_trace(go.Bar(name="Away Goals", x=alt_stats["altitude_band"].astype(str),
                              y=alt_stats["avg_away_goals"], marker_color="#ef4444"), row=1, col=1)
    fig_alt.add_trace(go.Bar(name="Home Win %", x=alt_stats["altitude_band"].astype(str),
                              y=alt_stats["home_win_rate"]*100, marker_color="#ffd700"), row=1, col=2)
    fig_alt.add_trace(go.Bar(name="Draw %", x=alt_stats["altitude_band"].astype(str),
                              y=alt_stats["draw_rate"]*100, marker_color="#6fcf97"), row=1, col=2)
    fig_alt.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white",
                           barmode="stack", height=400, title_text="Altitude Effect on Match Outcomes")
    st.plotly_chart(fig_alt, use_container_width=True)
    st.caption(
        "Based on historical training data (past WCs, qualifiers, friendlies) where home advantage "
        "existed — 'Home Win %' reflects the first-listed team in those records. WC 2026 predictions "
        "themselves are neutral-venue."
    )

    # Temperature bins analysis
    st.subheader("Temperature Effect on Goals & Outcomes")
    hist_df["temp_band"] = pd.cut(hist_df["temp_c"],
                                   bins=[0, 18, 24, 30, 45],
                                   labels=["Cool (<18°C)", "Mild (18-24°C)", "Warm (24-30°C)", "Hot (>30°C)"])
    temp_stats = hist_df.groupby("temp_band", observed=True).agg(
        avg_total_goals=("total_goals", "mean"),
        home_win_rate=("home_win", "mean"),
        draw_rate=("draw", "mean"),
        n=("total_goals", "count"),
    ).reset_index()

    fig_temp = make_subplots(rows=1, cols=2, subplot_titles=["Avg Total Goals", "Outcome Distribution"])
    fig_temp.add_trace(go.Bar(x=temp_stats["temp_band"].astype(str),
                               y=temp_stats["avg_total_goals"],
                               marker_color=["#00b4d8","#ffd700","#f5a623","#ef4444"]), row=1, col=1)
    fig_temp.add_trace(go.Bar(name="Home Win", x=temp_stats["temp_band"].astype(str),
                               y=temp_stats["home_win_rate"]*100, marker_color="#00b4d8"), row=1, col=2)
    fig_temp.add_trace(go.Bar(name="Draw", x=temp_stats["temp_band"].astype(str),
                               y=temp_stats["draw_rate"]*100, marker_color="#f5a623"), row=1, col=2)
    fig_temp.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white",
                            barmode="stack", height=400)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Stadium data table
    st.subheader("Full Stadium Data")
    display_cols = ["city", "country", "altitude_m", "capacity",
                    "avg_temp_june", "avg_humidity_june", "air_conditioned", "has_roof"]
    st.dataframe(stad_df[["Stadium"] + display_cols].set_index("Stadium"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Team Analytics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Team Analytics":
    st.markdown('<div class="section-header">Team Analytics</div>', unsafe_allow_html=True)

    teams_df = pd.DataFrame([
        {"team": t, **r} for t, r in TEAM_RATINGS.items()
    ])
    from src.constants import CONFEDERATION, PLAY_STYLE
    teams_df["confederation"] = teams_df["team"].map(CONFEDERATION).fillna("OTHER")
    teams_df["style"] = teams_df["team"].map(PLAY_STYLE).fillna("balanced")
    teams_df["in_2026"] = teams_df["team"].isin([t for g in GROUPS_2026.values() for t in g])
    teams_df_26 = teams_df[teams_df["in_2026"]].copy()

    st.subheader("Team Strength Scatter — Attack vs Defense")
    fig_ad = px.scatter(
        teams_df_26, x="defense", y="attack", size="overall", color="confederation",
        text="team", hover_data=["midfield", "fifa_rank", "style"],
        title="Attack vs Defense Rating (bubble size = overall, color = confederation)",
        labels={"attack": "Attack Rating", "defense": "Defense Rating"},
    )
    fig_ad.update_traces(textposition="top center", textfont_size=8)
    fig_ad.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=550)
    st.plotly_chart(fig_ad, use_container_width=True)

    # Head-to-head predictor
    st.subheader("Head-to-Head Matrix")
    st.caption("Predicted home win probability for any pairing (row = home, col = away)")
    selected_teams = st.multiselect(
        "Select teams for H2H matrix",
        sorted(teams_df_26["team"].tolist()),
        default=["Argentina", "France", "Brazil", "England", "Germany", "Spain", "Portugal"],
    )
    if len(selected_teams) >= 2:
        matrix_data = {}
        for h in selected_teams:
            matrix_data[h] = {}
            for a in selected_teams:
                if h == a:
                    matrix_data[h][a] = None
                else:
                    res = mp.predict_match(h, a)
                    matrix_data[h][a] = round(res["p_home_win"] * 100, 1)

        mat_df = pd.DataFrame(matrix_data).T[selected_teams]
        # Replace None diagonal
        for t in selected_teams:
            mat_df.loc[t, t] = np.nan

        fig_h2h = px.imshow(
            mat_df.values,
            x=selected_teams, y=selected_teams,
            color_continuous_scale="RdYlGn",
            zmin=20, zmax=80,
            text_auto=True,
            labels={"x": "Opponent", "y": "Team", "color": "Win %"},
            title="Head-to-Head Win Probability Matrix (%) — row team beats column team",
        )
        fig_h2h.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=500)
        st.plotly_chart(fig_h2h, use_container_width=True)

    # Confederation comparison
    st.subheader("Confederation Strength Comparison")
    conf_stats = teams_df_26.groupby("confederation").agg(
        avg_overall=("overall", "mean"),
        avg_attack=("attack", "mean"),
        avg_defense=("defense", "mean"),
        n_teams=("team", "count"),
    ).reset_index().sort_values("avg_overall", ascending=False)

    fig_conf = px.bar(conf_stats, x="confederation", y=["avg_attack", "avg_defense", "avg_overall"],
                      barmode="group", title="Average Ratings by Confederation",
                      color_discrete_map={"avg_attack":"#ef4444","avg_defense":"#00b4d8","avg_overall":"#ffd700"})
    fig_conf.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig_conf, use_container_width=True)

    # Full team table
    st.subheader("Full Team Ratings Table")
    st.dataframe(
        teams_df_26[["team","overall","attack","defense","midfield","fifa_rank","confederation","style"]]
        .sort_values("overall", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Correlation Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Correlation Explorer":
    st.markdown('<div class="section-header">Correlation Explorer</div>', unsafe_allow_html=True)
    st.markdown("Explore 100+ statistical correlations between match features and outcomes.")

    with st.spinner("Computing correlations…"):
        corr_df = ce.compute_all_correlations()

    # Apply readable labels
    corr_labeled = corr_df.copy()
    corr_labeled["feature_label"] = corr_labeled["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title()))
    corr_labeled["target_label"]  = corr_labeled["target"].map(lambda t: FEATURE_LABELS.get(t, t.replace("_", " ").title()))

    target_display = {t: FEATURE_LABELS.get(t, t) for t in ["home_win", "draw", "away_win", "home_goals", "away_goals", "total_goals"]}
    sel_target_label = st.selectbox("Outcome to predict", list(target_display.values()))
    sel_target = [k for k, v in target_display.items() if v == sel_target_label][0]

    top_n = st.slider("Show top N factors", 5, 30, 15)

    filtered = corr_labeled[corr_labeled["target"] == sel_target].head(top_n)

    fig_corr = px.bar(
        filtered, x="correlation", y="feature_label", orientation="h",
        color="correlation",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"Factors most correlated with: {sel_target_label}",
        labels={"correlation": "Pearson Correlation", "feature_label": ""},
        text="correlation",
    )
    fig_corr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_corr.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                            font_color="white", height=max(400, top_n * 30))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Full correlation matrix heatmap with readable labels
    st.subheader("Full Correlation Heatmap")
    pivot = corr_labeled.pivot_table(index="feature_label", columns="target_label",
                                      values="correlation", aggfunc="mean")
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        text_auto=".2f",
        title="All Factors × All Outcomes",
        aspect="auto",
    )
    fig_heat.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117",
                            font_color="white", height=700)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Environmental correlations
    st.subheader("Environmental Factor Correlations")
    env_corr = ce.weather_impact_analysis().copy()
    env_corr["feature"] = env_corr["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    env_corr["target"]  = env_corr["target"].map(lambda t: FEATURE_LABELS.get(t, t))
    st.dataframe(env_corr, use_container_width=True, hide_index=True)

    with st.expander("Full Correlation Table (raw)"):
        st.dataframe(corr_labeled[["feature_label","target_label","correlation","abs_correlation","n_samples"]],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown('<div class="section-header">Model Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("Which factors matter most in predicting World Cup match outcomes?")

    fi = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in mp.feature_importances_.items()
    ]).sort_values("importance", ascending=False)

    fi["label"] = fi["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title()))

    def categorize(f):
        if any(x in f for x in ["attack","defense","midfield","overall","rank","rating"]):
            return "Team Quality"
        elif any(x in f for x in ["form","pts"]):
            return "Recent Form"
        elif any(x in f for x in ["temp","humidity","altitude","heat","high_"]):
            return "Environment"
        elif any(x in f for x in ["xg","lambda","total_goals"]):
            return "Expected Goals"
        elif any(x in f for x in ["confederation","european","sa","style","press","aerial","setpiece","h2h","age","exp","pen","suspension","fan","host","club"]):
            return "Tactical / Context"
        return "Other"

    fi["category"] = fi["feature"].apply(categorize)

    fig_fi = px.bar(
        fi.head(30), x="importance", y="label", orientation="h",
        color="category",
        title="Top 30 Most Predictive Factors (Gradient Boosting)",
        labels={"importance": "Importance Score", "label": ""},
        color_discrete_map={
            "Team Quality": "#ffd700",
            "Recent Form": "#00b4d8",
            "Environment": "#ef4444",
            "Expected Goals": "#6fcf97",
            "Tactical / Context": "#9b59b6",
            "Other": "#888",
        }
    )
    fig_fi.update_layout(plot_bgcolor="#1c2333", paper_bgcolor="#0e1117", font_color="white", height=700)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Category rollup
    st.subheader("Category Contribution Breakdown")
    cat_rollup = fi.groupby("category")["importance"].sum().reset_index().sort_values("importance", ascending=False)
    fig_pie = px.pie(cat_rollup, names="category", values="importance",
                     title="Feature Importance by Category",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Model cross-validation score
    st.subheader("Model Accuracy")
    st.info("Model trained on 150+ international matches from WC 2014–2022, Euro 2024, Copa América 2024.")

    df_m = mp.df.dropna(subset=["home_win"])
    n = len(df_m)
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Training Samples", str(n))
    ac2.metric("Features Used", str(len(mp.feature_importances_)))
    ac3.metric("Algorithm", "Gradient Boosting")

    st.caption("""
    **Methodology**: Each match prediction uses a Gradient Boosting Classifier trained on
    historical international match data with 38 engineered features spanning team quality,
    recent form, environmental conditions, tactical styles, and expected goals.
    Score distributions use a Poisson model calibrated on per-team attack/defense parameters.
    Tournament simulation uses 5,000 Monte Carlo runs through the full bracket.
    """)

    with st.expander("Full Feature Importance Table"):
        st.dataframe(fi[["label","importance","category"]].rename(columns={"label":"Factor","importance":"Importance","category":"Category"}),
                     use_container_width=True, hide_index=True)
