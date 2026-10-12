"""
Generates a rich historical dataset of international football matches
with environmental and contextual features for ML training.
"""

import numpy as np
import pandas as pd
from src.constants import (
    TEAM_RATINGS, CONFEDERATION, STADIUMS, PLAY_STYLE,
    SQUAD_AVG_AGE, WC_EXPERIENCE, PENALTY_RECORD, H2H_ADVANTAGE,
    CLUB_FORM_BONUS, SUSPENSION_RISK, PRESS_INTENSITY,
    SET_PIECE_STRENGTH, AERIAL_STRENGTH, WC_2026_RESULTS,
)

RNG = np.random.default_rng(42)

# ── Historical match results (sample of real WC + EURO + Copa matches) ────────
_RAW_MATCHES = [
    # WC 2022
    ("2022-11-20","Argentina","Saudi Arabia",1,2,"Lusail Stadium WC22",28,55),
    ("2022-11-21","England","Iran",6,2,"Khalifa WC22",25,50),
    ("2022-11-21","Senegal","Netherlands",0,2,"Al Thumama WC22",24,55),
    ("2022-11-22","France","Australia",4,1,"Al Janoub WC22",23,58),
    ("2022-11-22","Germany","Japan",1,2,"Khalifa WC22",25,50),
    ("2022-11-22","Spain","Costa Rica",7,0,"Al Thumama WC22",24,55),
    ("2022-11-23","Belgium","Canada",1,0,"Al Rayyan WC22",26,52),
    ("2022-11-23","Morocco","Croatia",0,0,"Al Bayt WC22",22,55),
    ("2022-11-24","Brazil","Serbia",2,0,"Lusail Stadium WC22",28,55),
    ("2022-11-24","Portugal","Ghana",3,2,"Stadium 974 WC22",26,60),
    ("2022-11-25","Uruguay","South Korea",0,0,"Education City WC22",23,55),
    ("2022-11-26","Japan","Costa Rica",0,1,"Al Rayyan WC22",26,52),
    ("2022-11-26","Spain","Germany",1,1,"Al Bayt WC22",22,55),
    ("2022-11-27","France","Denmark",2,1,"Stadium 974 WC22",26,60),
    ("2022-11-27","Argentina","Mexico",2,0,"Lusail Stadium WC22",28,55),
    ("2022-11-28","England","USA",0,0,"Al Bayt WC22",22,55),
    ("2022-11-28","Morocco","Belgium",2,0,"Al Thumama WC22",24,55),
    ("2022-11-29","Netherlands","Ecuador",1,1,"Khalifa WC22",25,50),
    ("2022-11-29","Senegal","Ecuador",2,1,"Khalifa WC22",25,50),
    ("2022-11-30","Spain","Japan",1,2,"Khalifa WC22",25,50),
    ("2022-12-02","Argentina","Poland",2,0,"Stadium 974 WC22",26,60),
    ("2022-12-02","France","Tunisia",1,0,"Education City WC22",23,55),
    ("2022-12-03","Australia","Denmark",1,0,"Al Rayyan WC22",26,52),
    ("2022-12-04","England","Senegal",3,0,"Al Bayt WC22",22,55),
    ("2022-12-05","Japan","Croatia",1,1,"Al Janoub WC22",23,58),
    ("2022-12-06","Brazil","South Korea",4,1,"Stadium 974 WC22",26,60),
    ("2022-12-06","Morocco","Spain",0,0,"Education City WC22",23,55),
    ("2022-12-09","Netherlands","Argentina",2,2,"Lusail Stadium WC22",28,55),
    ("2022-12-10","France","England",2,1,"Al Bayt WC22",22,55),
    ("2022-12-10","Morocco","Portugal",1,0,"Al Thumama WC22",24,55),
    ("2022-12-13","Argentina","Croatia",3,0,"Lusail Stadium WC22",28,55),
    ("2022-12-14","France","Morocco",2,0,"Al Bayt WC22",22,55),
    ("2022-12-17","Argentina","France",3,3,"Lusail Stadium WC22",28,55),

    # WC 2018
    ("2018-06-14","Russia","Saudi Arabia",5,0,"Luzhniki 2018",15,55),
    ("2018-06-15","Uruguay","Egypt",1,0,"Ekaterinburg 2018",23,50),
    ("2018-06-15","Morocco","Iran",0,1,"St Petersburg 2018",20,58),
    ("2018-06-15","Portugal","Spain",3,3,"Fisht 2018",25,62),
    ("2018-06-16","France","Australia",2,1,"Kazan 2018",22,55),
    ("2018-06-16","Argentina","Iceland",1,1,"Spartak 2018",18,58),
    ("2018-06-16","Peru","Denmark",0,1,"Mordovia 2018",25,50),
    ("2018-06-17","Croatia","Nigeria",2,0,"Kaliningrad 2018",22,60),
    ("2018-06-17","Costa Rica","Serbia",0,1,"Samara 2018",28,48),
    ("2018-06-17","Germany","Mexico",0,1,"Luzhniki 2018",15,55),
    ("2018-06-18","Brazil","Switzerland",1,1,"Rostov 2018",28,65),
    ("2018-06-19","Sweden","South Korea",1,0,"Nizhny 2018",22,55),
    ("2018-06-19","Belgium","Panama",3,0,"Fisht 2018",25,62),
    ("2018-06-19","Tunisia","England",1,2,"Volgograd 2018",30,55),
    ("2018-06-21","Colombia","Japan",1,2,"Mordovia 2018",25,50),
    ("2018-06-22","Germany","Sweden",2,1,"Fisht 2018",25,62),
    ("2018-06-22","South Korea","Mexico",2,1,"Rostov 2018",28,65),
    ("2018-06-26","England","Belgium",0,1,"Kaliningrad 2018",22,60),
    ("2018-06-26","Japan","Poland",0,1,"Volgograd 2018",30,55),
    ("2018-06-28","Germany","South Korea",0,2,"Kazan 2018",22,55),
    ("2018-06-30","France","Argentina",4,3,"Kazan 2018",22,55),
    ("2018-07-01","Uruguay","Portugal",2,1,"Fisht 2018",25,62),
    ("2018-07-02","Russia","Spain",1,1,"Luzhniki 2018",15,55),
    ("2018-07-02","Croatia","Denmark",1,1,"Nizhny 2018",22,55),
    ("2018-07-06","Uruguay","France",0,2,"Nizhny 2018",22,55),
    ("2018-07-06","Brazil","Belgium",1,2,"Kazan 2018",22,55),
    ("2018-07-07","Sweden","England",0,2,"Samara 2018",28,48),
    ("2018-07-07","Russia","Croatia",2,2,"Fisht 2018",25,62),
    ("2018-07-10","France","Belgium",1,0,"St Petersburg 2018",20,58),
    ("2018-07-11","Croatia","England",2,1,"Luzhniki 2018",15,55),
    ("2018-07-15","France","Croatia",4,2,"Luzhniki 2018",15,55),

    # WC 2014
    ("2014-06-12","Brazil","Croatia",3,1,"Arena Corinthians",22,62),
    ("2014-06-13","Colombia","Greece",3,0,"Mineirao",22,65),
    ("2014-06-15","Germany","Portugal",4,0,"Arena Fonte Nova",26,70),
    ("2014-06-17","Argentina","Bosnia",2,1,"Maracana",24,65),
    ("2014-06-17","Germany","Ghana",2,2,"Castelao",28,68),
    ("2014-06-19","Brazil","Mexico",0,0,"Castelao",28,68),
    ("2014-06-21","Germany","USA",1,0,"Arena Pernambuco",30,72),
    ("2014-06-22","France","Switzerland",5,2,"Arena Fonte Nova",26,70),
    ("2014-06-23","USA","Portugal",2,2,"Arena Amazonia",31,80),
    ("2014-06-26","Germany","USA",1,0,"Arena Pernambuco",30,72),
    ("2014-06-26","Argentina","Nigeria",3,2,"Beira Rio",20,62),
    ("2014-06-28","Colombia","Uruguay",2,0,"Maracana",24,65),
    ("2014-06-28","France","Nigeria",2,0,"Brasilia",28,60),
    ("2014-06-30","Brazil","Chile",1,1,"Mineirao",22,65),
    ("2014-07-01","Colombia","Uruguay",2,0,"Maracana",24,65),
    ("2014-07-04","Brazil","Colombia",2,1,"Castelao",28,68),
    ("2014-07-04","France","Germany",0,1,"Mineirao",22,65),
    ("2014-07-05","Argentina","Belgium",1,0,"Mineirao",22,65),
    ("2014-07-05","Netherlands","Costa Rica",0,0,"Arena Fonte Nova",26,70),
    ("2014-07-08","Brazil","Germany",1,7,"Mineirao",22,65),
    ("2014-07-09","Netherlands","Argentina",0,0,"Arena Corinthians",22,62),
    ("2014-07-13","Germany","Argentina",1,0,"Maracana",24,65),

    # Euro 2024
    ("2024-06-14","Germany","Scotland",5,1,"Allianz Arena EURO",20,55),
    ("2024-06-15","Spain","Croatia",3,0,"Olympiastadion EURO",22,50),
    ("2024-06-16","England","Serbia",1,0,"Arena AufSchalke EURO",21,60),
    ("2024-06-17","France","Austria",1,0,"Dusseldorf Arena EURO",25,55),
    ("2024-06-20","Portugal","Czechia",2,1,"Leipzig Stadium EURO",24,52),
    ("2024-06-22","Spain","Italy",1,0,"Gelsenkirchen EURO",23,55),
    ("2024-06-25","Germany","Switzerland",0,0,"Frankfurt Arena EURO",22,60),
    ("2024-06-26","England","Slovenia",0,0,"Cologne EURO",24,55),
    ("2024-06-29","Germany","Denmark",2,0,"Dortmund EURO",22,58),
    ("2024-06-30","England","Slovakia",2,1,"Gelsenkirchen EURO",23,55),
    ("2024-06-30","Spain","Georgia",4,1,"Cologne EURO",24,55),
    ("2024-07-01","France","Belgium",1,0,"Dusseldorf Arena EURO",25,55),
    ("2024-07-02","Portugal","Slovenia",0,0,"Frankfurt Arena EURO",22,60),
    ("2024-07-05","Germany","Spain",1,2,"Stuttgart EURO",25,52),
    ("2024-07-05","France","Portugal",0,0,"Hamburg EURO",23,58),
    ("2024-07-06","England","Switzerland",1,1,"Dusseldorf Arena EURO",25,55),
    ("2024-07-06","Netherlands","Türkiye",2,1,"Berlin EURO",26,52),
    ("2024-07-09","Spain","France",2,1,"Allianz Arena EURO",20,55),
    ("2024-07-10","England","Netherlands",2,1,"Dortmund EURO",22,58),
    ("2024-07-14","Spain","England",2,1,"Berlin EURO",26,52),

    # Copa América 2024
    ("2024-06-20","Argentina","Canada",2,0,"Mercedes-Benz COPA",29,65),
    ("2024-06-23","Uruguay","Bolivia",5,0,"MetLife COPA",24,68),
    ("2024-06-25","Ecuador","Venezuela",1,2,"Levi's COPA",22,60),
    ("2024-06-27","Argentina","Chile",1,0,"MetLife COPA",28,65),
    ("2024-06-27","Colombia","Costa Rica",3,0,"NRG COPA",34,75),
    ("2024-06-29","Colombia","Paraguay",2,1,"NRG COPA",34,75),
    ("2024-07-04","Argentina","Ecuador",1,0,"NRG COPA",34,75),
    ("2024-07-04","Venezuela","Canada",1,0,"AT&T COPA",35,58),
    ("2024-07-05","Colombia","Panama",5,0,"State Farm COPA",38,52),
    ("2024-07-06","Uruguay","Brazil",0,0,"Allegiant COPA",40,25),
    ("2024-07-09","Argentina","Canada",2,0,"MetLife COPA",28,65),
    ("2024-07-09","Uruguay","Colombia",0,1,"Bank of America COPA",32,68),
    ("2024-07-14","Argentina","Colombia",1,0,"Hard Rock COPA",31,80),
]


def _wc_results_to_raw() -> list:
    """Convert WC_2026_RESULTS (date, home, away, hg, ag) to _RAW_MATCHES format."""
    out = []
    for r in WC_2026_RESULTS:
        date_str, home, away, hg, ag = r
        out.append((date_str, home, away, hg, ag, "WC2026", 25, 60))
    return out


def build_match_dataframe() -> pd.DataFrame:
    """
    Converts raw match tuples into a feature-rich DataFrame.
    Each row is a match; features cover team quality, environment,
    tactical style, rest days, and derived stats.
    """
    all_matches = _RAW_MATCHES + _wc_results_to_raw()
    rows = []
    for rec in all_matches:
        date_str, home, away, hg, ag, stadium_key, temp_c, humidity = rec

        h_rat = TEAM_RATINGS.get(home, {"attack": 68, "defense": 68, "midfield": 68, "overall": 68, "fifa_rank": 80})
        a_rat = TEAM_RATINGS.get(away, {"attack": 68, "defense": 68, "midfield": 68, "overall": 68, "fifa_rank": 80})

        altitude = 0
        if "WC22" in stadium_key or "Lusail" in stadium_key:
            altitude = 10
        elif "2018" in stadium_key:
            altitude = 100
        elif "Arena" in stadium_key or "EURO" in stadium_key or "COPA" in stadium_key:
            altitude = 50
        if "Ekaterinburg" in stadium_key:
            altitude = 247
        if "Amazonia" in stadium_key:
            altitude = 92
        if "Pernambuco" in stadium_key:
            altitude = 10

        h_conf = CONFEDERATION.get(home, "OTHER")
        a_conf = CONFEDERATION.get(away, "OTHER")
        h_style = PLAY_STYLE.get(home, "balanced")
        a_style = PLAY_STYLE.get(away, "balanced")

        result = "H" if hg > ag else ("A" if ag > hg else "D")

        rows.append({
            "date": pd.to_datetime(date_str),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "total_goals": hg + ag,
            "result": result,
            "home_win": int(result == "H"),
            "draw": int(result == "D"),
            "away_win": int(result == "A"),
            # Team ratings
            "home_overall": h_rat["overall"],
            "away_overall": a_rat["overall"],
            "rating_diff": h_rat["overall"] - a_rat["overall"],
            "home_attack": h_rat["attack"],
            "away_attack": a_rat["attack"],
            "home_defense": h_rat["defense"],
            "away_defense": a_rat["defense"],
            "home_midfield": h_rat["midfield"],
            "away_midfield": a_rat["midfield"],
            "home_fifa_rank": h_rat["fifa_rank"],
            "away_fifa_rank": a_rat["fifa_rank"],
            "rank_diff": a_rat["fifa_rank"] - h_rat["fifa_rank"],  # positive = home better
            # Environment
            "temp_c": temp_c,
            "humidity": humidity,
            "altitude_m": altitude,
            "heat_index": temp_c + 0.33 * (humidity / 100 * 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))) - 4.0,
            "high_altitude": int(altitude > 1500),
            "high_heat": int(temp_c > 30),
            "high_humidity": int(humidity > 70),
            # Tactical style encodings
            "home_possession_style": int(h_style in ("possession", "pressing")),
            "away_possession_style": int(a_style in ("possession", "pressing")),
            "home_attacking_style": int(h_style in ("attacking", "direct")),
            "away_attacking_style": int(a_style in ("attacking", "direct")),
            "home_defensive_style": int(h_style in ("defensive", "counter")),
            "away_defensive_style": int(a_style in ("defensive", "counter")),
            # Confederation context
            "same_confederation": int(h_conf == a_conf),
            "home_confederation": h_conf,
            "away_confederation": a_conf,
            "home_european": int(h_conf == "UEFA"),
            "away_european": int(a_conf == "UEFA"),
            "home_sa": int(h_conf == "CONMEBOL"),
            "away_sa": int(a_conf == "CONMEBOL"),
            # Squad age
            "home_squad_age": SQUAD_AVG_AGE.get(home, 27.5),
            "away_squad_age": SQUAD_AVG_AGE.get(away, 27.5),
            "squad_age_diff": SQUAD_AVG_AGE.get(home, 27.5) - SQUAD_AVG_AGE.get(away, 27.5),
            # Tournament experience
            "home_wc_exp": WC_EXPERIENCE.get(home, 0),
            "away_wc_exp": WC_EXPERIENCE.get(away, 0),
            "wc_exp_diff": WC_EXPERIENCE.get(home, 0) - WC_EXPERIENCE.get(away, 0),
            # Penalty record
            "home_pen_record": PENALTY_RECORD.get(home, 0.5),
            "away_pen_record": PENALTY_RECORD.get(away, 0.5),
            # H2H advantage (home team's historical win rate vs this opponent)
            "home_h2h_adv": H2H_ADVANTAGE.get((home, away), 0.5),
            # Club form bonus
            "home_club_form": CLUB_FORM_BONUS.get(home, 0.0),
            "away_club_form": CLUB_FORM_BONUS.get(away, 0.0),
            "club_form_diff": CLUB_FORM_BONUS.get(home, 0.0) - CLUB_FORM_BONUS.get(away, 0.0),
            # Suspension risk (normalised 0–1)
            "home_suspension_risk": SUSPENSION_RISK.get(home, 5.0) / 10.0,
            "away_suspension_risk": SUSPENSION_RISK.get(away, 5.0) / 10.0,
            # Press intensity (normalised 0–1)
            "home_press": PRESS_INTENSITY.get(home, 5.0) / 10.0,
            "away_press": PRESS_INTENSITY.get(away, 5.0) / 10.0,
            "press_diff": (PRESS_INTENSITY.get(home, 5.0) - PRESS_INTENSITY.get(away, 5.0)) / 10.0,
            # Set piece & aerial (normalised 0–1)
            "home_setpiece": SET_PIECE_STRENGTH.get(home, 5.5) / 10.0,
            "away_setpiece": SET_PIECE_STRENGTH.get(away, 5.5) / 10.0,
            "home_aerial": AERIAL_STRENGTH.get(home, 6.0) / 10.0,
            "away_aerial": AERIAL_STRENGTH.get(away, 6.0) / 10.0,
            # Set piece threat vs opponent's aerial defence
            "home_sp_vs_aerial": (SET_PIECE_STRENGTH.get(home, 5.5) - AERIAL_STRENGTH.get(away, 6.0)) / 10.0,
            "away_sp_vs_aerial": (SET_PIECE_STRENGTH.get(away, 5.5) - AERIAL_STRENGTH.get(home, 6.0)) / 10.0,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ── Derived rolling features ───────────────────────────────────────────────
    df = _add_rest_days(df)
    df = _add_rolling_form(df)
    df = _add_goal_expectation_features(df)
    return df


def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """For each match, compute days since each team's last match (capped at 14)."""
    last_match: dict = {}
    home_rest, away_rest = [], []
    for _, row in df.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"]
        home_rest.append(min(int((d - last_match[h]).days) if h in last_match else 10, 14))
        away_rest.append(min(int((d - last_match[a]).days) if a in last_match else 10, 14))
        last_match[h] = d
        last_match[a] = d
    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    df["rest_advantage"] = df["home_days_rest"] - df["away_days_rest"]
    return df


def _add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """For each match, compute home/away team's last-5-game form."""
    all_teams = pd.concat([
        df[["date", "home_team", "home_goals", "away_goals"]].rename(
            columns={"home_team": "team", "home_goals": "gf", "away_goals": "ga"}),
        df[["date", "away_team", "away_goals", "home_goals"]].rename(
            columns={"away_team": "team", "away_goals": "gf", "home_goals": "ga"}),
    ]).sort_values("date")
    all_teams["points"] = np.where(all_teams["gf"] > all_teams["ga"], 3,
                           np.where(all_teams["gf"] == all_teams["ga"], 1, 0))

    form = {}
    for _, row in all_teams.iterrows():
        t = row["team"]
        if t not in form:
            form[t] = []
        form[t].append((row["date"], row["gf"], row["ga"], row["points"]))

    def get_form(team, before_date, n=5):
        games = [g for g in form.get(team, []) if g[0] < before_date][-n:]
        if not games:
            return 1.5, 1.5, 3.0  # neutral defaults
        gf = np.mean([g[1] for g in games])
        ga = np.mean([g[2] for g in games])
        pts = np.mean([g[3] for g in games])
        return gf, ga, pts

    h_form = [get_form(r.home_team, r.date) for _, r in df.iterrows()]
    a_form = [get_form(r.away_team, r.date) for _, r in df.iterrows()]

    df["home_form_gf"] = [f[0] for f in h_form]
    df["home_form_ga"] = [f[1] for f in h_form]
    df["home_form_pts"] = [f[2] for f in h_form]
    df["away_form_gf"] = [f[0] for f in a_form]
    df["away_form_ga"] = [f[1] for f in a_form]
    df["away_form_pts"] = [f[2] for f in a_form]
    df["form_pts_diff"] = df["home_form_pts"] - df["away_form_pts"]
    return df


def _add_goal_expectation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add xG proxies and goal-market features."""
    # Simplified xG proxy: attack vs opponent defense delta
    df["home_xg_proxy"] = (df["home_attack"] - df["away_defense"]) / 20 + 1.3
    df["away_xg_proxy"] = (df["away_attack"] - df["home_defense"]) / 20 + 1.1
    df["home_xg_proxy"] = df["home_xg_proxy"].clip(0.3, 4.5)
    df["away_xg_proxy"] = df["away_xg_proxy"].clip(0.3, 4.5)

    # Altitude penalty on xG
    alt_penalty = df["altitude_m"].apply(lambda a: 0.05 if a > 1500 else 0.0)
    df["home_xg_proxy"] -= alt_penalty
    df["away_xg_proxy"] -= alt_penalty

    # Heat effect (fatigue reduces overall goal flow slightly in extreme heat)
    heat_effect = df["temp_c"].apply(lambda t: -0.1 if t > 32 else 0.0)
    df["home_xg_proxy"] += heat_effect
    df["away_xg_proxy"] += heat_effect

    df["total_xg"] = df["home_xg_proxy"] + df["away_xg_proxy"]
    df["xg_diff"] = df["home_xg_proxy"] - df["away_xg_proxy"]
    return df
