"""
Prediction models:
  - Poisson / Dixon-Coles match outcome model
  - Random Forest ensemble for win/draw/loss probabilities
  - Player goalscorer probability model
  - Correlation engine for 100+ factors
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.constants import (
    TEAM_RATINGS, KEY_PLAYERS, STADIUMS, GROUPS_2026, CONFEDERATION, PLAY_STYLE,
    SQUAD_AVG_AGE, WC_EXPERIENCE, PENALTY_RECORD, H2H_ADVANTAGE,
    CLUB_FORM_BONUS, SUSPENSION_RISK, PRESS_INTENSITY,
    SET_PIECE_STRENGTH, AERIAL_STRENGTH, HOST_NATIONS_2026, FAN_SUPPORT,
)
from src.historical_data import build_match_dataframe


# ── Feature columns used by the RF model ────────────────────────────────────
FEATURE_COLS = [
    "home_overall", "away_overall", "rating_diff",
    "home_attack", "away_attack", "home_defense", "away_defense",
    "home_midfield", "away_midfield",
    "rank_diff",
    "temp_c", "humidity", "altitude_m", "heat_index",
    "high_altitude", "high_heat", "high_humidity",
    "home_form_gf", "home_form_ga", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_pts",
    "form_pts_diff",
    "home_possession_style", "away_possession_style",
    "home_attacking_style", "away_attacking_style",
    "home_defensive_style", "away_defensive_style",
    "same_confederation",
    "home_european", "away_european",
    "home_sa", "away_sa",
    "home_xg_proxy", "away_xg_proxy", "total_xg", "xg_diff",
    # New correlations
    "home_days_rest", "away_days_rest", "rest_advantage",
    "home_squad_age", "away_squad_age", "squad_age_diff",
    "home_wc_exp", "away_wc_exp", "wc_exp_diff",
    "home_pen_record", "away_pen_record",
    "home_h2h_adv",
    "home_club_form", "away_club_form", "club_form_diff",
    "home_suspension_risk", "away_suspension_risk",
    "home_press", "away_press", "press_diff",
    "home_setpiece", "away_setpiece",
    "home_aerial", "away_aerial",
    "home_sp_vs_aerial", "away_sp_vs_aerial",
]


class MatchPredictor:
    """
    Dual-model predictor:
      1. Poisson model for exact score distribution
      2. Random Forest for 1X2 probabilities
    """

    def __init__(self):
        self.df = build_match_dataframe()
        self.rf = None
        self.team_attack_params = {}
        self.team_defense_params = {}
        self._train()

    # ── Training ─────────────────────────────────────────────────────────────
    def _train(self):
        df = self.df.dropna(subset=FEATURE_COLS)

        # 1. Random Forest on 1X2 outcome
        le = LabelEncoder()
        y = le.fit_transform(df["result"])
        self.le = le
        X = df[FEATURE_COLS]
        self.rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
        ])
        self.rf.fit(X, y)
        self.feature_importances_ = dict(zip(
            FEATURE_COLS,
            self.rf.named_steps["clf"].feature_importances_,
        ))

        # 2. Poisson attack/defense parameters per team
        self._fit_poisson_params()

    def _fit_poisson_params(self):
        """Fit per-team attack and defense strength via simple log-linear model."""
        df = self.df
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))

        # Init from ratings
        for t in teams:
            r = TEAM_RATINGS.get(t, {"attack": 70, "defense": 70})
            self.team_attack_params[t] = r["attack"] / 75.0
            self.team_defense_params[t] = r["defense"] / 75.0

        # One round of moment matching on the data
        for t in teams:
            home_matches = df[df["home_team"] == t]
            away_matches = df[df["away_team"] == t]
            scored = pd.concat([home_matches["home_goals"], away_matches["away_goals"]])
            conceded = pd.concat([home_matches["away_goals"], away_matches["home_goals"]])
            if len(scored) > 0:
                self.team_attack_params[t] = max(0.3, scored.mean() / 1.2)
                self.team_defense_params[t] = max(0.3, conceded.mean() / 1.2)

    # ── Prediction ───────────────────────────────────────────────────────────
    def predict_match(self, home: str, away: str, stadium_name: str = "MetLife Stadium",
                      home_form: float = 1.5, away_form: float = 1.5,
                      days_rest_home: int = 4, days_rest_away: int = 4,
                      is_knockout: bool = False):
        """
        Returns full prediction dict with:
        - win/draw/loss probabilities
        - most likely score
        - Poisson goal distribution
        - key factor analysis
        """
        stadium = STADIUMS.get(stadium_name, list(STADIUMS.values())[3])
        h_rat = TEAM_RATINGS.get(home, {"attack": 70, "defense": 70, "midfield": 70, "overall": 70, "fifa_rank": 80})
        a_rat = TEAM_RATINGS.get(away, {"attack": 70, "defense": 70, "midfield": 70, "overall": 70, "fifa_rank": 80})

        temp_c = stadium["avg_temp_june"]
        humidity = stadium["avg_humidity_june"]
        altitude = stadium["altitude_m"]
        heat_index = temp_c + 0.33 * (humidity / 100 * 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))) - 4.0

        h_conf = CONFEDERATION.get(home, "OTHER")
        a_conf = CONFEDERATION.get(away, "OTHER")
        from src.constants import PLAY_STYLE
        h_style = PLAY_STYLE.get(home, "balanced")
        a_style = PLAY_STYLE.get(away, "balanced")

        # Neutral venue — equal base xG (no home advantage in a World Cup)
        h_xg = (h_rat["attack"] - a_rat["defense"]) / 20 + 1.2
        a_xg = (a_rat["attack"] - h_rat["defense"]) / 20 + 1.2

        # Blend in actual WC goals-per-game once teams have played
        from src.constants import WC_2026_RESULTS
        from src.tournament import xg_adjustment
        h_xg = xg_adjustment(home, WC_2026_RESULTS, h_xg)
        a_xg = xg_adjustment(away, WC_2026_RESULTS, a_xg)

        # Host nation boost (USA/Mexico/Canada playing on home soil)
        if home in HOST_NATIONS_2026:
            h_xg += 0.15
        if away in HOST_NATIONS_2026:
            a_xg += 0.15

        # Fan / crowd support differential
        h_fans = FAN_SUPPORT.get(home, 5.0)
        a_fans = FAN_SUPPORT.get(away, 5.0)
        fan_delta = (h_fans - a_fans) * 0.005  # subtle atmosphere effect, not a rating buff
        h_xg += fan_delta
        a_xg -= fan_delta

        alt_pen = 0.05 if altitude > 1500 else 0.0
        heat_eff = -0.1 if temp_c > 32 else 0.0
        h_xg = h_xg - alt_pen + heat_eff
        a_xg = a_xg - alt_pen + heat_eff

        # Club season form
        h_xg += CLUB_FORM_BONUS.get(home, 0.0)
        a_xg += CLUB_FORM_BONUS.get(away, 0.0)

        # Rest days (±0.03 per day relative to a 4-day baseline)
        h_xg += (days_rest_home - 4) * 0.03
        a_xg += (days_rest_away - 4) * 0.03

        # Suspension risk penalty
        h_xg -= (SUSPENSION_RISK.get(home, 5.0) / 10.0) * 0.1
        a_xg -= (SUSPENSION_RISK.get(away, 5.0) / 10.0) * 0.1

        # H2H advantage
        h2h = H2H_ADVANTAGE.get((home, away))
        if h2h is None:
            rev = H2H_ADVANTAGE.get((away, home))
            h2h = (1.0 - rev) if rev is not None else 0.5
        h2h_delta = (h2h - 0.5) * 0.2
        h_xg += h2h_delta
        a_xg -= h2h_delta

        # Press vs deep-block: high-press neutralised by a well-drilled defensive block
        h_press = PRESS_INTENSITY.get(home, 5.0)
        a_press = PRESS_INTENSITY.get(away, 5.0)
        h_style = PLAY_STYLE.get(home, "balanced")
        a_style = PLAY_STYLE.get(away, "balanced")
        if h_press > 7.5 and a_style in ("defensive", "counter"):
            a_xg -= 0.10
        if a_press > 7.5 and h_style in ("defensive", "counter"):
            h_xg -= 0.10

        # Set piece vs aerial mismatch
        sp_adv_h = (SET_PIECE_STRENGTH.get(home, 5.5) - AERIAL_STRENGTH.get(away, 6.0)) / 100.0
        sp_adv_a = (SET_PIECE_STRENGTH.get(away, 5.5) - AERIAL_STRENGTH.get(home, 6.0)) / 100.0
        h_xg += sp_adv_h
        a_xg += sp_adv_a

        # WC experience edge in knockout pressure
        if is_knockout:
            exp_delta = (WC_EXPERIENCE.get(home, 0) - WC_EXPERIENCE.get(away, 0)) * 0.02
            h_xg += exp_delta
            a_xg -= exp_delta

        h_xg = max(0.3, h_xg)
        a_xg = max(0.3, a_xg)

        h2h_adv = H2H_ADVANTAGE.get((home, away), 0.5)
        X = pd.DataFrame([{
            "home_overall": h_rat["overall"],
            "away_overall": a_rat["overall"],
            "rating_diff": h_rat["overall"] - a_rat["overall"],
            "home_attack": h_rat["attack"],
            "away_attack": a_rat["attack"],
            "home_defense": h_rat["defense"],
            "away_defense": a_rat["defense"],
            "home_midfield": h_rat["midfield"],
            "away_midfield": a_rat["midfield"],
            "rank_diff": a_rat["fifa_rank"] - h_rat["fifa_rank"],
            "temp_c": temp_c,
            "humidity": humidity,
            "altitude_m": altitude,
            "heat_index": heat_index,
            "high_altitude": int(altitude > 1500),
            "high_heat": int(temp_c > 30),
            "high_humidity": int(humidity > 70),
            "home_form_gf": home_form,
            "home_form_ga": 1.5,
            "home_form_pts": home_form * 1.5,
            "away_form_gf": away_form,
            "away_form_ga": 1.5,
            "away_form_pts": away_form * 1.5,
            "form_pts_diff": (home_form - away_form) * 1.5,
            "home_possession_style": int(h_style in ("possession", "pressing")),
            "away_possession_style": int(a_style in ("possession", "pressing")),
            "home_attacking_style": int(h_style in ("attacking", "direct")),
            "away_attacking_style": int(a_style in ("attacking", "direct")),
            "home_defensive_style": int(h_style in ("defensive", "counter")),
            "away_defensive_style": int(a_style in ("defensive", "counter")),
            "same_confederation": int(h_conf == a_conf),
            "home_european": int(h_conf == "UEFA"),
            "away_european": int(a_conf == "UEFA"),
            "home_sa": int(h_conf == "CONMEBOL"),
            "away_sa": int(a_conf == "CONMEBOL"),
            "home_xg_proxy": h_xg,
            "away_xg_proxy": a_xg,
            "total_xg": h_xg + a_xg,
            "xg_diff": h_xg - a_xg,
            # New features
            "home_days_rest": days_rest_home,
            "away_days_rest": days_rest_away,
            "rest_advantage": days_rest_home - days_rest_away,
            "home_squad_age": SQUAD_AVG_AGE.get(home, 27.5),
            "away_squad_age": SQUAD_AVG_AGE.get(away, 27.5),
            "squad_age_diff": SQUAD_AVG_AGE.get(home, 27.5) - SQUAD_AVG_AGE.get(away, 27.5),
            "home_wc_exp": WC_EXPERIENCE.get(home, 0),
            "away_wc_exp": WC_EXPERIENCE.get(away, 0),
            "wc_exp_diff": WC_EXPERIENCE.get(home, 0) - WC_EXPERIENCE.get(away, 0),
            "home_pen_record": PENALTY_RECORD.get(home, 0.5),
            "away_pen_record": PENALTY_RECORD.get(away, 0.5),
            "home_h2h_adv": h2h_adv,
            "home_club_form": CLUB_FORM_BONUS.get(home, 0.0),
            "away_club_form": CLUB_FORM_BONUS.get(away, 0.0),
            "club_form_diff": CLUB_FORM_BONUS.get(home, 0.0) - CLUB_FORM_BONUS.get(away, 0.0),
            "home_suspension_risk": SUSPENSION_RISK.get(home, 5.0) / 10.0,
            "away_suspension_risk": SUSPENSION_RISK.get(away, 5.0) / 10.0,
            "home_press": h_press / 10.0,
            "away_press": a_press / 10.0,
            "press_diff": (h_press - a_press) / 10.0,
            "home_setpiece": SET_PIECE_STRENGTH.get(home, 5.5) / 10.0,
            "away_setpiece": SET_PIECE_STRENGTH.get(away, 5.5) / 10.0,
            "home_aerial": AERIAL_STRENGTH.get(home, 6.0) / 10.0,
            "away_aerial": AERIAL_STRENGTH.get(away, 6.0) / 10.0,
            "home_sp_vs_aerial": (SET_PIECE_STRENGTH.get(home, 5.5) - AERIAL_STRENGTH.get(away, 6.0)) / 10.0,
            "away_sp_vs_aerial": (SET_PIECE_STRENGTH.get(away, 5.5) - AERIAL_STRENGTH.get(home, 6.0)) / 10.0,
        }])

        # Poisson score matrix (blended with per-team fit params)
        h_lambda = self.team_attack_params.get(home, 1.2) * (1 / max(0.5, self.team_defense_params.get(away, 1.0)))
        a_lambda = self.team_attack_params.get(away, 1.0) * (1 / max(0.5, self.team_defense_params.get(home, 1.0)))
        h_lambda = 0.5 * h_lambda + 0.5 * h_xg
        a_lambda = 0.5 * a_lambda + 0.5 * a_xg

        score_matrix = self._score_matrix(h_lambda, a_lambda)
        most_likely_score = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

        # Win/draw/loss probabilities from Poisson score matrix
        # (more reliable than the RF which overfits on ~120 training samples)
        tril = np.tril(np.ones((7, 7), dtype=bool), k=-1)  # home goals > away goals
        triu = np.triu(np.ones((7, 7), dtype=bool), k=1)   # away goals > home goals
        p_home_poisson = float(score_matrix[tril].sum())
        p_draw_poisson = float(np.diag(score_matrix).sum())
        p_away_poisson = float(score_matrix[triu].sum())

        # Use Poisson score matrix for win probabilities — the RF is kept for
        # feature importance / correlation analysis only (it overfits on ~120 samples)
        p_home = p_home_poisson
        p_draw  = p_draw_poisson
        p_away  = p_away_poisson
        # Still run RF so feature_importances_ remain available
        _ = self.rf.predict_proba(X)

        # Environmental impact summary
        env_factors = self._env_impact(temp_c, humidity, altitude, h_rat, a_rat)
        extra_factors = self._extra_factors(
            home, away, days_rest_home, days_rest_away, is_knockout
        )

        return {
            "home_team": home,
            "away_team": away,
            "stadium": stadium_name,
            "temp_c": temp_c,
            "humidity": humidity,
            "altitude_m": altitude,
            "p_home_win": round(p_home, 3),
            "p_draw": round(p_draw, 3),
            "p_away_win": round(p_away, 3),
            "home_xg": round(h_xg, 2),
            "away_xg": round(a_xg, 2),
            "home_lambda": round(h_lambda, 2),
            "away_lambda": round(a_lambda, 2),
            "most_likely_score": f"{most_likely_score[0]}-{most_likely_score[1]}",
            "score_matrix": score_matrix,
            "env_factors": env_factors,
            "extra_factors": extra_factors,
            "feature_row": X.iloc[0].to_dict(),
        }

    def _score_matrix(self, h_lam: float, a_lam: float, max_goals: int = 7) -> np.ndarray:
        mat = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                mat[i, j] = poisson.pmf(i, h_lam) * poisson.pmf(j, a_lam)
        return mat

    def _env_impact(self, temp, humidity, altitude, h_rat, a_rat) -> dict:
        """Qualitative + quantitative environmental impact."""
        factors = {}

        # Heat impact: teams from hot climates adapt better
        if temp > 30:
            h_heat_pen = 0.0 if CONFEDERATION.get(h_rat.get("team", ""), "OTHER") in ("CAF", "AFC") else -2.5
            a_heat_pen = 0.0 if CONFEDERATION.get(a_rat.get("team", ""), "OTHER") in ("CAF", "AFC") else -2.5
            factors["heat"] = {
                "label": f"High heat ({temp}°C)",
                "impact": "Reduces sprint intensity ~12% in temps >32°C",
                "home_adj": h_heat_pen, "away_adj": a_heat_pen,
            }

        # Altitude impact
        if altitude > 1500:
            factors["altitude"] = {
                "label": f"High altitude ({altitude}m)",
                "impact": "VO₂ max drops ~3% per 300m above 1500m; passing accuracy falls",
                "home_adj": 0.0, "away_adj": -3.0,
            }

        # Humidity
        if humidity > 70:
            factors["humidity"] = {
                "label": f"High humidity ({humidity}%)",
                "impact": "Increases perceived temperature; hydration demands rise",
                "home_adj": 0.0, "away_adj": -1.5,
            }

        return factors

    def _extra_factors(self, home: str, away: str,
                       days_rest_home: int, days_rest_away: int,
                       is_knockout: bool) -> dict:
        factors = {}

        # Host nation / crowd support
        h_fans = FAN_SUPPORT.get(home, 5.0)
        a_fans = FAN_SUPPORT.get(away, 5.0)
        h_is_host = home in HOST_NATIONS_2026
        a_is_host = away in HOST_NATIONS_2026
        if h_is_host and a_is_host:
            factors["host_nation"] = {
                "label": "Both teams playing on home soil",
                "impact": f"Both {home} and {away} are 2026 host nations — crowd boost cancels out",
                "home_adj": 0.0, "away_adj": 0.0,
            }
        elif h_is_host:
            factors["host_nation"] = {
                "label": f"Host nation — {home}",
                "impact": f"{home} plays on home soil with full crowd support in 2026",
                "home_adj": 3.0, "away_adj": 0.0,
            }
        elif a_is_host:
            factors["host_nation"] = {
                "label": f"Host nation — {away}",
                "impact": f"{away} plays on home soil with full crowd support in 2026",
                "home_adj": 0.0, "away_adj": 3.0,
            }
        elif abs(h_fans - a_fans) >= 2.0:
            adv = home if h_fans > a_fans else away
            factors["crowd_support"] = {
                "label": f"Crowd support advantage — {adv}",
                "impact": (
                    f"{home} fan support: {h_fans:.0f}/10 vs {away}: {a_fans:.0f}/10 "
                    f"— larger diaspora and travelling support in North American venues"
                ),
                "home_adj": (h_fans - a_fans) * 0.4,
                "away_adj": -(h_fans - a_fans) * 0.4,
            }

        # 5. Days rest
        rest_diff = days_rest_home - days_rest_away
        if abs(rest_diff) >= 1:
            adv = home if rest_diff > 0 else away
            factors["rest"] = {
                "label": f"Rest advantage — {adv} ({'+' if rest_diff > 0 else ''}{rest_diff} days)",
                "impact": f"{adv} has more recovery time, reducing injury risk and fatigue",
                "home_adj": rest_diff * 3.0, "away_adj": -rest_diff * 3.0,
            }

        # 6. Head-to-head record
        h2h = H2H_ADVANTAGE.get((home, away))
        if h2h is None:
            rev = H2H_ADVANTAGE.get((away, home))
            if rev is not None:
                h2h = 1.0 - rev
        if h2h is not None and abs(h2h - 0.5) > 0.05:
            adv = home if h2h > 0.5 else away
            rate = max(h2h, 1.0 - h2h)
            factors["h2h"] = {
                "label": f"Head-to-head edge — {adv}",
                "impact": f"{adv} wins {rate*100:.0f}% of WC/major tournament meetings",
                "home_adj": (h2h - 0.5) * 10.0, "away_adj": -(h2h - 0.5) * 10.0,
            }

        # 7. WC knockout experience (especially relevant in knockouts)
        h_exp = WC_EXPERIENCE.get(home, 0)
        a_exp = WC_EXPERIENCE.get(away, 0)
        if abs(h_exp - a_exp) >= 2:
            adv = home if h_exp > a_exp else away
            factors["wc_experience"] = {
                "label": f"Tournament experience — {adv}",
                "impact": (
                    f"{home}: {h_exp} WC knockout appearances (2006–2022) vs "
                    f"{away}: {a_exp} — experience matters under pressure"
                ),
                "home_adj": (h_exp - a_exp) * 1.5, "away_adj": (a_exp - h_exp) * 1.5,
            }

        # 8. Must-win / knockout pressure
        if is_knockout:
            h_pen = PENALTY_RECORD.get(home, 0.5)
            a_pen = PENALTY_RECORD.get(away, 0.5)
            if abs(h_pen - a_pen) > 0.15:
                adv = home if h_pen > a_pen else away
                factors["penalties"] = {
                    "label": f"Penalty shootout record — {adv} stronger",
                    "impact": (
                        f"{home}: {h_pen*100:.0f}% WC shootout win rate vs "
                        f"{away}: {a_pen*100:.0f}% — could be decisive if level after 120 min"
                    ),
                    "home_adj": (h_pen - a_pen) * 5.0, "away_adj": (a_pen - h_pen) * 5.0,
                }

        # 9. Squad age
        h_age = SQUAD_AVG_AGE.get(home, 27.5)
        a_age = SQUAD_AVG_AGE.get(away, 27.5)
        if abs(h_age - a_age) >= 2.0:
            young = home if h_age < a_age else away
            old = away if h_age < a_age else home
            factors["squad_age"] = {
                "label": f"Squad age gap — {young} younger",
                "impact": (
                    f"{young} avg {min(h_age, a_age):.1f} yrs vs {old} avg {max(h_age, a_age):.1f} yrs "
                    f"— younger squad has stamina edge in extra time"
                ),
                "home_adj": (a_age - h_age) * 0.5, "away_adj": (h_age - a_age) * 0.5,
            }

        # 10. Club season form
        h_form = CLUB_FORM_BONUS.get(home, 0.0)
        a_form = CLUB_FORM_BONUS.get(away, 0.0)
        if abs(h_form - a_form) > 0.07:
            adv = home if h_form > a_form else away
            factors["club_form"] = {
                "label": f"Club season form — {adv}",
                "impact": f"{adv}'s key players arrive from strong club campaigns (2025–26 season)",
                "home_adj": h_form * 10, "away_adj": a_form * 10,
            }

        # 11. Suspension risk
        h_susp = SUSPENSION_RISK.get(home, 5.0)
        a_susp = SUSPENSION_RISK.get(away, 5.0)
        if h_susp > 6.5:
            factors["suspension_home"] = {
                "label": f"Suspension risk — {home}",
                "impact": f"{home} prone to card accumulation; key players may play cautiously to avoid bans",
                "home_adj": -(h_susp - 5.0) * 0.8, "away_adj": 0.0,
            }
        if a_susp > 6.5:
            factors["suspension_away"] = {
                "label": f"Suspension risk — {away}",
                "impact": f"{away} prone to card accumulation; key players may play cautiously to avoid bans",
                "home_adj": 0.0, "away_adj": -(a_susp - 5.0) * 0.8,
            }

        # 12 + 13. Press vs deep block
        h_press = PRESS_INTENSITY.get(home, 5.0)
        a_press = PRESS_INTENSITY.get(away, 5.0)
        h_style = PLAY_STYLE.get(home, "balanced")
        a_style = PLAY_STYLE.get(away, "balanced")
        if h_press > 7.5 and a_style in ("defensive", "counter"):
            factors["press_vs_block"] = {
                "label": f"High press neutralised — {away} deep block",
                "impact": (
                    f"{home}'s press ({h_press:.0f}/10) likely nullified by {away}'s "
                    f"compact defensive shape — expect fewer open chances, low scoring"
                ),
                "home_adj": -1.5, "away_adj": 1.5,
            }
        elif a_press > 7.5 and h_style in ("defensive", "counter"):
            factors["press_vs_block"] = {
                "label": f"High press neutralised — {home} deep block",
                "impact": (
                    f"{away}'s press ({a_press:.0f}/10) likely nullified by {home}'s "
                    f"compact defensive shape — expect fewer open chances, low scoring"
                ),
                "home_adj": 1.5, "away_adj": -1.5,
            }

        # 14. Set piece vs aerial mismatch
        h_sp = SET_PIECE_STRENGTH.get(home, 5.5)
        a_aerial = AERIAL_STRENGTH.get(away, 6.0)
        a_sp = SET_PIECE_STRENGTH.get(away, 5.5)
        h_aerial = AERIAL_STRENGTH.get(home, 6.0)
        if h_sp >= 8.0 and a_aerial <= 5.5:
            factors["setpiece_aerial"] = {
                "label": f"Set piece threat — {home} vs {away} aerial weakness",
                "impact": (
                    f"{home}'s set piece danger ({h_sp:.0f}/10) exploits "
                    f"{away}'s aerial vulnerability ({a_aerial:.0f}/10)"
                ),
                "home_adj": 2.0, "away_adj": 0.0,
            }
        elif a_sp >= 8.0 and h_aerial <= 5.5:
            factors["setpiece_aerial"] = {
                "label": f"Set piece threat — {away} vs {home} aerial weakness",
                "impact": (
                    f"{away}'s set piece danger ({a_sp:.0f}/10) exploits "
                    f"{home}'s aerial vulnerability ({h_aerial:.0f}/10)"
                ),
                "home_adj": 0.0, "away_adj": 2.0,
            }

        return factors


class GoalScorerPredictor:
    """
    Predicts probability that a player scores in a given match.
    Model: base rate × team role × match difficulty × environment
    """

    def __init__(self, match_predictor: MatchPredictor):
        self.mp = match_predictor
        self.players = pd.DataFrame(KEY_PLAYERS)

    def predict_scorers(self, match_result: dict, top_n: int = 10):
        """
        Returns top_n likely scorers for a given match prediction dict.
        """
        home = match_result["home_team"]
        away = match_result["away_team"]
        h_xg = match_result["home_xg"]
        a_xg = match_result["away_xg"]
        temp_c = match_result["temp_c"]
        altitude = match_result["altitude_m"]

        candidates = self.players[
            self.players["team"].isin([home, away]) & (self.players["pos"].isin(["FW", "MF"]))
        ].copy()

        if candidates.empty:
            return pd.DataFrame()

        env_penalty = 0.0
        if temp_c > 32:
            env_penalty -= 0.05
        if altitude > 1500:
            env_penalty -= 0.03

        rows = []
        for _, p in candidates.iterrows():
            team_xg = h_xg if p["team"] == home else a_xg
            base_rate = p["club_goals_per90"]
            team_share = 0.35 if p["pos"] == "FW" else 0.15
            wc_bonus = min(0.05, p["wc_goals"] * 0.005)
            age_factor = 1.0
            if p["age"] > 33:
                age_factor = 0.90
            elif p["age"] < 23:
                age_factor = 1.05

            prob = base_rate * team_xg * team_share * age_factor + wc_bonus + env_penalty
            prob = max(0.0, min(0.95, prob))
            rows.append({
                "player": p["name"],
                "team": p["team"],
                "position": p["pos"],
                "age": p["age"],
                "rating": p["rating"],
                "wc_goals_history": p["wc_goals"],
                "club_goals_per90": p["club_goals_per90"],
                "score_probability": round(prob, 3),
                "team_xg": team_xg,
            })

        out = pd.DataFrame(rows).sort_values("score_probability", ascending=False).head(top_n)
        out["score_probability_pct"] = (out["score_probability"] * 100).round(1)
        return out.reset_index(drop=True)


class CorrelationEngine:
    """
    Computes 100+ correlations between features and match outcomes.
    """

    def __init__(self):
        self.df = build_match_dataframe()
        self._results = None

    def compute_all_correlations(self) -> pd.DataFrame:
        if self._results is not None:
            return self._results

        df = self.df.copy()
        records = []

        numeric_targets = ["home_goals", "away_goals", "total_goals", "home_win", "draw", "away_win"]
        predictors = [c for c in FEATURE_COLS if c in df.columns]

        for pred in predictors:
            for tgt in numeric_targets:
                if pred in df.columns and tgt in df.columns:
                    valid = df[[pred, tgt]].dropna()
                    if len(valid) < 5:
                        continue
                    corr = valid[pred].corr(valid[tgt])
                    records.append({
                        "feature": pred,
                        "target": tgt,
                        "correlation": round(corr, 3),
                        "abs_correlation": round(abs(corr), 3),
                        "n_samples": len(valid),
                    })

        # Extra derived correlations
        derived_pairs = [
            # (new_feature_expr, new_feature_name, target)
            ("home_attack - away_defense", "attack_vs_def_mismatch", "home_goals"),
            ("away_attack - home_defense", "opp_attack_vs_def_mismatch", "away_goals"),
            ("(home_overall + away_overall) / 2", "avg_match_quality", "total_goals"),
            ("home_form_pts - away_form_pts", "form_advantage", "home_win"),
            ("temp_c * humidity / 100", "heat_humidity_product", "total_goals"),
            ("altitude_m / 1000", "altitude_km", "total_goals"),
            ("home_fifa_rank - away_fifa_rank", "ranking_gap", "home_win"),
        ]
        for expr, name, tgt in derived_pairs:
            try:
                feat_vals = df.eval(expr)
                if tgt in df.columns:
                    combined = pd.DataFrame({"feat": feat_vals, "tgt": df[tgt]}).dropna()
                    if len(combined) >= 5:
                        corr = combined["feat"].corr(combined["tgt"])
                        records.append({
                            "feature": name,
                            "target": tgt,
                            "correlation": round(corr, 3),
                            "abs_correlation": round(abs(corr), 3),
                            "n_samples": len(combined),
                        })
            except Exception:
                pass

        self._results = pd.DataFrame(records).sort_values("abs_correlation", ascending=False)
        return self._results

    def top_factors_for(self, target: str = "home_win", n: int = 20) -> pd.DataFrame:
        df = self.compute_all_correlations()
        return df[df["target"] == target].head(n)

    def weather_impact_analysis(self) -> pd.DataFrame:
        """Slice of correlations focused purely on environmental variables."""
        df = self.compute_all_correlations()
        env_features = ["temp_c", "humidity", "altitude_m", "heat_index",
                        "high_altitude", "high_heat", "high_humidity",
                        "heat_humidity_product", "altitude_km"]
        return df[df["feature"].isin(env_features)].sort_values("abs_correlation", ascending=False)


def simulate_tournament(predictor: MatchPredictor, n_sims: int = 1000) -> dict:
    """
    Monte Carlo simulation of the 2026 World Cup.
    Returns: winner probabilities, finalist probabilities, SF probabilities.
    """
    from src.constants import GROUPS_2026

    win_counts = {}
    final_counts = {}
    sf_counts = {}

    all_teams = [t for group in GROUPS_2026.values() for t in group]
    for t in all_teams:
        win_counts[t] = 0
        final_counts[t] = 0
        sf_counts[t] = 0

    group_pos1 = {g: Counter() for g in GROUPS_2026}
    group_pos2 = {g: Counter() for g in GROUPS_2026}
    group_pos3 = {g: Counter() for g in GROUPS_2026}

    # Pre-cache all group-stage predictions — same matchups repeat every simulation
    _pred_cache: dict = {}

    def _predict(h: str, a: str) -> dict:
        if (h, a) not in _pred_cache:
            _pred_cache[(h, a)] = predictor.predict_match(h, a)
        return _pred_cache[(h, a)]

    for _ in range(n_sims):
        # Simulate group stage
        qualifiers = []
        third_place_pool = []
        for group, teams in GROUPS_2026.items():
            standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
            for i, h in enumerate(teams):
                for a in teams[i+1:]:
                    res = _predict(h, a)
                    probs = np.array([res["p_home_win"], res["p_draw"], res["p_away_win"]], dtype=float)
                    probs /= probs.sum()
                    r = np.random.choice(["H", "D", "A"], p=probs)
                    hg = int(np.random.poisson(res["home_lambda"]))
                    ag = int(np.random.poisson(res["away_lambda"]))
                    if r == "H":
                        standings[h]["pts"] += 3
                    elif r == "D":
                        standings[h]["pts"] += 1
                        standings[a]["pts"] += 1
                    else:
                        standings[a]["pts"] += 3
                    standings[h]["gd"] += hg - ag
                    standings[a]["gd"] += ag - hg
                    standings[h]["gf"] += hg
                    standings[a]["gf"] += ag

            sorted_teams = sorted(teams, key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]), reverse=True)
            qualifiers.extend(sorted_teams[:2])
            group_pos1[group][sorted_teams[0]] += 1
            group_pos2[group][sorted_teams[1]] += 1
            group_pos3[group][sorted_teams[2]] += 1
            third = sorted_teams[2]
            third_place_pool.append((standings[third]["pts"], standings[third]["gd"], standings[third]["gf"], third))

        # Best 8 third-place finishers advance (WC 2026 format → 32 teams)
        third_place_pool.sort(reverse=True)
        qualifiers.extend(t for _, _, _, t in third_place_pool[:8])
        np.random.shuffle(qualifiers)

        # Knockout rounds (simple bracket)
        round_teams = qualifiers[:32]
        while len(round_teams) > 1:
            next_round = []
            for i in range(0, len(round_teams), 2):
                if i + 1 >= len(round_teams):
                    next_round.append(round_teams[i])
                    continue
                h, a = round_teams[i], round_teams[i+1]
                res = _predict(h, a)
                p_h = res["p_home_win"] / (res["p_home_win"] + res["p_away_win"])
                winner = h if np.random.random() < p_h else a
                next_round.append(winner)

            if len(round_teams) == 4:
                for t in round_teams:
                    sf_counts[t] += 1
            if len(round_teams) == 2:
                for t in round_teams:
                    final_counts[t] += 1

            round_teams = next_round

        if round_teams:
            win_counts[round_teams[0]] += 1

    total = n_sims
    return {
        "win_probability":      {t: round(c / total, 4) for t, c in win_counts.items()},
        "final_probability":    {t: round(c / total, 4) for t, c in final_counts.items()},
        "semifinal_probability":{t: round(c / total, 4) for t, c in sf_counts.items()},
        "group_advance": {
            g: {
                "first":       group_pos1[g].most_common(1)[0][0],
                "second":      group_pos2[g].most_common(1)[0][0],
                "third":       group_pos3[g].most_common(1)[0][0],
                "first_prob":  round(group_pos1[g].most_common(1)[0][1] / total, 3),
                "second_prob": round(group_pos2[g].most_common(1)[0][1] / total, 3),
            }
            for g in group_pos1
        },
    }
