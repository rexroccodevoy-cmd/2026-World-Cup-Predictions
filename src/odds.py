"""
Bookmaker odds integration:
  - Fetch live odds from The Odds API (the-odds-api.com)
  - Strip the vig / overround to get true implied probabilities
  - Compare market vs model probabilities → edge, EV, Kelly stake
  - Brier score calibration tracking as real results come in
"""

import requests
import numpy as np
import pandas as pd

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"   # The Odds API sport key for the World Cup

# ── Sample odds fallback (no API key needed) ──────────────────────────────────
# Realistic decimal odds (best-price across major books) for a selection of
# group stage fixtures. Format: (home, away): (home_odds, draw_odds, away_odds)
SAMPLE_ODDS = {
    ("United States", "Paraguay"):    (1.95, 3.40, 4.20),
    ("Mexico", "South Africa"):       (1.75, 3.50, 5.00),
    ("Canada", "Qatar"):              (1.80, 3.45, 4.60),
    ("Brazil", "Morocco"):            (1.85, 3.50, 4.40),
    ("England", "Croatia"):           (2.10, 3.20, 3.80),
    ("France", "Senegal"):            (1.65, 3.80, 5.50),
    ("Argentina", "Algeria"):         (1.50, 4.00, 7.00),
    ("Uruguay", "Spain"):             (4.30, 3.40, 1.90),
    ("Germany", "Ivory Coast"):       (1.70, 3.70, 5.20),
    ("Colombia", "Portugal"):         (3.90, 3.30, 2.05),
    ("Netherlands", "Japan"):         (1.85, 3.45, 4.50),
    ("Belgium", "Egypt"):             (1.75, 3.50, 5.00),
    ("Ecuador", "Germany"):           (4.10, 3.30, 1.95),
    ("Qatar", "Switzerland"):         (6.50, 4.00, 1.55),
}


def fetch_live_odds(api_key: str) -> tuple[pd.DataFrame, str]:
    """
    Fetch h2h (match winner) odds for upcoming WC matches.
    Returns (dataframe, status_message). Empty df on failure.
    """
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": "uk,eu,us",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        if resp.status_code == 401:
            return pd.DataFrame(), "Invalid API key."
        if resp.status_code == 404:
            return pd.DataFrame(), ("World Cup odds market not yet open on The Odds API "
                                    "(markets typically open closer to kickoff). Using sample odds.")
        resp.raise_for_status()
        events = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
    except requests.RequestException as e:
        return pd.DataFrame(), f"Request failed: {e}"

    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        best = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    price = float(outcome.get("price", 0))
                    if outcome["name"] == home:
                        best["home"] = max(best["home"], price)
                    elif outcome["name"] == away:
                        best["away"] = max(best["away"], price)
                    else:
                        best["draw"] = max(best["draw"], price)
        if all(v > 1.0 for v in best.values()):
            rows.append({"home": home, "away": away,
                         "home_odds": best["home"], "draw_odds": best["draw"],
                         "away_odds": best["away"], "source": "live"})

    if not rows:
        return pd.DataFrame(), "API responded but no h2h odds found. Using sample odds."
    return pd.DataFrame(rows), f"Live odds loaded ({len(rows)} matches, {remaining} API requests remaining)."


def sample_odds_df() -> pd.DataFrame:
    rows = [{"home": h, "away": a, "home_odds": ho, "draw_odds": do,
             "away_odds": ao, "source": "sample"}
            for (h, a), (ho, do, ao) in SAMPLE_ODDS.items()]
    return pd.DataFrame(rows)


def strip_vig(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float, float]:
    """
    Convert decimal odds to true implied probabilities by removing the overround.
    Returns (p_home, p_draw, p_away, overround_pct).
    """
    raw = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    overround = raw.sum()
    true = raw / overround
    return float(true[0]), float(true[1]), float(true[2]), float((overround - 1) * 100)


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked. Positive = theoretically profitable."""
    return model_prob * decimal_odds - 1.0


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Kelly criterion optimal stake as fraction of bankroll.
    f* = (bp - q) / b   where b = odds-1, p = model prob, q = 1-p
    Clamped at 0 (never bet negative edge) and capped at 25% (full Kelly is wild).
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * model_prob - (1.0 - model_prob)) / b
    return float(np.clip(f, 0.0, 0.25))


def build_value_table(odds_df: pd.DataFrame, predictor) -> pd.DataFrame:
    """
    For each match with odds, run the model and compute edge / EV / Kelly
    for all three outcomes. Returns one row per (match, outcome).
    """
    rows = []
    for _, r in odds_df.iterrows():
        home, away = r["home"], r["away"]
        try:
            pred = predictor.predict_match(home, away)
        except Exception:
            continue

        p_imp_h, p_imp_d, p_imp_a, overround = strip_vig(
            r["home_odds"], r["draw_odds"], r["away_odds"])

        for outcome, odds, p_market, p_model in [
            (f"{home} win", r["home_odds"], p_imp_h, pred["p_home_win"]),
            ("Draw",        r["draw_odds"], p_imp_d, pred["p_draw"]),
            (f"{away} win", r["away_odds"], p_imp_a, pred["p_away_win"]),
        ]:
            edge = p_model - p_market
            rows.append({
                "match": f"{home} vs {away}",
                "outcome": outcome,
                "best_odds": round(odds, 2),
                "market_prob": round(p_market * 100, 1),
                "model_prob": round(p_model * 100, 1),
                "edge_pct": round(edge * 100, 1),
                "ev_per_unit": round(expected_value(p_model, odds), 3),
                "kelly_pct": round(kelly_fraction(p_model, odds) * 100, 1),
                "overround_pct": round(overround, 1),
                "source": r["source"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge_pct", ascending=False)
    return df


def brier_scores(results: list, predictor, odds_lookup: dict | None = None) -> pd.DataFrame:
    """
    Calibration tracking: for each completed WC match, score the model
    (and the market if odds are available) with the Brier score.
    Lower = better calibrated. results rows: (date, home, away, hg, ag).

    Brier here = mean squared error across the 3-outcome vector.
    """
    rows = []
    for res in results:
        date_str, home, away, hg, ag = res
        actual = np.array([1, 0, 0]) if hg > ag else (np.array([0, 1, 0]) if hg == ag else np.array([0, 0, 1]))

        try:
            pred = predictor.predict_match(home, away)
        except Exception:
            continue
        model_vec = np.array([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])
        model_brier = float(np.mean((model_vec - actual) ** 2))

        market_brier = None
        if odds_lookup and (home, away) in odds_lookup:
            ho, do, ao = odds_lookup[(home, away)]
            ph, pd_, pa, _ = strip_vig(ho, do, ao)
            market_vec = np.array([ph, pd_, pa])
            market_brier = float(np.mean((market_vec - actual) ** 2))

        rows.append({
            "date": date_str,
            "match": f"{home} {hg}–{ag} {away}",
            "model_brier": round(model_brier, 4),
            "market_brier": round(market_brier, 4) if market_brier is not None else None,
        })
    return pd.DataFrame(rows)
