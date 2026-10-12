"""
Live tournament state: standings, WC-specific form, upcoming fixtures.
All functions are pure (no side effects) and work from WC_2026_RESULTS + GROUPS_2026.
"""

from collections import defaultdict
import pandas as pd
from datetime import datetime


def compute_standings(results: list, groups: dict) -> dict[str, pd.DataFrame]:
    """
    Compute current group standings from WC_2026_RESULTS.
    Returns {group_letter: DataFrame} sorted by pts, gd, gf.
    """
    team_stats: dict[str, dict] = {}
    for group, teams in groups.items():
        for t in teams:
            team_stats[t] = {"group": group, "played": 0, "w": 0, "d": 0,
                             "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}

    for date, t1, t2, g1, g2 in results:
        if t1 not in team_stats or t2 not in team_stats:
            continue
        for t, gf, ga in [(t1, g1, g2), (t2, g2, g1)]:
            team_stats[t]["played"] += 1
            team_stats[t]["gf"] += gf
            team_stats[t]["ga"] += ga
            team_stats[t]["gd"] += gf - ga
            if gf > ga:
                team_stats[t]["w"]   += 1
                team_stats[t]["pts"] += 3
            elif gf == ga:
                team_stats[t]["d"]   += 1
                team_stats[t]["pts"] += 1
            else:
                team_stats[t]["l"]   += 1

    tables = {}
    for group, teams in groups.items():
        rows = [{"team": t, **team_stats[t]} for t in teams]
        df = pd.DataFrame(rows).sort_values(
            ["pts", "gd", "gf"], ascending=False
        ).reset_index(drop=True)
        df.index += 1
        tables[group] = df[["team", "played", "w", "d", "l", "gf", "ga", "gd", "pts"]]
    return tables


def wc_form(team: str, results: list) -> dict:
    """
    Compute a team's actual WC 2026 performance stats.
    Returns attack_avg (goals scored/game), defense_avg (goals conceded/game), n_games.
    """
    gf, ga, n = 0, 0, 0
    for _, t1, t2, g1, g2 in results:
        if t1 == team:
            gf += g1; ga += g2; n += 1
        elif t2 == team:
            gf += g2; ga += g1; n += 1
    if n == 0:
        return {"attack_avg": None, "defense_avg": None, "n_games": 0}
    return {"attack_avg": round(gf / n, 2), "defense_avg": round(ga / n, 2), "n_games": n}


def all_teams_form(results: list) -> dict[str, dict]:
    """Return wc_form dict for every team that has played."""
    teams = set()
    for _, t1, t2, *_ in results:
        teams.add(t1); teams.add(t2)
    return {t: wc_form(t, results) for t in teams}


def upcoming_fixtures(results: list, fixtures: list) -> list[tuple]:
    """Return fixtures not yet in results, in date order."""
    played = {(t1, t2) for _, t1, t2, *_ in results}
    played |= {(t2, t1) for _, t1, t2, *_ in results}
    upcoming = [(d, t1, t2) for d, t1, t2 in fixtures if (t1, t2) not in played]
    return sorted(upcoming, key=lambda x: x[0])


def xg_adjustment(team: str, results: list, base_xg: float) -> float:
    """
    Blend pre-tournament base xG with actual WC goals-per-game.
    Weight: 0 games → 100% base; 1 game → 30% WC; 2+ games → 55% WC.
    """
    f = wc_form(team, results)
    n = f["n_games"]
    if n == 0 or f["attack_avg"] is None:
        return base_xg
    wc_weight = 0.30 if n == 1 else 0.55
    return round((1 - wc_weight) * base_xg + wc_weight * f["attack_avg"], 3)
