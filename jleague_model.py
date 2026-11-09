"""Goal-only J.League predictors and walk-forward backtests.

No in-match statistics or future scores are used to predict an earlier match.
The four methods are deliberately comparable on identical games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

MODEL_A = "Attack × Defence"
MODEL_B = "Simple average"
MODEL_C = "Recent Attack × Defence"
MODEL_D = "50/50 blend"
MODELS = (MODEL_A, MODEL_B, MODEL_C, MODEL_D)
MAX_GOALS = 12
HISTORY_DAYS = 730


def poisson_grid(home: float, away: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    g = np.outer(poisson.pmf(k, home), poisson.pmf(k, away))
    return g / g.sum()


def probs_for_grid(g: np.ndarray) -> np.ndarray:
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def score_for_grid(g: np.ndarray) -> tuple[str, float]:
    h, a = np.unravel_index(int(g.argmax()), g.shape)
    return f"{h}–{a}", float(g[h, a])


def _smooth(vals: pd.Series, league: float, k: float) -> float:
    v = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    return float((v.sum() + k * league) / (len(v) + k))


def _venue(past: pd.DataFrame, home: str, away: str, n: int, prior: float):
    hmean = max(float(past.FTHG.mean()), .5) if len(past) else 1.45
    amean = max(float(past.FTAG.mean()), .5) if len(past) else 1.15
    h = past.loc[past.HomeTeam.eq(home)].tail(n)
    a = past.loc[past.AwayTeam.eq(away)].tail(n)
    home_att = _smooth(h.FTHG, hmean, prior)
    home_con = _smooth(h.FTAG, amean, prior)
    away_att = _smooth(a.FTAG, amean, prior)
    away_con = _smooth(a.FTHG, hmean, prior)
    rating = {
        "Home attack": home_att / hmean,
        "Home defence vulnerability": home_con / amean,
        "Away attack": away_att / amean,
        "Away defence vulnerability": away_con / hmean,
    }
    return (home_att, home_con, away_att, away_con, hmean, amean), rating, (len(h), len(a))


def predict_models(past: pd.DataFrame, home: str, away: str) -> dict:
    if home == away:
        raise ValueError("Choose two different clubs.")
    if len(past) < 20:
        raise ValueError("At least 20 historical J1 matches are required.")
    v, rating, counts = _venue(past, home, away, 20, 7)
    ha, hc, aa, ac, lh, la = v
    attack = np.clip([lh * (ha / lh) * (ac / lh), la * (aa / la) * (hc / la)], .12, 4.8)
    simple = np.clip([(ha + ac) / 2, (aa + hc) / 2], .12, 4.8)
    rv, _, _ = _venue(past, home, away, 6, 5)
    rh, rhc, ra, rac, rlh, rla = rv
    recent = np.clip([rlh * (rh / rlh) * (rac / rlh), rla * (ra / rla) * (rhc / rla)], .12, 4.8)
    grids = {
        MODEL_A: poisson_grid(*attack), MODEL_B: poisson_grid(*simple),
        MODEL_C: poisson_grid(*recent),
    }
    grids[MODEL_D] = .5 * grids[MODEL_A] + .5 * grids[MODEL_C]
    goals = {MODEL_A: attack, MODEL_B: simple, MODEL_C: recent, MODEL_D: .5 * attack + .5 * recent}
    return {"grids": grids, "goals": goals, "ratings": rating, "counts": counts}


def prediction_rows(result: dict) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        grid = result["grids"][name]
        probs = probs_for_grid(grid)
        score, exact = score_for_grid(grid)
        mu = result["goals"][name]
        rows.append({
            "Model": name, "Expected home goals": round(float(mu[0]), 2),
            "Expected away goals": round(float(mu[1]), 2),
            "Most likely score": score, "Exact score %": round(100*exact, 1),
            "Home win %": round(100*probs[0], 1), "Draw %": round(100*probs[1], 1),
            "Away win %": round(100*probs[2], 1),
        })
    return pd.DataFrame(rows)


def historical_window(matches: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return matches.loc[(matches.Date < cutoff) & (matches.Date >= cutoff - pd.Timedelta(days=HISTORY_DAYS))]


def club_ratings(past: pd.DataFrame, clubs: list[str]) -> pd.DataFrame:
    rows = []
    for club in clubs:
        _, hr, _ = _venue(past, club, club, 20, 7)
        rows.append({"Club": club, "Home attack": hr["Home attack"],
                     "Home defence vulnerability": hr["Home defence vulnerability"],
                     "Away attack": hr["Away attack"],
                     "Away defence vulnerability": hr["Away defence vulnerability"]})
    return pd.DataFrame(rows)


def backtest(matches: pd.DataFrame, years: tuple[str, ...] = ("2023", "2024", "2025", "2026/27")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-by-date walk-forward test. Never use any games from the test date itself."""
    rows = []
    sorted_matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    for day, todays in sorted_matches.groupby("Date", sort=True):
        if not any(s in years for s in todays.SeasonCode.unique()):
            continue
        past = historical_window(sorted_matches, day)
        if len(past) < 180:
            continue
        for m in todays.itertuples(index=False):
            try:
                predictions = predict_models(past, m.HomeTeam, m.AwayTeam)
            except ValueError:
                continue
            actual_idx = 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2)
            for model in MODELS:
                p = probs_for_grid(predictions["grids"][model])
                mu = predictions["goals"][model]
                target = np.eye(3)[actual_idx]
                rows.append({"Season": m.SeasonCode, "Date": day, "Model": model,
                             "Brier": float(np.sum((p-target)**2)),
                             "Goal MAE": float((abs(mu[0]-m.FTHG)+abs(mu[1]-m.FTAG))/2),
                             "Home": m.HomeTeam, "Away": m.AwayTeam})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["Season", "Model", "Brier", "Goal MAE", "Matches"]), detail
    summary = detail.groupby(["Season", "Model"], as_index=False, sort=True).agg(
        **{"Brier": ("Brier", "mean"), "Goal MAE": ("Goal MAE", "mean"), "Matches": ("Brier", "size")})
    return summary, detail
