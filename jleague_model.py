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
SEASON_WINDOW = 2  # latest two completed-season labels, not a fixed number of days
MODEL_VERSION = "j1-attack-defence-two-season-v2"


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


def _venue(past: pd.DataFrame, home: str, away: str, n: int | None, prior: float):
    hmean = max(float(past.FTHG.mean()), .5) if len(past) else 1.45
    amean = max(float(past.FTAG.mean()), .5) if len(past) else 1.15
    h = past.loc[past.HomeTeam.eq(home)]
    a = past.loc[past.AwayTeam.eq(away)]
    if n is not None:
        h, a = h.tail(n), a.tail(n)
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


def recent_season_codes(matches: pd.DataFrame, n: int = SEASON_WINDOW) -> list[str]:
    """The latest n J1 seasons represented by *completed* results.

    Use each season's first match date, not lexical season labels or 730 days.
    The 2026 transitional competition is not part of the ordinary J1 model.
    This works for historical walk-forward windows as well as today's app.
    """
    if matches.empty:
        return []
    if "SeasonCode" not in matches.columns:
        raise ValueError("SeasonCode is required for the two-season model.")
    valid = matches.loc[matches.SeasonCode.notna() & matches.SeasonCode.ne("2026 special")]
    if valid.empty:
        return []
    first_dates = valid.groupby("SeasonCode")["Date"].min().sort_values()
    return first_dates.index.astype(str).tolist()[-n:]


def two_season_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """All completed games from the most recent two ordinary J1 seasons."""
    seasons = recent_season_codes(matches)
    return matches.loc[matches.SeasonCode.astype(str).isin(seasons)].copy()


def predict_models(past: pd.DataFrame, home: str, away: str) -> dict:
    if home == away:
        raise ValueError("Choose two different clubs.")
    past = two_season_matches(past).sort_values("Date")
    if len(past) < 20:
        raise ValueError("At least 20 historical J1 matches are required from the last two seasons.")
    # Baseline and simple average use every game in the two-season window.
    v, rating, counts = _venue(past, home, away, None, 7)
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
    """Pre-kickoff matches from the two latest seasons available at that date.

    The rule must be evaluated AFTER excluding the target match and every later
    match; no future information, including same-day scores, enters a forecast.
    No 730-day filter: J.League changed its calendar in 2026.
    """
    previous = matches.loc[matches.Date.lt(pd.Timestamp(cutoff))]
    return two_season_matches(previous).sort_values("Date").copy()


def club_ratings(past: pd.DataFrame, clubs: list[str]) -> pd.DataFrame:
    """Consistent team-strength profile for the same two seasons as predictions."""
    current = two_season_matches(past).sort_values("Date")
    if current.empty:
        raise ValueError("No completed matches in the latest two seasons.")
    rows = []
    for club in clubs:
        _, hr, (home_n, away_n) = _venue(current, club, club, None, 7)
        rows.append({"Club": club, "Home attack": hr["Home attack"],
                     "Home defence vulnerability": hr["Home defence vulnerability"],
                     "Away attack": hr["Away attack"],
                     "Away defence vulnerability": hr["Away defence vulnerability"],
                     "Home matches": home_n, "Away matches": away_n})
    return pd.DataFrame(rows)



def season_club_ratings(matches: pd.DataFrame, clubs: list[str], season: str,
                        prior: float = 7) -> pd.DataFrame:
    """Team profiles from *only* completed matches belonging to the specified season.

    Every match in that season is used (no last-20 cap), and both league averages
    and venue-specific club rates come from the same season. Unobserved home or
    away performances shrink to a neutral 1.00 rating rather than borrowing
    previous-season results. The prior is equivalent to prior matches per venue.
    """
    if "SeasonCode" not in matches.columns:
        raise ValueError("SeasonCode is required to calculate season-only ratings.")
    current = matches.loc[matches.SeasonCode.eq(season)].sort_values("Date")
    if current.empty:
        raise ValueError(f"No completed matches available for {season}.")
    records = []
    for club in clubs:
        _, ratings, (home_n, away_n) = _venue(current, club, club,
                                               n=len(current), prior=prior)
        records.append({
            "Club": club,
            "Home attack": ratings["Home attack"],
            "Away attack": ratings["Away attack"],
            "Home defence vulnerability": ratings["Home defence vulnerability"],
            "Away defence vulnerability": ratings["Away defence vulnerability"],
            "Home matches": home_n,
            "Away matches": away_n,
        })
    return pd.DataFrame(records)


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
