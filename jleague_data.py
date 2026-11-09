"""J1 results loader from Football-Data.co.uk. No bundled match data.

JPN.csv differs from EPL data: Home, Away, HG, AG, and (usually) Season/League.
Dates are interpreted day-first as the provider documents.
"""
from __future__ import annotations

import io
import pandas as pd
import requests

DATA_URL = "https://www.football-data.co.uk/new/JPN.csv"
DATA_PAGE = "https://www.football-data.co.uk/japan.php"
CORE = ["Date", "Home", "Away", "HG", "AG"]
STANDARD = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "SeasonCode"]


def season_for_date(date: pd.Timestamp) -> str:
    """Do not confuse the 2026 transitional tournament with the 2026/27 J1 season."""
    y, month = date.year, date.month
    if y == 2026 and month < 8:
        return "2026 special"
    if y >= 2026 and month >= 8:
        return f"{y}/{str(y + 1)[-2:]}"
    if y >= 2027 and month < 8:
        return f"{y-1}/{str(y)[-2:]}"
    return str(y)


def _read_csv(payload: bytes) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding, low_memory=False, on_bad_lines="skip")
        except Exception as e:
            errors.append(str(e))
    raise ValueError("Cannot parse the CSV. " + "; ".join(errors[:2]))


def clean_japan_csv(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return completed J1 matches and potential future fixture rows (without scores)."""
    d = raw.copy()
    d.columns = d.columns.astype(str).str.strip()
    aliases = {"HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "HG", "FTAG": "AG"}
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d and v not in d})
    missing = [x for x in CORE if x not in d.columns]
    if missing:
        raise ValueError(f"Expected a Football-Data Japan CSV; missing: {', '.join(missing)}. Found: {', '.join(d.columns[:12])}")
    # The Japanese extra league file is one J1 dataset. If other leagues are present,
    # only retain known top-flight codes instead of merging divisions.
    if "League" in d.columns:
        leagues = d["League"].dropna().astype(str).str.strip()
        codes = leagues.unique().tolist()
        if len(codes) > 1:
            is_j1 = d["League"].astype(str).str.contains(r"(?:j1|j-league|j league|division 1|top division)", case=False, regex=True)
            if is_j1.any():
                d = d.loc[is_j1].copy()
            else:
                raise ValueError("Multiple leagues present but J1 could not be identified safely.")
    dates = pd.to_datetime(d["Date"].astype(str), errors="coerce", dayfirst=True, format="mixed")
    d["Date"] = dates.dt.normalize()
    d["HG"] = pd.to_numeric(d["HG"], errors="coerce")
    d["AG"] = pd.to_numeric(d["AG"], errors="coerce")
    d["Home"] = d["Home"].astype("string").str.strip()
    d["Away"] = d["Away"].astype("string").str.strip()
    d = d.dropna(subset=["Date", "Home", "Away"])
    d = d.loc[(d.Home != "") & (d.Away != "") & (d.Home != d.Away)].copy()
    d["SeasonCode"] = d["Date"].map(season_for_date)
    completed_mask = (d.HG.notna() & d.AG.notna() &
                      d.HG.between(0, 15) & d.AG.between(0, 15) &
                      d.HG.mod(1).eq(0) & d.AG.mod(1).eq(0))
    completed = d.loc[completed_mask].copy()
    completed["HG"] = completed["HG"].astype(int)
    completed["AG"] = completed["AG"].astype(int)
    matches = completed.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})[STANDARD]
    matches = matches.loc[matches.SeasonCode.ne("2026 special")].copy()
    matches = matches.drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="last")
    matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    fixtures = d.loc[~completed_mask, ["Date", "Home", "Away"]].copy()
    fixtures = fixtures.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
    fixtures = fixtures.loc[fixtures.Date.ge(pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize())]
    return matches, fixtures.drop_duplicates().sort_values("Date").reset_index(drop=True)


def parse_japan_bytes(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    return clean_japan_csv(_read_csv(payload))


def download_japan_csv(session=requests) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = session.get(DATA_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0 (personal J.League study project)"})
    response.raise_for_status()
    return parse_japan_bytes(response.content)


def current_clubs(matches: pd.DataFrame) -> tuple[list[str], str]:
    if matches.empty:
        raise ValueError("No completed J1 matches loaded.")
    most_recent = matches.iloc[-1]["SeasonCode"]
    this = matches.loc[matches.SeasonCode.eq(most_recent)]
    clubs = sorted(set(this.HomeTeam.astype(str)).union(this.AwayTeam.astype(str)))
    return clubs, str(most_recent)
