"""Optional upcoming J1 fixtures and confirmed finals.

Historical J1 results ALWAYS come from Football-Data.co.uk/JPN.csv.
ESPN's undocumented public endpoint is only a convenience for schedules and
recent final-score confirmation. If unavailable the app has a manual fallback.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

JST = ZoneInfo("Asia/Tokyo")
API_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/jpn.1/scoreboard"

# ESPN and Football-Data.co.uk sometimes differ; be conservative if ambiguous.
ALIASES = {
    "vissel kobe": ["kobe", "vissel kobe"],
    "kashima antlers": ["kashima", "kashima antlers"],
    "urawa red diamonds": ["urawa", "urawa reds", "urawa red diamonds"],
    "urawa reds": ["urawa", "urawa reds"],
    "sanfrecce hiroshima": ["hiroshima", "sanfrecce hiroshima"],
    "kawasaki frontale": ["kawasaki", "kawasaki frontale"],
    "gamba osaka": ["gamba osaka", "gamba"],
    "cerezo osaka": ["cerezo osaka", "cerezo"],
    "nagoya grampus": ["nagoya", "nagoya grampus"],
    "kyoto sanga": ["kyoto", "kyoto sanga"],
    "yokohama f marinos": ["yokohama m", "yokohama fm", "yokohama f marinos"],
    "yokohama marinos": ["yokohama m", "yokohama fm"],
    "fc tokyo": ["tokyo", "fc tokyo"],
    "tokyo verdy": ["tokyo v", "tokyo verdy"],
    "avispa fukuoka": ["fukuoka", "avispa fukuoka"],
    "kashiwa reysol": ["kashiwa", "kashiwa reysol"],
    "fagiano okayama": ["okayama", "fagiano okayama"],
    "shimizu s pulse": ["shimizu", "shimizu s pulse"],
    "machida zelvia": ["machida", "machida zelvia"],
    "fc machida zelvia": ["machida", "machida zelvia"],
    "mito hollyhock": ["mito", "mito hollyhock"],
    "jef united chiba": ["chiba", "jef united chiba"],
    "v varen nagasaki": ["nagasaki", "v varen nagasaki"],
    "sagan tosu": ["tosu", "sagan tosu"],
    "jubilo iwata": ["iwata", "jubilo iwata"],
}


def norm(s: str) -> str:
    text = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def match_club(raw: str, clubs: list[str]) -> str | None:
    n = norm(raw)
    options = {norm(club): club for club in clubs}
    if n in options:
        return options[n]
    results = {options[alias] for alias in ALIASES.get(n, []) if alias in options}
    return next(iter(results)) if len(results) == 1 else None


def parse_espn(body: dict, clubs: list[str], now: datetime | None = None) -> tuple[list[dict], pd.DataFrame, list[str]]:
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        raise ValueError("Fixture source returned an unexpected response shape.")
    now = now or datetime.now(timezone.utc)
    future, finals, skipped = [], [], []
    for event in body["events"]:
        try:
            kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(timezone.utc)
            competitors = (event.get("competitions") or [{}])[0].get("competitors") or []
            home = next(c for c in competitors if c.get("homeAway") == "home")
            away = next(c for c in competitors if c.get("homeAway") == "away")
            rh = (home.get("team") or {}).get("displayName", "")
            ra = (away.get("team") or {}).get("displayName", "")
            ch, ca = match_club(rh, clubs), match_club(ra, clubs)
            if not ch or not ca or ch == ca:
                skipped.append(f"{rh} / {ra}")
                continue
            event_status = event.get("status") or {}
            status_type = event_status.get("type") or {}
            status_name = str(status_type.get("name", "")).upper()
            finished = bool(status_type.get("completed", False)) or status_name in {"STATUS_FINAL", "STATUS_FULL_TIME"}
            if finished:
                hg, ag = int(home["score"]), int(away["score"])
                if not (0 <= hg <= 15 and 0 <= ag <= 15):
                    continue
                finals.append({"Date": pd.Timestamp(kickoff.astimezone(JST).date()),
                               "HomeTeam": ch, "AwayTeam": ca, "FTHG": hg, "FTAG": ag})
            elif kickoff > now and status_name in {"STATUS_SCHEDULED", "STATUS_PRE", "STATUS_POSTPONED"}:
                if status_name == "STATUS_POSTPONED":
                    continue
                future.append({"home": ch, "away": ca, "kickoff_utc": kickoff,
                               "source": "ESPN public scoreboard (unofficial)",
                               "label": f"{kickoff.astimezone(JST):%a %d %b %H:%M} JST · {ch} vs {ca}"})
        except (KeyError, ValueError, TypeError, StopIteration):
            continue
    future.sort(key=lambda x: x["kickoff_utc"])
    return future, pd.DataFrame(finals, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]), skipped


def download_fixtures_scores(clubs: list[str], now: datetime | None = None, session=requests):
    now = now or datetime.now(timezone.utc)
    # Query J1 games from the last 10 days through 28 days ahead.
    start = (now - timedelta(days=10)).astimezone(JST).date()
    end = (now + timedelta(days=28)).astimezone(JST).date()
    response = session.get(API_URL, params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 250},
                           timeout=18, headers={"User-Agent": "JLeague-Match-Lab personal project"})
    response.raise_for_status()
    return parse_espn(response.json(), clubs, now)


def group_weeks(fixtures: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for fixture in fixtures:
        day = fixture["kickoff_utc"].astimezone(JST).date()
        monday = day - timedelta(days=day.weekday())
        label = f"{monday:%d %b}–{monday + timedelta(days=6):%d %b %Y} (JST)"
        grouped.setdefault(label, []).append(fixture)
    return grouped
