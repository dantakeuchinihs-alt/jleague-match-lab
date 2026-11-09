"""Read/write timestamped J.League predictions as GitHub Issues.

A fine-grained GitHub token needs Issues: read/write only for the chosen repository.
No secret is stored in the repository. Public repo = public saved predictions.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import requests

GITHUB_API = "https://api.github.com"
RECORD_START = "<!-- JLEAGUE_MATCH_LAB_RECORD_V1\n"
RECORD_END = "\n-->"
ISSUE_PREFIX = "[J.League Match Lab prediction]"
JST_ZONE = ZoneInfo("Asia/Tokyo")


def gh_headers(token: str = "") -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "JLeague-Match-Lab-Prediction-Tracker"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _response_error(resp: requests.Response) -> RuntimeError:
    try:
        message = str(resp.json().get("message", ""))
    except Exception:
        message = ""
    if resp.status_code == 401:
        return RuntimeError("GitHub rejected the token. Check the Streamlit Secrets configuration.")
    if resp.status_code == 403:
        return RuntimeError("GitHub denied this request. Check the token's Issues: Read and write permission or API rate limit.")
    if resp.status_code == 404:
        return RuntimeError("Repository not found. Check GITHUB_REPO and its visibility.")
    if resp.status_code == 410:
        return RuntimeError("GitHub Issues is disabled for this repository. Enable Issues in repository Settings → General.")
    return RuntimeError(f"GitHub API returned HTTP {resp.status_code}: {message[:160]}")


def _validate_repo(repo: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("GITHUB_REPO should look like owner/repository.")
    return repo


def list_predictions(repo: str, token: str = "", session=requests) -> list[dict]:
    """Fetch public or authenticated repository issues; record format is versioned."""
    repo = _validate_repo(repo)
    records = []
    for page in range(1, 11):
        resp = session.get(f"{GITHUB_API}/repos/{repo}/issues", headers=gh_headers(token),
                           params={"state":"all", "per_page":100, "page":page, "sort":"created", "direction":"desc"},
                           timeout=20)
        if not resp.ok:
            raise _response_error(resp)
        issues = resp.json()
        if not isinstance(issues, list):
            raise RuntimeError("Unexpected response from GitHub Issues.")
        for issue in issues:
            if "pull_request" in issue or not str(issue.get("title", "")).startswith(ISSUE_PREFIX):
                continue
            body = issue.get("body") or ""
            start = body.find(RECORD_START)
            if start == -1:
                continue
            end = body.find(RECORD_END, start + len(RECORD_START))
            if end == -1:
                continue
            try:
                data = json.loads(body[start + len(RECORD_START):end])
                if not _valid_record(data):
                    continue
                record = dict(data)
                record["github_created_at_utc"] = issue["created_at"]
                record["issue_url"] = issue["html_url"]
                record["issue_number"] = issue["number"]
                records.append(record)
            except (ValueError, KeyError, TypeError):
                continue
        if len(issues) < 100:
            break
    return records


def _valid_record(record: Any) -> bool:
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("model") != "Attack × Defence":
        return False
    required = ("home", "away", "kickoff_utc", "data_cutoff_utc_date", "score", "p_home", "p_draw", "p_away",
                "expected_home", "expected_away")
    if any(key not in record for key in required):
        return False
    try:
        p = [float(record[k]) for k in ("p_home", "p_draw", "p_away")]
        if any(not np.isfinite(x) or x < 0 or x > 1 for x in p) or abs(sum(p)-1) > .01:
            return False
        return isinstance(record["home"], str) and isinstance(record["away"], str)
    except (ValueError, TypeError):
        return False


def _utc_dt(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Kickoff time must include a timezone.")
    return value.astimezone(timezone.utc)


def make_prediction_record(home: str, away: str, kickoff: datetime, source: str, cutoff_date: str,
                           grid: np.ndarray, goals: np.ndarray, score: str, exact_prob: float,
                           probs: np.ndarray) -> dict:
    kickoff = _utc_dt(kickoff)
    if home == away:
        raise ValueError("Home and away team must be different.")
    if kickoff <= datetime.now(timezone.utc):
        raise ValueError("The scheduled kickoff is not in the future; this would not be a pre-match prediction.")
    if not np.isclose(float(np.sum(probs)), 1, atol=1e-4):
        raise ValueError("Invalid outcome probabilities.")
    return {
        "schema_version":1, "model":"Attack × Defence", "home":home, "away":away,
        "kickoff_utc":kickoff.isoformat().replace("+00:00", "Z"), "fixture_source":source,
        "data_cutoff_utc_date":str(cutoff_date),
        "score":score, "p_exact":round(float(exact_prob), 7),
        "p_home":round(float(probs[0]), 7), "p_draw":round(float(probs[1]), 7), "p_away":round(float(probs[2]), 7),
        "expected_home":round(float(goals[0]), 5), "expected_away":round(float(goals[1]), 5),
    }


def same_fixture(a: dict, b: dict) -> bool:
    """One locked prediction per matchup near scheduled date (within 7 days)."""
    try:
        return (a["home"] == b["home"] and a["away"] == b["away"] and
                abs((_utc_dt(a["kickoff_utc"]) - _utc_dt(b["kickoff_utc"])).total_seconds()) <= 7*86400)
    except (KeyError, ValueError, TypeError):
        return False


def save_prediction(repo: str, token: str, record: dict, session=requests,
                    existing_records: list[dict] | None = None) -> str:
    repo = _validate_repo(repo)
    if not token:
        raise ValueError("No GitHub token configured.")
    if not _valid_record(record):
        raise ValueError("Invalid prediction record.")
    if _utc_dt(record["kickoff_utc"]) <= datetime.now(timezone.utc):
        raise ValueError("Kickoff has already passed. Prediction not saved.")
    # Prevent duplicates based on existing GitHub records. This is a convenience, not a concurrency lock.
    existing = existing_records if existing_records is not None else list_predictions(repo, token=token, session=session)
    if any(same_fixture(x, record) for x in existing):
        raise ValueError("A prediction has already been saved for this fixture. Predictions are locked before kickoff.")
    date_jst = _utc_dt(record["kickoff_utc"]).astimezone(JST_ZONE).strftime("%d %b %Y %H:%M JST")
    title = f"{ISSUE_PREFIX} {record['home']} vs {record['away']} — {date_jst}"
    body = ("Locked pre-kickoff prediction from J.League Match Lab. GitHub's issue creation time is the audit timestamp.\n\n"
            f"**Model:** {record['model']}  \n"
            f"**Most likely score:** {record['home']} {record['score']} {record['away']}  \n"
            f"**Probabilities:** {100*record['p_home']:.1f}% home / {100*record['p_draw']:.1f}% draw / {100*record['p_away']:.1f}% away  \n"
            f"**Kickoff (JST):** {date_jst}  \n"
            f"**Data cutoff:** {record['data_cutoff_utc_date']}  \n\n"
            + RECORD_START + json.dumps(record, separators=(",", ":"), ensure_ascii=True) + RECORD_END)
    resp = session.post(f"{GITHUB_API}/repos/{repo}/issues", headers=gh_headers(token),
                        json={"title":title, "body":body}, timeout=20)
    if resp.status_code != 201:
        raise _response_error(resp)
    return str(resp.json()["html_url"])


def find_result(record: dict, matches: pd.DataFrame) -> dict | None:
    """Auto-settle exact home/away pair when result date is within three JST days of schedule.

    The result is never used to change saved predictions.
    Postponed/rescheduled games outside the window remain pending.
    """
    kickoff_utc = _utc_dt(record["kickoff_utc"])
    scheduled_date = kickoff_utc.astimezone(JST_ZONE).date()
    saved = _utc_dt(record.get("github_created_at_utc", record.get("saved_at_utc", record["kickoff_utc"])))
    if saved >= kickoff_utc:
        return None  # Not verifiably pre-kickoff
    home, away = record["home"], record["away"]
    candidates = matches.loc[matches.HomeTeam.eq(home) & matches.AwayTeam.eq(away)].copy()
    candidates["date_difference"] = candidates.Date.dt.date.map(lambda d: abs((d-scheduled_date).days))
    candidates = candidates.loc[candidates.date_difference.le(3)].sort_values(["date_difference", "Date"])
    if candidates.empty:
        return None
    match = candidates.iloc[0]
    h, a = int(match.FTHG), int(match.FTAG)
    actual = 0 if h > a else (1 if h == a else 2)
    probs = np.array([float(record[k]) for k in ("p_home", "p_draw", "p_away")])
    brier = float(np.sum((probs - np.eye(3)[actual])**2))
    predicted_outcome = int(np.argmax(probs))
    pred_score = record.get("score", "")
    return {"result":f"{h}–{a}", "outcome":("Home", "Draw", "Away")[actual],
            "correct_outcome":predicted_outcome == actual,
            "correct_exact_score":pred_score.replace("-", "–") == f"{h}–{a}",
            "brier":brier, "match_date":match.Date.date().isoformat()}


def table_for_predictions(records: list[dict], matches: pd.DataFrame, now: datetime | None = None,
                          provider_finals: pd.DataFrame | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    rows = []
    for record in records:
        try:
            kickoff = _utc_dt(record["kickoff_utc"])
            created = _utc_dt(record["github_created_at_utc"])
            # Prefer fresher confirmed provider scores, falling back to the historical CSV.
            # Do not use in-progress scores to settle a prediction.
            outcome = None
            result_source = None
            if provider_finals is not None and not provider_finals.empty:
                outcome = find_result(record, provider_finals)
                if outcome is not None:
                    result_source = "ESPN (confirmed final)"
            if outcome is None:
                outcome = find_result(record, matches)
                if outcome is not None:
                    result_source = "Football-Data.co.uk historical CSV"
            if created >= kickoff:
                status = "Not verified pre-match"
            elif outcome:
                status = "Final"
            elif now < kickoff:
                status = "Upcoming"
            else:
                status = "Awaiting result"
            rows.append({
                "Saved (UTC)":created.strftime("%Y-%m-%d %H:%M"),
                "Fixture (JST)":kickoff.astimezone(JST_ZONE).strftime("%Y-%m-%d %H:%M"),
                "Home":record["home"], "Away":record["away"],
                "Predicted score":record["score"],
                "Home win %":round(float(record["p_home"])*100, 1),
                "Draw %":round(float(record["p_draw"])*100, 1),
                "Away win %":round(float(record["p_away"])*100, 1),
                "Status":status,
                "Actual score":outcome["result"] if outcome else "—",
                "Result source": result_source or "Pending",
                "Brier":round(outcome["brier"], 4) if outcome else np.nan,
                "Exact correct":outcome["correct_exact_score"] if outcome else None,
                "Outcome correct":outcome["correct_outcome"] if outcome else None,
                "GitHub record":record.get("issue_url", ""),
                "Source":record.get("fixture_source", ""),
            })
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return pd.DataFrame(rows)


def save_predictions_batch(repo: str, token: str, records: list[dict], session=requests) -> dict:
    """Save up to 20 verified future forecasts, with one GitHub duplicate check.

    Each forecast is its own immutable GitHub Issue. Partial failures are
    returned and can safely be retried; already-saved fixtures are skipped.
    This is not an atomic transaction across several GitHub API calls.
    """
    if not records:
        return {"saved": [], "skipped": [], "failed": []}
    if len(records) > 20:
        raise ValueError("A batch may contain at most 20 matches.")
    if not token:
        raise ValueError("Configure GITHUB_TOKEN in Streamlit Secrets before saving.")
    existing = list_predictions(repo, token=token, session=session)
    result = {"saved": [], "skipped": [], "failed": []}
    for record in records:
        label = f"{record.get('home', '?')} vs {record.get('away', '?')}"
        if any(same_fixture(r, record) for r in existing):
            result["skipped"].append(label)
            continue
        try:
            url = save_prediction(repo, token, record, session=session, existing_records=existing)
            result["saved"].append({"fixture": label, "url": url})
            existing.append(record)  # Prevent duplicates inside this very batch.
        except Exception as exc:
            result["failed"].append({"fixture": label, "error": str(exc)})
    return result
