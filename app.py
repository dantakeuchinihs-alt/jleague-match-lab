"""J.League Match Lab — an independent J1 Streamlit companion to EPL Match Lab.

Primary source: https://www.football-data.co.uk/japan.php
Streamlit Community Cloud: point to this file in a separate GitHub repository.
"""
from __future__ import annotations

import html
import hmac
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from jleague_data import DATA_PAGE, DATA_URL, current_clubs, download_japan_csv, parse_japan_bytes
from jleague_model import (
    MODEL_A, MODEL_B, MODEL_C, MODEL_D, MODELS, backtest, club_ratings,
    historical_window, predict_models, prediction_rows, probs_for_grid, score_for_grid,
)
from jleague_fixtures import JST, download_fixtures_scores, group_weeks
from prediction_tracker import (
    list_predictions, make_prediction_record, save_prediction, save_predictions_batch,
    same_fixture, table_for_predictions,
)

ROOT = Path(__file__).resolve().parent
FAVICON = ROOT / "assets" / "favicon.png"
st.set_page_config(page_title="J.League Match Lab | Independent football analysis",
                   page_icon=Image.open(FAVICON) if FAVICON.is_file() else Image.new("RGB", (64,64), "#111412"),
                   layout="wide", initial_sidebar_state="expanded")
style_file = ROOT / "assets" / "site.css"
if style_file.is_file():
    st.markdown("<style>" + style_file.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""

REPO, GITHUB_TOKEN, SAVE_PIN = secret("GITHUB_REPO"), secret("GITHUB_TOKEN"), secret("SAVE_PIN")

@st.cache_data(ttl=6 * 3600, show_spinner="Downloading Japan J1 results…")
def get_results():
    return download_japan_csv()

@st.cache_data(ttl=30 * 60, show_spinner="Checking upcoming J1 fixtures…")
def get_scores(clubs: tuple[str, ...]):
    return download_fixtures_scores(list(clubs))

@st.cache_data(ttl=90, show_spinner=False)
def get_records(repo: str, token: str):
    return list_predictions(repo, token)

@st.cache_data(show_spinner="Evaluating historical J1 predictions…")
def get_evidence(matches: pd.DataFrame):
    return backtest(matches)


def e(text: str) -> str:
    return html.escape(str(text), quote=True)


def score_card(home: str, away: str, result: dict, name: str = MODEL_A):
    grid, mu = result["grids"][name], result["goals"][name]
    score, exact = score_for_grid(grid)
    h, a = score.split("–")
    st.markdown(f"""
    <div class="score-panel">
      <div class="score-panel-top"><span class="score-label">Model forecast / {e(name)}</span>
      <span class="score-prob">Exact score probability · {exact*100:.1f}%</span></div>
      <div class="score-line"><div class="score-club">{e(home)}<small>Home</small></div>
      <div class="score-numbers">{h}<span>–</span>{a}</div>
      <div class="score-club">{e(away)}<small>Away</small></div></div>
      <div class="score-bottom"><span class="score-goals">Expected goals: {mu[0]:.2f} – {mu[1]:.2f}</span>
      <span class="score-method">Independent Poisson score model</span></div>
    </div>""", unsafe_allow_html=True)


def prob_chart(probs: np.ndarray, home: str, away: str):
    labels = [f"{home} win", "Draw", f"{away} win"]
    fig = go.Figure(go.Bar(x=list(100 * probs), y=labels, orientation="h",
                           marker_color=["#C9B38E", "#819085", "#9DA8A1"],
                           text=[f"{100*p:.1f}%" for p in probs], textposition="outside"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#D1D5CE", "size": 12}, height=200,
                      xaxis={"range": [0, 105], "visible": False},
                      yaxis={"autorange": "reversed"},
                      margin={"l": 0, "r": 30, "t": 12, "b": 12}, showlegend=False)
    return fig


def heatmap(grid: np.ndarray, home: str, away: str):
    scores = grid[:7, :7] * 100
    fig = go.Figure(go.Heatmap(z=scores, x=list(range(7)), y=list(range(7)),
        colorscale=[[0, "#191D1A"], [.4, "#4B574C"], [.75, "#8C947F"], [1, "#C9B38E"]],
        text=np.round(scores, 1), texttemplate="%{text:.1f}",
        hovertemplate=f"{e(home)} %{{y}} – %{{x}} {e(away)}<br>%{{z:.1f}}%<extra></extra>",
        colorbar={"title": "%", "thickness": 11}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#D1D5CE", "size": 12}, height=425,
                      xaxis={"title": f"{away} goals", "dtick": 1},
                      yaxis={"title": f"{home} goals", "autorange": "reversed", "dtick": 1},
                      margin={"l": 45, "r": 20, "t": 15, "b": 55})
    return fig


def new_record(fixture: dict, matches: pd.DataFrame) -> dict:
    kickoff = fixture["kickoff_utc"]
    cutoff = pd.Timestamp(kickoff.astimezone(JST).date())
    past = historical_window(matches, cutoff)
    pred = predict_models(past, fixture["home"], fixture["away"])
    grid = pred["grids"][MODEL_A]
    score, exact = score_for_grid(grid)
    probs = probs_for_grid(grid)
    data_date = past.Date.max().date().isoformat()
    return make_prediction_record(fixture["home"], fixture["away"], kickoff,
                                  fixture["source"], data_date,
                                  grid, pred["goals"][MODEL_A], score, exact, probs)


def records_or_empty():
    if not REPO:
        return []
    try:
        return get_records(REPO, GITHUB_TOKEN)
    except Exception as exc:
        st.warning(f"Prediction history could not be loaded: {exc}")
        return []


def validate_pin(pin: str) -> bool:
    return bool(SAVE_PIN and pin and hmac.compare_digest(pin, SAVE_PIN))


def normalize_display(table: pd.DataFrame):
    st.dataframe(table, use_container_width=True, hide_index=True)


# Sidebar — file fallback keeps app useful when source hosting restricts requests.
st.sidebar.markdown('<div class="sidebar-logo"><div class="sidebar-wordmark">J.League <i>Lab.</i></div><div class="sidebar-subline">Independent match intelligence</div></div>', unsafe_allow_html=True)
upload = st.sidebar.file_uploader("Optional: upload JPN.csv from Football-Data.co.uk", type=["csv"])
if st.sidebar.button("Refresh historical data", use_container_width=True):
    get_results.clear()
    st.rerun()
try:
    matches, provider_pending = parse_japan_bytes(upload.getvalue()) if upload else get_results()
except Exception as exc:
    st.error("Could not load Japan match data. The provider might be temporarily blocking downloads.")
    st.code(str(exc))
    st.markdown(f"Download the CSV at [Football-Data Japan]({DATA_PAGE}) and upload it through the sidebar. The project never invents scores when data is unavailable.")
    st.stop()
if matches.empty:
    st.error("The uploaded file has no completed J1 league matches.")
    st.stop()
clubs, season = current_clubs(matches)
if len(clubs) < 6:
    st.error("Too few J1 clubs were detected in the latest season. Check the CSV.")
    st.stop()
now_jst = datetime.now(JST)
latest = matches.Date.max().strftime("%d %b %Y")
st.sidebar.caption("Data source: Football-Data.co.uk / Japan")
st.sidebar.metric("Completed results loaded", f"{len(matches):,}")
st.sidebar.metric("Latest published match", latest)
st.sidebar.caption(f"Most recent competition: {season}")
st.sidebar.caption("Historical scores may be published with a delay. The 2026 transitional tournament is excluded from modelling.")
st.sidebar.markdown(f"[Source and update notes]({DATA_PAGE})")

# Fixture service separate from goal model. Never block entire app if feed fails.
fixtures, espn_finals, feed_notes = [], pd.DataFrame(), []
try:
    fixtures, espn_finals, feed_notes = get_scores(tuple(clubs))
except Exception as exc:
    feed_notes = [f"Fixture/score feed unavailable: {str(exc)[:160]}"]

st.markdown(f"""<div class="masthead"><div class="brand-row"><span class="brand-kicker">Japan / J1 League / Football intelligence</span>
<span class="brand-edition">Independent research edition</span></div>
<h1 class="hero-title">J.League <em>Match Lab.</em></h1>
<div class="hero-subrow"><p class="hero-subtitle">Attack meets defence. Explore goal probabilities, team strengths, forthcoming matches and a permanent record of predictions.</p>
<span class="hero-metadata">Historical data through {e(latest)} · {e(season)}</span></div></div>""", unsafe_allow_html=True)

t_predict, t_scores, t_fixtures, t_tracker, t_ratings, t_evidence = st.tabs(
    ["Predict", "Scores", "Fixtures", "Tracker", "Team ratings", "Method"])

with t_predict:
    st.subheader("Choose your matchup")
    a, b = st.columns(2)
    home = a.selectbox("Home club", clubs, index=0, key="predict_home")
    away = b.selectbox("Away club", clubs, index=1 if len(clubs)>1 else 0, key="predict_away")
    if home == away:
        st.info("Select two different clubs.")
    else:
        cutoff = pd.Timestamp(now_jst.date()) + pd.Timedelta(days=1)
        past = historical_window(matches, cutoff)
        pred = predict_models(past, home, away)
        score_card(home, away, pred)
        st.plotly_chart(prob_chart(probs_for_grid(pred["grids"][MODEL_A]), home, away), use_container_width=True)
        st.caption("Hypothetical matchup, not an official fixture. Player availability, injuries and tactical formations are not included.")
        st.subheader("Compare the models")
        normalize_display(prediction_rows(pred))
        st.caption("All four methods use historical goals only. The blend averages probability distributions, not scores alone. Model selection is not based on J1 out-of-sample validation yet.")
        left, right = st.columns([1,1])
        with left:
            st.subheader("Attack / defence ratings")
            r = pred["ratings"]
            normalize_display(pd.DataFrame({"Rating":list(r.keys()),"Relative to J1 average":[round(x,2) for x in r.values()]}))
            st.caption("Attack above 1.00 indicates more goals than the venue league average. Defensive vulnerability below 1.00 indicates fewer goals conceded.")
            st.caption(f"Home club sample: {pred['counts'][0]} recent home matches; away club sample: {pred['counts'][1]} recent away matches. Values are smoothed toward league means.")
        with right:
            st.subheader("Exact-score probability map")
            st.plotly_chart(heatmap(pred["grids"][MODEL_A], home, away), use_container_width=True)

with t_scores:
    st.subheader("Latest results")
    st.caption("Football-Data historical records are the primary source. The optional ESPN score feed is used only to show newer confirmed finals; it does not retrain the model.")
    historical_rows = matches.sort_values("Date", ascending=False).head(25).copy()
    historical_rows["Date"] = historical_rows.Date.dt.strftime("%d %b %Y")
    normalize_display(historical_rows[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].rename(columns={"HomeTeam":"Home", "AwayTeam":"Away", "FTHG":"Home goals", "FTAG":"Away goals"}))
    if not espn_finals.empty:
        st.subheader("Additional provider-confirmed finals")
        newest = espn_finals.loc[espn_finals.Date.gt(matches.Date.max() - pd.Timedelta(days=14))].sort_values("Date", ascending=False)
        if len(newest):
            normalize_display(newest.rename(columns={"Date":"Date (JST)", "HomeTeam":"Home", "AwayTeam":"Away", "FTHG":"Home goals", "FTAG":"Away goals"}))
    else:
        st.info("No additional confirmed final scores are currently available from the optional feed.")

with t_fixtures:
    st.subheader("Upcoming fixtures")
    st.caption("Automatic schedules use ESPN's unofficial public J1 scoreboard when available. They must be checked against the official J.League fixture list before saving; weekly groups are not official round numbers.")
    # Append one user-entered fixture without claiming it was verified by a provider.
    manual = []
    with st.expander("Add a fixture manually (if no automatic schedule appears)"):
        with st.form("manual_fixture"):
            c1, c2 = st.columns(2)
            mh = c1.selectbox("Home", clubs, key="manual_home")
            ma = c2.selectbox("Away", clubs, index=1, key="manual_away")
            fd = c1.date_input("Scheduled date (Japan)", value=now_jst.date() + timedelta(days=1), min_value=now_jst.date())
            ft = c2.time_input("Kickoff time (JST)", value=time(19,0))
            added = st.form_submit_button("Use this fixture")
            if added:
                kickoff = datetime.combine(fd, ft, tzinfo=JST).astimezone(timezone.utc)
                if mh == ma:
                    st.error("Choose two different clubs.")
                elif kickoff <= datetime.now(timezone.utc):
                    st.error("Kickoff must be in the future.")
                else:
                    st.session_state["manual_fixture_record"] = {"home":mh, "away":ma, "kickoff_utc":kickoff,
                                                             "source":"Manually entered kickoff (not provider-verified)",
                                                             "label": f"{kickoff.astimezone(JST):%a %d %b %H:%M} JST · {mh} vs {ma}"}
    if "manual_fixture_record" in st.session_state:
        manual.append(st.session_state["manual_fixture_record"])
    future = sorted(fixtures + manual, key=lambda f: f["kickoff_utc"])
    if not future:
        st.warning("The fixture feed provided no confidently matched upcoming games. You can enter a verified fixture above, or try again later.")
    else:
        weeks = group_weeks(future)
        selected = st.selectbox("Calendar window", list(weeks))
        group = weeks[selected]
        forecast_rows, records = [], []
        for fixture in group:
            try:
                rec = new_record(fixture, matches)
                records.append(rec)
                forecast_rows.append({"Kickoff (JST)":fixture["kickoff_utc"].astimezone(JST).strftime("%d %b %H:%M"),
                    "Home":rec["home"],"Away":rec["away"],"Score":rec["score"],
                    "Home %":round(rec["p_home"]*100,1),"Draw %":round(rec["p_draw"]*100,1),"Away %":round(rec["p_away"]*100,1),
                    "Data through":rec["data_cutoff_utc_date"],"Fixture source":rec["fixture_source"]})
            except Exception as exc:
                st.warning(f"Could not forecast {fixture['home']} vs {fixture['away']}: {exc}")
        if forecast_rows:
            normalize_display(pd.DataFrame(forecast_rows))
            st.download_button("Download gameweek CSV", pd.DataFrame(forecast_rows).to_csv(index=False).encode("utf-8-sig"),
                               "jleague_gameweek_forecasts.csv", "text/csv")
            with st.form("batch_save"):
                pin = st.text_input("Private save password", type="password")
                confirmed = st.checkbox("I checked the fixture dates and kickoff times.")
                save = st.form_submit_button("Save this set of predictions", type="primary")
                if save:
                    if not (REPO and GITHUB_TOKEN and SAVE_PIN):
                        st.error("Configure GITHUB_REPO, GITHUB_TOKEN and SAVE_PIN in this new app's Streamlit Secrets first.")
                    elif not validate_pin(pin):
                        st.error("Incorrect save password.")
                    elif not confirmed:
                        st.warning("Confirm the kickoff details before saving.")
                    else:
                        try:
                            results = save_predictions_batch(REPO, GITHUB_TOKEN, records)
                            st.success(f"Saved {len(results['saved'])}; already saved {len(results['skipped'])}; failed {len(results['failed'])}.")
                            for failure in results["failed"]:
                                st.warning(f"{failure['fixture']}: {failure['error']}")
                            get_records.clear()
                        except Exception as exc:
                            st.error(f"Could not save: {exc}")
    if feed_notes:
        with st.expander("Fixture matching / source notes"):
            st.caption("Unmatched team names are skipped rather than guessed: " + "; ".join(feed_notes[:10]))

with t_tracker:
    st.subheader("Prediction history")
    st.caption("Saved records are locked as GitHub Issues in the J.League repository. Public repositories have public prediction histories; the save password remains private.")
    if st.button("Refresh prediction history"):
        get_records.clear()
        get_scores.clear()
        st.rerun()
    if not REPO:
        st.info("To enable permanent predictions, add GITHUB_REPO, GITHUB_TOKEN and SAVE_PIN in this new app's Streamlit Secrets.")
    else:
        records = records_or_empty()
        if records:
            history = table_for_predictions(records, matches, provider_finals=espn_finals)
            if len(history):
                ordered = history.sort_values("Saved (UTC)", ascending=False)
                normalize_display(ordered)
                st.download_button("Download prediction history CSV", history.to_csv(index=False).encode("utf-8-sig"),
                                   "jleague_saved_predictions.csv", "text/csv")
                settled = history.loc[history.Status.eq("Final")]
                if len(settled):
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Settled forecasts", len(settled))
                    c2.metric("Mean Brier score", f"{settled.Brier.mean():.3f}")
                    c3.metric("Correct outcomes", f"{100*settled['Outcome correct'].mean():.1f}%")
                else:
                    st.caption("No saved predictions have a verifiably finished result yet.")
        else:
            st.info("No saved J1 predictions yet. Use the Fixtures tab to save one before kickoff.")

with t_ratings:
    st.subheader("Team strength profiles")
    st.caption("Venue-specific attacking strength and defensive vulnerability (1.00 = J1 league average). A LOWER defensive vulnerability is better.")
    cutoff = pd.Timestamp(now_jst.date()) + pd.Timedelta(days=1)
    rated = club_ratings(historical_window(matches, cutoff), clubs)
    sort_col = st.selectbox("Rank by", ["Home attack", "Away attack", "Home defence vulnerability", "Away defence vulnerability"])
    ascending = "vulnerability" in sort_col
    rated = rated.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    normalize_display(rated.round(3))
    selected = st.selectbox("Inspect club", clubs, key="rating_team")
    profile = rated.loc[rated.Club.eq(selected)].iloc[0]
    st.subheader(selected)
    c1,c2,c3,c4 = st.columns(4)
    for col, label in zip([c1,c2,c3,c4], ["Home attack", "Away attack", "Home defence vulnerability", "Away defence vulnerability"]):
        col.metric(label, f"{profile[label]:.2f}")

with t_evidence:
    st.subheader("Model methodology and evidence")
    st.markdown("**Four competing models.** Attack × Defence is the initial reference, not a proven J1 winner. The simple average uses a team’s scoring rate and the opponent’s conceded rate; Recent Attack × Defence uses a shorter venue window; the blend is a 50/50 probability mixture.")
    st.markdown("**Data discipline.** Historical evaluations use only matches strictly before each match date, with venue smoothing for small samples. The transitional 2026 special competition is excluded. These historical seasons have been inspected during development, so prospective tracking is still required.")
    if st.button("Calculate historical backtest", type="primary"):
        with st.spinner("Evaluating walk-forward predictions…"):
            summary, detailed = get_evidence(matches)
        if summary.empty:
            st.warning("Not enough historical matches to evaluate the selected seasons.")
        else:
            result = summary.copy()
            result["Brier"] = result["Brier"].round(4)
            result["Goal MAE"] = result["Goal MAE"].round(3)
            normalize_display(result)
            two = result.loc[result.Season.isin(["2024", "2025"])]
            if len(two):
                mean = two.groupby("Model", as_index=False).apply(lambda x: pd.Series({
                    "Brier": np.average(x.Brier, weights=x.Matches),
                    "Goal MAE": np.average(x["Goal MAE"], weights=x.Matches),
                    "Matches": int(x.Matches.sum())}), include_groups=False)
                st.subheader("Historical two-season comparison")
                normalize_display(mean.sort_values("Brier").round(4))
            st.caption("Brier: lower is better for three-way win/draw/loss probabilities. Goal MAE: lower is better for the predicted numbers of goals. Results are retrospective, not a validated promise of future accuracy.")
    st.markdown("**Limitations.** The Japan CSV does not document EPL-style shots-on-target columns, so this application does not pretend to reproduce the EPL shots model. Fixture availability relies on a separate, unofficial provider and may be incomplete. The historical source is updated intermittently, not minute-by-minute.")
    st.markdown(f"Data: [Football-Data.co.uk Japan]({DATA_PAGE}) · Current J1 format: [J.LEAGUE official site](https://www.jleague.jp/en/outline/j1/)")
