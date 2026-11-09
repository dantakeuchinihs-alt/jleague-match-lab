# J.League Match Lab

A standalone J1 League match prediction and analytics website, adapted from the EPL Match Lab architecture. This repository is **independent** of the Premier League app. The data model uses [Football-Data.co.uk's Japan CSV](https://www.football-data.co.uk/japan.php); it does not include or redistribute the provider's records.

## What it does

- **Predict** — score probabilities and expected goals for any two current J1 clubs, using four models on the same match.
- **Scores** — latest completed J1 results from Football-Data.co.uk; optionally show newer ESPN-confirmed final results.
- **Fixtures** — upcoming J1 matches from an optional, undocumented ESPN public endpoint. Predict and save a whole Monday–Sunday calendar group, or enter a verified kickoff manually if fixture coverage is missing.
- **Tracker** — lock predictions before kickoff as individual GitHub Issues, review the history, check settled matches, and inspect outcome accuracy/Brier scores. No GitHub key is needed just to use the predictor.
- **Team ratings** — compare home/away attack and defensive vulnerability, relative to league averages.
- **Method** — historical walk-forward Brier and goal-MAE comparisons. Historical scores are *not* prospective validation.

## Prediction models

1. **Attack × Defence** *(default)*: estimated home goals = league home goals/game × home attacking ratio × away defensive vulnerability ratio; away equivalent uses away goals baseline. Current predictions use **every available venue match from the latest two ordinary J1 seasons**, smoothed using a seven-game league-average prior.
2. **Simple average**: mean of the smoothed attack and opponent's conceding rates, separately home and away.
3. **Recent Attack × Defence**: the same model using up to six venue matches **within the latest two seasons**, with a five-match prior.
4. **50/50 blend**: an equal mixture of the full Poisson score distributions from models 1 and 3.

All models use the same 90-minute full-time goals. Win/draw/loss probabilities are calculated from each Poisson score matrix. JPN.csv does **not** document the shots-on-target columns used by EPL Match Lab, so its shots model is deliberately not copied.

## J.League calendar transition

- Historical J1 league seasons through 2025 are calendar-year tournaments.
- The **2026 J1 100 Year Vision League** was a special transitional competition. To avoid conflating its group/penalty/playoff structure with a normal league, this app excludes January–July 2026 match records from training, ratings and evaluation.
- The **2026/27 J1 season started 7 August 2026**, with a conventional home-and-away 38-matchday format. Records from August 2026 to June 2027 are labelled `2026/27` based on their *dates*, not the CSV's season label.
- Current-season selection is derived from the latest completed league results. Newly promoted clubs have heavily smoothed ratings until more J1 matches accumulate.

Official information: [J.League J1 competition overview](https://www.jleague.jp/en/outline/j1/).

## Deploy — for beginners

**Create a new GitHub repository** such as `jleague-match-lab`. **Do not upload these files to your EPL repository.**

1. Download the ZIP, unzip it, and upload the contents of `JLeague_Match_Lab` to the **root** of the new GitHub repository (not the ZIP, and not an extra enclosing folder).
2. Make sure GitHub shows `app.py`, `jleague_data.py`, `jleague_model.py`, `jleague_fixtures.py`, `prediction_tracker.py`, `requirements.txt`, `README.md`, plus folders `assets` and `.streamlit`.
3. If GitHub's browser uploader misses the hidden theme folder, use **Add file > Create new file**, enter `.streamlit/config.toml` and copy its contents from the downloaded folder.
4. At [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app > From GitHub** and select your **new** repo, `main` branch, and `app.py` as entry point. Deploy.
5. The prediction pages work without any API key. Streamlit downloads the Japan data on first launch. If the source returns an HTTP error, download `JPN.csv` from [Football-Data Japan](https://www.football-data.co.uk/japan.php) to your computer and upload it using the app's sidebar. The provider's CSV is not included in this project.
6. To save predictions, enable **Issues** under your *new* GitHub repository's settings, and create a **fine-grained personal access token** restricted to **only that J.League repository** with **Issues: Read and write**. A token restricted to the EPL repository will not work on the new one.
7. Open your *new* Streamlit app's **Settings > Secrets** and enter:

```toml
GITHUB_REPO = "YOUR_GITHUB_USERNAME/jleague-match-lab"
GITHUB_TOKEN = "PASTE_YOUR_NEW_PRIVATE_TOKEN"
SAVE_PIN = "CHOOSE_A_LONG_PRIVATE_SAVE_PASSWORD"
```

**Never put tokens, PINs or `.streamlit/secrets.toml` in GitHub.** GitHub Issues in a public repository are themselves public, so stored predictions are visible to everyone; secrets are not stored in issues.

## Updating results

- The app caches Football-Data results for **six hours** and checks again when used. You can force a re-download with **Refresh historical data**.
- The provider says its archives are updated intermittently; an app refresh can't make unpublished matches appear.
- Fixtures and optional additional final scores are queried through ESPN's *undocumented, unofficial* public J1 scoreboard, cached **30 minutes**. That service is not guaranteed: it might stop working, use inconsistent club names, or have incomplete schedules. Uncertain names are skipped; the form allows a manual, verified fixture. This feed is **never used to fit the predictor**.
- Predictions saved before kickoff are evaluated when you reopen the Tracker, using only completed results. Predictions are not automatically written to GitHub unless you click Save and confirm your private PIN.

## Testing

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m streamlit run app.py
```

The project code has been syntax checked and its importer, model math, fixtures parser, and tracking have been tested with **synthetic** data. The live Japan download and Streamlit hosting cannot be exercised in this build environment; check both after deployment.

## Data and licensing

Source: [Football-Data.co.uk Japan](https://www.football-data.co.uk/japan.php) (`https://www.football-data.co.uk/new/JPN.csv`). The website specifies that free data are intended for private individual use, not commercial data products. This project fetches historical data without bundling them. Check source permissions before commercial use or redistribution.

Scores may be delayed; this is an analysis tool, not a live betting or betting-advice service.


## Two-season model + Team Strength Profiles (latest update)

**Both predictions and Team ratings now use the two most recent ordinary J1 seasons with completed results.** The training window is based on season labels, **not a fixed 730-day period**. For the current 2026/27 J1 league, that means **2025 and 2026/27**, while the transitional **2026 special tournament is always excluded**. As soon as 2027/28 match results become available, the window moves forward to **2026/27 and 2027/28**.

- Attack × Defence and Simple Average use all home/away matches from those two seasons and a seven-game neutral league-average prior.
- Recent Attack × Defence uses up to the most recent six home/away matches in the same two-season dataset. The 50/50 blend is built from those updated models.
- Team ratings use the **same two seasons and prior** as the main model. The table includes each club's home and away sample sizes.
- Older results remain available for **historical backtesting**. For each backtested match, the model chooses the two latest seasons whose match results existed *before that match day*, so no future match data leaks into training.
- Prediction history stays intact. **New saved forecasts** include a `model_version` and the exact `training_seasons` in the locked GitHub record; old records retain their original values rather than being recalculated.
- A newly promoted team with little J1 history is smoothed toward the league average; this is a limitation, not evidence that it is an average-strength team.

**Model choice has not been independently validated for J1.** Two seasons should better represent present-day squads than distant historical seasons, but its actual predictive advantage must be evaluated prospectively.

### Apply this update

Download the separate two-season update ZIP and upload **`app.py`, `jleague_model.py`, `prediction_tracker.py`** to the root of your **J.League** GitHub repository, replacing the existing versions. The `README.md` is optional. **Do not upload these files to your Premier League repository.** All existing Streamlit Secrets remain unchanged. Streamlit will normally redeploy after you commit changes.
