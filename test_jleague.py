import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jleague_data import clean_japan_csv, parse_japan_bytes, season_for_date
from jleague_model import MODELS, MODEL_A, backtest, predict_models, probs_for_grid, score_for_grid
from jleague_fixtures import parse_espn, match_club, group_weeks, JST
from prediction_tracker import make_prediction_record, _valid_record, same_fixture, find_result


def make_raw():
    teams = ["Kobe", "Kashima", "Urawa", "Hiroshima", "Nagoya", "Kyoto", "Kawasaki", "Kashiwa", "Gamba Osaka", "Cerezo Osaka"]
    rows = []
    for year in (2022, 2023, 2024, 2025):
        for j in range(140):
            h = teams[j % 10]
            a = teams[(j * 3 + 1) % 10]
            if h == a:
                a = teams[(j * 3 + 2) % 10]
            dt = pd.Timestamp(year, 2, 1) + pd.Timedelta(days=j*2)
            rows.append({"Country":"Japan", "League":"J1", "Season":year,
                        "Date":dt.strftime('%d/%m/%Y'), "Home":h, "Away":a,
                        "HG":(j+year)%4, "AG":j%3})
    rows.append({"League":"J1", "Date":"04/05/2026", "Home":"Kobe", "Away":"Kashima", "HG":2, "AG":2})
    rows.append({"League":"J1", "Date":"15/08/2026", "Home":"Kobe", "Away":"Kashima", "HG":1, "AG":0})
    rows.append({"League":"J1", "Date":"25/12/2026", "Home":"Kobe", "Away":"Kashima", "HG":np.nan, "AG":np.nan})
    return pd.DataFrame(rows)


def test_japan_csv_format_and_special_tournament():
    raw = make_raw()
    matches, unfinished = clean_japan_csv(raw)
    assert len(matches) == 561
    assert '2026 special' not in matches.SeasonCode.unique()
    assert matches.iloc[-1].SeasonCode == '2026/27'
    assert {'Date','HomeTeam','AwayTeam','FTHG','FTAG','SeasonCode'}.issubset(matches.columns)
    assert season_for_date(pd.Timestamp('2027-05-04')) == '2026/27'
    assert len(unfinished) <= 1


def test_all_four_models_valid_probabilities():
    matches, _ = clean_japan_csv(make_raw())
    result = predict_models(matches.loc[matches.Date < '2025-01-01'], 'Kobe', 'Kashima')
    assert set(result['grids']) == set(MODELS)
    for name in MODELS:
        grid = result['grids'][name]
        p = probs_for_grid(grid)
        assert grid.shape == (13, 13)
        assert abs(p.sum()-1) < 1e-9
        assert (p > 0).all()
        assert len(score_for_grid(grid)[0].split('–')) == 2
        assert (result['goals'][name] > 0).all()


def test_backtest_has_consistent_match_counts():
    matches, _ = clean_japan_csv(make_raw())
    summary, detail = backtest(matches, years=('2024','2025'))
    assert not summary.empty
    assert set(summary.Model) == set(MODELS)
    c = detail.groupby('Model').size().to_numpy()
    assert np.all(c == c[0])
    assert detail.Brier.between(0, 2).all()


def test_espn_parser_conservative_finals_and_fixtures():
    now = datetime(2026, 11, 9, 0, tzinfo=timezone.utc)
    def event(name, dt, finish, s1, s2):
        return {'date':dt, 'status':{'type':{'name':name,'completed':finish}},'competitions':[{'competitors':[
            {'homeAway':'home','team':{'displayName':'Vissel Kobe'},'score':s1},
            {'homeAway':'away','team':{'displayName':'Kashima Antlers'},'score':s2}]}]}
    body = {'events':[event('STATUS_FINAL','2026-11-07T10:00:00Z',True,'1','0'),
                      event('STATUS_SCHEDULED','2026-11-12T10:00:00Z',False,None,None),
                      event('STATUS_IN_PROGRESS','2026-11-09T10:00:00Z',False,'2','2')]}
    fixtures, finals, unknown = parse_espn(body, ['Kobe','Kashima'], now)
    assert len(fixtures) == 1 and len(finals) == 1
    assert finals.iloc[0].FTHG == 1
    assert match_club('Some unknown name', ['Kobe','Kashima']) is None
    assert group_weeks(fixtures)


def test_prematch_records_and_settlement():
    future = datetime.now(timezone.utc) + timedelta(days=3)
    from jleague_model import poisson_grid
    g = poisson_grid(1.4, 1.1)
    probs = probs_for_grid(g)
    score, prob = score_for_grid(g)
    record = make_prediction_record('Kobe','Kashima',future,'manual','2026-11-01',g,np.array([1.4,1.1]),score,prob,probs)
    assert _valid_record(record)
    assert same_fixture(record,dict(record))
    saved = future - timedelta(days=1)
    record['github_created_at_utc'] = saved.isoformat()
    games = pd.DataFrame([{'Date':pd.Timestamp(future.astimezone(JST).date()),'HomeTeam':'Kobe','AwayTeam':'Kashima','FTHG':1,'FTAG':0}])
    r = find_result(record, games)
    assert r and r['result'] == '1–0'


def test_missing_fields_are_not_silently_accepted():
    with pytest.raises(ValueError, match='missing'):
        clean_japan_csv(pd.DataFrame({'Date':['2026-01-01']}))
