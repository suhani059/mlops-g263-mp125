"""Tests for the seven data checks. Small data is built inside the test:
the real CSVs are not in git."""
import numpy as np
import pandas as pd

from mp125 import data_checks as dc


def make_df(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "record_id": [f"R{i:03d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "hour": rng.integers(6, 22, n),
        "driver_exp_yr": rng.integers(1, 25, n),
        "weather": ["clear", "rain", "fog", "rain"] * (n // 4),
        "delay_min": rng.uniform(10, 30, n).round(1),
    })


def test_sentinel_found():
    df = make_df()
    df.loc[:4, "hour"] = 9999
    res = dc.check_sentinel(df, [-1, 9999])
    assert res == {"present": True, "column": "hour", "value": 9999, "count": 5}


def test_clean_data_has_no_sentinel_or_duplicates():
    df = make_df()
    assert dc.check_sentinel(df, [-1, 9999])["present"] is False
    assert dc.check_duplicates(df)["present"] is False


def test_duplicates_ignore_record_id():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]].assign(record_id="R999")], ignore_index=True)
    assert dc.check_duplicates(df) == {"present": True, "count": 1}


def test_stuck_sensor_window():
    df = make_df()
    df.loc[10:24, "driver_exp_yr"] = 5
    res = dc.check_stuck_sensor(df, "date", min_run=10)
    assert res["present"] and res["column"] == "driver_exp_yr"
    assert res["start_date"] == df.loc[10, "date"]
    assert res["end_date"] == df.loc[24, "date"]


def test_target_typos_found():
    df = make_df()
    df.loc[[3, 7, 11], "delay_min"] *= 10
    assert dc.check_target_typos(df, "delay_min") == {"present": True, "count": 3}


def test_spelling_levels():
    df = make_df()
    df.loc[0, "weather"] = " Rain"
    df.loc[1, "weather"] = "FOG"
    df.loc[2, "weather"] = "cle."
    res = dc.check_spelling(df)
    assert res == {"present": True, "column": "weather", "true_levels": 3}
