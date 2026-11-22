import numpy as np
import pandas as pd

from mp125 import cleaning as cl
from tests.test_data_checks import make_df

PARAMS = {
    "data": {"target": "delay_min", "date_col": "date"},
    "checks": {"sentinel_candidates": [-1, 9999], "stuck_min_run": 10,
               "typo_gap_ratio": 3.0},
}


def dirty():
    df = make_df()
    df.loc[:4, "hour"] = 9999
    df.loc[10:24, "driver_exp_yr"] = 5
    df.loc[[3, 7], "delay_min"] *= 10
    df.loc[0, "weather"] = " Rain"
    return df


def test_cleaning_keeps_row_count():
    assert len(cl.clean_train(dirty(), PARAMS)) == len(dirty())


def test_cleaning_removes_problems():
    raw = dirty()
    out = cl.clean_train(raw, PARAMS)
    assert not (out["hour"] == 9999).any()
    assert out.loc[10:24, "driver_exp_yr"].isna().all()
    assert out["delay_min"].max() < 40
    assert out["delay_min"][3] == raw["delay_min"][3] / 10
    assert set(out["weather"]) == {"clear", "rain", "fog"}


def test_fill_uses_train_medians_only():
    out = cl.clean_train(dirty(), PARAMS)
    fill = {"hour": 12.0, "driver_exp_yr": 9.0}
    filled = cl.apply_fill(out, fill)
    assert not filled["hour"].isna().any()
    assert not filled["driver_exp_yr"].isna().any()
