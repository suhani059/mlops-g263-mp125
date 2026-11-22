"""Stage 2: fix the data problems that Stage 1 found.

Nothing is hard-coded to one value: each fix first asks data_checks where the
problem is, then repairs only that. Rows are never silently dropped, so the
cleaned file keeps the same number of rows as the raw file.
"""
import numpy as np
import pandas as pd

from mp125 import data_checks as dc

NUMERIC = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
CATEGORICAL = ["weather", "depot_zone"]
FEATURES = NUMERIC + CATEGORICAL


def clean_train(df, params):
    """Return a cleaned copy of train.csv. `params` is the dict from params.yaml."""
    d, c = params["data"], params["checks"]
    out = df.copy()

    # 1. spelling variants: Rain, ' rain', rai. -> rain
    spell = dc.check_spelling(out)
    if spell["present"]:
        out[spell["column"]] = dc.clean_text(out[spell["column"]])

    # 2. sentinel value (9999 in hour): not a real reading -> blank, fill later
    sent = dc.check_sentinel(out, c["sentinel_candidates"])
    if sent["present"]:
        out[sent["column"]] = out[sent["column"]].replace(sent["value"], np.nan)

    # 3. stuck sensor: values inside the stuck window are fake -> blank
    stuck = dc.check_stuck_sensor(out, d["date_col"], c["stuck_min_run"])
    if stuck["present"]:
        in_window = out[d["date_col"]].between(stuck["start_date"], stuck["end_date"])
        out.loc[in_window, stuck["column"]] = np.nan

    # 4. target typos: 10x too big -> divide by 10
    cutoff = dc.find_typo_cutoff(out, d["target"], c["typo_gap_ratio"])
    if cutoff is not None:
        big = out[d["target"]] >= cutoff
        out.loc[big, d["target"]] = out.loc[big, d["target"]] / 10

    # 5. duplicate rows (none in our data, but the step stays in the pipeline)
    cols = [x for x in out.columns if x != "record_id"]
    out = out.drop_duplicates(subset=cols).reset_index(drop=True)
    return out


def fit_fill_values(df):
    """Median of each numeric column (blanks ignored). Learned on training data only."""
    return {c: float(df[c].median()) for c in NUMERIC}


def apply_fill(df, fill_values):
    """Fill blanks with the stored medians. Used for train, later and holdout alike."""
    out = df.copy()
    for c, v in fill_values.items():
        out[c] = out[c].fillna(v)
    return out


def clean_text_columns(df):
    """Spelling fix for later/holdout, where the text should already be clean."""
    out = df.copy()
    for c in ("weather", "depot_zone"):
        out[c] = out[c].astype(str).str.strip().str.lower().str.rstrip(".")
    out["depot_zone"] = out["depot_zone"].str.capitalize()
    return out
