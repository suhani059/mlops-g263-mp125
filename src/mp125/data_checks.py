"""Stage 1: detect the seven kinds of data problem in train.csv.

Every function takes a DataFrame and returns a dict that maps directly onto
one block of findings.yaml. Nothing here is hard-coded to our data: the
answers come from looking at the file.
"""
import numpy as np
import pandas as pd


def check_leaking_column(train, holdout, target, id_cols=("record_id",)):
    """A feature in train that is missing from holdout_features.csv."""
    extra = [c for c in train.columns
             if c not in holdout.columns and c != target and c not in id_cols]
    if not extra:
        return {"present": False, "column": None}
    num = train[extra].select_dtypes("number")
    corr = num.corrwith(train[target]).abs().sort_values(ascending=False)
    return {"present": True, "column": corr.index[0] if len(corr) else extra[0]}


def check_unit_mixup(train, group_col, target, factors, tol=0.15):
    """One group whose mean for a column is ~factor x (or 1/factor of) the others."""
    for col in train.select_dtypes("number").columns:
        if col == target:
            continue
        means = train.groupby(group_col)[col].median()
        for g, m in means.items():
            rest = means.drop(g).median()
            if rest == 0 or m == 0:
                continue
            ratio = max(m / rest, rest / m)
            for f in factors:
                if abs(ratio - f) / f < tol:
                    return {"present": True, "column": col, "group_value": g,
                            "factor": f if m > rest else 1 / f}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel(train, candidates):
    """An impossible special number repeated in a numeric column."""
    for col in train.select_dtypes("number").columns:
        for v in candidates:
            n = int((train[col] == v).sum())
            if n > 0:
                return {"present": True, "column": col, "value": v, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicates(train):
    """Number of EXTRA copies (rows you would delete). record_id is ignored."""
    cols = [c for c in train.columns if c != "record_id"]
    n = int(train.duplicated(subset=cols).sum())
    return {"present": n > 0, "count": n if n else None}


def check_stuck_sensor(train, date_col, min_run=10, skip=()):
    """A column with the same value on every row for a long run of days."""
    ds = train.sort_values([date_col, "record_id"]).reset_index(drop=True)
    best = None
    for col in ds.select_dtypes("number").columns:
        if col in skip:
            continue
        run_id = (ds[col] != ds[col].shift()).cumsum()
        runs = ds.groupby(run_id).agg(n=(col, "size"), start=(date_col, "first"),
                                      end=(date_col, "last"))
        top = runs.sort_values("n", ascending=False).iloc[0]
        if top.n >= min_run and (best is None or top.n > best[1]):
            best = (col, int(top.n), top.start, top.end)
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    return {"present": True, "column": best[0], "start_date": best[2], "end_date": best[3]}


def check_target_typos(train, target, gap_ratio=3.0):
    """Targets typed 10x too big: a small cluster above a large empty gap."""
    s = np.sort(train[target].values)
    top = s[-max(5, len(s) // 5):]                    # look in the top 20%
    ratios = top[1:] / top[:-1]
    i = int(np.argmax(ratios))
    if ratios[i] < gap_ratio:
        return {"present": False, "count": None}
    cutoff = top[i + 1]
    return {"present": True, "count": int((train[target] >= cutoff).sum())}


def check_spelling(train):
    """Text column whose cleaned categories are fewer than its raw categories."""
    for col in train.select_dtypes(exclude="number").columns:
        if col in ("record_id", "date"):
            continue
        raw = train[col].nunique()
        clean = clean_text(train[col]).nunique()
        if raw > clean:
            return {"present": True, "column": col, "true_levels": int(clean)}
    return {"present": False, "column": None, "true_levels": None}


def clean_text(s):
    """Lower-case, strip spaces and dots, then group spellings that share their
    first 3 letters (cle., CLEAR, ' clear' -> clear) and use the commonest form."""
    s = s.str.strip().str.lower().str.rstrip(".")
    key = s.str[:3]
    label = s.groupby(key).agg(lambda x: x.value_counts().idxmax())
    return key.map(label)
