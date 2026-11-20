"""Run all seven checks and print/save the result. Usage: python -m mp125.run_checks"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from mp125 import data_checks as dc


def main(params_path="params.yaml"):
    p = yaml.safe_load(open(params_path))
    d, c = p["data"], p["checks"]
    train = pd.read_csv(d["raw_train"])
    holdout = pd.read_csv(d["raw_holdout"])
    sha = hashlib.sha256(open(d["raw_train"], "rb").read()).hexdigest()

    res = {
        "data_sha256": sha,
        "leaking_column": dc.check_leaking_column(train, holdout, d["target"]),
        "unit_mixup": dc.check_unit_mixup(train, d["group_col"], d["target"],
                                          c["unit_factors"], c["unit_tolerance"]),
        "sentinel_value": dc.check_sentinel(train, c["sentinel_candidates"]),
        "duplicate_rows": dc.check_duplicates(train),
        "stuck_sensor": dc.check_stuck_sensor(train, d["date_col"], c["stuck_min_run"]),
        "target_typos": dc.check_target_typos(train, d["target"], c["typo_gap_ratio"]),
        "spelling_variants": dc.check_spelling(train),
    }
    print(json.dumps(res, indent=2, default=str))
    Path("reports").mkdir(exist_ok=True)
    json.dump(res, open("reports/data_checks.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
