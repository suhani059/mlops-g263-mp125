"""Stage 4: drift, per-group error, retrain v2, promotion rule, holdout predictions.

Run from the repo root (after python -m mp125.train):
    python -m mp125.monitor
"""
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from mp125 import cleaning as cl
from mp125.findings import set_values
from mp125.train import build_model, prepare_data, sha256_of


# ---------------------------------------------------------------- pure functions
def psi(expected, actual, bins=10, eps=1e-4):
    """Population Stability Index of `actual` against `expected`.

    The bin edges come from `expected` (train): `bins` equal-width bins between
    its min and max. The two outer edges are opened to -inf/+inf so a later
    value outside the train range still lands in a bin.
    PSI = sum( (a_i - e_i) * ln(a_i / e_i) ), with a_i, e_i the share of rows
    in bin i. Empty bins get a tiny share (eps) so the logarithm is defined.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.linspace(expected.min(), expected.max(), bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train, later, columns, bins=10):
    return {c: psi(train[c], later[c], bins) for c in columns}


def mae_by_group(model, df, group_col, target, features):
    out = {}
    for g, part in df.groupby(group_col):
        out[g] = float(mean_absolute_error(part[target], model.predict(part[features])))
    return out


def decide(mae_v1, mae_v2, threshold=0.15):
    """Client rule: promote v2 only if its MAE is at least `threshold` (15%) lower."""
    return "promote" if mae_v2 <= (1 - threshold) * mae_v1 else "keep"


# ---------------------------------------------------------------- the pipeline
def raw_numeric_without_blanks(params):
    """Cleaned train.csv with the unreliable readings left blank (not filled).
    PSI is computed on real readings only, so a filled-in median cannot fake a spike."""
    d = params["data"]
    return cl.clean_train(pd.read_csv(d["raw_train"]), params)


def main(params_path="params.yaml"):
    params = yaml.safe_load(open(params_path))
    d, t, m = params["data"], params["train"], params["monitor"]
    target, group = d["target"], d["group_col"]
    name = t["registered_model_name"]

    mlflow.set_tracking_uri(t["tracking_uri"])
    mlflow.set_experiment(t["experiment_name"])
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if len(versions) != 1:
        raise SystemExit(f"Expected exactly 1 registered version, found {len(versions)}. "
                         "To rerun from scratch: delete mlflow.db and mlruns, then run "
                         "python -m mp125.train first.")

    # v1 = the model the service serves today. Keep a frozen copy of it.
    model_path, v1_path = Path(t["model_path"]), Path(m["v1_model_path"])
    if not v1_path.exists():
        bundle = joblib.load(model_path)
        if bundle["version"] != 1:
            raise SystemExit("models/model.joblib is not v1. Run python -m mp125.train first.")
        shutil.copy(model_path, v1_path)
    v1 = joblib.load(v1_path)

    train, later, fill = prepare_data(params)          # filled, ready for the model
    later = later.sort_values([d["date_col"], "record_id"]).reset_index(drop=True)

    # 1. PSI per numeric column: cleaned train vs later
    psi_vals = psi_table(raw_numeric_without_blanks(params), later, cl.NUMERIC, m["psi_bins"])
    most_drifted = max(psi_vals, key=psi_vals.get)

    # 2. v1 MAE per depot_zone on all of later.csv
    by_group = mae_by_group(v1["model"], later, group, target, cl.FEATURES)
    worst = max(by_group, key=by_group.get)

    # 3. v2 = same setting as v1, trained on train + first 70% of later
    cut = int(len(later) * m["retrain_fraction"])
    later_fit, later_check = later.iloc[:cut], later.iloc[cut:]
    v1_run = client.get_run(client.get_model_version_by_alias(name, "champion").run_id)
    cfg = next(c for c in t["settings"] if c["name"] == v1_run.data.tags["mlflow.runName"])
    fit = pd.concat([train, later_fit], ignore_index=True)
    v2_model = build_model(cfg["kind"], cfg["params"], t["seed"])
    v2_model.fit(fit[cl.FEATURES], fit[target])

    mae_v1 = float(mean_absolute_error(later_check[target], v1["model"].predict(later_check[cl.FEATURES])))
    mae_v2 = float(mean_absolute_error(later_check[target], v2_model.predict(later_check[cl.FEATURES])))
    decision = decide(mae_v1, mae_v2, m["promotion_threshold"])

    # 4. log v2 and register it as version 2 (always); champion moves only if promoted
    with mlflow.start_run(run_name=f"{cfg['name']}_v2_retrain") as run:
        mlflow.log_param("data_sha256", sha256_of(d["raw_train"]))
        mlflow.log_param("model_kind", cfg["kind"])
        mlflow.log_param("trained_on", f"train + first {cut} rows of later")
        mlflow.log_params({f"hp_{k}": v for k, v in cfg["params"].items()})
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.sklearn.log_model(v2_model, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id
    mv2 = mlflow.register_model(f"runs:/{run_id}/model", name)
    if decision == "promote":
        client.set_registered_model_alias(name, "champion", mv2.version)
        joblib.dump({"model": v2_model, "fill_values": fill, "version": int(mv2.version)}, model_path)
    champion = joblib.load(model_path)                 # v1 if kept, v2 if promoted

    # 5. holdout predictions with the champion
    hold = pd.read_csv(d["raw_holdout"])
    hold_x = cl.apply_fill(cl.clean_text_columns(hold), champion["fill_values"])
    Path("predictions").mkdir(exist_ok=True)
    pd.DataFrame({"record_id": hold["record_id"],
                  "prediction": champion["model"].predict(hold_x[cl.FEATURES])}
                 ).to_csv("predictions/holdout_predictions.csv", index=False)

    # 6. findings.yaml
    updates = {("monitoring", "psi", c): round(v, 4) for c, v in psi_vals.items()}
    updates.update({("monitoring", "most_drifted_feature"): most_drifted,
                    ("monitoring", "worst_served_group"): worst,
                    ("monitoring", "mae_v1_on_check"): round(mae_v1, 4),
                    ("monitoring", "mae_v2_on_check"): round(mae_v2, 4),
                    ("monitoring", "decision"): decision})
    updates.update({("monitoring", "mae_by_group", g): round(v, 4) for g, v in by_group.items()})
    updates[("tracking", "champion_version")] = int(champion["version"])
    missing = set_values("reports/findings.yaml", updates)
    assert not missing, f"keys not found in findings.yaml: {missing}"

    result = {"psi": psi_vals, "most_drifted_feature": most_drifted,
              "mae_by_group": by_group, "worst_served_group": worst,
              "check_rows": len(later_check), "mae_v1_on_check": mae_v1,
              "mae_v2_on_check": mae_v2, "improvement": 1 - mae_v2 / mae_v1,
              "decision": decision, "champion_version": int(champion["version"])}
    Path("reports").mkdir(exist_ok=True)
    json.dump(result, open("reports/monitoring.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
