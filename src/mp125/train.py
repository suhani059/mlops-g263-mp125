"""Stage 2: train v1 with MLflow tracking and register the champion.

Run from the repo root:   python -m mp125.train        (PYTHONPATH=src)
"""
import hashlib
import re
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp125 import cleaning as cl


def build_model(kind, hp, seed):
    """One sklearn Pipeline: one-hot the text columns, then a regressor."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cl.CATEGORICAL)],
        remainder="passthrough",
    )
    if kind == "random_forest":
        reg = RandomForestRegressor(random_state=seed, n_jobs=-1, **hp)
    elif kind == "gradient_boosting":
        reg = GradientBoostingRegressor(random_state=seed, **hp)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("prep", pre), ("model", reg)])


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def prepare_data(params):
    d = params["data"]
    raw = pd.read_csv(d["raw_train"])
    train = cl.clean_train(raw, params)
    fill = cl.fit_fill_values(train)
    train = cl.apply_fill(train, fill)
    later = cl.apply_fill(cl.clean_text_columns(pd.read_csv(d["raw_later"])), fill)
    return train, later, fill


def time_split(df, date_col, frac):
    """Oldest rows train, newest rows validate (the future is never in training)."""
    df = df.sort_values([date_col, "record_id"]).reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def update_findings(path, **values):
    """Edit the tracking: block of findings.yaml in place, keeping comments."""
    lines = Path(path).read_text().splitlines()
    out, in_tracking = [], False
    for line in lines:
        if line.startswith("tracking:"):
            in_tracking = True
        elif line and not line.startswith(" ") and not line.startswith("#"):
            in_tracking = False
        if in_tracking:
            m = re.match(r"^(\s+)(\w+):\s*[^#\n]*?(\s*#.*)?$", line)
            if m and m.group(2) in values:
                v = values[m.group(2)]
                v = f'"{v}"' if isinstance(v, str) else v
                line = f"{m.group(1)}{m.group(2)}: {v}{m.group(3) or ''}"
        out.append(line)
    Path(path).write_text("\n".join(out) + "\n")


def main(params_path="params.yaml"):
    params = yaml.safe_load(open(params_path))
    d, t = params["data"], params["train"]
    target = d["target"]

    train, later, fill = prepare_data(params)
    tr, val = time_split(train, d["date_col"], t["train_fraction"])
    data_sha = sha256_of(d["raw_train"])

    mlflow.set_tracking_uri(t["tracking_uri"])
    mlflow.set_experiment(t["experiment_name"])
    client = MlflowClient()

    results = []
    for cfg in t["settings"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            model = build_model(cfg["kind"], cfg["params"], t["seed"])
            model.fit(tr[cl.FEATURES], tr[target])
            mae_val = mean_absolute_error(val[target], model.predict(val[cl.FEATURES]))
            mae_later = mean_absolute_error(later[target], model.predict(later[cl.FEATURES]))

            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_kind", cfg["kind"])
            mlflow.log_params({f"hp_{k}": v for k, v in cfg["params"].items()})
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("mae_later", mae_later)   # for information only
            mlflow.sklearn.log_model(model, name="model",
                                     serialization_format="cloudpickle")
            results.append((mae_val, run.info.run_id, cfg["name"]))
            print(f"{cfg['name']:<22} mae_val={mae_val:.3f}  mae_later={mae_later:.3f}")

    # the best setting is chosen on validation MAE only (never on later.csv)
    mae_best, best_id, best_name = min(results)
    print(f"\nbest: {best_name}  run_id={best_id}  mae_val={mae_best:.3f}")

    name = t["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_id}/model", name)
    client.set_registered_model_alias(name, "champion", mv.version)

    # refit the champion on all of train so the service has the strongest v1
    best_cfg = next(c for c in t["settings"] if c["name"] == best_name)
    final = build_model(best_cfg["kind"], best_cfg["params"], t["seed"])
    final.fit(train[cl.FEATURES], train[target])
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "fill_values": fill, "version": int(mv.version)},
                t["model_path"])

    update_findings("reports/findings.yaml",
                    experiment_name=t["experiment_name"], best_run_id=best_id,
                    registered_model_name=name, champion_version=int(mv.version))
    print(f"registered {name} v{mv.version} with alias champion; saved {t['model_path']}")


if __name__ == "__main__":
    main()
