import numpy as np
import pandas as pd

from mp125 import monitor as mon
from mp125.findings import set_values


def test_psi_same_distribution_is_near_zero():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    assert mon.psi(a, b) < 0.05


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(3, 1, 2000)
    assert mon.psi(a, b) > 1.0


def test_psi_survives_values_outside_train_range_and_blanks():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    later = np.array([50.0, 60.0, 70.0])
    assert np.isfinite(mon.psi(train, later))


def test_most_drifted_column_is_found():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    later = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(4, 1, 500)})
    table = mon.psi_table(train, later, ["a", "b"])
    assert max(table, key=table.get) == "b"


def test_promotion_rule_needs_15_percent():
    assert mon.decide(10.0, 8.4) == "promote"      # 16% lower
    assert mon.decide(10.0, 8.5) == "promote"      # exactly 15% lower
    assert mon.decide(10.0, 8.6) == "keep"         # only 14% lower
    assert mon.decide(10.0, 11.0) == "keep"        # worse


def test_mae_by_group():
    class Const:
        def predict(self, X):
            return np.zeros(len(X))
    df = pd.DataFrame({"g": ["N", "N", "S"], "y": [2.0, 4.0, 5.0], "x": [0, 0, 0]})
    out = mon.mae_by_group(Const(), df, "g", "y", ["x"])
    assert out == {"N": 3.0, "S": 5.0}


def test_set_values_keeps_comments_and_quoted_keys(tmp_path):
    f = tmp_path / "f.yaml"
    f.write_text("monitoring:\n  psi:            # note\n    stops: null\n"
                 "  mae_by_group:\n    'North': null\n  decision: null   # promote or keep\n")
    missing = set_values(f, {("monitoring", "psi", "stops"): 1.5,
                             ("monitoring", "mae_by_group", "North"): 2.5,
                             ("monitoring", "decision"): "keep"})
    text = f.read_text()
    assert missing == []
    assert "stops: 1.5" in text and "'North': 2.5" in text
    assert 'decision: "keep"' in text and "# promote or keep" in text and "# note" in text
