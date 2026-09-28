"""After training, run: python3 test_logistic_results.py [gender|adhd]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def test_saved_results(target="gender"):
    folder = Path(__file__).resolve().parent / "dataStorage"
    metrics = pd.read_csv(folder / f"logistic_{target}_metrics.csv")
    predictions = pd.read_csv(folder / f"logistic_{target}_predictions.csv")
    assert not metrics.duplicated(["C", "split"]).any()
    assert set(metrics["C"]) == {1, 0.1, 0.01}
    assert set(metrics["split"]) == {"Train_set", "Val_set", "Test_set"}
    assert len(metrics) == 9
    assert not predictions.duplicated(["C", "split", "participant_id"]).any()
    for row in metrics.itertuples():
        saved = predictions.loc[
            (predictions["C"] == row.C) & (predictions["split"] == row.split)
        ]
        assert len(saved) > 0
        assert saved["probability"].between(0, 1).all()
        assert np.array_equal(saved["prediction"], saved["probability"] >= 0.5)
        assert np.isclose(row.accuracy, accuracy_score(saved.y_true, saved.prediction))
        assert np.isclose(row.f1, f1_score(saved.y_true, saved.prediction))
    print("Saved metrics match predictions for all three C values.")


if __name__ == "__main__":
    test_saved_results(sys.argv[1] if len(sys.argv) > 1 else "gender")
