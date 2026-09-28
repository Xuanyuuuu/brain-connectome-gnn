import pickle
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from time import perf_counter
from sklearn.metrics import(
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    log_loss
)

from config import (
    data_store,
    demography_data_imputed,
    load_graph_features,
    split_indices,
    y_results,
    categorical_values,
    seed
)

TARGET = "gender"  # "adhd" or "gender"; also names the output files

train_idx = split_indices["train_idx"]
val_idx = split_indices["val_idx"]
test_idx = split_indices["test_idx"]

y_adhd = y_results["y_adhd"]
y_gender = y_results["y_gender"]
y_target = y_results[f"y_{TARGET}"]


features = load_graph_features()

node_strengths = features["node_strengths"]["node_strengths"]
node_clusterings = features["node_clusterings"]["node_clusterings"]
modularities = features["graphs_modularities"]["graphs_modularities"]
region_features = features["region_features"]

# Keep only the two informative interhemispheric columns: density is constant (~1)
# on full matrices, and the strength variants are rescaled copies of mean_interhemispheric_fc.
interhemispheric_cols = ["mean_interhemispheric_fc", "hemisphere_connection_ratio"]
participant_ids = demography_data_imputed["participant_id"].to_numpy()
interhemispheric_features = features["interhemispheric_features"].loc[
    participant_ids, interhemispheric_cols
]


print(demography_data_imputed.columns)
start_col_idx = demography_data_imputed.columns.get_loc('Basic_Demos_Study_Site')
categorical_cols = demography_data_imputed.columns[start_col_idx:].tolist()
categorical_cols.remove("age_missing")
print(categorical_cols)

categories = [categorical_values[col] for col in categorical_cols]

graph_features = pd.concat ([pd.DataFrame(node_strengths,
                                          columns=[f"nodo{i}_strength" 
                                                   for i in range(node_strengths.shape[1])]),
                            pd.DataFrame(node_clusterings,
                                          columns=[f"nodo{i}_cluster" 
                                                   for i in range(node_clusterings.shape[1])]),
                            pd.DataFrame(modularities,
                                        columns=["modularity"]),
                            region_features.reset_index(drop=True),
                            interhemispheric_features.reset_index(drop=True),
                         ],axis=1)
                             
                             
demographic_numeric_col = [
    col 
    for col in demography_data_imputed.columns[:start_col_idx]
    if col != 'participant_id'
]

numeric_cols = demographic_numeric_col+graph_features.columns.tolist()

binary_cols = ['age_missing']

preprocess = ColumnTransformer([
    ('numeric',StandardScaler(),numeric_cols),
    ('categorical',OneHotEncoder(categories=categories,handle_unknown='ignore'),categorical_cols),
    ('binary','passthrough',binary_cols)
])

x = pd.concat([demography_data_imputed.reset_index(drop=True),
               graph_features],axis=1).drop(columns='participant_id')


x_train = x.iloc[train_idx]
x_val = x.iloc[val_idx]
x_test = x.iloc[test_idx]

y_train = y_target.iloc[train_idx]
y_val = y_target.iloc[val_idx]
y_test = y_target.iloc[test_idx]

# Save the original tables before scaling or one-hot encoding; y_* follow TARGET.
with (data_store / "training_data.pkl").open("wb") as f:
    pickle.dump({
        "target": TARGET,
        "x": x,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_adhd": y_adhd,
        "y_gender": y_gender,
        "participant_id": demography_data_imputed["participant_id"],
        "split_indices": split_indices,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "binary_cols": binary_cols,
        "categorical_values": categorical_values,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

C_values = [1, 0.1, 0.01]
metrics_rows = []
prediction_frames = []

for C in C_values:
    model = make_pipeline(
        clone(preprocess),
        LogisticRegression(C=C, max_iter=2000, random_state=seed, verbose=1),
        verbose=True,
    )
    print(f"Start training: C={C}", flush=True)
    start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start

    for name, X_part, y_part in [
        ("Train_set", x_train, y_train),
        ("Val_set", x_val, y_val),
        ("Test_set", x_test, y_test),  # reported only; C is chosen on Val_set
    ]:
        probability = model.predict_proba(X_part)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = {
            "C": C,
            "split": name,
            "accuracy": accuracy_score(y_part, prediction),
            "log_loss": log_loss(y_part, probability),
            "roc_auc": roc_auc_score(y_part, probability),
            "ap": average_precision_score(y_part, probability),
            "f1": f1_score(y_part, prediction, zero_division=0),
            "fit_seconds": fit_seconds,
            "n_iter": int(model[-1].n_iter_[0]),
        }
        metrics_rows.append(metrics)
        prediction_frames.append(pd.DataFrame({
            "C": C,
            "split": name,
            "participant_id": demography_data_imputed.iloc[X_part.index]["participant_id"].to_numpy(),
            "y_true": y_part.to_numpy(),
            "probability": probability,
            "prediction": prediction,
        }))
        print(
            f"C={C} | {name} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Log loss: {metrics['log_loss']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"AP: {metrics['ap']:.4f} | F1: {metrics['f1']:.4f}",
            flush=True,
        )

    # Save after each completed C; these files summarize the current sweep.
    pd.DataFrame(metrics_rows).to_csv(data_store / f"logistic_{TARGET}_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        data_store / f"logistic_{TARGET}_predictions.csv", index=False,
    )
    print(f"C={C} saved; {fit_seconds:.2f}s, {model[-1].n_iter_[0]} iterations", flush=True)
