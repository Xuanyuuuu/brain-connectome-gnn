"""Assemble node inputs: (participants, brain regions, node features)."""
import pickle
from sklearn.preprocessing import StandardScaler
from config import split_indices,data_store, node_strengths_path, node_clusterings_path, y_results, load_graph_features,demography_data_imputed, categorical_values
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

target = y_results["y_gender"]
top_fraction = 0.2

def graph_fraction(graph,top_fraction):
    upper_indices = np.triu_indices(len(graph),k=1)
    upper_graph = graph[upper_indices]
    cutoff = np.quantile(upper_graph,1-top_fraction)
    result = np.where(graph >= cutoff,graph,0)
    return result

with (data_store / "atlas_node_features.pkl").open("rb") as f:
    atlas = pickle.load(f)
with node_strengths_path.open("rb") as f:
    strengths = pickle.load(f)
with node_clusterings_path.open("rb") as f:
    clusterings = pickle.load(f)

train_idx = split_indices["train_idx"]
val_idx = split_indices["val_idx"]
test_idx = split_indices["test_idx"]

atlas_features = atlas["atlas_features"]                 # (200, 74)
node_strengths = strengths["node_strengths"]            # (N, 200)
node_clusterings = clusterings["node_clusterings"]      # (N, 200)
participant_id = strengths["participant_id"]
network_id = atlas["network_id"]
network_name = atlas["network_name"]
compo_id = atlas["compo_id"]
compo_name = atlas["compo_name"]

assert np.array_equal(participant_id, clusterings["participant_id"]), "Participant order differs"
assert np.array_equal(atlas["roi_labels"], np.arange(1, 201)), "Unexpected ROI order"
assert node_strengths.shape == node_clusterings.shape == (len(participant_id), 200)

# Stack the two subject-dependent attributes along the feature axis.
# Values remain unscaled here; training-only scaling is a separate step.
subject_features = np.stack([node_strengths, node_clusterings], axis=-1)

# Every participant shares the same atlas attributes.
shared_atlas = np.broadcast_to(
    atlas_features,
    (len(participant_id), *atlas_features.shape),
)
x = np.concatenate([subject_features, shared_atlas], axis=-1).astype(np.float32)
feature_names = ["strength", "clustering"] + atlas["feature_names"]

assert x.shape == (len(participant_id), 200, len(feature_names))
assert np.isfinite(x).all(), "Non-finite node features"
print(f"Node input x: {x.shape}")

train_values = x[train_idx,:,:2]
train_table = train_values.reshape(len(train_values),200*2)
node_scaler = StandardScaler()
node_scaler.fit(train_table)

all_values = x[:,:,:2]
all_table = all_values.reshape(len(all_values),200*2)
scaled_table = node_scaler.transform(all_table)

x[:,:,:2] = scaled_table.reshape(len(scaled_table),200,2)

demo = demography_data_imputed.reset_index(drop=True)
assert np.array_equal(demo["participant_id"].to_numpy(), participant_id), "Demography order differs"

graph_level = load_graph_features()
region = graph_level["region_features"]
assert np.array_equal(region.index.to_numpy(), participant_id), "Region order differs"
region=region.reset_index(drop=True)                      
modularity = graph_level["graphs_modularities"]["graphs_modularities"]
assert np.array_equal(graph_level["graphs_modularities"]["participant_id"], participant_id)

interhemispheric_cols = ["mean_interhemispheric_fc", "hemisphere_connection_ratio"]
interhemispheric_features = graph_level["interhemispheric_features"].loc[:,interhemispheric_cols]
assert np.array_equal(interhemispheric_features.index.to_numpy(), participant_id), "Interhemispheric order differs"


start_col = demo.columns.get_loc("Basic_Demos_Study_Site")
categorical_cols = [c for c in demo.columns[start_col:] if c != "age_missing"]

demo = pd.concat([demo,region,interhemispheric_features.reset_index(drop=True)],axis=1)
demo['modularity'] = modularity
numeric_cols = [c for c in demo.columns[:start_col] if c != "participant_id"]
numeric_cols += region.columns.tolist() + ['modularity'] + interhemispheric_cols

global_preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_cols),
    ("categorical", OneHotEncoder(
        categories=[categorical_values[c] for c in categorical_cols],
        handle_unknown="ignore",
        sparse_output=False,
    ), categorical_cols),
    ("binary", "passthrough", ["age_missing"]),
])

global_preprocess.fit(demo.iloc[train_idx])
global_x = global_preprocess.transform(demo).astype(np.float32)
assert np.isfinite(global_x).all(), "Non-finite global features"
print(f"Global input: {global_x.shape}")



# Each participant has their own connectivity matrix and Gender label.
with (data_store / "graph_lists.pkl").open("rb") as f:
    graphs = pickle.load(f)


assert len(graphs) == len(x) == len(target), "Participant counts differ"
assert target.isin([0, 1]).all(), "Gender labels must be 0 or 1"

graph_dataset = []
for i in range(len(graphs)):
    # np.abs creates a new array, leaving the saved matrix unchanged.
    adjacency = np.abs(graphs[i])
    assert adjacency.shape == (200, 200) and np.isfinite(adjacency).all()
    assert np.allclose(adjacency, adjacency.T), "Expected a symmetric matrix"

    if top_fraction < 1.0:
        adjacency = graph_fraction(adjacency,top_fraction)

    # A symmetric matrix supplies both directions of each connection.
    source_nodes, target_nodes = np.nonzero(adjacency)
    edge_index = np.stack([source_nodes, target_nodes], axis=0)
    edge_weight = adjacency[source_nodes, target_nodes]

    # PyG expects integer node indices and floating-point features/weights.
    # One float label per graph is suitable for BCEWithLogitsLoss later.
    graph = Data(
        x=torch.tensor(x[i], dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
        global_x=torch.tensor(global_x[i], dtype=torch.float32).unsqueeze(0),
        y=torch.tensor([target.iloc[i]], dtype=torch.float32),
        participant_id=str(participant_id[i]),
        network_id = torch.tensor(network_id,dtype=torch.long),
        compo_id = torch.tensor(compo_id,dtype=torch.long)
    )
    graph_dataset.append(graph)

train_graphs = [graph_dataset[i] for i in train_idx]
val_graphs = [graph_dataset[i] for i in val_idx]
test_graphs = [graph_dataset[i] for i in test_idx]

print(f"Graphs: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
print(f"First graph: {graph_dataset[0]}")
