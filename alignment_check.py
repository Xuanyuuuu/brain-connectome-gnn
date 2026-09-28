import pickle
import numpy as np
import pandas as pd
from graph_data import list_to_matrix

from config import (
    data_store,
    fMRI_Connectome,
    demography_data_imputed,
    seed
)

ref_ids = pd.Index(
    fMRI_Connectome["participant_id"],
    name="participant_id",
)


assert ref_ids.is_unique and not ref_ids.hasnans

def align(df, name):
    assert df.index.is_unique, f"{name}: repeat participant_id"
    assert not df.index.hasnans, f"{name}: missing participant_id"

    missing = ref_ids.difference(df.index)
    extra = df.index.difference(ref_ids)
    assert missing.empty and extra.empty, (
        f"{name}: Missing {len(missing)} ，Extra {len(extra)} "
    )
    assert df.index.equals(ref_ids), f"{name}: participant_id or order misaligned"
    return df

demo = align(
    demography_data_imputed.set_index("participant_id"),
    "demography",
)

egion = align(
    pd.read_csv(
        data_store / "region_connectivity_features.csv",
        index_col="participant_id",
    ),
    "region",
)


with open(data_store / 'graph_lists.pkl', 'rb') as f:
    graphs = pickle.load(f)

test_row = fMRI_Connectome.iloc[seed,1:].to_numpy(dtype=float)
test_indi = list_to_matrix(test_row)
target_indi = graphs[seed]

assert np.array_equal(test_indi, target_indi), "Two sample graphs are not equal"
print('Alignment checks passed.')

def load_feature(filename, key):
    with (data_store / filename).open("rb") as f:
        saved = pickle.load(f)

    values = np.asarray(saved[key])
    assert not np.isnan(values).any(), "NaNs in Net features"


strength = load_feature("nodes_strengths", "node_strengths")
clustering = load_feature("nodes_clusterings", "node_clusterings")
modularity = load_feature("graphs_modularities", "graphs_modularities")