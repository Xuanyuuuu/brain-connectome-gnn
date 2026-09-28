"""Prepare fixed atlas attributes; no model training is performed here."""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    from config import atlas_labels_path as atlas_path, data_store

    # ROI 1 is matrix node 0, ROI 2 is matrix node 1, and so on.
    atlas = pd.read_csv(atlas_path).sort_values("ROI Label").reset_index(drop=True)
    assert atlas["ROI Label"].tolist() == list(range(1, 201)), "Expected ROI labels 1–200"
    for column in ["Network Name", "Full Component Name"]:
        assert atlas[column].notna().all(), f"Missing labels in {column}"
        atlas[column] = atlas[column].str.strip()
        assert atlas[column].ne("").all(), f"Empty labels in {column}"
    assert np.isfinite(atlas[["R", "A", "S"]].to_numpy()).all(), "Invalid coordinates"
    network_id, network_name = pd.factorize(atlas['Network Name'])
    assert len(network_id)==200 and len(network_name)==17, "Network count wrong"
    assert (network_id>=0).all(), "Missing network labels"
    

    # The same component name may occur in several networks.
    atlas["submodule"] = atlas["Network Name"] + "__" + atlas["Full Component Name"]
    compo_id, compo_name = pd.factorize(atlas['submodule'])

    # This atlas is fixed for everyone, so fit on its 200 ROIs once.
    # Subject-dependent strength/clustering will be processed separately.
    atlas_preprocess = ColumnTransformer([
        ("coordinates", StandardScaler(), ["R", "A", "S"]),
        ("categories", OneHotEncoder(sparse_output=False), ["Network Name", "submodule"]),
    ])
    atlas_features = atlas_preprocess.fit_transform(atlas).astype(np.float32)
    feature_names = atlas_preprocess.get_feature_names_out().tolist()

    # Runnable checks: scaling and exactly one active category in each group.
    network_count = atlas["Network Name"].nunique()
    submodule_count = atlas["submodule"].nunique()
    assert atlas_features.shape == (200, 3 + network_count + submodule_count)
    assert len(feature_names) == atlas_features.shape[1]
    assert np.isfinite(atlas_features).all()
    assert (atlas_features[:, 3:3 + network_count].sum(axis=1) == 1).all()
    assert (atlas_features[:, 3 + network_count:].sum(axis=1) == 1).all()

    decoded = atlas_preprocess.named_transformers_["categories"].inverse_transform(
        atlas_features[:, 3:]
    )
    assert np.array_equal(decoded, atlas[["Network Name", "submodule"]].to_numpy())

    data_store.mkdir(parents=True, exist_ok=True)
    output_path = data_store / "atlas_node_features.pkl"
    with output_path.open("wb") as f:
        pickle.dump({
            "atlas_features": atlas_features,
            "feature_names": feature_names,
            "roi_labels": atlas["ROI Label"].to_numpy(),
            "atlas_preprocess": atlas_preprocess,
            "network_id":network_id,
            "network_name":network_name,
            'compo_id':compo_id,
            'compo_name':compo_name
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Networks: {network_count}; submodules: {submodule_count}")
    print(f"Saved atlas_features {atlas_features.shape} to {output_path}")
    print(f"len(compo_name):{len(compo_name)}")


if __name__ == "__main__":
    main()
