from pathlib import Path
import pickle

seed = 42
root_folder = Path(__file__).resolve().parent
raw_data_file = Path(__file__).resolve().parent.parent / "widsdatathon2025_new/TRAIN_NEW"
atlas_labels_path = raw_data_file.parent / "Schaefer200_merged_labels.csv"
data_store = Path(__file__).resolve().parent/"dataStorage"
data_store.mkdir(parents=True, exist_ok=True)

graph_lists_path = data_store / 'graph_lists.pkl'
node_strengths_path = data_store / 'nodes_strengths'
node_clusterings_path = data_store / 'nodes_clusterings'
graphs_modularities_path = data_store / 'graphs_modularities'
region_features_path = data_store / 'region_connectivity_features.csv'
interhemispheric_features_path = data_store / 'interhemispheric_features.csv'
training_data_path = data_store / 'training_data.pkl'


def load_training_data():
    """Read original logistic-regression tables without running training."""
    with training_data_path.open('rb') as f:
        return pickle.load(f)


def load_graph_features():
    """Load generated features; pickle dictionaries retain IDs and weight_mode."""
    import pandas as pd

    with node_strengths_path.open('rb') as f:
        strengths = pickle.load(f)
    with node_clusterings_path.open('rb') as f:
        clusterings = pickle.load(f)
    with graphs_modularities_path.open('rb') as f:
        modularities = pickle.load(f)
    return {
        'node_strengths': strengths,
        'node_clusterings': clusterings,
        'graphs_modularities': modularities,
        'region_features': pd.read_csv(region_features_path, index_col='participant_id'),
        'interhemispheric_features': pd.read_csv(
            interhemispheric_features_path, index_col='participant_id'
        ),
    }

required_files = [
    'demography_data_imputed.pkl',
    'fMRI_Connectome_data.pkl',
    'split_indices.pkl',
    'y_results.pkl',
    'categorical_values.pkl'
]

if not all ((data_store/name).exists() for name in required_files):
    import data_access
    data_access.main()


with open(data_store/'demography_data_imputed.pkl','rb') as f:
    demography_imputed = pickle.load(f)
    demography_data_imputed = demography_imputed['demography_data_imputed']
    imputer_age= demography_imputed['imputer_age']

with open(data_store/'fMRI_Connectome_data.pkl','rb') as f:
    fMRI_Connectome = pickle.load(f)['fMRI_Connectome']

with open(data_store/'split_indices.pkl','rb') as f:
    split_indices = pickle.load(f)


with open(data_store/'y_results.pkl','rb') as f:
    y_results = pickle.load(f)

with open(data_store / "categorical_values.pkl", "rb") as f:
    categorical_values = pickle.load(f)
    
