"""Run with: python3 2026_revise/test_graph_data.py"""
import pickle
import runpy
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
import networkx as nx


def test_region_features():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_store = root / 'dataStorage'
        data_store.mkdir()
        # Shuffled rows and whitespace must not change ROI membership.
        labels = pd.DataFrame({'ROI Label': range(1, 201), 'Network Name': 'other'})
        labels['ROI Name'] = ['_LH_'] * 100 + ['_RH_'] * 100
        labels.loc[[2, 8, 103, 110], 'Network Name'] = 'target '
        labels.iloc[::-1].to_csv(root / 'Schaefer200_merged_labels.csv', index=False)
        matrix = np.eye(200)
        matrix[2, 8] = matrix[8, 2] = -0.6
        matrix[103, 110] = matrix[110, 103] = 0.2
        matrix[np.ix_([2, 8], [103, 110])] = 0.3
        matrix[np.ix_([103, 110], [2, 8])] = 0.3
        with (data_store / 'graph_lists.pkl').open('wb') as f:
            pickle.dump([matrix, matrix * 2], f)

        config = ModuleType('config')
        config.seed = 42
        config.data_store = data_store
        config.atlas_labels_path = root / 'Schaefer200_merged_labels.csv'
        config.split_indices = dict.fromkeys(
            ['train_idx', 'test_idx', 'val_idx', 'train_id', 'test_id', 'val_id'], []
        )
        config.fMRI_Connectome = pd.DataFrame({'participant_id': ['second', 'first']})
        config.demography_data_imputed = None
        config.y_results = {'y_adhd': [], 'y_gender': []}
        # Importing helpers must neither load configuration nor compute features.
        with patch.dict(sys.modules, {'config': None}), patch.object(
            nx, 'clustering', side_effect=AssertionError('Clustering ran on import')
        ):
            imported = runpy.run_path(str(Path(__file__).with_name('graph_data.py')))
        assert callable(imported['list_to_matrix'])
        assert not (data_store / 'region_connectivity_features.csv').exists()
        with patch.dict(sys.modules, {'config': config}):
            result = runpy.run_path(
                str(Path(__file__).with_name('graph_data.py')), run_name='__main__'
            )

        saved = pd.read_csv(data_store / 'region_connectivity_features.csv', index_col=0)
        assert saved.index.tolist() == ['second', 'first']
        columns = ['target__' + key for key in
                   ['mean_LL', 'mean_RR', 'mean_LR', 'difference_L_minus_R']]
        assert np.allclose(saved[columns], [[0.6, 0.2, 0.3, 0.4], [1.2, 0.4, 0.6, 0.8]])
        missing = result['calculate_region_connectivity']([2], matrix)
        assert all(np.isnan(missing[key]) for key in
                   ['mean_LL', 'mean_RR', 'mean_LR', 'difference_L_minus_R'])
        with (data_store / 'nodes_clusterings').open('rb') as f:
            clustering = pickle.load(f)
        assert clustering['node_clusterings'].shape == (2, 200)
        assert clustering['node_clusterings'].dtype != object
        assert clustering['participant_id'].tolist() == ['second', 'first']
        assert np.allclose(clustering['node_clusterings'][0], clustering['node_clusterings'][1])
        with (data_store / 'nodes_strengths').open('rb') as f:
            strength = pickle.load(f)['node_strengths']
        assert np.allclose(strength[:, 2], [1.2, 2.4])
        with (data_store / 'graphs_modularities').open('rb') as f:
            modularity = pickle.load(f)['graphs_modularities']
        assert modularity.shape == (2,)
        assert np.isfinite(modularity).all()
        # Absolute weights form one isolated module; its fixed-partition Q is zero.
        assert np.allclose(modularity, 0)
        toy = nx.Graph()
        toy.add_weighted_edges_from([(0, 1, 1), (2, 3, 1)])
        assert np.isclose(nx.community.modularity(toy, [{0, 1}, {2, 3}]), 0.5)
    print('Region feature checks passed.')


if __name__ == '__main__':
    test_region_features()
