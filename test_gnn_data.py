"""Run with: python3 2026_revise/test_gnn_data.py"""
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import GNN_data as GNN


def test_graph_dataset():
    for i in [0, len(GNN.graph_dataset) - 1]:
        graph = GNN.graph_dataset[i]
        expected = np.abs(GNN.graphs[i])
        np.fill_diagonal(expected, 0)
        if GNN.top_fraction < 1.0:
            expected = GNN.graph_fraction(expected, GNN.top_fraction)
        restored = np.zeros_like(expected)
        restored[graph.edge_index[0].numpy(), graph.edge_index[1].numpy()] = graph.edge_weight.numpy()
        assert np.allclose(restored, expected)
        assert np.allclose(restored, restored.T)  # thresholding keeps the graph symmetric

        # Density: top_fraction of the 200*199 directed pairs (within 1%)
        expected_edges = GNN.top_fraction * 200 * 199
        assert abs(graph.edge_index.shape[1] - expected_edges) / expected_edges < 0.01

        # Every kept edge is at least as strong as every dropped edge
        upper = np.triu_indices(200, k=1)
        kept = restored[upper] > 0
        if not kept.all():
            full = np.abs(GNN.graphs[i])[upper]
            assert full[kept].min() >= full[~kept].max()
        assert np.array_equal(graph.x.numpy(), GNN.x[i])
        assert np.array_equal(graph.global_x.numpy(), GNN.global_x[i:i + 1])
        assert graph.y.item() == GNN.target.iloc[i]

    # Global features: one row per participant, no NaN from misaligned concat.
    assert GNN.global_x.shape[0] == len(GNN.participant_id)
    assert np.isfinite(GNN.global_x).all()
    # Scaler was fit on train only, so train numeric columns have mean ~0.
    n_num = len(GNN.numeric_cols)
    assert np.allclose(GNN.global_x[GNN.train_idx, :n_num].mean(axis=0), 0, atol=1e-5)

    for name in ["train", "val", "test"]:
        dataset = getattr(GNN, f"{name}_graphs")
        indices = getattr(GNN, f"{name}_idx")
        assert len(dataset) == len(indices)
        assert all(graph is GNN.graph_dataset[i] for graph, i in zip(dataset, indices))

    batch = next(iter(DataLoader(GNN.train_graphs, batch_size=2, shuffle=False)))
    assert batch.x.shape == (400, 76)
    assert batch.global_x.shape == (2, GNN.global_x.shape[1])  # [1, d] per graph -> [batch, d]
    assert batch.y.shape == (2,)
    source, target = batch.edge_index
    assert torch.equal(batch.batch[source], batch.batch[target])
    assert batch.edge_weight.numel() == batch.edge_index.shape[1]
    assert batch.network_id.shape == (400,)
    assert batch.network_id.min() == 0 and batch.network_id.max() == 16  # no per-graph offset
    print(f"top_fraction={GNN.top_fraction}: graph contents, density, global features, "
          f"splits and two-graph batching passed.")


if __name__ == "__main__":
    test_graph_dataset()
