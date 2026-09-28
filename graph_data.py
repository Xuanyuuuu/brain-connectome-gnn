import numpy as np
import pickle
import pandas as pd
import networkx as nx
from time import perf_counter

def list_to_matrix(fmri_list):
    matrix = np.zeros((200,200))
    indi = 0
    for i in range(200):
        for j in range(i+1,200):
            matrix[i,j] = fmri_list[indi]
            matrix[j,i] = matrix[i,j]
            indi = indi + 1 
    return matrix

def calculate_interhemispheric_connectivity(adj_matrix, left_nodes, right_nodes):
    """
    Graph-level connectivity calculation
    
    """
    left_left = adj_matrix[np.ix_(left_nodes, left_nodes)]
    right_right = adj_matrix[np.ix_(right_nodes, right_nodes)]
    left_right = adj_matrix[np.ix_(left_nodes, right_nodes)]
    
    left_left_strength = np.sum(np.abs(np.triu(left_left, k=1)))
    right_right_strength = np.sum(np.abs(np.triu(right_right, k=1)))
    left_right_strength = np.sum(np.abs(left_right))
    
    max_possible_lr_connections = len(left_nodes) * len(right_nodes)
    interhemispheric_density = np.count_nonzero(left_right) / max_possible_lr_connections if max_possible_lr_connections else np.nan
    mean_interhemispheric_fc = np.mean(np.abs(left_right)) if left_right.size else np.nan
    
    norm_interhemispheric_strength = mean_interhemispheric_fc
    
    within_strength = left_left_strength + right_right_strength
    hemisphere_connection_ratio = left_right_strength / within_strength if within_strength else np.nan
    
    return {
        "interhemispheric_density": interhemispheric_density,
        "interhemispheric_strength": left_right_strength,
        "normalized_interhemispheric_strength": norm_interhemispheric_strength,
        "hemisphere_connection_ratio": hemisphere_connection_ratio,
        "mean_interhemispheric_fc": mean_interhemispheric_fc
    }

def calculate_region_connectivity(nodes, adj_matrix):
    """
    region-level connectivity calculation
    
    """
    nodes = np.array(nodes,dtype=int)
    left_nodes = nodes[nodes < 100]
    right_nodes = nodes[nodes >= 100]
    LL_region = adj_matrix[np.ix_(left_nodes,left_nodes)]
    RR_region = adj_matrix[np.ix_(right_nodes,right_nodes)]
    LR_region = adj_matrix[np.ix_(left_nodes,right_nodes)]

    LL_edges = LL_region[np.triu_indices(len(left_nodes),k=1)]
    RR_edges = RR_region[np.triu_indices(len(right_nodes),k=1)]

    mean_LL = np.abs(LL_edges).mean() if LL_edges.size else np.nan
    mean_RR = np.abs(RR_edges).mean() if RR_edges.size else np.nan
    mean_LR = np.abs(LR_region).mean() if LR_region.size else np.nan

    sum_LL = np.abs(LL_edges).sum()
    sum_RR = np.abs(RR_edges).sum()
    sum_LR = np.abs(LR_region).sum()
    
    return {
        "mean_LL": mean_LL,
        "mean_RR": mean_RR,
        "mean_LR": mean_LR,
        "difference_L_minus_R": mean_LL - mean_RR,

        "sum_LL": sum_LL,
        "sum_RR": sum_RR,
        "sum_LR": sum_LR,
        "sum_difference_L_minus_R": sum_LL - sum_RR,
        }


def save_interhemispheric_features(graphs, id_ref, schaefer_tb, data_store):
    """Whole-brain left-vs-right connectivity for every participant -> (N, 5) CSV."""
    # Schaefer ordering: ROI 1-100 are left hemisphere, 101-200 are right.
    assert schaefer_tb.loc[:99, 'ROI Name'].str.contains('_LH_').all()
    assert schaefer_tb.loc[100:, 'ROI Name'].str.contains('_RH_').all()
    left_nodes = np.arange(100)
    right_nodes = np.arange(100, 200)

    rows = [
        calculate_interhemispheric_connectivity(adj_matrix, left_nodes, right_nodes)
        for adj_matrix in graphs
    ]
    hemisphere_features = pd.DataFrame(rows, index=pd.Index(id_ref, name='participant_id'))

    assert hemisphere_features.shape == (len(graphs), 5)
    assert not hemisphere_features.isna().any().any(), 'NaN in interhemispheric features'
    hemisphere_features.to_csv(data_store / 'interhemispheric_features.csv')
    print(f'Saved interhemispheric features: {hemisphere_features.shape}', flush=True)


def main():
    from config import data_store, fMRI_Connectome, atlas_labels_path

    id_ref = fMRI_Connectome['participant_id']
    ## save fMRI data as graphs
    if (data_store/"graph_lists.pkl").exists():
        with open(data_store/"graph_lists.pkl",'rb') as f:
            graphs = pickle.load(f)
    else:
        graphs = []
        for i in range(fMRI_Connectome.shape[0]):
            current_list = fMRI_Connectome.iloc[i,1:].to_numpy(dtype=float)
            single_graph = list_to_matrix(current_list)
            # single_graph[single_graph<0] = 0
            graphs.append(single_graph)

        with open(data_store/"graph_lists.pkl",'wb') as f:
            pickle.dump(graphs,f)

    # Regional connectivity information
    schaefer_tb = pd.read_csv(atlas_labels_path)
    # Keep DataFrame indices aligned with 0-based matrix indices, even if rows were reordered.
    schaefer_tb = schaefer_tb.sort_values('ROI Label').reset_index(drop=True)
    network_name = schaefer_tb['Network Name'].str.strip()
    schaefer_tb['Network Name'] = network_name

    network_group = {}
    for name,group in schaefer_tb.groupby('Network Name'):
        group_indices = group.index.to_numpy()
        network_group[name] = group_indices

    if len(graphs) != len(id_ref):
        raise ValueError('Graph count does not match participant IDs.')
    if sorted(schaefer_tb['ROI Label'].tolist()) != list(range(1, 201)):
        raise ValueError('Expected unique ROI labels from 1 to 200.')
    for i, graph in enumerate(graphs):
        if graph.shape != (200, 200) or not np.isfinite(graph).all() or not np.allclose(graph, graph.T):
            raise ValueError(f'Graph {i} must be a finite, symmetric 200x200 matrix; check graph_lists.pkl.')

    region_columns = {}
    for name, nodes in network_group.items():
        metrics = pd.DataFrame([
            calculate_region_connectivity(nodes, adj_matrix)
            for adj_matrix in graphs
        ], index=pd.Index(id_ref, name='participant_id'))
        for metric in metrics.columns:
            region_columns[f'{name}__{metric}'] = metrics[metric]

    region_features = pd.DataFrame(region_columns)
    assert region_features.index.equals(pd.Index(id_ref)), "ID not equal in region_feature!"
    assert not region_features.isna().any().any(), 'NAN in region features'
    region_features.to_csv(data_store / 'region_connectivity_features.csv')
    print(f'Saved region connectivity features: {region_features.shape}', flush=True)

    save_interhemispheric_features(graphs, id_ref, schaefer_tb, data_store)


    ## node strength + clustering
    node_strengths = []
    node_clusterings = []
    graphs_modularities = []
    start = perf_counter()
    for i, graph in enumerate(graphs, start=1):
        # Absolute FC measures association magnitude, not positive synchrony.
        A = np.abs(graph)
        np.fill_diagonal(A, 0)
        print(f'[{i}/{len(graphs)}] Computing weighted clustering...', flush=True)
        node_strength = A.sum(axis=1)
        node_strengths.append(node_strength)
        G = nx.from_numpy_array(A)
        node_clustering = nx.clustering(G, weight='weight')
        node_clusterings.append([node_clustering[node] for node in range(200)])
        Q = nx.community.modularity(G, network_group.values(), weight='weight') if A.any() else np.nan
        graphs_modularities.append(Q)
        elapsed = perf_counter() - start
        remaining = elapsed / i * (len(graphs) - i)
        print(f'[{i}/{len(graphs)}] Done; elapsed {elapsed:.1f}s; estimated remaining {remaining / 60:.1f} min', flush=True)

    node_strengths = np.array(node_strengths)
    print(node_strengths.shape)
    with open(data_store/"nodes_strengths", 'wb') as f:
        pickle.dump ({
            'node_strengths':node_strengths,
            'participant_id':id_ref.to_numpy(),
            'weight_mode':'absolute',
        },f)

    node_clusterings = np.array(node_clusterings)
    print(node_clusterings.shape)
    with open(data_store/"nodes_clusterings", 'wb') as f:
        pickle.dump ({
            'node_clusterings':node_clusterings,
            'participant_id':id_ref.to_numpy(),
            'weight_mode':'absolute',
        },f)

    graphs_modularities = np.array(graphs_modularities)
    print(graphs_modularities.shape)
    with open(data_store/"graphs_modularities", 'wb') as f:
        pickle.dump ({
            'graphs_modularities':graphs_modularities,
            'participant_id':id_ref.to_numpy(),
            'weight_mode':'absolute',
        },f)



if __name__ == '__main__':
    main()
