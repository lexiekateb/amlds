import numpy as np
import networkx as nx
import random

def add_random_edges(G, n_random):
    nodes = list(G.nodes())
    n = len(nodes)
    
    added = 0
    while added < n_random:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1

def assign_edge_weights(G, method='uniform', seed=42):
    np.random.seed(seed)
    
    if method == 'uniform':         # truly random assignment of edge weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.5)
    elif method == 'engagement':    # based on node's engagement
        for u, v in G.edges():
            v_u = np.random.randint(100, 10000)    # simulate views
            v_v = np.random.randint(100, 10000)    # simulate views
            raw = (v_u + v_v) / 20000.0
            G[u][v]['weight'] = max(raw, 0.01) # avoid insanely small weights

def create_sbm_graph(sizes, p_intra, p_inter, seed=42):

    n_communities = len(sizes)
    p = [[p_inter for _ in range(n_communities)] for _ in range(n_communities)]
    
    # set intra probabilities
    for i in range(n_communities):
        p[i][i] = p_intra

    G = nx.stochastic_block_model(sizes, p, seed=seed)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    
    return G

def create_influencer_graph(n_nodes=1000, bucket_counts=None, bucket_edges=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    num_buckets = len(bucket_counts)

    if len(bucket_edges) != num_buckets:
        raise ValueError(f"bucket_edges must have the same length as bucket_counts: got {len(bucket_edges)} and {num_buckets}")   
    
    if sum(bucket_counts) != n_nodes:
        raise ValueError(f"bucket_counts must sum to n_nodes: got {sum(bucket_counts)} and {n_nodes}")

    G = nx.Graph()
    node_id = 0

    for bucket_idx, (count, edge_count) in enumerate(zip(bucket_counts, bucket_edges)):
        
        influence_level = 1 - (bucket_idx / (num_buckets - 1))
        
        for _ in range(count):
            G.add_node(node_id, 
                       bucket_index=bucket_idx, 
                       influence=influence_level,
                       target_edges=edge_count)
            node_id += 1

    for node in G.nodes():
        edge_count = G.nodes[node]['target_edges']
        potential_connections = list(set(G.nodes()) - {node} - set(G.neighbors(node)))
        edge_count = min(edge_count, len(potential_connections))
        
        connections = random.sample(potential_connections, edge_count)
        for target in connections:
            G.add_edge(node, target)

    return G


def create_random_geometric_graph(n, radius, dim=2, seed=42, remove_self_loops=True):
    G = nx.random_geometric_graph(n, radius, dim=dim, seed=seed)
    G.remove_nodes_from(list(nx.isolates(G)))
    if remove_self_loops:
        G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)
    
    return G