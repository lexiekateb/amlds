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