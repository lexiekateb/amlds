import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

class DeGrootModel:
    def __init__(self, graph):
        # edge weights represent influence
        self.G = graph
        self.n = graph.number_of_nodes()
        
        # graph to matrix
        self.adj_matrix = nx.to_numpy_array(graph, nodelist=range(self.n), weight="weight")
        
        # normalize
        row_sums = self.adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        self.normalized_adj_matrix = self.adj_matrix / row_sums[:, np.newaxis] # degree normalized adjacency matrix
        
    def initialize_opinions(self, positive_value=0.8, negative_value=-0.8, positive_ratio=0.5):
        n_positive = int(self.n * positive_ratio)
        n_negative = self.n - n_positive
        
        # opinion vector
        opinions = np.zeros(self.n)
        opinions[:n_positive] = positive_value
        opinions[n_positive:] = negative_value
        
        # randomize assignments; not clustered initially
        np.random.shuffle(opinions)
        
        self.opinions = opinions
        self.opinion_history = [self.opinions.copy()]
        self.time_steps = 0
        
        return self.opinions
    
    def initialize_opinions_manual(self, initial_opinions, proportions):
        if len(initial_opinions) != len(proportions):
            raise ValueError(f"opinion and proportion length  must match")
        if abs(sum(proportions) != 1):
            raise ValueError(f"proportions must sum to 1")
        
        # num nodes per opinion 
        counts = []
        remaining = self.n
        
        for i in range(len(proportions)):
            count = int(self.n * proportions[i])
            counts.append(count)
            remaining -= count
        
        # opinion vector
        opinions = np.zeros(self.n)
        
        # assign opinions to each group
        start = 0
        for i, count in enumerate(counts):
            end = start + count
            opinions[start:end] = initial_opinions[i]
            start = end
        
        # randomize assignments; not clustered initially
        np.random.shuffle(opinions)

        # store opinions
        self.opinions = opinions
        self.opinion_history = [self.opinions.copy()]
        self.time_steps = 0
        
        return self.opinions
        
    def initialize_opinions_manual(self, initial_opinions, proportions):
        if len(initial_opinions) != len(proportions):
            raise ValueError(f"opinion and proportion length  must match")
        if abs(sum(proportions) != 1):
            raise ValueError(f"proportions must sum to 1")
        
        # num nodes per opinion 
        counts = []
        remaining = self.n
        
        for i in range(len(proportions)):
            count = int(self.n * proportions[i])
            counts.append(count)
            remaining -= count
        
        # opinion vector
        opinions = np.zeros(self.n)
        
        # assign opinions to each group
        start = 0
        for i, count in enumerate(counts):
            end = start + count
            opinions[start:end] = initial_opinions[i]
            start = end
        
        # randomize assignments; not clustered initially
        np.random.shuffle(opinions)

        # store opinions
        self.opinions = opinions
        self.opinion_history = [self.opinions.copy()]
        self.time_steps = 0
        
        return self.opinions

    def update(self):
        self.opinions = self.normalized_adj_matrix @ self.opinions
        self.opinions = np.clip(self.opinions, -1.0, 1.0)

        self.opinion_history.append(self.opinions.copy())
        self.time_steps += 1

    def run(self, steps):
        for _ in range(steps):
            self.update()
        return self.opinions
    
    def compute_polarization_variance(self):
        return np.var(self.opinions)
    
    def compute_polarization_std(self):
        return np.std(self.opinions)
    
    def compute_polarization_range(self):
        return np.max(self.opinions) - np.min(self.opinions)

    def compute_local_agreement(self):
        signs = np.sign(self.opinions - np.mean(self.opinions))
        local_agreements = []
        
        for i in range(self.n):
            neighbors = list(self.G.neighbors(i))
            if len(neighbors) == 0:
                local_agreements.append(0.5)  # if no neighbors
            else:
                agreement_count = sum(signs[i] == signs[j] for j in neighbors)
                local_agreements.append(agreement_count / len(neighbors))
        
        local_agreements = np.array(local_agreements)
        return np.mean(local_agreements), np.var(local_agreements)