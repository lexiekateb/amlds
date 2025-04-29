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
        self.transition_matrix = self.adj_matrix / row_sums[:, np.newaxis]
        
    def initialize_opinions_60_40_split(self, positive_value=0.8, negative_value=-0.8, positive_ratio=0.6):
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
        
    def initialize_opinions(self, initial_opinions=None):
        if initial_opinions is not None:
            if len(initial_opinions) != self.n:
                raise ValueError(f"error with opinion vector")
            self.opinions = np.array(initial_opinions)
        else:
            # random init from -1 to 1 if no vector given
            self.opinions = np.random.uniform(-1, 1, self.n)
        
        self.opinion_history = [self.opinions.copy()]
        self.time_steps = 0
        
    def update(self):
        self.opinions = self.transition_matrix @ self.opinions
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
        
        return np.mean(local_agreements)
    
    def plot_opinion_evolution(self):
        opinions = np.array(self.opinion_history)
        
        plt.figure(figsize=(10, 6))
        for i in range(self.n):
            plt.plot(range(self.time_steps + 1), opinions[:, i], alpha=0.5)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Opinion')
        plt.title('Evolution of Opinions in DeGroot Model')
        plt.grid(True)
        plt.show()
    
    def plot_polarization_metrics(self):
        variance = []
        std_dev = []
        opinion_range = []
        local_agreement = []
        
        for t in range(len(self.opinion_history)):
            current_opinions = self.opinions

            self.opinions = self.opinion_history[t]
            variance.append(self.compute_polarization_variance())
            std_dev.append(self.compute_polarization_std())
            opinion_range.append(self.compute_polarization_range())
            local_agreement.append(self.compute_local_agreement())
            
            # restore current opinions
            self.opinions = current_opinions

        plt.figure(figsize=(12, 10))        
        plt.subplot(4, 1, 1)
        plt.plot(range(self.time_steps + 1), variance)
        plt.title('Opinion Variance Over Time')
        plt.ylabel('Variance')
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(range(self.time_steps + 1), std_dev)
        plt.title('Opinion Standard Deviation Over Time')
        plt.ylabel('Std. Dev.')
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(range(self.time_steps + 1), opinion_range)
        plt.title('Opinion Range Over Time')
        plt.ylabel('Range')
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(range(self.time_steps + 1), local_agreement)
        plt.title('Average Local Agreement Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Local Agreement')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()