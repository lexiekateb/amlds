import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    
class DeGrootModel:
    def __init__(self, graph):
        # graph: NetworkX graph with edge weights representing influence
        self.G = graph
        self.n = graph.number_of_nodes()
        
        # graph to matrix
        self.adj_matrix = nx.to_numpy_array(graph, nodelist=range(self.n), weight="weight")
        
        # normalize
        row_sums = self.adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        self.transition_matrix = self.adj_matrix / row_sums[:, np.newaxis]
        
    def initialize_opinions_60_40_split(self, positive_value=0.8, negative_value=-0.8, positive_ratio=0.6):
        """
            positive_value: positive opinion val
            negative_value: negative opinion val
            positive_ratio: ratio of positive to neg users
        """
        n_positive = int(self.n * positive_ratio)
        n_negative = self.n - n_positive
        
        #opinion vector
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
        """
            initial_opinions: initial opinion vector
        """
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


class DeGrootThresholdModel(DeGrootModel):
    def __init__(self, graph, local_agreement_threshold=0.5):
        """
            graph: netx graph w weights
            local_agreement_threshold: threshold for when user posts
        """
        super().__init__(graph)
        self.threshold = local_agreement_threshold
        
    def initialize_opinions(self, initial_opinions=None):
        super().initialize_opinions(initial_opinions)
        
        # posting history
        self.posting_status = np.zeros(self.n, dtype=bool)
        self.local_agreements = np.zeros(self.n)
        self.posting_history = [self.posting_status.copy()]
        self.local_agreement_history = []
        self.post_count = np.zeros(self.n, dtype=int)
        self.positive_posts = np.zeros(self.n, dtype=int)
        self.negative_posts = np.zeros(self.n, dtype=int)
        
        self._update_posting_status()
        
    def initialize_opinions_60_40_split(self, positive_value=0.8, negative_value=-0.8, positive_ratio=0.6):
        opinions = super().initialize_opinions_60_40_split(
            positive_value, negative_value, positive_ratio
        )
        
        # Initialize posting history
        self.posting_status = np.zeros(self.n, dtype=bool)
        self.local_agreements = np.zeros(self.n)
        
        # Store history for analysis
        self.posting_history = [self.posting_status.copy()]
        self.local_agreement_history = []
        self.post_count = np.zeros(self.n, dtype=int)
        self.positive_posts = np.zeros(self.n, dtype=int)
        self.negative_posts = np.zeros(self.n, dtype=int)
        
        # Update initial posting status
        self._update_posting_status()
        
        return opinions
        
    def _update_posting_status(self):
        # local agreement for each node
        signs = np.sign(self.opinions - np.mean(self.opinions))
        local_agreements = np.zeros(self.n)
        
        for i in range(self.n):
            neighbors = list(self.G.neighbors(i))
            if len(neighbors) == 0:
                local_agreements[i] = 0.5  # default if no neighbors
            else:
                agreement_count = sum(signs[i] == signs[j] for j in neighbors)
                local_agreements[i] = agreement_count / len(neighbors)

        # determine if posting
        self.posting_status = local_agreements > self.threshold
        self.local_agreements = local_agreements
        new_posts = self.posting_status.astype(int)
        self.post_count += new_posts
        
        # post amount tracking
        signs = np.sign(self.opinions - np.mean(self.opinions))
        positive_posts = (signs > 0) & self.posting_status
        negative_posts = (signs < 0) & self.posting_status
        
        self.positive_posts += positive_posts.astype(int)
        self.negative_posts += negative_posts.astype(int)

        self.local_agreement_history.append(local_agreements.copy())
        
    def update(self):
        self.opinions = self.transition_matrix @ self.opinions
        self.opinions = np.clip(self.opinions, -1.0, 1.0)
        
        # Update posting status based on new opinions
        self._update_posting_status()
        
        # Store updated opinions and posting status
        self.opinion_history.append(self.opinions.copy())
        self.posting_history.append(self.posting_status.copy())
        self.time_steps += 1
    
    def get_posting_statistics(self):
        """Get statistics about posting behavior"""
        # Get total post counts
        total_posts = self.post_count.sum()
        total_positive = self.positive_posts.sum()
        total_negative = self.negative_posts.sum()
        
        # Calculate proportion of positive posts
        if total_posts > 0:
            positive_proportion = total_positive / total_posts
            negative_proportion = total_negative / total_posts
        else:
            positive_proportion = 0.5
            negative_proportion = 0.5
        
        # Get user-level proportions
        user_post_count = self.post_count.copy()
        user_positive_prop = np.zeros(self.n)
        
        for i in range(self.n):
            if user_post_count[i] > 0:
                user_positive_prop[i] = self.positive_posts[i] / user_post_count[i]
            else:
                user_positive_prop[i] = 0.5
        
        return {
            'total_posts': total_posts,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'positive_proportion': positive_proportion,
            'negative_proportion': negative_proportion,
            'user_post_count': user_post_count,
            'user_positive_proportion': user_positive_prop
        }
    
    def plot_opinion_evolution(self, highlight_posting=True):
        opinions = np.array(self.opinion_history)
        posts = np.array(self.posting_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot all opinions in a lighter color
        for i in range(self.n):
            plt.plot(range(self.time_steps + 1), opinions[:, i], 
                     color='lightgray', alpha=0.3)
        
        if highlight_posting:
            # Highlight opinions of users who post at each time step
            for t in range(self.time_steps + 1):
                for i in range(self.n):
                    if posts[t, i]:
                        plt.scatter(
                            t, opinions[t, i], 
                            color='blue' if opinions[t, i] > np.mean(opinions[t]) else 'red',
                            alpha=0.7, zorder=3
                        )
        
        plt.xlabel('Time Steps')
        plt.ylabel('Opinion')
        plt.title('Evolution of Opinions with Posting Status')
        plt.grid(True)
        plt.show()
    
    def plot_posting_and_variance(self):
        timesteps = range(self.time_steps + 1)
        
        posting_stats = []
        for t in range(self.time_steps + 1):
            posting_status = self.posting_history[t]
            opinions = self.opinion_history[t]
            mean_opinion = np.mean(opinions)
            
            positive_posts = sum((opinions > mean_opinion) & posting_status)
            negative_posts = sum((opinions < mean_opinion) & posting_status)
            total_posts = positive_posts + negative_posts
            
            posting_stats.append({
                'time': t,
                'positive_posts': positive_posts,
                'negative_posts': negative_posts,
                'total_posts': total_posts,
                'posting_ratio': sum(posting_status) / self.n
            })
            
        post_df = pd.DataFrame(posting_stats)
        
        variance_metrics = []
        for t in range(self.time_steps + 1):
            self.opinions = self.opinion_history[t]
            variance_metrics.append({
                'time': t,
                'variance': self.compute_polarization_variance(),
                'std_dev': self.compute_polarization_std(),
                'range': self.compute_polarization_range(),
                'local_agreement': np.mean(self.local_agreement_history[t]) if t < len(self.local_agreement_history) else 0
            })
            
        variance_df = pd.DataFrame(variance_metrics)
        
        # Plot results
        plt.figure(figsize=(12, 15))
        
        # Plot 1: Post counts over time
        plt.subplot(5, 1, 1)
        plt.bar(post_df['time'], post_df['positive_posts'], color='blue', label='Positive Posts', alpha=0.7)
        plt.bar(post_df['time'], post_df['negative_posts'], bottom=post_df['positive_posts'], 
                color='red', label='Negative Posts', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Post Count')
        plt.title('Number of Posts by Opinion Type Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Posting ratio over time
        plt.subplot(5, 1, 2)
        plt.plot(post_df['time'], post_df['posting_ratio'], marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Ratio of Users Posting')
        plt.title('Proportion of Users Posting Over Time')
        plt.grid(True)
        
        # Plot 3: Variance over time
        plt.subplot(5, 1, 3)
        plt.plot(variance_df['time'], variance_df['variance'], marker='s')
        plt.xlabel('Time Steps')
        plt.ylabel('Opinion Variance')
        plt.title('Opinion Variance Over Time')
        plt.grid(True)
        
        # Plot 4: Standard deviation over time
        plt.subplot(5, 1, 4)
        plt.plot(variance_df['time'], variance_df['std_dev'], marker='d')
        plt.xlabel('Time Steps')
        plt.ylabel('Opinion Std. Dev.')
        plt.title('Opinion Standard Deviation Over Time')
        plt.grid(True)
        
        # Plot 5: Average local agreement over time
        plt.subplot(5, 1, 5)
        plt.plot(range(self.time_steps + 1), [np.mean(la) for la in self.local_agreement_history], marker='s')
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Posting Threshold')
        plt.xlabel('Time Steps')
        plt.ylabel('Average Local Agreement')
        plt.title('Average Local Agreement Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return post_df, variance_df
    
    def visualize_initial_distribution(self):
        initial_opinions = self.opinion_history[0]
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(initial_opinions, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(initial_opinions), color='red', linestyle='--', label='Mean')
        plt.xlabel('Opinion Value')
        plt.ylabel('Count')
        plt.title('Distribution of Initial Opinions')
        plt.legend()
        
        # Network visualization with opinions
        plt.subplot(1, 2, 2)
        pos = nx.spring_layout(self.G, seed=42)  # Position nodes using spring layout
        
        # Node colors based on opinions
        colors = []
        for opinion in initial_opinions:
            if opinion > 0:
                colors.append('blue')
            else:
                colors.append('red')
        
        nx.draw_networkx(
            self.G, pos, 
            node_color=colors,
            node_size=50,
            with_labels=False,
            edge_color='gray',
            alpha=0.7
        )
        
        # Add legend
        plt.plot([0], [0], 'o', color='blue', label='Positive Opinion (60%)')
        plt.plot([0], [0], 'o', color='red', label='Negative Opinion (40%)')
        plt.legend()
        
        plt.title('Network with Initial Opinion Distribution')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # generate a SBM graph
    G = nx.stochastic_block_model(
        sizes=[10, 20, 75, 15, 10],  # sizes of communities
        p=[[0.5, 0.02, 0.01, 0.005, 0.001],  # define connections between communities
        [0.02, 0.5, 0.02, 0.001, 0.005],  
        [0.01, 0.02, 0.5, 0.02, 0.001],
        [0.005, 0.001, 0.02, 0.5, 0.02],
        [0.001, 0.005, 0.001, 0.02, 0.5]]
    )
    G.remove_nodes_from(list(nx.isolates(G))) # nodes with no edges; isolated nerds
    G = nx.convert_node_labels_to_integers(G)
    
    num_edges = int(G.number_of_edges() * 0.3)  # 30% extra random edges
    add_random_edges(G, num_edges)
    assign_edge_weights(G, method='engagement', seed=42)     # alt - uniform
    
    model = DeGrootThresholdModel(G, local_agreement_threshold=0.75)    
    # initialize opinions
    model.initialize_opinions_60_40_split(
        positive_value=0.8,      
        negative_value=-0.8,     
        positive_ratio=0.6       # % of positive nodes initially
    )
    
    model.visualize_initial_distribution()
    model.run(100)
    
    # results
    model.plot_opinion_evolution()
    model.plot_posting_and_variance()
    
    stats = model.get_posting_statistics()
    print(f"Total posts: {stats['total_posts']}")
    print(f"Positive posts: {stats['total_positive']} ({stats['positive_proportion']:.2f})")
    print(f"Negative posts: {stats['total_negative']} ({stats['negative_proportion']:.2f})")
    
    print(f"\nFinal opinion variance: {model.compute_polarization_variance():.4f}")
    print(f"Final opinion std dev: {model.compute_polarization_std():.4f}")
    print(f"Final opinion range: {model.compute_polarization_range():.4f}")
    print(f"Final local agreement: {model.compute_local_agreement():.4f}")