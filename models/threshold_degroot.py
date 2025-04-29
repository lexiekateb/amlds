import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from .base_degroot import DeGrootModel

class DeGrootThresholdModel(DeGrootModel):
    def __init__(self, graph, local_agreement_threshold=0.75):       # default threshold to .75
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
        
        # initialize posting history
        self.posting_status = np.zeros(self.n, dtype=bool)
        self.local_agreements = np.zeros(self.n)
        
        # store history
        self.posting_history = [self.posting_status.copy()]
        self.local_agreement_history = []
        self.post_count = np.zeros(self.n, dtype=int)
        self.positive_posts = np.zeros(self.n, dtype=int)
        self.negative_posts = np.zeros(self.n, dtype=int)
        
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
        
        # update posting status based on new opinions
        self._update_posting_status()
        
        # store updated opinions and posting status
        self.opinion_history.append(self.opinions.copy())
        self.posting_history.append(self.posting_status.copy())
        self.time_steps += 1
    
    def get_posting_statistics(self):
        # get total post counts
        total_posts = self.post_count.sum()
        total_positive = self.positive_posts.sum()
        total_negative = self.negative_posts.sum()
        
        # calculate proportion of positive posts
        positive_proportion = total_positive / total_posts
        negative_proportion = total_negative / total_posts

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
        
        for i in range(self.n):
            plt.plot(range(self.time_steps + 1), opinions[:, i], 
                     color='lightgray', alpha=0.3)
        
        if highlight_posting:
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