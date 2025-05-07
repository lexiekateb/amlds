import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from .base_degroot import DeGrootModel

class DeGrootThresholdModel(DeGrootModel):
    def __init__(self, graph, local_agreement_threshold=0.75):       # default threshold to .75
        super().__init__(graph)
        self.threshold = local_agreement_threshold
    
    def initialize_opinions_manual(self, initial_opinions, proportions, SBM_bias_blocks=None):
        opinions = super().initialize_opinions_manual(initial_opinions, proportions, SBM_bias_blocks)

        # Calculate initial opinion distribution statistics
        initial_pos_opinions = sum(proportions[i] for i in range(len(proportions)) if initial_opinions[i] > 0)
        initial_neg_opinions = sum(proportions[i] for i in range(len(proportions)) if initial_opinions[i] < 0)
        initial_neutral_opinions = sum(proportions[i] for i in range(len(proportions)) if initial_opinions[i] == 0)
        
        # Store the statistics
        self.initial_pos_opinions = initial_pos_opinions
        self.initial_neg_opinions = initial_neg_opinions 
        self.initial_neutral_opinions = initial_neutral_opinions
        self.initial_pos_to_neg_ratio = initial_pos_opinions / initial_neg_opinions if initial_neg_opinions > 0 else float('inf')
        self.initial_proportion_positive = initial_pos_opinions / (initial_pos_opinions + initial_neg_opinions)

        # Print the initial distribution statistics
        print(f"Initial opinion distribution:")
        print(f"  Positive opinions: {initial_pos_opinions:.2%}")
        print(f"  Negative opinions: {initial_neg_opinions:.2%}")
        print(f"  Neutral opinions: {initial_neutral_opinions:.2%}")
        print(f"  Positive-to-negative ratio: {self.initial_pos_to_neg_ratio:.2f}")
        print(f"  Initial proportion of opinions on positive side: {self.initial_proportion_positive:.2%}")
        
        # initialize posting history
        self.posting_status = np.zeros(self.n, dtype=bool)
        self.local_agreements = np.zeros(self.n)
        
        # store history of changes
        self.posting_history = [self.posting_status.copy()]
        self.local_agreement_history = []
        self.post_count = np.zeros(self.n, dtype=int)
        self.positive_posts = np.zeros(self.n, dtype=int)
        self.negative_posts = np.zeros(self.n, dtype=int)
        
        self._update_posting_status()
        
        return opinions

    def _update_posting_status(self):
        # Compute local agreement for each node based on the paper
        signs = np.sign(self.opinions - np.mean(self.opinions))
        local_agreements = np.zeros(self.n)
        
        for i in range(self.n):
            neighbors = list(self.G.neighbors(i))
            if len(neighbors) == 0:
                local_agreements[i] = 0.5  # default if no neighbors
            else:
                agreement_count = sum(signs[i] == signs[j] for j in neighbors)
                local_agreements[i] = agreement_count / len(neighbors)

        # determine if posting, post amount tracking
        self.posting_status = local_agreements > self.threshold
        self.local_agreements = local_agreements
        new_posts = self.posting_status.astype(int)
        self.post_count += new_posts
        
        # count positive posts for time step
        signs = np.sign(self.opinions)
        positive_posts = (signs > 0) & self.posting_status # people on the positive side of the opinion who are supposed to post 
        negative_posts = (signs < 0) & self.posting_status
        self.positive_posts += positive_posts.astype(int)
        self.negative_posts += negative_posts.astype(int)

        self.local_agreement_history.append(local_agreements.copy())
        
    def update(self):
        # update DeGroot model step
        self.opinions = self.normalized_adj_matrix @ self.opinions
        self.opinions = np.clip(self.opinions, -1.0, 1.0)
        
        # update posting status based on new opinions
        self._update_posting_status()
        
        # store updated opinions and posting status
        self.opinion_history.append(self.opinions.copy())
        self.posting_history.append(self.posting_status.copy())
        self.time_steps += 1
    

    def plot_opinion_evolution(self, highlight_posting=True):
        # Plot opinion evolution with posting status
        opinions = np.array(self.opinion_history)
        posts = np.array(self.posting_history)
        
        plt.figure(figsize=(8, 6))

        for i in range(self.n):
            plt.plot(range(self.time_steps + 1), opinions[:, i], 
                     color='lightgray', alpha=0.3)
        
        if highlight_posting:
            for t in range(self.time_steps + 1):
                for i in range(self.n):
                    if posts[t, i]:
                        threshold = 0 #np.mean(opinions[t])
                        plt.scatter(
                            t, opinions[t, i], 
                            color='blue' if opinions[t, i] > threshold else 'red',
                            alpha=0.7, zorder=3, s=5
                        )
        plt.xlabel('Time Steps')
        plt.ylabel('Opinion')
        plt.title('Evolution of Opinions with Posting Status')
        plt.grid(True)
        plt.show()
    
    def _plot_post_counts(self, post_df):
        plt.bar(post_df['time'], post_df['positive_posts'], color='blue', label='Positive Posts', alpha=0.7)
        plt.bar(post_df['time'], post_df['negative_posts'], bottom=post_df['positive_posts'], 
                color='red', label='Negative Posts', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Post Count')
        plt.title('Number of Posts by Opinion Type Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(post_df['positive_posts'] + post_df['negative_posts']) * 1.1)
        plt.show()
            
    def _plot_cumulative_posts(self, post_df):
        cumulative_positive = np.cumsum(post_df['positive_posts'])
        cumulative_negative = np.cumsum(post_df['negative_posts'])
        plt.plot(post_df['time'], cumulative_positive, color='blue', label='Cumulative Positive Posts')
        plt.plot(post_df['time'], cumulative_negative, color='red', label='Cumulative Negative Posts')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Posts')
        plt.title('Cumulative Number of Posts Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(max(cumulative_positive), max(cumulative_negative)) * 1.1)
        plt.show()

    def _plot_posting_ratio(self, post_df):
        plt.plot(post_df['time'], post_df['posting_ratio'], marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Ratio of Users Posting')
        plt.title('Proportion of Users Posting Over Time')
        plt.grid(True)
        plt.ylim(0, max(post_df['posting_ratio']) * 1.1)
        plt.show()


    def _plot_local_agreement(self, local_agreement_history, threshold):
        local_agreements = [np.mean(la) for la in local_agreement_history]
        plt.plot(range(self.time_steps + 1), local_agreements, marker='s')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Posting Threshold')
        plt.xlabel('Time Steps')
        plt.ylabel('Average Local Agreement')
        plt.title('Average Local Agreement Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(min(min(local_agreements), threshold) * 0.9,
                max(max(local_agreements), threshold) * 1.1)
        plt.tight_layout()
        plt.show()
    
    def get_final_posting_statistics(self):
        # get total post counts
        total_posts = self.post_count.sum()
        total_positive = self.positive_posts.sum()
        total_negative = self.negative_posts.sum()
        
        # calculate proportion of positive posts
        positive_proportion = total_positive / total_posts
        negative_proportion = total_negative / total_posts
        cumulative_pos_to_neg_ratio = total_positive / total_negative

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
            'cumulative_pos_to_neg_ratio': cumulative_pos_to_neg_ratio,
            'user_post_count': user_post_count,
            'user_positive_proportion': user_positive_prop
        }

    def plot_posting_and_variance(self, visualize=False):
        timesteps = range(self.time_steps + 1)
        
        posting_stats = []
        for t in range(self.time_steps + 1):
            posting_status = self.posting_history[t]
            opinions = self.opinion_history[t]

            threshold = 0
            positive_posts = sum((opinions > threshold) & posting_status)
            negative_posts = sum((opinions < threshold) & posting_status)
            pos_to_neg_ratio = positive_posts / negative_posts
            proportion_positive = positive_posts / (positive_posts + negative_posts)
            total_posts = positive_posts + negative_posts
            
            posting_stats.append({
                'time': t,
                'positive_posts': positive_posts,
                'negative_posts': negative_posts,
                'pos_to_neg_ratio': pos_to_neg_ratio,
                'total_posts': total_posts,
                'posting_ratio': sum(posting_status) / self.n,
                'proportion_positive': proportion_positive
            })
        
        post_df = pd.DataFrame(posting_stats)
        
        variance_metrics = []
        for t in range(self.time_steps + 1):
            self.opinions = self.opinion_history[t]
            variance_metrics.append({
                'time': t,
                'opinion variance': self.compute_polarization_variance(),
                'opinion std_dev': self.compute_polarization_std(),
                'opinion range': self.compute_polarization_range(),
                'avg_local_agreement': np.mean(self.local_agreement_history[t]) if t < len(self.local_agreement_history) else 0
            })
        
        variance_df = pd.DataFrame(variance_metrics)
        
        if visualize:
            # Plot 1: Post counts over time
            self._plot_post_counts(post_df)

            # Plot 2: Cumulative posts over time
            self._plot_cumulative_posts(post_df)

            # Plot 3: Posting ratio over time
            self._plot_posting_ratio(post_df)

            # Plot 4: Average local agreement over time
            self._plot_local_agreement(self.local_agreement_history, self.threshold)
        
        return post_df, variance_df

    def visualize_distribution(self, layout='spring', timestep=0):
        initial_opinions = self.opinion_history[timestep]
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(initial_opinions, bins=5, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(initial_opinions), color='red', linestyle='--', label='Mean')
        plt.xlabel('Opinion Value')
        plt.ylabel('Count')
        plt.title('Distribution of Initial Opinions')
        plt.legend()
        
        # Network visualization with opinions
        plt.subplot(1, 2, 2)
        if layout == 'spring':
            pos = nx.spring_layout(self.G, seed=42)  # Position nodes using spring layout
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.G)  # Position nodes using spectral layout
        
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
        
        plt.title('Network with Initial Opinion Distribution')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()