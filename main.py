import matplotlib.pyplot as plt
from models import DeGrootThresholdModel
from utils import add_random_edges, assign_edge_weights
from utils.graph_utils import create_sbm_graph
from visualization.plot_utils import plot_network, plot_posting_heatmap
import numpy as np

def run_experiment(graph, threshold=0.75, steps=100, positive_ratio=0.6, visualize=True):
    # initialization
    model = DeGrootThresholdModel(graph, local_agreement_threshold=threshold)

    # initial opinions and belief distribution
    ispal_op = [-1, -.5, 0, .5, 1]
    ispal_prop = [.1, .2, .29, .22, .19] # target 90/10
    abortion_op = [0.9, 0.3, -0.3, -0.9]
    abortion_prop = [0.31, 0.34, 0.27, 0.08] # 95/5

    model.initialize_opinions_manual(
        initial_opinions=ispal_op,
        proportions=ispal_prop
    )
    
    if visualize:
        model.visualize_distribution(layout='spring', timestep=0)
    
    model.run(steps)
    
    if visualize:
        model.plot_opinion_evolution()
    
    post_df, variance_df = model.plot_posting_and_variance(visualize=visualize)
        
    stats = model.get_final_posting_statistics()
    print('EXPERIMENT STATS:')
    print(f"Total posts: {stats['total_posts']}")
    print(f"Positive posts: {stats['total_positive']} ({stats['positive_proportion']:.2f})")
    print(f"Negative posts: {stats['total_negative']} ({stats['negative_proportion']:.2f})")
    cumulative_pos_to_neg_ratio = stats['cumulative_pos_to_neg_ratio']
    print(f"Overall pos/neg ratio: {cumulative_pos_to_neg_ratio:.2f}")
    print(f"Overall proportion of positive posts: {stats['total_positive'] / (stats['total_positive'] + stats['total_negative']):.2f}")
    
    proportion_positives = post_df['proportion_positive']
    final_proportion_positive = proportion_positives.iloc[-1]
    proportion_positive_var = proportion_positives.dropna().var()
    print(f"Variance in proportion of positive posts over time: {proportion_positive_var:.4f}")
    
    print(f"\nFinal opinion range: {model.compute_polarization_range():.4f}")
    print(f"Final opinion variance: {model.compute_polarization_variance():.4f}")
    print(f"Final opinion std dev: {model.compute_polarization_std():.4f}")
    average_local_agreement = model.compute_local_agreement()[0]
    print(f"Final average local agreement: {average_local_agreement:.4f}")
    local_agreement_variance = model.compute_local_agreement()[1]
    print(f"Final local agreement variance: {local_agreement_variance:.4f}")

    # Compute second eigenvalue of normalized adjacency matrix
    normalized_adj = model.normalized_adj_matrix
    eigenvalues = np.linalg.eigvals(normalized_adj)
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    second_eigenvalue = sorted_eigenvalues[1]
    print(f"Second largest eigenvalue: {second_eigenvalue:.4f}")

    return model, cumulative_pos_to_neg_ratio, proportion_positive_var, average_local_agreement, second_eigenvalue

def main():
    # generate a simple sbm graph
    G = create_sbm_graph(
        sizes=[25, 25, 25, 25],  # sizes of communities
        p_intra=0.5,                    # probability within communities
        p_inter=0.01                     # probability between communities
    )
    
    # add random edge weights to make the graph more realistic to TikTok Randomness

    num_edges = int(G.number_of_edges() * 0.1)  # 10% extra random edges

    add_random_edges(G, num_edges)
    assign_edge_weights(G, method='engagement', seed=42)
    
    model = run_experiment(
        graph=G,
        threshold=0.75,
        steps=50,
        positive_ratio=0.6
    )
    
    return model

if __name__ == "__main__":
    model = main()
    plt.show()