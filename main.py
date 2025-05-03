import matplotlib.pyplot as plt
from models import DeGrootModel, DeGrootThresholdModel
from utils import add_random_edges, assign_edge_weights
from utils.graph_utils import create_sbm_graph
from visualization.plot_utils import plot_network, plot_posting_heatmap

def run_experiment(graph, threshold=0.75, steps=100, positive_ratio=0.6):
    # initialization
    model = DeGrootThresholdModel(graph, local_agreement_threshold=threshold)
    # model.initialize_opinions_60_40_split(
    #     positive_value=0.8,      
    #     negative_value=-0.8,     
    #     positive_ratio=positive_ratio
    # )
    
    model.initialize_opinions_manual(
        initial_opinions=[0.8, 0.4, -0.4, -0.8],
        proportions=[0.25, 0.25, 0.25, 0.25]
    )
    
    model.visualize_initial_distribution()

    model.run(steps)
    
    # stats + plots
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
    
    return model

def main():
    # generate a simple sbm graph
    G = create_sbm_graph(
        sizes=[25, 25, 25, 25],  # sizes of communities
        p_intra=0.5,                    # probability within communities
        p_inter=0.01                     # probability between communities
    )
    
    # add random edge weights to make the graph more realistic to TikTok Randomness
    num_edges = int(G.number_of_edges() * 0.5)  # 0% extra random edges
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