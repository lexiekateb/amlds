import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

def plot_network(G, opinions, pos=None, node_size=50, with_labels=False, title=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Create a colormap from opinions
    mean_opinion = np.mean(opinions)
    node_colors = []
    for opinion in opinions:
        if opinion > mean_opinion:
            intensity = min(1.0, (opinion - mean_opinion) / 0.5 + 0.5)
            node_colors.append((0, 0, intensity))
        else:
            intensity = min(1.0, (mean_opinion - opinion) / 0.5 + 0.5)
            node_colors.append((intensity, 0, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw network
    nx.draw_networkx(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        with_labels=with_labels,
        edge_color='gray',
        alpha=0.8,
        ax=ax
    )
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Opinion')
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax

def plot_posting_heatmap(model, time_steps=None):
    if time_steps is None:
        time_steps = model.time_steps + 1
    else:
        time_steps = min(time_steps, model.time_steps + 1)
    
    # Create posting matrix
    posting_matrix = np.zeros((model.n, time_steps))
    for t in range(time_steps):
        opinions = model.opinion_history[t]
        mean_opinion = np.mean(opinions)
        
        for i in range(model.n):
            if model.posting_history[t][i]:
                posting_matrix[i, t] = 1 if opinions[i] > mean_opinion else -1
            else:
                posting_matrix[i, t] = 0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort nodes by final opinion
    final_opinions = model.opinion_history[-1]
    sorted_indices = np.argsort(final_opinions)
    
    # Create the heatmap
    sns.heatmap(
        posting_matrix[sorted_indices, :],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        ax=ax
    )
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Nodes (sorted by final opinion)')
    ax.set_title('Posting Behavior Over Time')
    
    # Add colorbar legend
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Negative Post', 'Silent', 'Positive Post'])
    
    plt.tight_layout()
    
    return fig, ax