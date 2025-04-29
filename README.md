# DeGroot Threshold Model for Social Media Posting Behavior

## Usage

To run a basic simulation:

```python
import networkx as nx
from models import DeGrootThresholdModel
from utils import add_random_edges, assign_edge_weights

G = nx.stochastic_block_model(
    sizes=[10, 20, 75, 15, 10],  # sizes of communities
    p=[[0.5, 0.02, 0.01, 0.005, 0.001],
       [0.02, 0.5, 0.02, 0.001, 0.005],
       [0.01, 0.02, 0.5, 0.02, 0.001],
       [0.005, 0.001, 0.02, 0.5, 0.02],
       [0.001, 0.005, 0.001, 0.02, 0.5]]
)
G.remove_nodes_from(list(nx.isolates(G)))
G = nx.convert_node_labels_to_integers(G)

add_random_edges(G, int(G.number_of_edges() * 0.3))
assign_edge_weights(G, method='engagement')

model = DeGrootThresholdModel(G, local_agreement_threshold=0.75)

model.initialize_opinions_60_40_split(
    positive_value=0.8,      
    negative_value=-0.8,     
    positive_ratio=0.6
)

model.run(100)

model.plot_opinion_evolution()
model.plot_posting_and_variance()
```

Or, you can use the `run_experiment()` function in `main.py`