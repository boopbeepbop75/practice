import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
import HyperParameters

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.LEARNING_RATE

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_UNITS, output_dim=OUTPUT_SHAPE):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Add this check
        assert edge_index.max() < x.size(0), f"Max edge index {edge_index.max()} is >= number of nodes {x.size(0)}"
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x