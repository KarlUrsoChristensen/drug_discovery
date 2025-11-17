import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_channels = hidden_channels
        # default aggregation function for GCNConv is a sum scaled by degrees 
        # (i.e. number of bonds between the given molecules)
        self.conv1 = GCNConv(num_node_features, hidden_channels, add_self_loops = True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops = True)
        self.linear = torch.nn.Linear(hidden_channels+num_edge_features, 1)

    def forward(self, data):
        # data contains ['z', 'y', 'edge_attr', 'batch', 'idx', 'x', 'edge_index', 'pos', 'ptr', 'name']
        # z might be atomic number
        # y might be target label
        # edge_attr has size [num_edges, num_edge_features]
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # x represents nodes (atoms) and has size [num_notes, num_node_features]
        # edge_index represents edges (bonds), given by index of the two nodes it connects 
        # and has size [2, num_edges] = [2, num_bonds * 2], since bonds are undirected but edges are directed
        # batch represents the molecule index for each node and has size [num_notes]
        # i.e. we have packed mutiple molecules into a single disconnected graph

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # Aggregate edge features
        node_edge_agg = self.aggregate_edge_features(x, edge_index, edge_attr)

        # Concatenate embeddings with aggregated note features
        x = torch.cat([x, node_edge_agg], dim=-1) # x new size is [num_hidden_features + num_edge_features]

        # 2. Readout layer
        # averaging each embedding feature across all nodes in each graph
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = self.linear(x)

        return x
    
    def aggregate_edge_features(self, node_features, edge_index, edge_attr):
        """
        Aggregate edge features for each node's neighbors and update node features.
        """
        # Step 1: Create an empty tensor for edge feature aggregation
        num_nodes = node_features.size(0)
        edge_agg = torch.zeros(num_nodes, edge_attr.size(1), device=node_features.device)
        
        # Step 2: Aggregate the edge features for each node by summing the edge features of its neighbors
        for i in range(edge_index.size(1)):  # Iterate over all edges
            source_node = edge_index[0, i]
            target_node = edge_index[1, i]
            edge_feature = edge_attr[i]
            
            # Add the edge feature to the respective nodes' aggregated feature
            edge_agg[source_node] += edge_feature
            edge_agg[target_node] += edge_feature # maybe should use mean and not sum

        return edge_agg


class OLDGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(OLDGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x