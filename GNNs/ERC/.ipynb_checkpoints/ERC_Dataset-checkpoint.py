
import torch
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F



'''
##############################################################################################################################################

The dataset object taked in a list of graphs where each graph, 'G', is composed of a sub-graphs 'G_i' where each subgraph represents an utterance.  

Each subggraph G_i has an entry for Structure, 'X', and one for its emotion label, 'Y'. 

X contains node entries where each node has a corresponding embedding and list of edges. 

Consider entry 'X' in graph 'G_0':

A dictionary entry for a speaker node will have a full feature embedding and edges pointing to all other nodes in the time stamp as well as an edge pointing to its future state. 
- Ex. 'D_0': {'embedding': [-0.46728479862213135, -0.20498991012573242, -0.43848446011543274, ....] ,'edges': ['A_0', 'B_0', 'C_0']}

A dictionary entry for a silent node will have a null feature embedding and one edge pointing to its future state. 
- Ex. 'A_0': {'embedding': [NULL] ,'edges': []}

##############################################################################################################################################
'''

class ERC_Dataset(Dataset):
    def __init__(self, graph_list, emotion_mapping):
        """
        Initialize the ERC_Dataset object.

        :param graph_list: List of graphs where each graph contains a list of subgraphs.
        """
        super(ERC_Dataset, self).__init__()
        self.emotion_mapping = emotion_mapping
        self.graph_list = graph_list
        self.processed_data = self._process_graphs(graph_list)

    def _process_graphs(self, graph_list):
        """
        Process the input list of graphs and combine all subgraphs in each graph into a single PyTorch Geometric Data object.

        :param graph_list: List of graphs where each graph contains a list of subgraphs.
        :return: A list of combined subgraphs represented as a single PyTorch Geometric Data object.
        """
        processed_graphs = []

        for graph in graph_list:
            node_embeddings = []
            edge_list = []
            y_labels = []
    


            # Process each subgraph
            for subgraph in graph:  # Each subgraph contains 'X' and 'Y'
                node_to_idx = {}
                start_idx = len(node_embeddings)
                
                # Process nodes
                for node_name, node_data in subgraph['X'].items():
                    node_idx = len(node_to_idx)
                    node_to_idx[node_name] = start_idx + node_idx
                    node_embeddings.append(node_data['embedding'])

                # Process edges
                for node_name, node_data in subgraph['X'].items():
                    node_idx = node_to_idx[node_name]
                    for edge in node_data['edges']:
                        if edge in node_to_idx:
                            edge_list.append((node_idx, node_to_idx[edge]))

                # Process labels
                emotion_index = self.emotion_mapping[subgraph['Y']]
                y_labels.append(emotion_index)

            # Convert embeddings and edges to tensors
            x = torch.tensor(node_embeddings, dtype=torch.float)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
            y = torch.tensor(y_labels, dtype=torch.long)
            turns_list = [i for i in range(len(graph))]
            turns = torch.tensor(turns_list, dtype=torch.long)

            # Create a PyTorch Geometric Data object for the big graph
            graph_data = Data(x=x, edge_index=edge_index, y=y, turns=turns)
            processed_graphs.append(graph_data)

        return processed_graphs

    def len(self):
        """
        Return the number of graphs in the dataset.
        """
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get a specific graph by index.

        :param idx: Index of the graph to retrieve.
        :return: A PyTorch Geometric Data object representing the combined graph of subgraphs.
        """
        return self.processed_data[idx]








'''
##############################################################################################################################################
Function below prints out an lement of a dataloader so we can knwo how to parse it in forward passes
##############################################################################################################################################
'''




def inspect_dataLoader(train_loader):
    for batch_idx, batch in enumerate(train_loader):
        # Use the batch attribute of batches to get number of graphs 
        num_graphs = batch.batch.unique().shape[0]

        current_label_start = 0
        for i in range(num_graphs):

            # Get Nodes
            graph_nodes = (batch.batch == i).nonzero(as_tuple=False).view(-1)  
            x_subgraph = batch.x[graph_nodes]

            # Get Edges
            edge_mask = (batch.edge_index[0].unsqueeze(1) == graph_nodes).any(1) & \
                        (batch.edge_index[1].unsqueeze(1) == graph_nodes).any(1)
            edge_index_subgraph = batch.edge_index[:, edge_mask] - graph_nodes.min()  # Reindex edges to the subgraph node indices

            # Extract labels: subgraphs always equal num nodes - num edges given structure 
            num_subgraphs = x_subgraph.shape[0] -  edge_index_subgraph.shape[1] 
            y_subgraph = batch.y[current_label_start:current_label_start + num_subgraphs] 

            # extract turns
            turns = batch.turns  

            print(f"Graph {i + 1}:")
            print(f"  Subgraph node features shape: {x_subgraph.shape}")
            print(f"  Subgraph edge index shape: {edge_index_subgraph.shape}")
            print(f"  Subgraph labels: {y_subgraph}")

            current_label_start += num_subgraphs
            break
        
        break  
