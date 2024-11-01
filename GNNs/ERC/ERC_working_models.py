

###################################  BELOW ARE ALL THE WORKING ARCHITECTURES FOR ERC WITH THEIR DESCRIPTIONS ##############################################


'''
##############################################################################################################################################


                                                        DATA USED BY ALL MODELS 

    Given a conversational graph as a series of disconnected subgraphs where each subgraph represents an utterance. Each subgraph consists of n 
    nodes, where each node is a participant in conversation. The speaking participant node has edges directed at all other participants 
    (non-speakers). The speaker node originally contains an embedidng of size 768 (BERT encoded). All non-speaker nodes originally contain an 
    empty embedding of size 768. 

    All models are tasked with predicting the emotion conveyed by each utterance in a conversation. 

##############################################################################################################################################
'''

# IMPORTS 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.data import Data





'''
##############################################################################################################################################

                                                                MODEL #1: SIMPT

                                        SIMOULTANEOUS MESSAGE PASSING AND SUBSEQUENT TRANSFORMER UPDATE

    Forward Pass: 
    1. Apply message passing using GATConv to all subgraphs in a graph simoultaneoulsy. Since subgraphs are disconnected they well be updated separatley. 
    2. After message passing has been applied create a node sequence for each participant by linking their node embeddings across all turns. 
    3. Apply a transformer model to each sequence to update embeddings using bidirectional attention.
    4. Use Global Attantion Pooling to classify each group of nodes in each utterance 

    Rationale: 
    Onec information has been passes from speakers to listeners, both speaker and listeners can reference past and future states to 
    redefine the emotional context of the information they received/expressed. 


    Performance: 
    Promising - Weighted-F1 score of 0.61

##############################################################################################################################################
'''


# The transformer block nelow is used to update node embeddings with their past and future states (hence, a memory update) 
class TransformerMemoryUpdate(nn.Module):
    def __init__(self, embedding_dim, nhead=4, num_layers=2):
        super(TransformerMemoryUpdate, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, memory_states):
        """
        memory_states: (batch_size, seq_length, embedding_dim)
        The sequence of memory states is processed to predict the next embedding.
        """
        # Transpose to match PyTorch's transformer input format
        memory_states = memory_states.transpose(0, 1)  
        updated_memory = self.transformer(memory_states)
        updated_memory = updated_memory.transpose(0, 1)  
        return updated_memory


class SIMPT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=7, num_heads=4, num_transformer_layers=2):
        super(SIMPT, self).__init__()
        
        # GATConv for message passing
        self.gat_conv = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        
        # Transformer for memory update
        self.transformer = TransformerMemoryUpdate(embedding_dim=hidden_dim, nhead=num_heads, num_layers=num_transformer_layers)
        
        # Global Attention Pooling for classification
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.global_att_pool = GlobalAttention(self.attention_layer)
        
        # Output layer for classification
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):

        batch_outputs = []
        num_graphs = batch.batch.unique().shape[0]


        # Iterate through each graph in the batch 
        for i in range(num_graphs):

            # Get the nodes belonging to the current graph
            graph_nodes = (batch.batch == i).nonzero(as_tuple=False).view(-1)  
            x_graph = batch.x[graph_nodes]

            # Get the edges belonging to the current graph
            edge_mask = (batch.edge_index[0].unsqueeze(1) == graph_nodes).any(1) & \
                        (batch.edge_index[1].unsqueeze(1) == graph_nodes).any(1)
            edge_index_graph = batch.edge_index[:, edge_mask] - graph_nodes.min() 

            # Apply GATConv to the current graph 
            x_graph = self.gat_conv(x_graph, edge_index_graph)

            # Determine number of turns and participants
            num_turns = x_graph.shape[0] - edge_index_graph.shape[1]
            num_participants = int(x_graph.shape[0] / num_turns)
            
            # Create a list of node states for each participant
            all_participant_sequences = [[] for _ in range(num_participants)]

            # Gather participant sequences over turns
            for turn_id in range(num_turns):
                for participant_id in range(num_participants):
                    node_index = participant_id + turn_id * num_participants  
                    all_participant_sequences[participant_id].append(x_graph[node_index])

            # Stack the sequences and apply the transformer to update memory states
            updated_sequences = []
            for participant_sequence in all_participant_sequences:
                participant_sequence = torch.stack(participant_sequence) 
                updated_sequence = self.transformer(participant_sequence.unsqueeze(0))  
                updated_sequences.append(updated_sequence.squeeze(0))  

            # Stack all updated sequences (one per participant)
            updated_sequences = torch.stack(updated_sequences, dim=1)  

            # Apply global attention pooling to classify each turn (pooled across participants)
            subgraph_outputs = []
            for turn_idx in range(num_turns):
                turn_embeddings = updated_sequences[turn_idx, :, :]  
                pooled = self.global_att_pool(turn_embeddings)
                output = self.classifier(pooled)
                subgraph_outputs.append(output)

            # Stack subgraph outputs for the current graph
            graph_output = torch.stack(subgraph_outputs, dim=0)  
            batch_outputs.append(graph_output)

        # Stack outputs for the entire batch
        return torch.cat(batch_outputs, dim=0)



'''
##############################################################################################################################################

                                                                MODEL #2: ITERMEM

                                        ITERATIVE MESSSAGE PASSING AND UNIDIRECTIONAL RNN MEMORY UPDATE

    Forward Pass: 
    For each subgraph in a graph: 
        1. Feed sequence of past node states for each participant to an RNN and obtain a current memory embedding for each particpant 
        2. Assign memory embedding to node:
            For non-speaker: Simply Substitute current empty embedding with memory embedding 
            For speaker: Concatenat populated sentence embedding with memory embedding and use linear layer to project to dim 768
        3. Apply message passing with updated embeddings 
        4. Use Global Attantion Pooling to classify each group of nodes in each utterance 

    Rationale: 
    Participants update their memory before receiving information from the current speaker. 


    Performance: 
    Promising - Weighted-F1 score of 0.60

##############################################################################################################################################
'''


# The RNN below is used to update the current node embeddings only using their past states 
class RNNForNextState(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=1):
        super(RNNForNextState, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.scale_up_projection = nn.Linear(256, 768)

    def forward(self, memory_states):
        batch_size, _, embedding_dim = memory_states.shape
        
        zero_state = torch.zeros((batch_size, 1, embedding_dim)).to(memory_states.device)  
        extended_memory = torch.cat([memory_states, zero_state], dim=1)  

        # Apply RNN, but we only care about the last state (output and hidden state)
        _, hidden = self.rnn(extended_memory)  
        
        new_memory_state = hidden[-1]  # Shape: (batch_size, hidden_dim)

         # Substitute projection with padding
        padded_memory = torch.cat([new_memory_state, torch.zeros((batch_size, 768 - embedding_dim)).to(new_memory_state.device)], dim=-1)

        return padded_memory  # Shape: (batch_size, 768)

        # # Project the pooled memory from 256 to 768 dimensions
        # projected_memory = self.scale_up_projection(new_memory_state)  

        # return projected_memory

import torch
import torch.nn as nn

class TransformerForNextState(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, num_layers=2):
        super(TransformerForNextState, self).__init__()
        
        # Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project from 256 to 768 if needed (or use padding as an alternative)
        self.scale_up_projection = nn.Linear(embedding_dim, 768)

    def forward(self, memory_states):
        """
        memory_states: Tensor of shape (batch_size, seq_length, embedding_dim).
        The sequence of past memory states is processed by the transformer to predict the next embedding.
        """
        batch_size, seq_length, embedding_dim = memory_states.shape
        # Initialize a zeroed "current" state
        zero_state = torch.zeros((batch_size, 1, embedding_dim), device=memory_states.device)
        # Concatenate past states with the zeroed current state
        extended_memory = torch.cat([memory_states, zero_state], dim=1)  # Shape: (batch_size, seq_length + 1, embedding_dim)
        # Transpose to match Transformer input shape (seq_length, batch_size, embedding_dim)
        extended_memory = extended_memory.transpose(0, 1)
        # Apply the transformer encoder to the sequence of past and current states
        transformed_memory = self.transformer_encoder(extended_memory)  # Shape: (seq_length + 1, batch_size, embedding_dim)
        # Get the last output, which is the predicted embedding for the current state
        new_memory_state = transformed_memory[-1]  # Shape: (batch_size, embedding_dim)
        # Optionally, project to 768 dimensions
        projected_memory = self.scale_up_projection(new_memory_state)

        return projected_memory  # Shape: (batch_size, 768)



class ITERMEM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=7, num_heads=4):
        super(ITERMEM, self).__init__()
        
        # GATConv for message passing
        self.gat_conv = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        
        # RNN for memory contextualization 
        self.rnn = RNNForNextState()
        self.transformer = TransformerForNextState()

        # layer to project speaker sentence embedding and memory into a single and memory state 
        self.speaker_projection = nn.Sequential(nn.Linear(768 * 2, 768), nn.ReLU())
        
        # Global Attention Pooling for subgraph classification
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.global_att_pool = GlobalAttention(self.attention_layer)
        
        # Output layer for classification (after message passing)
        self.classifier = nn.Linear(hidden_dim, output_dim)



    # Function to return a list of subgraphs (Data objects) from the current graph. Each subgraph will contain a subset of nodes and edges from the graph.
    def get_subgraphs_as_list(self, x_graph, edge_index_graph):
        num_subgraphs = x_graph.shape[0] - edge_index_graph.shape[1]
        nodes_per_subgraph = int(x_graph.shape[0]/num_subgraphs)

        subgraphs = []
        for i in range(0, x_graph.shape[0], nodes_per_subgraph):
            # Select nodes and edges for each subgraph
            subgraph_nodes = x_graph[i:i + nodes_per_subgraph]  # Select nodes for this subgraph
            subgraph_edge_mask = (edge_index_graph[0] >= i) & (edge_index_graph[0] < i + nodes_per_subgraph)
            subgraph_edges = edge_index_graph[:, subgraph_edge_mask]

            # Adjust edge indices to be local to the subgraph
            subgraph_edges = subgraph_edges - i

            # Create subgraph Data object
            subgraph_data = Data(x=subgraph_nodes, edge_index=subgraph_edges)
            subgraphs.append(subgraph_data)

        return subgraphs



    def forward(self, batch):
        

        subgraph_outputs = []
        num_graphs = batch.batch.unique().shape[0]

        # Iterate through each graph in the batch 
        current_label_start = 0
        for i in range(num_graphs):

            # Get graph nodes
            graph_nodes = (batch.batch == i).nonzero(as_tuple=False).view(-1)  
            x_graph = batch.x[graph_nodes]

            # Get graph edges
            edge_mask = (batch.edge_index[0].unsqueeze(1) == graph_nodes).any(1) & \
                        (batch.edge_index[1].unsqueeze(1) == graph_nodes).any(1)
            edge_index_graph = batch.edge_index[:, edge_mask] - graph_nodes.min()  # Reindex edges to the subgraph node indices

            # Get subgraphs from the current graph
            subgraphs = self.get_subgraphs_as_list(x_graph, edge_index_graph)

            # Initialize memory states for current graph
            memory_states = []  
            speaker_node_embeddings = []
            # Iterate through subgraphs for message passing and memory updates
            for j, subgraph in enumerate(subgraphs):
                x, edge_index = subgraph.x, subgraph.edge_index
                
                if x.size(0) == 1:
                    speaker_node_index = torch.tensor([0], dtype=torch.long, device=x.device)  # Assign the only node as speaker
                else:
                    # Otherwise, extract the speaker node from outgoing edges
                    speaker_node_index = torch.unique(edge_index[0])   
                
                # Apply message passing on the first iteration (no memory updates). Embeddings change from 768 -> 256
                if j == 0:
                    x = self.gat_conv(x, edge_index)
                    speaker_node_embeddings.append(x[speaker_node_index])
                    memory_states.append(x)
                else:
                    # Stack current memory states 
                    memory_sequence = torch.stack(memory_states, dim=1)

                    # Apply RNN to generate memory contextualized embeddings 
                    # pooled_memory_state = self.rnn(memory_sequence)
                    # Apply transformer to generate memory contextualized embeddings 
                    pooled_memory_state = self.transformer(memory_sequence)

                    x_updated = x.clone()  # Clone x to avoid in-place operation

                    # Step 2: Update non-speaker node embeddings with memory state embeddings if there are non-speakers
                    if x.size(0) != 1:
                        non_speaker_indices = [idx for idx in range(x.size(0)) if not torch.equal(torch.tensor(idx), speaker_node_index)]
                        # Replace the non-speaker embeddings (768) with memory embeddings (256)
                        x_updated[non_speaker_indices] = pooled_memory_state[non_speaker_indices].squeeze(1)

                    # Step 3: Update speaker node by concatenating memory and info embedding (768 + 256), and projecting to 256
                    speaker_memory_embedding = pooled_memory_state[speaker_node_index].squeeze(1)  # Make shape [1, 768]
                    speaker_info_embedding = x[speaker_node_index]  # Shape [1, 768]
                    speaker_concat_embedding = torch.cat((speaker_memory_embedding, speaker_info_embedding), dim=-1)
                    # Project to 768
                    speaker_projected_embedding = self.speaker_projection(speaker_concat_embedding)  # Shape [1, 768]
                    x_updated[speaker_node_index] = speaker_projected_embedding

                    # Update x with the new embeddings
                    x = x_updated

                    # Step 4: Apply message passing with updated embeddings
                    x = self.gat_conv(x, edge_index)
                    # update embedding list used for classification
                    speaker_node_embeddings.append(x[speaker_node_index])
                    
                    # Step 5: Update all memory states based on the updated embeddings after message passing 
                    memory_states.append(x)

            # Update the index where labels for current graph in batch start 
            current_label_start += len(subgraphs)

            # After all iterations, append predictions for all subgraphs in current graph 
            # subgraph_outputs += [self.classifier(self.global_att_pool(x)) for x in memory_states]

            # Use only the final speaker node embedding for classification instead of global pooling
            subgraph_outputs += [self.classifier(speaker_node) for speaker_node in speaker_node_embeddings]
        
        return torch.stack(subgraph_outputs, dim=0)  


'''
##############################################################################################################################################

                                                                MODEL #3: ITERSTEP (UNDER DEVELOPMENT)

                                                        ITERATIVE TWO-STEP MESSSAGE PASSING 

    Forward Pass: 
    

    Rationale: 
    


    Performance: 


##############################################################################################################################################
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.data import Data


class ITERSTEP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, output_dim=256, num_heads=4):
        super(ITERSTEP, self).__init__()
        
        # GATConv for message passing
        self.gat_conv = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        
        # Global attention pooling
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.global_att_pool = GlobalAttention(self.attention_layer)
        
        # Output layer for classification
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def get_subgraph_for_turn(self, graph, turn_idx):
        """
        Generates a subgraph that includes all turns up to the current turn, 
        but only edges pointing to nodes in the current turn.
        
        :param graph: PyTorch Geometric Data object (entire graph)
        :param turn_idx: The current turn index to generate the subgraph for.
        :return: Subgraph of the current turn.
        """
        # Get the node indices for the current and previous turns
        subgraph_nodes = [i for i in range(graph.turns.size(0)) if graph.turns[i] <= turn_idx]
        
        # Get edges where the target node is in the current turn
        subgraph_edges = graph.edge_index[:, (graph.turns[graph.edge_index[1]] == turn_idx)]
        
        # Get the corresponding embeddings for the nodes in the subgraph
        subgraph_x = graph.x[subgraph_nodes]
        
        # Create a new PyTorch Geometric Data object for the subgraph
        subgraph = Data(x=subgraph_x, edge_index=subgraph_edges)
        
        return subgraph, subgraph_nodes  # Return the subgraph and the corresponding node indices in the original graph
    
    def forward(self, batch):
        """
        Forward pass for a batch of graphs.
        :param batch: A batch of graphs.
        """
        # Initialize to store all outputs
        batch_outputs = []
        
        # Get number of graphs in batch
        num_graphs = batch.batch.unique().shape[0]
        
        # Iterate through each graph in the batch
        for i in range(num_graphs):
            # Get the current graph from the batch
            graph_nodes = (batch.batch == i).nonzero(as_tuple=False).view(-1)
            valid_edges_mask = torch.isin(batch.edge_index[0], graph_nodes)
            edge_index_graph = batch.edge_index[:, valid_edges_mask]
            
            graph = Data(
                x=batch.x[graph_nodes], 
                edge_index=edge_index_graph,
                y=batch.y[graph_nodes],
                turns=batch.turns[graph_nodes]
            )
            
            
            # Iterate through the turns in the graph
            graph_outputs = []
            for turn_idx in range(graph.turns.max().item() + 1):
                # Generate subgraph for the current turn
                subgraph, subgraph_nodes = self.get_subgraph_for_turn(graph, turn_idx)
                
                # Apply message passing using GATConv
                subgraph.x = self.gat_conv(subgraph.x, subgraph.edge_index)
                
                # Update the embeddings of the original graph with the updated embeddings from the subgraph
                graph.x[subgraph_nodes] = subgraph.x
                
                # Apply global attention pooling
                pooled_output = self.global_att_pool(subgraph.x)
                
                # Classify using the output from the attention pooling
                output = self.classifier(pooled_output)
                
                # Store the output for the current subgraph (turn)
                graph_outputs.append(output)
            
            # Stack the outputs for all turns in the current graph
            graph_outputs = torch.stack(graph_outputs, dim=0)  # (num_turns, output_dim)
            batch_outputs.append(graph_outputs)
        
        # Stack all graph outputs
        return torch.stack(batch_outputs, dim=0)  # (num_graphs, num_turns, output_dim)
