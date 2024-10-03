# for data handling 
import pandas as pd
import pickle

# for data set 
from torch_geometric.data import Data, Dataset

# for model architetcture 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention

# for training process 
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader  
from torch.utils.data import Subset

from torch.utils.data import random_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# to save images
from io import BytesIO
import base64

# for confusion matrix 
from sklearn.metrics import confusion_matrix
import seaborn as sns

# for perf metrics
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score


from denseweight import DenseWeight
from model_scripts.optimize_alpha import find_optimal_alpha

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== THE TWO FUNCTIONS BELOW ARE TASKED WITH FILTERING THE DATA AND ASSIGNING LABELS ===================

def filter_and_associate_data_question_binary(graph_data, df, question, max_std_dev=1.0, floor=4.5, ceiling=3.5):

    # Filter DataFrame for entries where the standard deviation is below the max_std_dev
    filtered_df = df[(df[f'{question}_std'] <= max_std_dev) & 
                ((df[f'{question}_mean'] <= ceiling) | (df[f'{question}_mean'] >= floor))]
    filtered_graph_data = []

    # Iterate over each graph entry and find corresponding rows in the DataFrame
    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        # Find the matching row in the filtered DataFrame
        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        if not match.empty:
            average_score = match[f'{question}_mean'].values[0]
            graph_entry['score'] = 1 if average_score >= floor else 0
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data


def filter_and_associate_data_category_binary(graph_data, df_kappa, category, min_kappa=0.2, floor=3.5, ceiling=4.5):

    kappa_column = f'{category}_Kappa'
    average_column = f'{category}_Average'

    filtered_df = df_kappa[(df_kappa[kappa_column] >= min_kappa) & 
                        ((df_kappa[average_column] <= ceiling) | (df_kappa[average_column] >= floor))]

    filtered_graph_data = []

    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        
        if not match.empty:
            # Binary classification: 1 if average score is above the floor, 0 if below the ceiling
            average_score = match[average_column].values[0]
            graph_entry['score'] = 1 if average_score >= floor else 0
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data



# ===================================== BELOW WE DEFINE THE MODEL ARCHITECTURE =======

NUM_EDGE_FEATURES = 3
NUM_NODE_FEATURES = 12

class CohesionGNN(torch.nn.Module):
    def __init__(self, num_features=NUM_NODE_FEATURES, hidden_size=32, target_size=1, dropout=0.5):
        super(CohesionGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.dropout = dropout

        # GAT convolution layer
        self.conv1 = GATConv(self.num_features, self.hidden_size, edge_dim=NUM_EDGE_FEATURES)
        
        # Attention mechanism for global graph-level pooling
        self.att = nn.Linear(self.hidden_size, 1)  # To compute attention scores
        
        # Global attention pooling layer
        self.global_att_pool = GlobalAttention(self.att)
        
        # Linear layer for binary classification (output size 1 for binary labels)
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply GAT convolution layer
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph attention pooling to aggregate node embeddings into graph-level representation
        x = self.global_att_pool(x, batch)  
        
        # Linear layer for final binary prediction
        x = self.linear(x)  
        
        # Apply sigmoid for binary classification (optional, depends on the loss function used)
        x = torch.sigmoid(x)
        
        return x



#  ===================== BELOW WE DEFINE EVALUATE, LOSS, AND TRAIN FUNCTIONS FOR THE MODEL ==============
def evaluate_model(model, data_loader, device):
    model.eval()  
    total_loss = 0
    with torch.no_grad():  
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            
            loss = F.mse_loss(out.squeeze(-1), data.y)  
            total_loss += loss.item() * data.num_graphs  
    return total_loss / len(data_loader.dataset)  


def weighted_bce_loss(pred, target, weight):

    # Apply sigmoid, BCE, and multiply weighted loss (greater loss for laerger predicitons). 
    pred = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    weighted_loss = weight * bce_loss
    
    return torch.mean(weighted_loss)


# Training loop with k-fold cross-validation
def k_fold_train_model(dataset, k_folds=5, hyperparams=None, patience=50, device='cpu'):

    # apply denseweights to correct for imbalance

    train_val_targets = [data.y.item() for data in dataset]
    
    dw = DenseWeight(alpha = hyperparams['alpha_weight'])
    train_val_weights = dw.fit(train_val_targets)
    for i, data in enumerate(dataset):
        data.weight = torch.tensor([train_val_weights[i]], dtype=torch.float)

    weights = [data.weight for data in dataset ]
    print("Alpha:", hyperparams['alpha_weight'])
    print("weights:", weights)
    kfold = KFold(n_splits=k_folds, shuffle=True)

     # get labels    
    fold_train_losses = []
    fold_val_losses = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold+1}/{k_folds}")
        print("==========================================")

        # Create training and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Initialize model (binary classification with 1 output)
        model = CohesionGNN(num_features=12, hidden_size=32, target_size=1, dropout=hyperparams['dropout_rate'])
        model = model.to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        train_losses = []
        val_losses = []

        # Train loop
        for epoch in range(hyperparams['n_epochs']):
            if early_stop:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

            model.train()
            running_loss = 0
            # Training step
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                
                # Use the custom weighted BCE loss
                loss = weighted_bce_loss(out.squeeze(-1), data.y, data.weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.num_graphs

            avg_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            
            # Evaluate on validation set
            val_loss = evaluate_model(model, val_loader, device)
            val_losses.append(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'models/best_model_fold_{fold+1}.pth') 
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                early_stop = True

            # Print progress
            if epoch % hyperparams['print_interval'] == 0 or early_stop:
                print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Store fold losses
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        print(f"Fold {fold+1} completed.")
        print("==========================================")
    
    # Average the losses across all folds
    avg_train_loss = [sum(folds) / len(folds) for folds in zip(*fold_train_losses)]
    avg_val_loss = [sum(folds) / len(folds) for folds in zip(*fold_val_losses)]
    
    return avg_train_loss, avg_val_loss

# ========================= BELOW WE DEFINE HELPER FUNCTIONS FOR THOSE USED BY FLASK =====================================
def get_confusion_matrix(all_labels, preds):
    # Clear the previous figure
    plt.clf()  
    plt.close()  

    plt.figure(figsize=(4, 4))

    cm = confusion_matrix(all_labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'], cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return img_base64



def get_perf_metrics(kmodel, data_test, hyperparams):

    kmodel.eval()  
    kmodel = kmodel.to(device)


    # Prepare test data loader
    test_loader = DataLoader(data_test, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loss = evaluate_model(kmodel, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Collect predictions and true labels
    all_probs = []
    all_labels = []

    with torch.no_grad():  
        for data in data_test:
            out = kmodel(data).squeeze(-1)  
            probs = torch.sigmoid(out)  
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    # compute metrics across all test samples
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Filter out NaN values from F1 scores
    valid_idx = ~np.isnan(f1_scores)
    valid_f1_scores = f1_scores[valid_idx]
    valid_thresholds = thresholds[valid_idx[:-1]]  # thresholds are one element shorter than f1_scores

    # Find the threshold that maximizes the valid F1 score
    optimal_idx = np.argmax(valid_f1_scores)
    optimal_threshold = valid_thresholds[optimal_idx]
    optimal_f1 = valid_f1_scores[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Evaluate the model using the optimal threshold
    preds = (all_probs >= optimal_threshold).astype(int)

    # Now calculate the final evaluation metrics based on this threshold
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)  
    recall = recall_score(all_labels, preds, zero_division=0)  
    f1 = f1_score(all_labels, preds, zero_division=0) 

    perf_dic = {"Test Loss": f"{test_loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "Final F1 Score": f"{f1:.4f}"}
    conf_matrix = get_confusion_matrix(all_labels, preds)

    return perf_dic, conf_matrix



# ============================ BELOW WE DEFINE THE FUNCTIONS USED BY FLASK APP ================================


def train_model(data_train_val, hyperparams):
    k_fold_train_model(data_train_val, 
                        k_folds=5, 
                        hyperparams=hyperparams, 
                        patience=50, 
                        device=device)


def get_results(data_test, hyperparams):
    model = CohesionGNN(num_features=12, hidden_size=32, target_size=1, dropout=hyperparams['dropout_rate'])
    models = []
    for i in range(5):
        model.load_state_dict(torch.load(f'models/best_model_fold_{i + 1}.pth'))
        perf_dic, conf_matrix = get_perf_metrics(model, data_test, hyperparams)
        models.append((perf_dic, conf_matrix))
    
    return models




