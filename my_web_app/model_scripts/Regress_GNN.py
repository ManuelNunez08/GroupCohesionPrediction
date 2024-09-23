# for data handling 
import pandas as pd
import pickle
import numpy as np

# for performance metrics 
from sklearn.metrics import mean_squared_error, r2_score

# for data set 
from torch_geometric.data import Data, Dataset

# for model architetcture 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

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

# we use dense weight to balance data set
from denseweight import DenseWeight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================== BELOW WE DEFINE THE MODEL ARCHITECTURE =========================

NUM_EDGE_FEATURES = 3
NUM_NODE_FEATURES = 12

class CohesionGNN(torch.nn.Module):
    def __init__(self, num_features= NUM_NODE_FEATURES, hidden_size=32, target_size=1, dropout=0.5):
        super(CohesionGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.dropout = dropout

        self.conv1 = GATConv(self.num_features, self.hidden_size, edge_dim=NUM_EDGE_FEATURES)
        
        self.linear = nn.Linear(self.hidden_size, self.target_size)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply GAT convolution layer
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global mean pooling: aggregates node embeddings 
        x = global_mean_pool(x, batch)  
        
        # Apply final linear layer for regression
        x = self.linear(x)  
        
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



def weighted_mse_loss(pred, target, weight):
    loss = weight * (pred - target) ** 2
    return torch.mean(loss)


def k_fold_train_model(dataset, k_folds=5, hyperparams=None, patience=50, device='cpu'):

    train_val_targets = [data.y.item() for data in dataset]
    
    dw = DenseWeight(alpha = hyperparams['alpha_weight'])
    train_val_weights = dw.fit(train_val_targets)
    for i, data in enumerate(dataset):
        data.weight = torch.tensor([train_val_weights[i]], dtype=torch.float)

    weights = [data.weight for data in dataset ]
    print("Alpha:", hyperparams['alpha_weight'])
    print("weights:", weights)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    fold_train_losses = []
    fold_val_losses = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold+1}/{k_folds}")
        print("==========================================")

        # we randomly define training and validation subsets for each training process 
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # These data loaders are used to train and evaaluate 
        train_loader = DataLoader(train_subset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Initialize model
        model = CohesionGNN(num_features=12, hidden_size=32, target_size=1, dropout=hyperparams['dropout_rate'])
        model = model.to(device)
        
        # We use an Adam to perform gradient descent and optimize model weights 
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        train_losses = []
        val_losses = []

        # Train
        for epoch in range(hyperparams['n_epochs']):
            if early_stop:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

            model.train()
            running_loss = 0
            # calculate loss across training batches 
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                # use customized loss function
                loss = weighted_mse_loss(out, data.y, data.weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.num_graphs  # Multiply loss by batch size

            avg_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            
            # Evaluate on validation 
            val_loss = evaluate_model(model, val_loader, device)
            val_losses.append(val_loss)

            # Early stop after 50 epochs of declining validation loss 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'models/best_model_fold_{fold+1}.pth') 
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                early_stop = True

            # Print progress
            if epoch % hyperparams['print_interval'] == 0 or early_stop == True:
                print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Store losses for this fold
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        print(f"Fold {fold+1} completed.")
        print("==========================================")
    
    # average the losses across all folds
    avg_train_loss = [sum(folds) / len(folds) for folds in zip(*fold_train_losses)]
    avg_val_loss = [sum(folds) / len(folds) for folds in zip(*fold_val_losses)]
    
    return avg_train_loss, avg_val_loss

# =================================== HELPER FUNCTIONS ==============================

def get_regression(actuals, predictions):
    plt.clf()  
    plt.close()  

    plt.figure(figsize=(4, 4))
    plt.scatter(actuals, predictions, alpha=0.7, color='b', label="Predicted vs Actual")
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='r', linestyle='--', label="Ideal Line (y=x)")
    plt.xlabel("Actual Cohesion Values")
    plt.ylabel("Predicted Cohesion Values")
    plt.title("Predicted vs Actual Cohesion Values")
    plt.ylim([1,7])
    plt.xlim([1,7])
    plt.legend()
    plt.grid(True)

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

    # List to store predictions and actual values
    predictions = []
    actuals = []

    # Disable gradient 
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = kmodel(data)  
            predictions.append(out.squeeze(-1).cpu().numpy())  
            actuals.append(data.y.cpu().numpy())  

    # Flatten predictions and actual values lists
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    perf_dic = {"Test Loss": f"{test_loss:.4f}",
                "Mean Squared Error": f"{mse:.4f}",
                "R-squared": f"{r2:.4f}"}
    
    regress_plot = get_regression(actuals, predictions)

    return perf_dic, regress_plot


# ===================================== BELOW ARE THE FUNTIONS USED BY THE FLASK APP ===============

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
        perf_dic, regress_plot = get_perf_metrics(model, data_test, hyperparams)
        models.append((perf_dic, regress_plot))
    
    return models