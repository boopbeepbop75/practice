import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import Data_cleanup
from GNN_Model import GNN
from Dataset import GraphDataset
import HyperParameters
import random

from scipy.ndimage import find_objects
from itertools import combinations
from torch_geometric.data import Batch


#Load and or preprocess Data
project_folder = HyperParameters.PROJECT_FOLDER
# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load(f'{project_folder}processed_training_graphs.pt')
    testing_data = torch.load(f'{project_folder}processed_testing_graphs.pt')
    training_labels = np.load(f'{project_folder}training_labels.npy')
    testing_labels = np.load(f'{project_folder}testing_labels.npy')

    # Extract the images, graphs, and edges
    '''normalized_training_images = training_data['images']  # Images (already tensors)
    training_edge_indices = training_data['edge_indices']  # Edge indices
    training_node_features = training_data['node_features']  # Node features

    normalized_testing_images = testing_data['images']  # Images (already tensors)
    testing_edge_indices = testing_data['edge_indices']  # Edge indices
    testing_node_features = testing_data['node_features']  # Node features'''

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    training_data = torch.load(f'{project_folder}processed_training_graphs.pt')
    testing_data = torch.load(f'{project_folder}processed_testing_graphs.pt')
    training_labels = np.load(f'{project_folder}training_labels.npy')
    testing_labels = np.load(f'{project_folder}testing_labels.npy')

    # Further preprocessing (assuming you generate node features and edges during cleanup)
    # Example: create_edge_index_from_slic(segments) and compute_node_features(image, segments)

print(training_data[0])

###Finish loading data###

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.EPOCHS

# Create the Dataset
Training_Dataset = GraphDataset(training_data, training_labels)
Testing_Dataset = GraphDataset(testing_data, testing_labels)

def collate_fn(data):
    graphs, labels = zip(*data)
    batched_graphs = Batch.from_data_list(graphs)
    
    # Ensure x is a tensor
    if isinstance(batched_graphs.x, list):
        batched_graphs.x = torch.tensor(batched_graphs.x, dtype=torch.float)
    
    return batched_graphs, torch.tensor(labels)

train_loader = DataLoader(Training_Dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(Testing_Dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Create new model
# Assuming each node has a feature vector of length 50 (n_segments)
# Determine input_dim based on the actual data
sample_data = next(iter(train_loader))[0]
input_dim = sample_data.x.shape[-1] if isinstance(sample_data.x, torch.Tensor) else 1
Model_0 = GNN(input_dim=input_dim)

#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model_0.parameters(), lr=LEARNING_RATE)

#Make Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''
# Training Loop
'''for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}\n---------")
    train_loss = 0
    for batch_idx, (batch_graphs, batch_labels) in enumerate(train_loader):
        # Add these checks
        print(f"Type of batch_graphs: {type(batch_graphs)}")
        print(f"Type of batch_graphs.x: {type(batch_graphs.x)}")
        
        if isinstance(batch_graphs.x, list):
            print(f"Number of nodes: {len(batch_graphs.x)}")
        else:
            print(f"Number of nodes: {batch_graphs.x.size(0)}")
        
        print(f"Max edge index: {batch_graphs.edge_index.max().item()}")
        print(f"Number of edges: {batch_graphs.edge_index.size(1)}")
        print(f"Shape of batch_labels: {batch_labels.shape}")
        
        # Convert batch_graphs.x to tensor if it's a list
        if isinstance(batch_graphs.x, list):
            batch_graphs.x = torch.tensor(batch_graphs.x, dtype=torch.float)
        
        # Your existing code...
        y_pred = Model_0(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)

        # Calculate Loss
        loss = loss_fn(y_pred, batch_labels)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate Average loss for the epoch
    train_loss /= len(train_loader)
    
    print("Testing the model...")
    test_loss, test_acc = 0, 0
    Model_0.eval()
    with torch.inference_mode():
        for batch_graphs, y_test in test_loader:
            test_pred = Model_0(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}%")'''