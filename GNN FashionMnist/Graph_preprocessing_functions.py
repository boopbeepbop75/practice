import numpy as np
from skimage.color import rgb2lab
from skimage.measure import regionprops
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.ndimage import find_objects
from itertools import combinations

def create_edge_index_from_slic(segments):
    unique_segments = np.unique(segments)
    num_segments = len(unique_segments)
    
    # Create a mapping from segment label to index
    segment_to_index = {seg: idx for idx, seg in enumerate(unique_segments)}
    
    # Find bounding box for each segment
    bounding_boxes = find_objects(segments)
    
    # Function to check if two segments are neighbors
    def are_neighbors(seg1, seg2):
        bb1 = bounding_boxes[segment_to_index[seg1]]
        bb2 = bounding_boxes[segment_to_index[seg2]]
        return (
            (bb1[0].start <= bb2[0].stop and bb2[0].start <= bb1[0].stop) and
            (bb1[1].start <= bb2[1].stop and bb2[1].start <= bb1[1].stop)
        )
    
    # Create edges
    edges = []
    for seg1, seg2 in combinations(unique_segments, 2):
        if are_neighbors(seg1, seg2):
            # Add edges in both directions
            edges.append([segment_to_index[seg1], segment_to_index[seg2]])
            edges.append([segment_to_index[seg2], segment_to_index[seg1]])
    
    # Convert to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Ensure that all indices are within the valid range
    assert edge_index.max() < num_segments, f"Max edge index {edge_index.max()} is >= number of segments {num_segments}"
    
    return edge_index

def compute_node_features(image, segments):
    # Check if the image is grayscale (i.e., 2D)
    if len(image.shape) == 2:  # Grayscale image
        image_features = image  # Use the grayscale pixel values as node features
    else:
        # For color images, convert from RGB to LAB color space
        image_lab = rgb2lab(image)
        image_features = image_lab

    # Initialize node features
    node_features = []
    for segment_label in np.unique(segments):
        # Get the pixels belonging to the current segment
        segment_mask = segments == segment_label
        # Compute the mean feature value for the segment
        mean_feature = image_features[segment_mask].mean(axis=0)
        node_features.append(mean_feature)

    return np.array(node_features)