import os
import glob
from PIL import Image
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic
import Graph_preprocessing_functions
import HyperParameters
import Get_Data
import matplotlib.pyplot as plt
import random
from skimage.color import label2rgb  # Import label2rgb
from skimage.color import rgb2gray 

# Display the image
def show_img(x, y):
    plt.imshow(x, cmap='gray')  # Use 'gray' for grayscale images
    plt.title(y)
    plt.axis('off')  # Optional: turn off axis labels
    plt.show()

def show_image_slic(x):
    fig = plt.figure("Superpixels -- %d segments" % (50))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    plt.axis("off")
    plt.show()

def load_and_preprocess_images(data):
    images = []
    labels = []

    for image, label in data:
        # Convert the tensor to a numpy array (28x28 image)
        image_np = image.numpy().squeeze()  # Remove single channel dimension
        images.append(image_np)
        labels.append(label)
    
    return np.array(images), np.array(labels)

def process_images_to_graphs(images, labels, n_segments, sigma):
    processed_graphs = []
    for i, image in enumerate(images):
        segments = slic(image, n_segments=n_segments, sigma=sigma, channel_axis=None)
        # Visualize SLIC segmentation for the first image
        if i in HyperParameters.STOPS:
            # Create a label image where each segment is colored
            segmented_image = label2rgb(segments, image=image, kind='avg')
            
            # Display the original and segmented images
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')  # Display original image (assumed grayscale)
            plt.title(HyperParameters.CLASSES[labels[i]])
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(segmented_image)
            plt.title('SLIC Segmentation')
            plt.axis('off')

            plt.show()

        edge_index = Graph_preprocessing_functions.create_edge_index_from_slic(segments)
        node_features = Graph_preprocessing_functions.compute_node_features(image, segments)
        
        # Ensure node_features is a tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Add these checks
        print(f"Image {i}:")
        print(f"Number of segments: {len(np.unique(segments))}")
        print(f"Node features shape: {node_features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Max node index in edge_index: {edge_index.max().item()}")
        print("---")
        
        graph = Data(x=node_features, edge_index=edge_index)
        processed_graphs.append(graph)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(images)} images")
    return processed_graphs

def clean_data():
    project_folder = HyperParameters.PROJECT_FOLDER
    '''training_folders = [os.path.join(project_folder, "GNN_Dataset/seg_train/seg_train", class_name) 
                        for class_name in ["buildings", "forest", "glacier", "mountain", "sea", "street"]]
    testing_folders = [os.path.join(project_folder, "GNN_Dataset/seg_test/seg_test", class_name) 
                       for class_name in ["buildings", "forest", "glacier", "mountain", "sea", "street"]]
    '''
    print("Loading and preprocessing images...")
    training_data = Get_Data.get_training_data()
    testing_data = Get_Data.get_testing_data()

    training_data, training_labels = load_and_preprocess_images(training_data)
    testing_data, testing_labels = load_and_preprocess_images(testing_data)

    random_index = random.randint(0, len(training_data)-1)
    CLASSES = HyperParameters.CLASSES
    
    show_img(training_data[random_index], CLASSES[training_labels[random_index]])

    print("Processing training data to graphs...")
    processed_training_graphs = process_images_to_graphs(training_data, training_labels, HyperParameters.n_segments, HyperParameters.sigma)

    print("Processing testing data to graphs...")
    processed_testing_graphs = process_images_to_graphs(testing_data, testing_labels, HyperParameters.n_segments, HyperParameters.sigma)

    print("Saving processed graphs...")
    torch.save(processed_training_graphs, f'{project_folder}processed_training_graphs.pt')
    torch.save(processed_testing_graphs, f'{project_folder}processed_testing_graphs.pt')
    
    print("Saving labels...")
    np.save(f'{project_folder}training_labels.npy', training_labels)
    np.save(f'{project_folder}testing_labels.npy', testing_labels)

    print("Data cleanup completed successfully.")

'''if __name__ == "__main__":
    clean_data()'''