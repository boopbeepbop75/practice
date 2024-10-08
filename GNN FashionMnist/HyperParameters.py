#PROJECT FOLDER
PROJECT_FOLDER = 'GNN FashionMnist/'

### SLIC HYPER PARAMETERS ###
n_segments = 50
sigma = 1

### MODEL HYPER PARAMETERS ###
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
BATCH_SIZE = 16
HIDDEN_UNITS = 10
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = .001
EPOCHS = 3

#VISUALIZATION PARAMETERS
STOPS = [555, 876, 1057, 26882, 9411, 8419, 47916, 40896, 14107, 38054]