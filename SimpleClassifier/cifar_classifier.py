from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn
import random

from ImageSimilarityEngine.SimpleClassifier.cnn_network import ConvCNN

# Define transformations for augmenting data from CIFAR-10 (improves performance)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset from CIFAR
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


# Create balanced subset, sampling 1000 images per class
def create_balanced_subset(dataset, num_samples_per_class=1000):
    class_indices = defaultdict(list)

    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Sample images per class
    subset_indices = []
    for label in range(10):  # CIFAR-10 has 10 classes (0 to 9)
        class_subset = random.sample(class_indices[label], num_samples_per_class)
        subset_indices.extend(class_subset)

    # Return a Subset object
    return Subset(dataset, subset_indices)

balanced_train_subset = create_balanced_subset(train_dataset)

# Define K-Fold validation params
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=21)

# Params for Early Stoppage
patience = 5  # Init max num of epochs to wait for improvement
best_val_loss = float('inf')  # Init best validation loss
patience_counter = 0  # Init counter for patience

# Define hyperparameters for learning.
batch_size = 64
epochs = 50
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Recommended online for torch


# Initialize the model to ConvCNN (custom model)
model = ConvCNN().to(device)
# Initialize loss function to minimize cross-entropy
criterion = nn.CrossEntropyLoss()
# Initialize Adam to optimize
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Begin training using K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(balanced_train_subset)))):
    print(f'Current Fold {fold + 1}')

    # Use Subset to partition data and create training and validation sets.
    train_subset = Subset(balanced_train_subset, train_idx)
    val_subset = Subset(balanced_train_subset, val_idx)
    # Use DataLoader to provide iterable data sampler for both sets.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Loop through training epochs
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        # Define running loss for training.
        training_loss = 0.0
        # Begin iteration over data sample
        for inputs, labels in train_loader:
            # Sends inputs to device to process
            inputs, labels = inputs.to(device), labels.to(device)
            # Resets gradient for model
            optimizer.zero_grad()
            # Calls model on inputs to get outputs and train (forward pass)
            outputs = model(inputs)
            # Determines loss for inputs
            loss = criterion(outputs, labels)
            # Back-propagates loss (gradient adjustment)
            loss.backward()
            # Updates model params
            optimizer.step()
            # Adjust running loss to compute loss for epoch
            training_loss += loss.item()

        # Calculate average training loss
        avg_training_loss = training_loss / len(train_loader)

        # Validation loop
        # Sets model to evaluation mode.
        model.eval()
        # Set count of correct and total for accuracy predictions.
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            # Iterate over inputs in validation set
            for inputs, labels in val_loader:
                # Prepare inputs in computational device
                inputs, labels = inputs.to(device), labels.to(device)
                # Get outputs of model based on inputs (classify)
                outputs = model(inputs)
                # Calculate current loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Get accuracy
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_training_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '../EvaluationModel/best_model_classifier.pth')  # Save the best model
            print("Validation loss improved. Saving the model.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

