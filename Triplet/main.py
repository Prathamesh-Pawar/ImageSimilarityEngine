import warnings
warnings.filterwarnings("ignore")
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from torch.optim import lr_scheduler
from torch import optim
from networks import EmbeddingNet, TripletNet
from losses import OnlineTripletLoss
from utils import  SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit

from datasets_s import BalancedBatchSampler



to_pil_image = transforms.ToPILImage()

# Transform for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Organize CIFAR-10 dataset by class
class_to_images_train = {i: [] for i in range(10)}
labels_train = []
for idx, (image, label) in enumerate(cifar10_train):
    if idx == 10000:
        break
    class_to_images_train[label].append((image, label))
    labels_train.append(label)

labels_train = torch.tensor(labels_train)

class_to_images_test = {i: [] for i in range(10)}
labels_test = []
for idx, (image, label) in enumerate(cifar10_test):
    if idx == 1000:
        break
    class_to_images_test[label].append((image, label))
    labels_test.append(label)

labels_test = torch.tensor(labels_test)


# Custom dataset class
class TripletCIFAR10Dataset(Dataset):
    def __init__(self, labels, class_to_images, num_samples):
        self.train_labels = labels
        self.class_to_images = class_to_images
        self.num_samples = num_samples
        self.classes = list(class_to_images.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Select a random class
        same_class = random.choice(self.classes)

        # Select two images from the same class
        img1, _ = random.choice(self.class_to_images[same_class])
        img2, _ = random.choice(self.class_to_images[same_class])

        # Select a different class
        diff_class = random.choice([c for c in self.classes if c != same_class])

        # Select one image from the different class
        img3, _ = random.choice(self.class_to_images[diff_class])

        # Return triplet (img1, img2, img3) and label (same_class)
        return (img1, img2, img3), same_class


# Create datasets
train_size, test_size = 5000, 1000
train_dataset = TripletCIFAR10Dataset(labels_train, class_to_images_train, train_size)
test_dataset = TripletCIFAR10Dataset(labels_test, class_to_images_test, test_size)


print("Datasets created successfully!")


# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(torch.tensor(cifar10_train.targets), n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(torch.tensor(cifar10_test.targets), n_classes=10, n_samples=25)

online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler)


margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)

loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50


fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, None, log_interval, metrics=[AverageNonzeroTripletsMetric()])
