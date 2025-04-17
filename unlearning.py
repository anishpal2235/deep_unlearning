import os
import subprocess
import requests
import tqdm
import random

from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset, Subset
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Helper functions for loading the hidden dataset.
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Path to the animal dataset folder
data_path = "C:/Users/palan/OneDrive/Desktop/Deep Unlearning/animals/animals"

# Load the full dataset using ImageFolder
full_dataset = datasets.ImageFolder(root=data_path, transform=data_transform)

# Get class indices for 'elephant'
elephant_class_idx = full_dataset.class_to_idx['elephant']

# Separate indices for retain (non-elephant) and forget (elephant)
retain_indices = [i for i, (_, label) in enumerate(full_dataset) if label != elephant_class_idx]
forget_indices = [i for i, (_, label) in enumerate(full_dataset) if label == elephant_class_idx]

# Create retain_loader and forget_loader
retain_dataset = Subset(full_dataset, retain_indices)
forget_dataset = Subset(full_dataset, forget_indices)

retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True)
forget_loader = DataLoader(forget_dataset, batch_size=64, shuffle=True)

# Optionally, create a validation_loader with full dataset or another subset
validation_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 512),  # Match this size with the output size of FeatureExtractor
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Create instances
generator = resnet18(weights = None, num_classes = 10)

discriminator = Discriminator()
feature_extractor = resnet18(weights = None, num_classes = 10)

# Move models to device (GPU/CPU)
generator.to(DEVICE)
discriminator.to(DEVICE)
feature_extractor.to(DEVICE)

# Unlearning Algorithm Implementation
def unlearning(generator, discriminator, feature_extractor, retain_loader, forget_loader, validation_loader):
    epochs = 20
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer_generator = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-4)
    scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, T_max=epochs)
    
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-4)
    scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, T_max=epochs)

    discriminator.train()
    generator.train()

    for epoch in range(epochs):
        for retain_data, forget_data in zip(retain_loader, forget_loader):
            retain_inputs, retain_labels = retain_data[0].to(DEVICE), retain_data[1].to(DEVICE)  
            forget_inputs, forget_labels = forget_data[0].to(DEVICE), forget_data[1].to(DEVICE)
            
            # Train Discriminator
            optimizer_discriminator.zero_grad()
            
            retain_features = feature_extractor(retain_inputs).detach()
            retain_predictions = discriminator(retain_features)
            retain_loss = criterion(retain_predictions, torch.ones_like(retain_predictions))
            
            forget_features = generator(forget_inputs).detach() 
            forget_predictions = discriminator(forget_features)
            forget_loss = criterion(forget_predictions, torch.zeros_like(forget_predictions))
            
            discriminator_loss = retain_loss + forget_loss
            discriminator_loss.backward(retain_graph=True)
            optimizer_discriminator.step()
            
            # Train Generator
            optimizer_generator.zero_grad()
            forget_features = generator(forget_inputs)
            generator_predictions = discriminator(forget_features)
            generator_loss = criterion(generator_predictions, torch.ones_like(generator_predictions))
            generator_loss.backward()
            optimizer_generator.step()
            
        scheduler_generator.step()
        scheduler_discriminator.step()

        print(f'Epoch {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss.item()}, Generator Loss: {generator_loss.item()}')

    discriminator.eval()
    generator.eval()

# Run the unlearning process
unlearning(generator, discriminator, feature_extractor, retain_loader, forget_loader, validation_loader)

from PIL import Image

# Helper function to load and preprocess an image
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)

# Path to the elephant image you want to test
elephant_image_path = "C:/Users/palan/OneDrive/Desktop/Deep Unlearning/animals/animals/elephant/0e81eb0721.jpg"  # Update this with the actual path

# Load and preprocess the elephant image
elephant_image = load_and_preprocess_image(elephant_image_path, data_transform)

# Put models in evaluation mode
feature_extractor.eval()
discriminator.eval()

# Pass the elephant image through the feature extractor
with torch.no_grad():
    elephant_features = feature_extractor(elephant_image)
    
    # Check the output of the discriminator
    elephant_prediction = discriminator(elephant_features)
    
    # The closer the prediction is to 1, the more the discriminator "remembers" the elephant
    print(f"Discriminator prediction for elephant image: {elephant_prediction.item()}")

# Output:
# If the discriminator output is closer to 0, the unlearning has worked.
# If the discriminator output is closer to 1, the elephant has not been forgotten.
import torch.nn.functional as F

# Helper function to map class indices back to their labels
def get_class_label(pred_idx, dataset):
    return dataset.classes[pred_idx]

# Pass the elephant image through the generator (ResNet18)
with torch.no_grad():
    generator_output = generator(elephant_image)

# Apply softmax to convert the output to probabilities
class_probabilities = F.softmax(generator_output, dim=1)

# Get the predicted class (index of the highest probability)
predicted_class_idx = torch.argmax(class_probabilities, dim=1).item()

# Get the actual class label from the dataset
predicted_class_label = get_class_label(predicted_class_idx, full_dataset)
predicted_probability = class_probabilities[0][predicted_class_idx].item()


print(f'Predicted class for the elephant image: {predicted_class_label}')