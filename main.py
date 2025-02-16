import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt # For data viz
import pandas
import numpy
import sys
from tqdm.auto import tqdm
from PIL import Image
from glob import glob
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Ensure all images are standardized
transformer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class PlayingCardDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform = transform)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# Import and verify datasets
train_folder = './dataset/train'
valid_folder = './dataset/valid'
test_folder = './dataset/test'

train_dataset = PlayingCardDataset(train_folder, transformer)
valid_dataset = PlayingCardDataset(valid_folder, transformer)
test_dataset = PlayingCardDataset(test_folder, transformer)

batchSize = 32
train_dataloader = DataLoader(train_dataset, batch_size = batchSize, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batchSize, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batchSize, shuffle = False)

# Import dict to transalte label to card values
label_to_card = {v: k for k, v in ImageFolder(train_folder).class_to_idx.items()}

# Create the structure of the nn
numCards = 53
class CardClassifier(nn.Module):
    def __init__(self, num_classes = numCards):
        super(CardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CardClassifier(num_classes = numCards)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if a saved model exists and load it if available
model_path = "card_classifier.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully!")
else:
    print("No saved model found. Training from scratch...")

    train_losses, val_losses = [], []
    numEpoch = 5
    for epoch in range(numEpoch):

        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc='Training loop'):

            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_dataloader, desc='Validation loop'):

                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(valid_dataloader.dataset)
        val_losses.append(val_loss)


        print(f"Epoch {epoch+1}/{numEpoch} - Train loss: {train_loss}, Validation loss: {val_loss}")

    torch.save(model.state_dict(), "card_classifier.pth")
    print("Model saved successfully!")

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show(block = True)

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device, class_names):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # Don't compute gradients
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            probabilities = predict(model, images, device)
            predicted_indices = probabilities.argmax(axis=1)  # Get the predicted indices

            # Compare predicted labels with true labels
            correct_predictions += (predicted_indices == labels).sum().item()  # Count correct predictions
            total_predictions += labels.size(0)  # Count total samples

    accuracy = correct_predictions / total_predictions * 100  # Percentage accuracy
    return accuracy

# Calculate accuracy on the validation set
val_accuracy = calculate_accuracy(model, valid_dataloader, device, test_dataset.classes)
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Calculate accuracy on the test set
test_accuracy = calculate_accuracy(model, test_dataloader, device, test_dataset.classes)
print(f"Test Accuracy: {test_accuracy:.2f}%")

