# train.py
import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import Flickr8KDataset  
from model import ImageCaptioningModel  

from dotenv import load_dotenv
load_dotenv()

# Image preprocessing transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a standard size (256x256)
    transforms.ToTensor(),  # Convert image to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Parameters setup
image_dir = os.getenv("IMAGE_DIR")  
caption_file = os.getenv("CAPTION_FILE")  
batch_size = 16  # Number of samples in each batch
num_epochs = 10  # Number of training epochs 
learning_rate = 0.001  # Learning rate for optimizer

# Load the dataset using the custom dataset class
dataset = Flickr8KDataset(image_dir=image_dir, caption_file=caption_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for batching and shuffling

# Model setup - Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU
model = ImageCaptioningModel(vocab_size=len(dataset.vocab), embedding_size=128, hidden_size=256)  # Instantiate the model
model.to(device)  # Move the model to the selected device (CPU/GPU)

# Loss function and optimizer setup
criterion = nn.CrossEntropyLoss(ignore_index=0)  # CrossEntropyLoss with ignore_index for padding token
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for training the model

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")  
    model.train()  # Set model to training mode (important for layers like dropout)
    total_loss = 0  # Variable to accumulate loss for the epoch

    # Iterate through batches of data
    for i, (imgs, captions, lengths) in enumerate(dataloader):
        imgs, captions = imgs.to(device), captions.to(device)  # Move images and captions to the device
        optimizer.zero_grad()  # Reset gradients from previous batch

        # Forward pass through the model
        outputs = model(imgs, captions)
        
        # Calculate loss by comparing predicted outputs with actual captions
        loss = criterion(outputs.view(-1, len(dataset.vocab)), captions.view(-1))
        total_loss += loss.item()  # Accumulate loss for the epoch

        loss.backward()  # Backpropagate the loss to calculate gradients
        optimizer.step()  # Update model weights using the optimizer

        
        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")
    # Compute average loss for the epoch
    avg_loss = total_loss / len(dataloader)  
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
    
# Function to plot training and test loss graphs
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))  
    plt.plot(train_losses, label='Training Loss') 
    plt.plot(test_losses, label='Test Loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss') 
    plt.title('Training and Test Loss Over Epochs')  
    plt.legend()  
    plt.show()  

# Function to display test set examples along with their predicted and ground truth captions
def display_test_examples(test_loader, model, device):
    model.eval()  # Set model to evaluation mode (important for layers like dropout)
    test_iter = iter(test_loader)  # Get an iterator for the test set
    images, captions, _ = next(test_iter)  # Get a batch of test images and captions
    images = images.to(device)  # Move images to the device

    with torch.no_grad():  # Disable gradient calculation (no backpropagation needed during evaluation)
        outputs = model(images, captions)  # Get model predictions for the images
        predicted_captions = torch.argmax(outputs, dim=-1)  # Get the predicted caption (highest probability)

    # Loop through the images and display the ground truth and predicted captions
    for i in range(len(images)):
        # Convert image tensor to numpy array for displaying
        image = images[i].cpu().permute(1, 2, 0).numpy()
        
        # Convert the ground truth and predicted captions to words using the vocabulary
        ground_truth = ' '.join([word for idx in captions[i].cpu().numpy() if idx in model.vocab])
        predicted = ' '.join([word for idx in predicted_captions[i].cpu().numpy() if idx in model.vocab])
        
        # Print ground truth and predicted captions
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted: {predicted}")
        
        # Display the image
        plt.imshow(image)
        plt.show()

print("Training complete.")  


