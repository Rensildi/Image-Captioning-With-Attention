# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data_loader import Flickr8KDataset
from model import ImageCaptioningModel

# Image preprocessing transformation (move here)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Parameters
# Update paths to match your dataset location
image_dir = "/wsu/home/hh/hh27/hh2781/ondemand/Image-Captioning-With-Attention/flickr8k/Images/"
image_files = os.listdir(image_dir)
print(f"Image files: {image_files[:5]}")  # Print the first 5 images in the directory
caption_file = "/wsu/home/hh/hh27/hh2781/ondemand/Image-Captioning-With-Attention/flickr8k/captions.txt"

batch_size = 8  # Reduced from 32 to 8
num_epochs = 10
learning_rate = 0.001

# Dataset and DataLoader
dataset = Flickr8KDataset(image_dir=image_dir, caption_file=caption_file, transform=transform)
# Debugging: print the first few image IDs and captions
for img_id in list(dataset.captions.keys())[:5]:  # print the first 5 image IDs
    print(f"Image ID: {img_id}, Captions: {dataset.captions[img_id]}")
    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
print(f"Vocab size: {len(dataset.vocab)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageCaptioningModel(vocab_size=len(dataset.vocab), embedding_size=128, hidden_size=256)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()

    total_loss = 0
    for i, (imgs, captions, lengths) in enumerate(dataloader):
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()

        print(f"Batch {i}: Training on {len(imgs)} images")  
        print(f"Batch {i}: Captions shape: {captions.shape}")
        
        # Debugging: Print the actual caption indices
        print(f"Caption indices in batch {i}: {captions}")
        

        # Ensure captions are within vocab size
        min_index = captions.min().item()
        max_index = captions.max().item()
        
        print(f"Caption indices - min: {min_index}, max: {max_index}")
        
        assert min_index >= 0 and max_index < len(dataset.vocab), \
            f"Caption contains an index out of range of vocab! Min index: {min_index}, Max index: {max_index}, Vocab size: {len(dataset.vocab)}"

        outputs = model(imgs, captions)

        loss = criterion(outputs.view(-1, len(dataset.vocab)), captions.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

print("Training complete.")


