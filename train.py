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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Parameters
image_dir = os.getenv("IMAGE_DIR")
caption_file = os.getenv("CAPTION_FILE")
batch_size = 8
num_epochs = 10
learning_rate = 0.001

dataset = Flickr8KDataset(image_dir=image_dir, caption_file=caption_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
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
        
        outputs = model(imgs, captions)
        loss = criterion(outputs.view(-1, len(dataset.vocab)), captions.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

print("Training complete.")


