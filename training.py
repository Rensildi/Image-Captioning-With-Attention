# training.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Initialize the model
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab))

# Choose the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss Function (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Procedure
def train_model(data_loader, num_epochs=10):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for imgs, captions in data_loader:
            imgs, captions = imgs.to(device), captions.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output, _ = model(imgs, captions[:, :-1])  # exclude <EOS> token from input

            # Calculate loss (Cross-Entropy)
            loss = criterion(output.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))  # ignore <SOS> token
            loss.backward()

            # Update model weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(data_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return train_losses


# Plot training loss
def plot_loss(train_losses):
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

# Assuming you have a DataLoader `data_loader`
# Train the model for 10 epochs and plot the training loss
train_losses = train_model(data_loader, num_epochs=10)
plot_loss(train_losses)
