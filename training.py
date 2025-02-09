import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Initialize NLTK
nltk.download('punkt')

# Vocabulary Class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return word_tokenize(text.lower())

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start indexing from 4 after reserved tokens
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

# Flickr Dataset Class
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])  # Assuming the first column contains image paths
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption = self.df.iloc[idx, 1]  # assuming caption is in the second column
        caption_tokens = nltk.word_tokenize(caption.lower())  # Tokenize caption
        caption_tokens = ['<SOS>'] + caption_tokens + ['<EOS>']  # Add special tokens

        caption_indices = [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"]) for word in caption_tokens]

        return image, torch.tensor(caption_indices)

    def collate_fn(self, batch):
        images, captions = zip(*batch)

        # Stack images (ensure consistent shape)
        images = torch.stack(images, 0)

        # Determine the max length in the batch of captions
        max_caption_len = max([len(caption) for caption in captions])

        # Pad the captions
        padded_captions = torch.full((len(captions), max_caption_len), self.vocab.stoi["<PAD>"], dtype=torch.long)

        for i, caption in enumerate(captions):
            padded_captions[i, :len(caption)] = caption  # Fill up to the caption's actual length

        return images, padded_captions

# Model class with ResNet for feature extraction
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        
        # Use a pre-trained ResNet for feature extraction
        resnet = models.resnet18(pretrained=True)  # or resnet18, resnet50, etc.
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head
        
        # Decoder (LSTM)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Linear layer to predict words
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # Extract features using ResNet
        features = self.resnet(images)  
        features = features.view(features.size(0), -1)  # Flatten to match LSTM input

        # Pass through LSTM decoder
        hiddens, _ = self.decoder(captions)
        
        # Output the predictions
        output = self.fc(hiddens)

        return output

# Define transformations
transform = T.Compose([
    T.Resize((256, 256)),  # Resize all images to 256x256
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization based on ImageNet
])

# Initialize dataset and vocab
dataset = FlickrDataset(root_dir="../Image-Captioning-With-Attention/flickr8k/Images", captions_file="../Image-Captioning-With-Attention/flickr8k/captions.txt", transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)

# Initialize the model
vocab_size = len(dataset.vocab)
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=vocab_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(data_loader, num_epochs=10):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for imgs, captions in data_loader:
            imgs, captions = imgs.to(device), captions.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output = model(imgs, captions[:, :-1])  # exclude <EOS> token from input

            # Calculate loss (Cross-Entropy)
            loss = criterion(output.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))  # ignore <SOS> token
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

# Train model for 10 epochs
train_losses = train_model(data_loader, num_epochs=10)
plot_loss(train_losses)

# Evaluation function
def evaluate_model(data_loader):
    model.eval()
    with torch.no_grad():
        for imgs, captions in data_loader:
            imgs, captions = imgs.to(device), captions.to(device)
            output = model(imgs, captions[:, :-1])  # Get model predictions
            
            # Get predicted words (use the word with the highest probability)
            predicted_ids = torch.argmax(output, dim=2)

            # Convert predicted IDs back to words
            predicted_words = [[dataset.vocab.itos[idx] for idx in caption] for caption in predicted_ids.cpu().numpy()]

            print(predicted_words)  # For now, print predictions

# Create a small data loader for evaluation
eval_data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

# Evaluate model
evaluate_model(eval_data_loader)
