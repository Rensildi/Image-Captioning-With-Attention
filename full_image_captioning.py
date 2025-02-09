# full_image_captioning.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# Initialize NLTK tokenizer
nltk.download('punkt')

# Define Vocabulary class
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

# Define dataset class for loading Flickr8k dataset
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
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)

# Define image transformations for resizing and normalization
transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# Define the Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        
        # Encoder: CNN using ResNet (feature extraction)
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        
        # Decoder: LSTM with attention mechanism
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size + 2048, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        batch_size = images.size(0)

        # Extract features from images using CNN (ResNet)
        image_features = self.resnet(images)

        # Initialize LSTM hidden state
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)

        # Embedding of captions
        captions = self.embed(captions)

        # Attention mechanism & LSTM
        outputs = []
        for t in range(captions.size(1)):
            caption_t = captions[:, t].unsqueeze(1)
            attn_input = torch.cat((caption_t, image_features.unsqueeze(1).repeat(1, 1, 1)), dim=-1)
            attn_weights = torch.softmax(self.attn(attn_input), dim=1)
            context = (attn_weights * image_features.unsqueeze(1)).sum(dim=1)

            # Concatenate the context vector and caption embedding
            lstm_input = torch.cat((caption_t, context.unsqueeze(1)), dim=-1)
            out, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            out = self.fc(out.squeeze(1))
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Training Algorithm with loss function and optimizer
def train_model(data_loader, model, vocab, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()

        for imgs, captions in data_loader:
            imgs, captions = imgs.to(device), captions.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output = model(imgs, captions[:, :-1])  # exclude <EOS> token from input

            # Calculate loss
            loss = criterion(output.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))  # exclude <SOS> token
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(data_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return train_losses

# Function to plot training loss
def plot_loss(train_losses):
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

# Load the dataset
data_location = "../Image-Captioning-With-Attention/flickr8k"
dataset = FlickrDataset(root_dir=data_location + "/Images", captions_file=data_location + "/captions.txt", transform=transforms)

# DataLoader
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(dataset.vocab))

# Train the model
train_losses = train_model(data_loader, model, dataset.vocab, num_epochs=10)

# Plot training loss
plot_loss(train_losses)
