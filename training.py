import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import matplotlib.pyplot as plt

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

# Define transformations
transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])

# Initialize dataset and vocab
dataset = FlickrDataset(root_dir="./Images", captions_file="./captions.txt", transform=transforms)

# Now you can access dataset.vocab to get vocab for the model
vocab = dataset.vocab
vocab_size = len(vocab)

# Model class
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Image feature extractor (e.g., ResNet-152 or your own CNN)
        self.encoder = nn.Conv2d(3, self.embed_size, kernel_size=3, stride=2, padding=1)

        # Decoder (LSTM)
        self.decoder = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)

        # Linear layer to predict words
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.view(features.size(0), -1)  # Flatten

        hiddens, _ = self.decoder(captions)
        output = self.fc(hiddens)

        return output

# Initialize model
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=vocab_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train function
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

# Create data loader
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# Train model for 10 epochs
train_losses = train_model(data_loader, num_epochs=10)
plot_loss(train_losses)
