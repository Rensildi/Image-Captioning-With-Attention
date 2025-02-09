import os
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import torchvision.transforms as T
import torch


'''
    Load the cpations into a DataFrame, 
    where each row corresponds 
    to an image and its caption
'''


# Ensure tokenization data is available
nltk.download('punkt')

# Dataset location
data_location = "../Image-Captioning-With-Attention/flickr8k"

# Read captions file
caption_file = os.path.join(data_location, 'captions.txt')
df = pd.read_csv(caption_file, delimiter='\t', names=["image", "caption"])

# Show a preview
print("There are {} images to captions.".format(len(df)))
print(df.head())


'''
    Vocabulary class builds a vocabulary based 
    on a frequency threshold, which helps filter
    out rare words.
    
    The vocabulary assigns an index to each word and 
    allows to convert words into their corresponding indicies
    and vice versa
    
'''

# Vocabulary Class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # reserved tokens
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return word_tokenize(text.lower())  # Tokenize using NLTK

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start indexing from 4 for actual words
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

# Test Vocabulary creation
vocab = Vocabulary(freq_threshold=5)  # Words must appear at least 5 times
vocab.build_vocab(df['caption'].tolist())
print(f"Vocabulary Size: {len(vocab)}")
print(f"Vocabulary Example: {vocab.stoi}")

# Preprocessing images (Resizing & Normalizing)
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),  # Convert image to Tensor, scales pixel values to [0, 1]
])

# Show a sample image and caption
sample_idx = 0
image_path = os.path.join(data_location, "Images", df.iloc[sample_idx, 0])
img = Image.open(image_path)
img = transform(img)

# Display sample image
import matplotlib.pyplot as plt
plt.imshow(img.permute(1, 2, 0))  # Convert Tensor to HxWxC format
plt.title(f"Sample Caption: {df.iloc[sample_idx, 1]}")
plt.show()
