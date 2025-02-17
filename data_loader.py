# data_loader.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nltk.tokenize import word_tokenize
from collections import Counter
from PIL import Image

# Define the custom dataset class for the Flickr8K dataset
class Flickr8KDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_caption_length=30):
        """
        Initialize the dataset.

        Parameters:
        - image_dir (str): Directory containing the images.
        - caption_file (str): Path to the caption file.
        - transform (callable, optional): Optional transform to be applied on an image.
        - max_caption_length (int, optional): Maximum length of the captions.
        """
        self.image_dir = image_dir
        self.captions = self.load_captions(caption_file)  # Load captions from the file
        self.img_ids = list(self.captions.keys())  # Get the list of image IDs
        self.transform = transform  # Set image transformation (if any)
        self.vocab = self.build_vocab()  # Build vocabulary from captions
        self.max_caption_length = max_caption_length  # Set max caption length

    def load_captions(self, caption_file):
        """
        Load captions from a file and organize them by image ID.

        Parameters:
        - caption_file (str): Path to the caption file.

        Returns:
        - dict: Mapping from image ID to list of captions.
        """
        with open(caption_file, 'r') as f:
            captions = f.readlines()  # Read all lines in the caption file

        img_to_captions = {}  # Dictionary to store captions by image ID
        malformed_line_count = 0  # Counter for malformed lines

        # Process each caption line
        for caption in captions:
            parts = caption.strip().split(',')
            if len(parts) != 2:  # Skip malformed lines (those not containing image ID and caption)
                print(f"Skipping malformed line: {caption.strip()}")
                malformed_line_count += 1
                continue

            img_id, caption_text = parts
            img_id = img_id.split('.')[0]  # Remove file extension from image ID

            # Add caption to the corresponding image ID
            if img_id not in img_to_captions:
                img_to_captions[img_id] = []
            img_to_captions[img_id].append(caption_text.strip())

        print(f"Total malformed lines skipped: {malformed_line_count}")

        if len(img_to_captions) == 0:  # Ensure there are valid captions
            raise ValueError("No valid captions found. Please check the dataset file.")

        return img_to_captions  # Return the dictionary of captions

    def build_vocab(self):
        """
        Build vocabulary from all captions in the dataset.

        Returns:
        - dict: Vocabulary mapping words to indices.
        """
        words = []  # List to store all words from captions
        for img_id in self.img_ids:
            for caption in self.captions[img_id]:
                words.extend(word_tokenize(caption.lower()))  # Tokenize and add words to the list

        word_counts = Counter(words)  # Count the frequency of each word
        # Create a vocabulary mapping word to index (starting from index 4)
        vocab = {word: idx+4 for idx, (word, _) in enumerate(word_counts.most_common())}
        
        # Special tokens: <UNK> (unknown), <START>, <END>, <PAD>
        vocab['<UNK>'] = 0
        vocab['<START>'] = 1
        vocab['<END>'] = 2
        vocab['<PAD>'] = 3

        print(f"Vocab size: {len(vocab)}")  # Print vocabulary size
        print(f"Sample vocab entries: {list(vocab.items())[:10]}")  
        return vocab 

    def __len__(self):
        """
        Return the total number of images (samples) in the dataset.

        Returns:
        - int: Number of image samples.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieve a sample (image, caption) from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (image, caption_indices, caption_length)
            - image (tensor): The transformed image.
            - caption_indices (tensor): Indices of words in the caption.
            - caption_length (int): The length of the caption (excluding padding).
        """
        img_id = self.img_ids[idx]  # Get the image ID for the given index
        captions = self.captions[img_id]  # Get all captions for this image
        caption = captions[np.random.randint(len(captions))]  # Select a random caption for the image

        img_path = os.path.join(self.image_dir, img_id + '.jpg')  # Construct the image path
        img = Image.open(img_path).convert("RGB")  # Open and convert the image to RGB

        # Apply transformations to the image (if any)
        if self.transform:
            img = self.transform(img)

        # Tokenize the caption and add <START> and <END> tokens
        caption_tokens = word_tokenize(caption.lower())
        caption_tokens = ['<START>'] + caption_tokens + ['<END>']
        
        # Truncate the caption if it's too long, or pad it if it's too short
        caption_tokens = caption_tokens[:self.max_caption_length]
        padding_needed = self.max_caption_length - len(caption_tokens)
        caption_tokens += ['<PAD>'] * padding_needed  # Add padding tokens

        # Convert tokens to indices using the vocabulary
        caption_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in caption_tokens]

        # Assert that all indices are within the vocabulary range
        assert all(0 <= idx < len(self.vocab) for idx in caption_indices), (
            f"Caption contains an index out of range! Indices: {caption_indices}, Vocab size: {len(self.vocab)}"
        )

        # Return the image, caption indices, and caption length
        return img, torch.tensor(caption_indices), len(caption_indices)

