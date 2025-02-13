# data_loader.py
import os
import torch
import numpy as np  # Added missing import
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from nltk.tokenize import word_tokenize
from collections import Counter
from PIL import Image

class Flickr8KDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_caption_length = 30):
        self.image_dir = image_dir
        self.captions = self.load_captions(caption_file)
        self.img_ids = list(self.captions.keys())
        self.transform = transform
        self.vocab = self.build_vocab()
        self.max_caption_length = max_caption_length #Max length for captions

    def load_captions(self, caption_file):
        with open(caption_file, 'r') as f:
            captions = f.readlines()

        img_to_captions = {}
        malformed_line_count = 0  # Initialize counter for malformed lines

        for caption in captions:
            parts = caption.strip().split(',')
            if len(parts) != 2:  # Skip malformed lines
                print(f"Skipping malformed line: {caption.strip()}")
                malformed_line_count += 1  # Increment the counter
                continue

            img_id, caption_text = parts
            img_id = img_id.split('.')[0]  # Remove the file extension if present

            if img_id not in img_to_captions:
                img_to_captions[img_id] = []
            img_to_captions[img_id].append(caption_text.strip())

        print(f"Total malformed lines skipped: {malformed_line_count}")  # Print the total count

        if len(img_to_captions) == 0:
            raise ValueError("No valid captions found. Please check the dataset file.")

        return img_to_captions

    def build_vocab(self):
        words = []
        for img_id in self.img_ids:
            for caption in self.captions[img_id]:
                words.extend(word_tokenize(caption.lower()))
        word_counts = Counter(words)
        vocab = {word: idx+4 for idx, (word, _) in enumerate(word_counts.most_common())}  # Starting at index 4
        vocab['<UNK>'] = 0  # Special token for unknown words
        vocab['<START>'] = 1  # Special token for start of sentence
        vocab['<END>'] = 2  # Special token for end of sentence
        vocab['<PAD>'] = 3  # Special token for padding

        # Debugging: Print some tokens and vocab size
        print(f"Vocab size: {len(vocab)}")
        print(f"Sample vocab entries: {list(vocab.items())[:10]}")  # Print the first 10 words in vocab

        return vocab


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        captions = self.captions[img_id]
        caption = captions[np.random.randint(len(captions))]

        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        caption_tokens = word_tokenize(caption.lower())
        caption_tokens = ['<START>'] + caption_tokens + ['<END>']
        
        # Truncate caption to max length if needed
        caption_tokens = caption_tokens[:self.max_caption_length]
        
        # Pad the caption with <PAD> tokens if it's shorter than max length
        padding_needed = self.max_caption_length - len(caption_tokens)
        caption_tokens += ['<PAD>'] * padding_needed
        
        caption_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in caption_tokens]

        # Ensure no indices exceed vocab size
        assert all(0 <= idx < len(self.vocab) for idx in caption_indices), \
            f"Caption contains an index out of range! Indices: {caption_indices}, Vocab size: {len(self.vocab)}"
        
        return img, torch.tensor(caption_indices), len(caption_indices)

