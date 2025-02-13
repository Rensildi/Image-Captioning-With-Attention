# utils.py
from nltk.tokenize import word_tokenize
from collections import Counter

# Tokenization function (can be reused if needed elsewhere)
def build_vocab(captions):
    words = []
    for caption in captions:
        words.extend(word_tokenize(caption.lower()))
    word_counts = Counter(words)
    vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.most_common())}
    vocab['<UNK>'] = len(vocab) + 1  # Add unknown token
    vocab['<START>'] = len(vocab) + 2  # Add start token
    vocab['<END>'] = len(vocab) + 3    # Add end token
    return vocab
