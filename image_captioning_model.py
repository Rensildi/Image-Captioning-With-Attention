# models/image_captioning_model.py

import torch
import torch.nn as nn
import torchvision.models as models


# Encoder Class
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        return features


# Attention Class (Bahdanau Attention)
class Attention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(embed_size + hidden_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden):
        combined = torch.cat((features, hidden), dim=1)
        attention_scores = self.attn(combined)
        attention_weights = torch.sigmoid(self.attn_weights(attention_scores))
        return attention_weights


# Decoder Class with Attention
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention):
        super(DecoderWithAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.attention = attention
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions, hidden=None):
        embedded = self.embed(captions)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attn_weights = self.attention(features, lstm_out)
        context_vector = attn_weights * features
        output = self.fc_out(lstm_out + context_vector)
        output = self.softmax(output)
        return output, hidden


# Full Image Captioning Model with Encoder, Decoder, and Attention
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(embed_size)
        self.attention = Attention(embed_size, hidden_size)
        self.decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size, self.attention)

    def forward(self, images, captions, hidden=None):
        features = self.encoder(images)
        output, hidden = self.decoder(features, captions, hidden)
        return output, hidden
