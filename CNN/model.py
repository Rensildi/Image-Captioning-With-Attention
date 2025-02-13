# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, hidden_size, feature_size, vocab_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size + feature_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)
        self.vocab_size = vocab_size

    def forward(self, hidden_state, encoder_out, captions):
        # Ensure hidden_state has correct dimensions
        hidden_state = hidden_state.squeeze(2)  # Remove the extra dimension if it exists

        # Debugging prints
        print(f"Adjusted hidden_state shape: {hidden_state.shape}")  # (batch_size, hidden_size)
        print(f"encoder_out shape: {encoder_out.shape}")             # (batch_size, feature_size)

        # Ensure the shapes match before concatenation
        hidden_state = hidden_state.unsqueeze(1)  # Add a dimension for sequence length (1 in this case)

        combined = torch.cat((hidden_state, encoder_out), dim=-1)  # Concatenate along the feature axis

        # Compute attention weights
        energy = torch.tanh(self.attn(combined))
        attention = self.attn_weights(energy)
        attention = torch.softmax(attention, dim=1)

        return attention, None


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256, feature_size=512):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = models.resnet18(weights="IMAGENET1K_V1")  # Using ResNet18
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove final fully connected layer
        self.encoder_out_size = feature_size  # Adjusted for ResNet18

        self.embedding = nn.Embedding(vocab_size, embedding_size)  # Caption embedding
        self.lstm = nn.LSTM(embedding_size + self.encoder_out_size, hidden_size)
        
        # Pass vocab_size to Attention
        self.attention = Attention(hidden_size, self.encoder_out_size, vocab_size)
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)  # Output layer for word prediction

    def forward(self, images, captions):
        # Step 1: Extract image features using the encoder (ResNet)
        encoder_out = self.encoder(images)  # [batch_size, feature_size, 1, 1]
        encoder_out = encoder_out.view(encoder_out.size(0), self.encoder_out_size)  # Flatten output

        captions_embed = self.embedding(captions)  # Embed captions

        # Initial hidden state for LSTM (batch size, hidden size)
        h, c = None, None  # Initialize to None outside loop
        outputs = []

        for t in range(captions_embed.size(1)):  # Loop over caption tokens
            # Get attention weights and context vector
            attn_weights, _ = self.attention(h[-1].unsqueeze(1).unsqueeze(2) if h is not None else torch.zeros(1, captions_embed.size(0), self.lstm.hidden_size).to(images.device), 
                                             encoder_out.unsqueeze(1), 
                                             captions[:, t].unsqueeze(1))

            context = torch.sum(attn_weights * encoder_out.unsqueeze(1), dim=1)
            
            # Combine caption embedding and context vector
            lstm_input = torch.cat((captions_embed[:, t, :], context), dim=1).unsqueeze(0)
            
            # Pass input to LSTM and get output
            out, (h, c) = self.lstm(lstm_input, (h, c) if h is not None else (torch.zeros(1, captions_embed.size(0), self.lstm.hidden_size).to(images.device), 
                                                                          torch.zeros(1, captions_embed.size(0), self.lstm.hidden_size).to(images.device)))

            out = self.fc_out(out.squeeze(0))  # Predict the next word
            outputs.append(out)

        # Stack the outputs to get shape [batch_size, sequence_length, vocab_size]
        return torch.stack(outputs, dim=1)
