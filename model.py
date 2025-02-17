# model.py
import torch
import torch.nn as nn
import torchvision.models as models

# Attention mechanism class definition
class Attention(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(Attention, self).__init__()
        # Define a linear layer for combining the hidden state and encoder output
        self.attn = nn.Linear(hidden_size + feature_size, hidden_size)
        # Define a linear layer for calculating the attention weights
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state, encoder_out):
        # Expand the hidden state to match the encoder output size (broadcasting)
        hidden_state = hidden_state.unsqueeze(1).expand(-1, encoder_out.size(1), -1)
        
        # Concatenate the hidden state and encoder output along the feature dimension
        combined = torch.cat((hidden_state, encoder_out), dim=-1)
        
        # Compute the energy term through a tanh activation function
        energy = torch.tanh(self.attn(combined))
        
        # Compute the attention scores using the attention weights
        attention = self.attn_weights(energy)
        
        # Apply softmax to the attention scores to obtain attention weights
        attention = torch.softmax(attention, dim=1)
        
        return attention  # Return the attention weights

# Main Image Captioning Model class definition
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256, feature_size=512):
        super(ImageCaptioningModel, self).__init__()

        # Use a pretrained ResNet18 as the encoder for extracting image features
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final classification layer
        self.encoder_out_size = feature_size  # Size of encoder output (image features)

        # Embedding layer for word indices in the captions
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # LSTM for generating captions (combines image features and previous words)
        self.lstm = nn.LSTM(embedding_size + feature_size, hidden_size, batch_first=True)
        
        # Attention layer for focusing on different parts of the image
        self.attention = Attention(hidden_size, feature_size)
        
        # Fully connected layer for outputting predicted word scores
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # Initialize weights for better training convergence
        self.init_weights()

    def init_weights(self):
        # Initialize weights of linear and embedding layers with Xavier uniform distribution
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):  # Initialize LSTM weights with Xavier
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)

    def forward(self, images, captions):
        # Extract features from the input images using the encoder (ResNet)
        encoder_out = self.encoder(images)
        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # Flatten the features
        encoder_out = encoder_out.unsqueeze(1).expand(-1, captions.size(1), -1)  # Expand for the sequence length

        # Embed the caption words into word vectors
        captions_embed = self.embedding(captions)
        
        # Initialize hidden states and cell states for the LSTM (set to zero initially)
        h, c = (torch.zeros(1, captions.size(0), self.lstm.hidden_size).to(images.device),
                torch.zeros(1, captions.size(0), self.lstm.hidden_size).to(images.device))
        
        # List to store the output at each time step (i.e., predicted word scores)
        outputs = []

        # Iterate through each word in the caption sequence (time step)
        for t in range(captions_embed.size(1)):
            # Compute attention weights based on the current hidden state and image features
            attn_weights = self.attention(h[-1], encoder_out)
            
            # Compute the context vector by taking a weighted sum of the encoder's output
            context = torch.sum(attn_weights * encoder_out, dim=1)
            
            # Concatenate the word embedding at time t and the context vector
            lstm_input = torch.cat((captions_embed[:, t, :], context), dim=1).unsqueeze(1)
            
            # Pass the input through the LSTM to get the next hidden state and output
            out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Pass the LSTM output through a fully connected layer to get word predictions
            out = self.fc_out(out.squeeze(1))
            
            # Append the output to the list of outputs
            outputs.append(out)
        
        # Stack all outputs to form the final tensor of predicted word scores for the sequence
        return torch.stack(outputs, dim=1)