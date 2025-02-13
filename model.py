import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size + feature_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state, encoder_out):
        # Compute attention weights
        combined = torch.cat((hidden_state, encoder_out), dim=-1)
        energy = torch.tanh(self.attn(combined))
        attention = self.attn_weights(energy)
        attention = torch.softmax(attention, dim=1)
        return attention

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256, feature_size=512):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.encoder_out_size = feature_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + feature_size, hidden_size)
        self.attention = Attention(hidden_size, feature_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)  # [batch_size, feature_size, 1, 1]
        encoder_out = encoder_out.squeeze(2).squeeze(2)  # [batch_size, feature_size]
        encoder_out = encoder_out.unsqueeze(1).expand(-1, captions.size(1), -1)  # [batch_size, seq_len, feature_size]

        captions_embed = self.embedding(captions)

        h, c = None, None
        outputs = []

        for t in range(captions_embed.size(1)):
            hidden_state = h[-1].squeeze(0) if h is not None else torch.zeros(captions.size(0), self.lstm.hidden_size).to(images.device)
            hidden_state = hidden_state.unsqueeze(1).expand(-1, captions.size(1), -1)  # [batch_size, seq_len, hidden_size]

            attn_weights = self.attention(hidden_state, encoder_out)
            context = torch.sum(attn_weights * encoder_out, dim=1)

            lstm_input = torch.cat((captions_embed[:, t, :], context), dim=1).unsqueeze(0)
            out, (h, c) = self.lstm(lstm_input, (h, c) if h is not None else (
                torch.zeros(1, captions_embed.size(0), self.lstm.hidden_size).to(images.device),
                torch.zeros(1, captions_embed.size(0), self.lstm.hidden_size).to(images.device)))

            out = self.fc_out(out.squeeze(0))
            outputs.append(out)

        return torch.stack(outputs, dim=1)