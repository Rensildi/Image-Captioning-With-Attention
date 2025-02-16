# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size + feature_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state, encoder_out):
        hidden_state = hidden_state.unsqueeze(1).expand(-1, encoder_out.size(1), -1)
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
        self.lstm = nn.LSTM(embedding_size + feature_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, feature_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        encoder_out = encoder_out.view(encoder_out.size(0), -1)
        encoder_out = encoder_out.unsqueeze(1).expand(-1, captions.size(1), -1)

        captions_embed = self.embedding(captions)
        h, c = (torch.zeros(1, captions.size(0), self.lstm.hidden_size).to(images.device),
                torch.zeros(1, captions.size(0), self.lstm.hidden_size).to(images.device))
        
        outputs = []
        for t in range(captions_embed.size(1)):
            attn_weights = self.attention(h[-1], encoder_out)
            context = torch.sum(attn_weights * encoder_out, dim=1)
            lstm_input = torch.cat((captions_embed[:, t, :], context), dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = self.fc_out(out.squeeze(1))
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)