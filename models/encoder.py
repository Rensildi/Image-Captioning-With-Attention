import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        # Load pre-trained ResNet-152 model
        resnet = models.resnet152(pretrained=True)
        # Remove the final fully connected layer (classification layer)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # Define a fully connected layer to convert resnet output to embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        # Pass the image through ResNet
        features = self.resnet(images)
        # Flatten the features and pass through the fully connected layer
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        return features
