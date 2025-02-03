import numpy as np
import torch
from torch.utils.data import DataLoader ,Dataset
import torchvision.transforms as T

from data_loader import FlickrDataset, get_dataloader

# Initiating the Dataset and Dataloader
data_location = "../input/flickr8k"
BATCH_SIZE = 256
NUM_WORKER = 4

# defining the transofrm to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# testing the dataset class
dataset = FlickrDataset (
    root_dir = data_location+"/Images",
    caption_file = data_location+"/captions.txt",
    transform = transforms
)

# writing the dataloader
data_loader = get_dataloader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True
)


# vocab size
vocab_size = len(dataset.vocab)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Implementing helper function to plot the Tensor image

import matplotlib.pyplot as plt
def show_image(img, title=None):
    '''Imshow for Tensor '''
    
    #unnormalize
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1,2,0))
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause for a moment for the plots to be updated
    
    # Initialiez 