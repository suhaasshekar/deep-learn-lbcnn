#
# Suhaas Shekar
# 118220499

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import matplotlib.pyplot as plt

def load_image(image_path):
    img = Image.open(image_path)
    img = demosaic(img, 'gbrg')
    im = Image.fromarray(np.uint8(img))
    return im

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# number of subprocesses to use for data loading
num_workers = 5
# how many samples per batch to load
batch_size = 100
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

DATA_PATH_TRAIN = Path("/home/ss20/sample-data")

train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=transform, loader=load_image)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# obtain training indices that will be used for validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    print(img.shape)
    #npimg = img.numpy()
    plt.imshow(np.transpose(img[9].numpy(), (1, 2, 0)))
    plt.show()

def main():
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(images)
    print(labels[9])

if __name__ == "__main__":
    main()














