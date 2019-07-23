import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import matplotlib.pyplot as plt
import lbcnn
import torch.optim as optim
import gc

def load_image(image_path):
    img = Image.open(image_path)
    img = demosaic(img, 'gbrg')
    im = Image.fromarray(np.uint8(img))
    return im

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
# train_on_gpu = False

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')

else:
    print('CUDA is available!  Training on GPU ...')

# number of subprocesses to use for data loading
num_workers = 10
# how many samples per batch to load
batch_size = 50
# percentage of training set to use as validation
#valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATA_PATH_TEST = Path("/home/ss20/sample-data")

test_data = datasets.ImageFolder(root=DATA_PATH_TEST, transform=transform, loader=load_image)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# create a complete CNN
model = lbcnn.Net(5)
if train_on_gpu:
    model.cuda()
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

no_of_classes = 2
classes = ['rain', 'sunny']
# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model_file = 'model_0.01_20000_5_200.pt'

model.load_state_dict(torch.load(model_file))
print('Load saved model for testing...')
print('Testing the model..')
model.eval()
with torch.no_grad():
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# average test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(no_of_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
