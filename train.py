#
# Suhaas Shekar
# 118220499

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

def sampler_(labels):
    counts = np.array([len(np.where(labels == t)[0]) for t in counts_array])
    weights = torch.tensor([1.0 / x if x else 0 for x in counts], dtype=torch.float)
    sample_weights = np.array([weights[t] for t in labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
#train_on_gpu = False

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# number of subprocesses to use for data loading
num_workers = 10
# how many samples per batch to load
batch_size = 100
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

DATA_PATH_TRAIN = Path("/home/ss20/dataset")
DATA_PATH_TEST = Path("/home/ss20/sample-data")

train_data = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=transform, loader=load_image)
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_data = datasets.ImageFolder(root=DATA_PATH_TEST, transform=transform, loader=load_image)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# obtain training indices that will be used for validation
labels = [x[1] for x in train_data.samples]
no_of_classes =  len(train_data.classes)
counts_array = np.array([x for x in range(no_of_classes)])
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
#train_idx, valid_idx = indices[split:], indices[:split]
train_idx, valid_idx = indices[:400], indices[400:500]
train_labels = [labels[x] for x in train_idx]
#print(train_labels)
val_labels = [labels[x] for x in valid_idx]
train_sampler, valid_sampler = sampler_(train_labels), sampler_(val_labels)

# define samplers for obtaining training and validation batches
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(Subset(train_data, train_idx), batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(Subset(train_data, valid_idx), batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# specify the image classes
classes = train_data.classes

# create a complete CNN
model = lbcnn.Net()
#print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# number of epochs to train the model
n_epochs = 1

valid_loss_min = np.Inf  # track change in validation loss

outfile = open("out.txt", "a")

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    val_class_correct = list(0. for i in range(len(classes)))
    val_class_total = list(0. for i in range(len(classes)))

    ###################
    # train the model #
    ###################
    model.train()
    print("Training the model..")
    outfile.write("Training the model..")
    outfile.flush()
    counter = 0
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        #print(target)
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)
        #print("output", output)
        counter += batch_size
        #print("Trained - %d" % counter)

        # calculate the batch loss
        loss = criterion(output, target)
        #print("loss", loss)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        #print("loss.item", loss.item(), data.size(0))
        train_loss += loss.item() * data.size(0)

        # convert output probabilites to predicted class
        _, pred = torch.max(output, 1)
        #print("torch.max", torch.max(output, 1))
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        #print(target)
        #print(pred)
        #print("correct tensor, pred.eq", correct_tensor, target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        #print("correct", correct)
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        del data, target, pred, correct, correct_tensor
        torch.cuda.empty_cache()
        gc.collect()

    ######################
    # validate the model #
    ######################
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                val_class_correct[label] += correct[i].item()
                val_class_total[label] += 1

            del data, target, pred, correct, correct_tensor
            torch.cuda.empty_cache()
            gc.collect()

    # calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    outfile.write('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    outfile.flush()

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        outfile.write('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        outfile.flush()
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

    # Calculating Train Accuracy
    for i in range(no_of_classes):
        if class_total[i] > 0:
            print('Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            outfile.write('Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            outfile.flush()
        else:
            print('Train Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            outfile.write('Train Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            outfile.flush()

    # Calculating Validation Accuracy
    for i in range(no_of_classes):
        if val_class_total[i] > 0:
            print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * val_class_correct[i] / val_class_total[i],
                np.sum(val_class_correct[i]), np.sum(val_class_total[i])))
            outfile.write('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * val_class_correct[i] / val_class_total[i],
                np.sum(val_class_correct[i]), np.sum(val_class_total[i])))
            outfile.flush()
        else:
            print('Val Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            outfile.write('Val Accuracy of %5s: N/A (no training examples)' % (classes[i]))
            outfile.flush()


    print('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    outfile.write('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    outfile.flush()

    print('Validation Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        100. * np.sum(val_class_correct) / np.sum(val_class_total),
        np.sum(val_class_correct), np.sum(val_class_total)))
    outfile.write('Validation Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        100. * np.sum(val_class_correct) / np.sum(val_class_total),
        np.sum(val_class_correct), np.sum(val_class_total)))
    outfile.flush()

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.load_state_dict(torch.load('model_cifar.pt'))
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
        test_loss += loss.item()*data.size(0)
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
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
outfile.write('Test Loss: {:.6f}\n'.format(test_loss))
outfile.flush()

for i in range(no_of_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
        outfile.write('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
        outfile.flush()
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        outfile.write('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        outfile.flush()

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
outfile.write('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
outfile.flush()

outfile.close()
