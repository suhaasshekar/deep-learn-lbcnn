#
# Suhaas Shekar
# 118220499


import torch.nn.functional as F
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
import torch.optim as optim
import gc
# defining the LBCNN architecture
class Scratch(nn.Module):

    def __init__(self, no_of_lbc_layers):
        super(Scratch, self).__init__()
        self.weights_3_channel = torch.tensor([[[[0, 1, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 1, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 1, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]]],

                                    [[[0, 0, 1],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 1],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 1],
                                      [0, -1, 0],
                                      [0, 0, 0]]],

                                    [[[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 1],
                                      [0, 0, 0]]],

                                    [[[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]]],

                                    [[[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 1, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 1, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [0, 1, 0]]],

                                    [[[0, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0]],
                                     [[0, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0]]],

                                    [[[0, 0, 0],
                                      [1, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [1, -1, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [1, -1, 0],
                                      [0, 0, 0]]],

                                    [[[1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]],
                                     [[1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 0]]]], dtype=torch.float)

        self.weights_1_channel = torch.tensor([[[[0, 1, 0],
                                        [0, -1, 0],
                                        [0, 0, 0]]],


                                      [[[0, 0, 1],
                                        [0, -1, 0],
                                        [0, 0, 0]]],


                                      [[[0, 0, 0],
                                        [0, -1, 1],
                                        [0, 0, 0]]],


                                      [[[0, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]]],


                                      [[[0, 0, 0],
                                        [0, -1, 0],
                                        [0, 1, 0]]],


                                      [[[0, 0, 0],
                                        [0, -1, 0],
                                        [1, 0, 0]]],


                                      [[[0, 0, 0],
                                        [1, -1, 0],
                                        [0, 0, 0]]],


                                      [[[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 0]]]], dtype=torch.float)


        self.layer1 = nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False)
        self.layer1.weight = nn.Parameter(self.weights_3_channel, requires_grad=True)
        self.layer2 = nn.Conv2d(8, 1, 1, stride=1, padding=0)

        self.lbc_layers = nn.ModuleList()
        self.conv = nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False)
        self.conv.weight = nn.Parameter(self.weights_1_channel, requires_grad=True)
        for i in range(no_of_lbc_layers - 1):
            self.lbc_layers.append(self.conv)
            self.lbc_layers.append(nn.ReLU())
            self.lbc_layers.append(nn.Conv2d(8, 1, 1, stride=1, padding=0))

        #self.pool = nn.AvgPool2d((930, 1250), stride=6)
        self.pool = nn.AvgPool2d((226, 226), stride=6)
        #self.pool = nn.MaxPool2d((226, 226), stride=6)
        self.fc1 = nn.Linear(36, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print('before network',x)
        x = F.relu(self.layer1(x))
        #print('after first', x)
        x = self.layer2(x)
        for layer in self.lbc_layers:
            x = layer(x)

        x = self.pool(x)
        #x = x.view(-1, 1228800)
        x = x.view(-1, 36)
        #print('after flatten', x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        print('x',x)

        return x

def load_image(image_path):
    img = Image.open(image_path)
    img = demosaic(img, 'gbrg')
    im = Image.fromarray(np.uint8(img))
    return im

def sampler_(labels, counts_array):
    counts = np.array([len(np.where(labels == t)[0]) for t in counts_array])
    weights = torch.tensor([1.0 / x if x else 0 for x in counts], dtype=torch.float)
    sample_weights = np.array([weights[t] for t in labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=False)
    return sampler

def train_n_test(model_file, output_file, learning_rate=0.01, data_size=5000, no_of_lbc_layers=10, epochs=100):

    print("Learning rate=%f Data size=%d No. of LBCs=%d Epochs=%d\n" % (
        learning_rate, data_size, no_of_lbc_layers, epochs
    ))

    #check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    #train_on_gpu = False

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')

    else:
        print('CUDA is available!  Training on GPU ...')

    # number of subprocesses to use for data loading
    num_workers = 10
    # how many samples per batch to load
    batch_size = 10
    # percentage of training set to use as validation
    #valid_size = 0.2

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
    #test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # obtain training indices that will be used for validation
    labels = [x[1] for x in train_data.samples]
    no_of_classes =  len(train_data.classes)
    counts_array = np.array([x for x in range(no_of_classes)])
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    #split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[:10], indices[10:20]
    #train_idx, valid_idx = indices[:int(data_size * 0.8)], indices[int(data_size * 0.8):data_size]
    train_labels = [labels[x] for x in train_idx]
    #print(train_labels)
    val_labels = [labels[x] for x in valid_idx]
    train_sampler, valid_sampler = sampler_(train_labels, counts_array), sampler_(val_labels, counts_array)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(Subset(train_data, train_idx), batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(Subset(train_data, valid_idx), batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    # specify the image classes
    classes = train_data.classes

    # create a complete CNN
    model = Scratch(1)
    #print(model)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # number of epochs to train the model
    n_epochs = epochs

    valid_loss_min = np.Inf  # track change in validation loss

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
        counter = 0
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            print('target',target)
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
            print("loss", loss)
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

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))

            torch.save(model.state_dict(), model_file)
            valid_loss_min = valid_loss

        # # Calculating Train Accuracy
        # for i in range(no_of_classes):
        #     if class_total[i] > 0:
        #         print('Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
        #             classes[i], 100 * class_correct[i] / class_total[i],
        #             np.sum(class_correct[i]), np.sum(class_total[i])))
        #
        #     else:
        #         print('Train Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        #
        #
        # # Calculating Validation Accuracy
        # for i in range(no_of_classes):
        #     if val_class_total[i] > 0:
        #         print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
        #             classes[i], 100 * val_class_correct[i] / val_class_total[i],
        #             np.sum(val_class_correct[i]), np.sum(val_class_total[i])))
        #
        #     else:
        #         print('Val Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        #
        #
        # print('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        #     100. * np.sum(class_correct) / np.sum(class_total),
        #     np.sum(class_correct), np.sum(class_total)))
        # print('Validation Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        #     100. * np.sum(val_class_correct) / np.sum(val_class_total),
        #     np.sum(val_class_correct), np.sum(val_class_total)))

train_n_test('scratch.pt', None, learning_rate=0.01, data_size=2, no_of_lbc_layers=10, epochs=40)