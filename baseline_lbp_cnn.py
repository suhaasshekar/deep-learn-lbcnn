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
from skimage import feature
import matplotlib.pyplot as plt
import cnn
import torch.optim as optim
import gc
import plotter
import cv2

def load_image(image_path):
    img = Image.open(image_path)
    img = demosaic(img, 'gbrg')
    cvimage = np.array(img, dtype=np.uint8)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_RGB2BGR)
    cvgray = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(cvgray, 8, 1, method='default')
    im = Image.fromarray(np.uint8(lbp))
    return im

def sampler_(labels, counts_array):
    counts = np.array([len(np.where(labels == t)[0]) for t in counts_array])
    weights = torch.tensor([1.0 / x if x else 0 for x in counts], dtype=torch.float)
    sample_weights = np.array([weights[t] for t in labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=False)
    return sampler

def train_n_test(model_file, output_file, learning_rate=0.01, data_size=5000, no_of_lbc_layers=10, epochs=100):
    global graph
    graph = plotter.VisdomLinePlotter(env_name='Baseline Plots')
    outfile = open(output_file, "a")
    print("Learning rate=%f Data size=%d No. of LBCs=%d Epochs=%d\n" % (
        learning_rate, data_size, no_of_lbc_layers, epochs
    ))
    outfile.write("Learning rate=%f Data size=%d No. of LBCs=%d Epochs=%d\n" % (
        learning_rate, data_size, no_of_lbc_layers, epochs
    ))
    outfile.flush()
    #check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    #train_on_gpu = False

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        outfile.write('CUDA is not available.  Training on CPU ...')
        outfile.flush()
    else:
        print('CUDA is available!  Training on GPU ...')
        outfile.write('CUDA is available!  Training on GPU ...')
        outfile.flush()

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
    train_idx, valid_idx = indices[:250], indices[250:300]
    #train_idx, valid_idx = indices[:int(data_size * 0.8)], indices[int(data_size * 0.8):data_size]
    train_labels = [labels[x] for x in train_idx]
    #print(train_labels)
    val_labels = [labels[x] for x in valid_idx]
    train_sampler, valid_sampler = sampler_(train_labels, counts_array), sampler_(val_labels, counts_array)

    test_indices = list(range(len(test_data)))
    np.random.shuffle(test_indices)
    test_idx = test_indices[:50]
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
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
    model = cnn.CNNNet()

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=1.0, dampening=0, nesterov=True)
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
        graph.plot('Loss', 'Train', 'Class Loss', epoch, train_loss)
        graph.plot('Loss', 'Val', 'Class Loss', epoch, valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        outfile.write('\n\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
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
            torch.save(model.state_dict(), model_file)
            valid_loss_min = valid_loss

        # Calculating Train Accuracy
        for i in range(no_of_classes):
            if class_total[i] > 0:
                print('Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                outfile.write('\nTrain Accuracy of %5s: %2d%% (%2d/%2d)' % (
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
                outfile.write('\nValidation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * val_class_correct[i] / val_class_total[i],
                    np.sum(val_class_correct[i]), np.sum(val_class_total[i])))
                outfile.flush()
            else:
                print('Val Accuracy of %5s: N/A (no training examples)' % (classes[i]))
                outfile.write('Val Accuracy of %5s: N/A (no training examples)' % (classes[i]))
                outfile.flush()


        print('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        outfile.write('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        outfile.flush()

        print('Validation Accuracy (Overall): %2d%% (%2d/%2d)\n' % (
            100. * np.sum(val_class_correct) / np.sum(val_class_total),
            np.sum(val_class_correct), np.sum(val_class_total)))
        outfile.write('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(val_class_correct) / np.sum(val_class_total),
            np.sum(val_class_correct), np.sum(val_class_total)))
        outfile.flush()
        graph.plot('Accuracy', 'Val', 'Accuracy', epoch, 100. * np.sum(val_class_correct) / np.sum(val_class_total))

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(no_of_classes))
    class_total = list(0. for i in range(no_of_classes))

    model.load_state_dict(torch.load(model_file))
    print('Load saved model for testing...')
    print('Testing the model..')
    outfile.write('\nLoad saved model for testing...\nTesting the model..')
    outfile.flush()
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
    outfile.write('\nTest Loss: {:.6f}'.format(test_loss))
    outfile.flush()

    for i in range(no_of_classes):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            outfile.write('\nTest Accuracy of %5s: %2d%% (%2d/%2d)' % (
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

train_n_test(model_file='baseline.pt', output_file='baseline.txt', learning_rate=0.01, data_size=1000,
                         no_of_lbc_layers=3, epochs=100)
