import numpy as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


# defining the LBCNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weights = torch.tensor([[[[0, 1, 0],
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

        self.weights1 = torch.tensor([[[[0, 1, 0],
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
        self.layer1.weight = nn.Parameter(self.weights, requires_grad=False)

        self.layer2 = nn.Conv2d(8, 1, 1, stride=1, padding=0)

        self.layer3 = nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False)
        self.layer3.weight = nn.Parameter(self.weights1, requires_grad=False)

        self.layer4 = nn.Conv2d(8, 1, 1, stride=1, padding=0)

        #self.pool = nn.AvgPool2d((930, 1250), stride=6)
        self.pool = nn.AvgPool2d((226, 226), stride=6)
        #self.pool = nn.MaxPool2d((226, 226), stride=6)
        self.fc1 = nn.Linear(36, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = self.pool(x)
        #x = x.view(-1, 1228800)
        x = x.view(-1, 36)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

#model = Net()
#print(model)
# print(model.layer1.weight)
# print(model.layer1.weight.shape)
# print(model.layer3.weight)

