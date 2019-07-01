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
                                      [0, 0, 0]]]], dtype=torch.float, requires_grad=False)

        self.layer1 = nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False)
        self.layer1.weight = nn.Parameter(self.weights, requires_grad=False)

        self.layer2 = nn.ReLU()

        self.layer3 = nn.Conv2d(8, 1, 1, padding=0)

        self.layer4 = nn.Conv2d(1, 8, 3, padding=1, bias=False)

        self.first_lbc_layer = nn.Sequential(self.layer1, self.layer2, self.layer3)
        self.lbc_layers = nn.Sequential(self.layer4, self.layer2, self.layer3)

        self.pool = nn.AvgPool2d((930, 1250), stride=6)

        self.fc1 = nn.Linear(36, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.first_lbc_layer(x)

        x = self.lbc_layers(x)

        for i in range(9):
            x = self.lbc_layers(x)
        x = self.pool(x)

        x = x.view(-1, 36)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

# model = Net()
# print(model)
# print(model.layer1.weight)
# print(model.layer1.weight.shape)
# print(model.layer3.weight)

