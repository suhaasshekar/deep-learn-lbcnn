import torch
import torch.nn as nn
import torch.nn.functional as F


# defining the LBCNN architecture
class Net(nn.Module):

    def __init__(self, no_of_lbc_layers):
        super(Net, self).__init__()
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
        self.pool = nn.AvgPool2d((5, 5), stride=5)
        #self.pool = nn.MaxPool2d((226, 226), stride=6)
        self.fc1 = nn.Linear(36, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        for layer in self.lbc_layers:
            x = layer(x)

        x = self.pool(x)
        #x = x.view(-1, 1228800)
        x = x.view(-1, 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#model = Net()
#print(model)
# print(model.layer1.weight)
# print(model.layer1.weight.shape)
# print(model.layer3.weight)

