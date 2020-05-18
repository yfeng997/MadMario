import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super(ConvNet, self).__init__()
        c, h, w = input_dim
        self.conv_1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(3136, 512)
        self.output = nn.Linear(512, output_dim)

    def forward(self, input):
        # input: B x C x H x W
        x = input
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.output(x)

        return x
