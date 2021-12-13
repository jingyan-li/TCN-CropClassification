'''
Model for crop classification using TCN
Add a FCL for classification
'''
import torch.nn.functional as F
from torch import nn
from tcn_models.TCN import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        '''

        :param input_size: integer, number of input channels
        :param output_size: integer, number of output classes
        :param num_channels: array of integer, number of output channels per level
        :param kernel_size:
        :param dropout:
        '''
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # Classification
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)