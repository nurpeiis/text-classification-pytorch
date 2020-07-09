import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim


class CnnText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_num, num_filters, window_sizes, dropout_prob, in_channel=1):
        """
            vocab_size : int = vocabulary size
            embedding_dim : int = dimensionality of the embeddings
            class_num : int = number of classes to classify
            num_filters : int = number of filters or out_channels
            window_size : list<int> = window size of each kernel, len(window_sizes) = number of kernels
            dropout_prob : float = 0.0 < p < 1.0, where p is probability of the element to be zeroed
        """
        super(CnnText, self).__init__()

        # Step 1: Initialize variable
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.in_channel = in_channel
        self.num_filters = num_filters
        self.window_sizes = window_sizes
        self.dropout_prob = dropout_prob

        # Step 2: Initialize layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(
            self.in_channel, self.num_filters, [window_size, embedding_dim],
            padding=(window_size-1, 0)) for window_size in self.window_sizes])
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.num_filters *
                            len(window_sizes), self.class_num)

    def conv_block(self, input, conv_layer):
        """ 
        input : list<list<list<list<float>>>> = 4 dimensional input sentences in a batch with size (batch_size, in_channel, num_sequences, embedding_dim)
        conv_layer : nn.Conv2d = convolutional layer that is performing the 
        """
        conv_out = conv_layer(
            input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        # activation.size() = (batch_size, out_channels, dim1)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input):
        """ 
        input : list<list<int>> = list of sentences in batch with shape: (batch_size, num_sequences)
        return logits, 
        """
        embeddings = self.embedding(
            input)  # (batch_size, num_sequences, embedding_dim)
        # input.size() = (batch_size, 1, num_sequences, embedding_length)
        embeddings = embeddings.unsqueeze(1)
        max_outs = []
        for conv in self.convs:
            max_outs.append(self.conv_block(embeddings, conv))
        all_out = torch.cat(max_outs, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size() = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in)

        return logits
