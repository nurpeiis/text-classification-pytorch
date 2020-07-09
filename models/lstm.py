import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim


class LstmText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_num, hidden_size, dropout_prob):
        """
            vocab_size : int = vocabulary size
            embedding_dim : int = dimensionality of the embeddings
            class_num : int = number of classes to classify
            hidden_size : int = size of hidden state of LSTM
            dropout_prob : float = 0.0 < p < 1.0, where p is probability of the element to be zeroed
        """
        super(LstmText, self).__init__()

        # Step 1: Initialize variable
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # Step 2: Initialize layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, input):
        """
        input : list<list<int>> = list of sentences in batch with shape: (batch_size, num_sequences)
        return logits,
        """
        embeddings = self.embedding(
            input)  # (batch_size, num_sequences, embedding_dim)
        # input.size() = (batch_size, 1, num_sequences, embedding_length)
        # input.size() = (num_sequences,30,  embedding_length)
        embeddings = embeddings.permute(1, 0, 2)
        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(1, 50,  self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, 50,  self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(1, 50,  self.hidden_size))
            c_0 = Variable(torch.zeros(1, 50,  self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(
            embeddings, (h_0, c_0))
        fc_in = self.dropout(output)
        # fc_in.size() = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in)

        return logits
