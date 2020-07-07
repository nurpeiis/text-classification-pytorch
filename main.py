import torch
from models.cnn import CnnText


def train():
    model = CnnText(len(vocab_size), embedding_dim, class_num,
                    num_filters, window_sizes, dropout_prob)
    if torch.cuda.is_available():
        model.cuda()
