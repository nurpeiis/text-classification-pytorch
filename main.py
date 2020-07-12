import torch
import csv
import pandas as pd
import torch.optim as optim
from torch import autograd, optim, nn
from torch.utils.data import DataLoader
from models.cnn import CnnText
from models.lstm import LstmText
from data import TextualDataset
from token_to_index import TokenToIndex


def train(model, data):
    dataloader = DataLoader(data, batch_size=50,
                            shuffle=True)

    if torch.cuda.is_available():
        model.cuda()
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_epoch_loss = 0
    total_epoch_acc = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        sequences = batch[0]
        labels = batch[1]
        if torch.cuda.is_available():
            sequences, labels = sequences.cuda(), labels.cuda()

        # Predict
        logits = model(sequences)
        # Backpropagation
        losses = loss_function(logits, labels)
        num_corrects = (torch.max(logits, 1)[1].view(
            labels.size()).data == labels.data).float().sum()
        acc = 100.0 * num_corrects/len(batch[0])
        losses.backward()
        optimizer.step()
        if i % 100 == 0:
            print(
                f'Training Loss: {losses.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += losses.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(dataloader), total_epoch_acc/len(dataloader)


def eval(model, data):
    dataloader = DataLoader(data, batch_size=50,
                            shuffle=True)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_epoch_loss = 0
    total_epoch_acc = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            sequences = batch[0]
            labels = batch[1]
            if torch.cuda.is_available():
                sequences, labels = sequences.cuda(), labels.cuda()

            # Predict
            logits = model(sequences)
            losses = loss_function(logits, labels)

            num_corrects = (torch.max(logits, 1)[1].view(
                labels.size()).data == labels.data).float().sum()
            acc = 100.0 * num_corrects/len(batch[0])
            total_epoch_loss += losses.item()
            total_epoch_acc += acc.item()
    return total_epoch_loss/len(dataloader), total_epoch_acc/len(dataloader)


def main():
    dict_files = []
    output_dict = ''
    train_file = ''
    label_to_id_file = ''
    if torch.cuda.is_available():
        torch.cuda.manual_seed(999)
        torch.cuda.manual_seed_all(999)
    else:
        torch.manual_seed(999)
    token_to_index = TokenToIndex(dict_files, output_dict)
    token_to_index.get_dict()
    labels = pd.read_csv(
        train_file, delimiter='\t', header=0)['dialect_city_id'].unique()
    label_to_id = dict()
    id_to_label = dict()
    counter = 0
    for l in labels:
        if (l in label_to_id) == False:
            label_to_id[l] = counter
            id_to_label[counter] = l
            counter += 1

    output = csv.writer(open(label_to_id_file, "w"))
    for key, val in label_to_id.items():
        output.writerow([key, val])
    data_train = TextualDataset(
        train_file, token_to_index, label_to_id)

    # model = CnnText(len(data_train.vocab), 128, len(data_train.classes)                128, [3, 4, 5], 0.1)
    model = LstmText(len(data_train.vocab), 128,
                     len(data_train.classes), 256, 0.1)
    data_eval = TextualDataset(
        '../Capstone/hierarchical-did/data_processed/madar_shared_task1/MADAR-Corpus-26-dev.tsv', token_to_index, label_to_id)
    data_test = TextualDataset(
        '../Capstone/hierarchical-did/data_processed/madar_shared_task1/MADAR-Corpus-26-test.tsv', token_to_index, label_to_id)

    max_val_accuracy = 0
    for i in range(10):
        print(
            "==================== Epoch: {} ====================".format(i))
        # seed
        train_loss, train_acc = train(model, data_train)
        val_loss, val_acc = eval(model, data_eval)
        test_loss, test_acc = eval(model, data_test)
        if val_acc > max_val_accuracy:
            torch.save(model.state_dict(), 'model.pt')
            max_val_accuracy = val_acc
        print(f'Epoch: {i+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Test. Loss: {test_loss:3f}, Test. Acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
