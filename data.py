import ast
import torch
import pandas as pd
from torch.utils.data import Dataset
from token_to_index import TokenToIndex


class TextualDataset(Dataset):

    def __init__(self, filename, token_to_index, label_to_id):
        """
            filename : string = input file
            token_to_index : list<string> = files from which dictionary is going to be create
            output_dict_file : string = file to which dictionary is going to be saved
        """
        self.dataset = pd.read_csv(filename, delimiter='\t', header=0)
        tokenized_sentences = self.dataset['tokenized']
        sentences = self.dataset['original_sentence']
        self.labels = self.dataset['dialect_city_id']
        self.classes = self.labels.unique()
        self.samples = []
        self.vocab = token_to_index.character_to_id_vocab
        self.sent_size = 30
        self.char_seq_size = 120
        """
        for i in range(len(tokenized_sentences)):
            indexed_sentence = [n.strip()
                                for n in ast.literal_eval(tokenized_sentences[i])]
            for j in range(len(indexed_sentence)):
                indexed_sentence[j] = token_to_index.token_to_id(
                    indexed_sentence[j])
            while len(indexed_sentence) < self.sent_size:
                indexed_sentence.append(
                    token_to_index.token_to_id_vocab['<PAD>'])
            if len(indexed_sentence) > self.sent_size:
                indexed_sentence = indexed_sentence[:30]
            self.samples.append(
                (indexed_sentence, label_to_id[self.labels[i]]))
        """
        for i in range(len(sentences)):
            indexed_sentence = list(sentences[i])
            for j in range(len(indexed_sentence)):
                indexed_sentence[j] = token_to_index.char_to_id(
                    indexed_sentence[j])
            while len(indexed_sentence) < self.char_seq_size:
                indexed_sentence.append(
                    token_to_index.character_to_id_vocab['<PAD>'])
            if len(indexed_sentence) > self.char_seq_size:
                indexed_sentence = indexed_sentence[:120]
            self.samples.append(
                (indexed_sentence, label_to_id[self.labels[i]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indexed_sentence = torch.tensor(
            self.samples[idx][0], dtype=torch.long)
        label = torch.tensor(
            self.samples[idx][1], dtype=torch.long)
        return indexed_sentence, label
