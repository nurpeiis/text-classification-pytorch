import ast
import pandas as pd
from torch.utils.data import Dataset
from token_to_index import TokenToIndex


class TextualDataset(Dataset):

    def __init__(self, filename, token_to_index_files, output_dict_file):
        """ 
            filename : string = input file  
            token_to_index_files : list<string> = files from which dictionary is going to be create
            output_dict_file : string = file to which dictionary is going to be saved
        """
        self.dataset = pd.read_csv(filename, delimiter='\t', header=0)
        tokenized_sentences = self.dataset['tokenized']
        self.labels = self.dataset['dialect_city_id']
        self.samples = []
        token_to_index = TokenToIndex(token_to_index_files, output_dict_file)
        token_to_index.get_dict()
        for i in range(len(tokenized_sentences)):
            indexed_sentence = [n.strip()
                                for n in ast.literal_eval(tokenized_sentences[i])]
            for j in range(len(indexed_sentence)):
                indexed_sentence[j] = token_to_index.token_to_id(
                    indexed_sentence[j])
            self.samples.append(indexed_sentence, self.labels[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
