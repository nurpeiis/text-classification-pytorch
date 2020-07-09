import csv
import ast
import pandas as pd

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class TokenToIndex:

    def __init__(self, input_files, output_dict):
        """
            input_files : list<string> = list of files from which the sentences is going to be extracted
            output_dict : string = name of the file where token to index will be saved
        """
        self.input_files = input_files
        self.output_dict = output_dict
        self.token_to_id = dict()

    def get_dict(self):
        words = dict()
        counter = 2
        words[PAD_TOKEN] = 0
        words[UNK_TOKEN] = 1
        for file in self.input_files:
            df = pd.read_csv(file, delimiter='\t', header=0)
            tmp_word_list = df['tokenized']
            for wl in tmp_word_list:
                wl = [n.strip()
                      for n in ast.literal_eval(wl)]
                for word in wl:
                    if (word in words) == False:
                        words[word] = counter
                        counter = counter + 1

        output = csv.writer(open(self.output_dict, "w"))
        for key, val in words.items():
            output.writerow([key, val])
        self.token_to_id = words

    def token_to_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id[UNK_TOKEN]

    def __len__(self):
        self.token_to_id
