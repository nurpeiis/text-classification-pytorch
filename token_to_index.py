import csv
import ast
import pandas as pd

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class TokenToIndex:

    def __init__(self, input_files, output_dict, output_char_dict):
        """
            input_files : list<string> = list of files from which the sentences is going to be extracted
            output_dict : string = name of the file where token to index will be saved
        """
        self.input_files = input_files
        self.output_dict = output_dict
        self.output_char_dict = output_char_dict
        self.token_to_id_vocab = dict()
        self.char_to_id_vocab = dict()

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
        self.token_to_id_vocab = words

    def get_char_dict(self):
        characters = dict()
        counter = 2
        characters[PAD_TOKEN] = 0
        characters[UNK_TOKEN] = 1
        for file in self.input_files:
            df = pd.read_csv(file, delimiter='\t', header=0)
            tmp_sentences = df['original_sentence']
            for sentence in tmp_sentences:
                sentence = list(sentence)
                for character in sentence:
                    if (character in characters) == False:
                        characters[character] = counter
                        counter = counter + 1

        output = csv.writer(open(self.output_char_dict, "w"))
        for key, val in characters.items():
            output.writerow([key, val])
        self.token_to_id_vocab = characters

    def token_to_id(self, token):
        """ 
            token: string = token
            return id : int
        """
        if token in self.token_to_id_vocab:
            return self.token_to_id_vocab[token]
        else:
            return self.token_to_id_vocab[UNK_TOKEN]

    def char_to_id_vocab(self, character):
        """ 
            character: char = character
            return id : int
        """
        if character in self.character_to_id_vocab:
            return self.character_to_id_vocab[character]
        else:
            return self.character_to_id_vocab[UNK_TOKEN]
