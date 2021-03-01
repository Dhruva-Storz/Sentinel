import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer, BartTokenizer

import numpy as np
import re


class TransformerDataLoader:
    def __init__(self, pretrained_model='bert-base-uncased', max_len=300, batch_size=64, shuffle=False):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.sep_token = ' ' + self.tokenizer.sep_token + ' '
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = False

    def encode_tweets(self, tweets):
        all_sentences = []
        input_ids = np.zeros(shape=(len(tweets), self.max_len))
        attention_masks = np.ones(shape=(len(tweets), self.max_len))
        token_type_ids = np.zeros(shape=(len(tweets), self.max_len))

        counts = 0
        for idx, tweet in enumerate(tweets):
            sentences = re.split(r'(?<=[\.\!\?])\s+', tweet) # Split each sentence in the tweet

            input_ids[idx] = self.tokenizer.encode(self.sep_token.join(sentences),
                                                   max_length=self.max_len,
                                                   pad_to_max_length=True)
            if self.pad_token_id in input_ids[idx]:
                attention_masks[idx, np.where(input_ids[idx] == self.pad_token_id)[0][0]:] = 0
            sep_indices = np.where(input_ids[idx] == self.sep_token_id)[0]

            sentences = sentences[:len(sep_indices)]
            all_sentences += sentences

            for i, sep in enumerate(sep_indices[:-1]):
                if i%2 == 0:
                    if i < len(sep_indices) - 2:
                        token_type_ids[idx, sep+1:sep_indices[i+1]+1] = 1
                    elif i == len(sep_indices) - 2:
                        token_type_ids[idx, sep+1:] = 1

        dataset = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(attention_masks), torch.LongTensor(token_type_ids))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return np.array(all_sentences), loader
