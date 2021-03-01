import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset, sampler

from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer


def read_data(data_path, seed, neutral=True):
    """
    Reads the data file as a pandas.DataFrame object and shuffles the data. If
    the data is unbalanced, it will make the number of samples in each class
    equal by downsampling to the smallest class size.

    :param data_path: path to the data file. It should end with either '.csv' or
            '.pickle'
    :param seed: random seed to use when shuffling the data to reproduce the
            result in the future
    :param neutral: If True, three sentiment classes where 0 for negative, 1 for
            neutral and 2 for positive. Else, 0 for negative and 1 for positive

    :return: pandas.DataFrame object with columns 'sentiment', 'text', 'id' and
            equal number of samples per sentiment class
    """
    # Read the data file
    if '.csv' in data_path:
        df = pd.read_csv(data_path)
    elif '.pickle' in data_path:
        with open(data_path, 'rb') as file:
            df = pickle.load(file)
    else:
        raise Exception('data file type not valid: ', data_path)

    print('Original data size = %.0f' %(len(df)))
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle

    if df.sentiment.dtype == 'object':
        df.sentiment.replace({'negative':0, 'neutral':1, 'positive':2},
                             inplace=True)
    df = df.astype({'sentiment':'int64'})

    if not neutral:
        df = df[df.sentiment != 1]
        df.sentiment.replace(2, 1, inplace=True)

    # Downsample the data by the smallest class size
    df = df.groupby('sentiment').head(min(df.sentiment.value_counts()))
    print('Downsapled data size = %.0f' %(len(df)))
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def tokenize_data(df, seq_len, pretrained_model, **kwarg):
    """
    Preprocess the tweets by converting its labels into integers and tokenizing
    the tweets. It then converts these tokens into input_ids and attention_masks
    which is recognised by BERT/ALBERT.

    :param df: pandas.DataFrame object of the data. It should include columns
            'text'. It should also have 'sentiment' if not a test set
    :param seq_len: maximum number of tokens to use for a sentence (tweet)
    :param pretrained_model: type of pre-trained model to use

    :return: torch.utils.data.TenosrDataset which contains (input_ids,
            attention_masks, ids, labels) - labels is included only if the given
            df contains 'sentiment' column
    """
    # Get the correct tokenizer for the pre-trained transformer model
    if 'albert' in pretrained_model:
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model)
    elif 'roberta' in pretrained_model:
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    elif 'bert' in pretrained_model:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    tweets = df.text.values
    # When tokenizing, adds special tokens required for the model e.g. <CLS> and
    # <SEP> for the start and end of each tweet for BERT
    input_ids = torch.LongTensor([tokenizer.encode(text,
                                    max_length=seq_len,
                                    add_special_tokens=True,
                                    pad_to_max_length=True) for text in tweets])
    # Create attention masks: 0 for padding, 1 otherwise
    attention_masks = torch.zeros(input_ids.shape).long()
    attention_masks[:] = tokenizer.pad_token_id
    attention_masks[attention_masks != input_ids] = 1
    attention_masks[attention_masks == tokenizer.pad_token_id] = 0

    model_input = [input_ids, attention_masks]
    if 'sentiment' in df: # Train/Validation set (has gold labels)
        model_input.append(torch.from_numpy(df.sentiment.values).unsqueeze(1))

    return TensorDataset(*model_input)
