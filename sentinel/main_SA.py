"""
Author: Sun Wha Whang
Created on 12/02/20

Takes a list of tweets as the input. It then process the tweets and pass through
our trained model to return the overall sentiment of the tweets.
"""

# -*- coding: utf-8 -*-
from __future__ import unicode_literals


from sentiment_analysis.preprocessing import Preprocessor, SentimentPreprocessor
from sentiment_analysis.model import SentimentClassificationModel
from sentiment_analysis.data_loader import tokenize_data
from sentiment_analysis.train import check_GPU_usage, check_GPU_availability, get_sentiment, majority_voting

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, sampler

import pandas as pd
import time
import datetime
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import textwrap
import re

import matplotlib

# matplotlib.rc('font', **{'sans-serif' : 'Arial',
#                          'family' : 'sans-serif'})



class SentimentAnalysisModel(object):
    def __init__(self, model_path=None, debug=True):
        """
        Sentiment analysis model using BERT. Make sure that GPU is available.

        :param model_path: if given, load the sentiment analysis model from the
                path. If None, use the default model (best trained model)
        :param debug: if True, print details of the model for debugging purposes
        """
        self.device = check_GPU_availability()
        print(self.device)

        if self.device == 'cpu' and debug:
            print('Warning: cpu is being used, make sure gpu is available for efficient computation!')

        if not model_path:
            model_path = 'sentiment_analysis/saved_model_new/bert-base-uncased_final_1.pt'

        start = time.time()

        if self.device == 'cpu':
            hyperparam_dict = torch.load(model_path,
                                     map_location=lambda storage,
                                     loc: storage.cpu()) # changed .cuda() to 'to()'
        else:
            hyperparam_dict = torch.load(model_path,
                                        map_location=lambda storage,
                                        loc: storage.cuda(self.device)) # changed .cuda() to 'to()'

        self.model = SentimentClassificationModel(**hyperparam_dict)
        self.model.load_state_dict(hyperparam_dict['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.hyperparam_dict = {k:hyperparam_dict[k] for k in hyperparam_dict if k != 'state_dict'}

        if debug:
            print('Time taken to load the model is %.0f seconds.' %(time.time()-start))
            print('====== Hyper-parameters of the model =======')
            for key, val in hyperparam_dict.items():
                if key != 'state_dict':
                    print('    - ', key, ': ', val)

        self.debug = debug


    def return_sentiment(self,
                         tweets,
                         tweet_col='tweets',
                         batch_size=1024,
                         apply_softmax=True,
                         emoji_voting=False,
                         plot_sa=True,
                         plot_n=5,
                         sentiment_col='sentiment',
                         plot_path='SA_plot.png',
                         n_bins=10):
        """
        Returns the sentiment of given tweets using the pre-trained model.

        :param tweets: either pandas.DataFrame or list that contains the tweets.
        :param tweet_col: only applicable if tweets is given as pandas.DataFrame
                Name of the column that contains the tweets. Please don't use
                'text' as the name if possible.
        :param batch_size: batch size of dataloader when passing to the model
        :param apply_softmax: if True, return softmax vectors in 'softmax'
                column. If False, return the raw logits vectors in the column.

        :return tweets: pandas.DataFrame object with all original columns and
                extra columns 'text' (processed tweets), 'sentiment' and
                'softmax'. If tweet_col was 'text', this is changed to
                'original_tweets'
        :return output_str: summary of the output results that gives the number
                of tweets in each sentiment catagory.
        """
        if isinstance(tweets, list):
            tweets = pd.DataFrame({tweet_col : tweets})

        if tweet_col == 'text':
            if self.debug:
                print(":INFO: original tweets column name changed from 'text' to 'original_tweets'")
            tweets['original_tweets'] = tweets[tweet_col]
            tweet_col = 'original_tweets'

        if self.hyperparam_dict['preprocess']:
            preprocessor = SentimentPreprocessor(tweets[tweet_col].tolist(), remove_short_tweets=False)
            processed_tweets, idx = preprocessor.preprocessed_corpus()
            if len(tweets) != len(idx) and self.debug:
                print(':INFO: Uninformative tweets removed. Using %.0f tweets out of %.0f.' %(len(idx), len(tweets)))
            tweets = tweets.iloc[idx].copy()
            tweets['text'] = processed_tweets
            emoji_scores = preprocessor.emoji_scores # returns idx after removing short tweets
        else:
            tweets['text'] = tweets[tweet_col]
            emoji_scores = None

        dataset = tokenize_data(tweets, **self.hyperparam_dict)
        loader = DataLoader(dataset, batch_size=batch_size)

        start = time.time()
        out, softmax = get_sentiment(self.model,
                                     loader,
                                     self.device,
                                     softmax=apply_softmax)
        if self.debug:
            print('Time taken to get SA on the tweets with dataloader length of %.0f is %.0f seconds.' %(len(loader), time.time()-start))
        tweets['sentiment'] = out
        tweets['softmax'] = list(softmax)

        if emoji_scores is not None and emoji_voting:
            majority_vote = majority_voting(out,
                                            emoji_scores,
                                            self.hyperparam_dict['neutral'])
            tweets['sentiment'] = majority_vote

        out = tweets['sentiment'].values
        if self.hyperparam_dict['neutral']:
            sa_counts = [len(out[out == 2]), len(out[out == 1]), len(out[out == 0])]
            output_str = '<%s> Total tweets: %.0f, Positive: %.0f, Neutral: %.0f, Negative: %.0f \n' %('sentiment', len(out), len(out[out == 2]), len(out[out == 1]), len(out[out == 0]))
            tweets.loc[tweets['sentiment'] == 0, 'sentiment'] = 'negative'
            tweets.loc[tweets['sentiment'] == 1, 'sentiment'] = 'neutral'
            tweets.loc[tweets['sentiment'] == 2, 'sentiment'] = 'positive'
        else:
            sa_counts = [len(out[out == 1]), len(out[out == 0])]
            output_str = '<%s> Total tweets: %.0f, Positive: %.0f, Negative: %.0f \n' %('sentiment', len(out), len(out[out == 1]), len(out[out == 0]))
            tweets.loc[tweets['sentiment'] == 0, 'sentiment'] = 'negative'
            tweets.loc[tweets['sentiment'] == 1, 'sentiment'] = 'positive'

        sa_counts = [i / len(out) for i in sa_counts]
        sa_counts = json.dumps(sa_counts)
        with open("./static/files/italian_flag.json", 'w') as f:
            json.dump(sa_counts, f)

        self.generate_histogram(tweets, n_bins)

        self.plot_sentiment(tweets, tweet_col, sentiment_col, plot_n,
                            plot_path, plot_sa)

        if not apply_softmax:
            tweets.rename(columns={'sfotmax':'logits'}, inplace=True)

        if self.debug:
            print('INFO: SA complete at ', datetime.datetime.now())
        return tweets, output_str


    def generate_histogram(self, df, n_bin=10, save_path="./static/files/SA_time_binned.json"):
        date_col = 'date'
        if self.hyperparam_dict['neutral']:
            sentiments = ['positive', 'neutral', 'negative']
        else:
            sentiments = ['positive', 'negative']

        dates = df[date_col].tolist()
        freq = (max(dates) - min(dates)) / n_bin

        for sent in sentiments:
            df[sent] = 0
            df.loc[df.sentiment == sent, sent] = 1

        cols = sentiments + [date_col]
        df_bin = df[cols].groupby(pd.Grouper(key=date_col, freq=freq)).sum().reset_index()

        json_data = []
        for sent in sentiments:
            sent_data = []
            for i in range(n_bin):
                sent_data.append({"time": str(df_bin[date_col][i]),
                                  "y": int(df_bin[sent][i])})
            json_data.append(sent_data)

        with open(save_path, 'w') as f:
            json.dump(json_data, f)


    def plot_sentiment(self, tweets_df, tweet_col, sentiment_col, n, save_path,
                       plot_sa):
        """
        Generates a png file that contains the first n tweets in the tweets_df
        and their sentiments

        :param tweets_df: pandas.DataFrame that contains the tweets
        :tweets_col: Name of the column that contains the tweets
        :sentiments_col: Name of the column that contains the sentiments
        :n: number of tweets to use in the plot

        :return: None
        """
        n = len(tweets_df)
        tweets = tweets_df[:n][tweet_col].tolist()
        sentiments = tweets_df[:n][sentiment_col].tolist()
        names = ['Tweet %s:'%(i+1) for i in range(len(tweets))]

        tweets_line = []
        for t in tweets:
            tweets_line.append(textwrap.wrap(t, width=50))

        json_data = [[n, t, s] for n, t, s in zip(names, sentiments, tweets_line)]
        with open("./static/files/sa_d3.json", 'w') as file:
            json.dump(json_data, file)

        if not plot_sa:
            return None

        fig, ax = plt.subplots(1, figsize=(12, 8))
        plt.axis('off')
        r = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        facecolor_dict = {'positive':'g',
                          'neutral':'w',
                          'negative':'r'}

        height = .95
        for i, t, s in zip(range(len(tweets)), tweets, sentiments):
            t = textwrap.fill(t, width=50)
            props = dict(boxstyle='round', facecolor=facecolor_dict[s], alpha=0.6)
            txt = plt.text(0.5, height, t, fontsize=14, bbox=props,
                        horizontalalignment='center', verticalalignment='top')

            bb = txt.get_window_extent(renderer=r)
            coord0, coord1 = bb.get_points()
            x0, y0 = inv.transform(coord0)
            x1, y1 = inv.transform(coord1)
            y_adj = (y1 - y0) / 2

            plt.text(0.1, height-y_adj, 'Tweet %s:'%(i+1), color='w', fontsize=14,
                     horizontalalignment='center', verticalalignment='center')
            plt.text(0.9, height-y_adj, s, color='w', fontsize=14,
                     horizontalalignment='center', verticalalignment='center')

            height -= y_adj*2 + 0.05

        bh = 1 + height*6
        plt.savefig(save_path, transparent=True, dpi=200,
                    bbox_inches=Bbox([[1.6, bh], [10.7, 7]]))
