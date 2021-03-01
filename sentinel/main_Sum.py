from summarisation.model import BERTEncoder, Clusters, AbstractiveSummariser
from summarisation.data_loader import TransformerDataLoader
from summarisation.preprocessing import SummarisationPreprocessor
from sentence_transformers import SentenceTransformer

import torch
import numpy as np

class TweetSummariser(object):
    """
    Abstractive summariser of tweets.
    """
    def __init__(self,
                 debug=True,
                 bert_pretrained_model='bert-base-cased',
                 bert_state_dict=None,
                 bert_pool='last',
                 bert_use_hidden=True,
                 bert_max_len=300,
                 bert_batch_size=16,
                 use_sbert=True,
                 dist_metric='cosine',
                 pca_components=None,
                 bart_pretrained_model='semsim',
                 filter_sentiment=True,
                 softmax_threshold_neg=0.98,
                 softmax_threshold_pos=0.98,
                 summary_max_len=70,
                 ):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + '0' if use_cuda else "cpu")

        self.bert_dataloader = TransformerDataLoader(bert_pretrained_model, bert_max_len, bert_batch_size)
        self.bert_encoder = BERTEncoder(bert_pretrained_model, bert_state_dict)
        self.use_hidden = bert_use_hidden
        self.pool = bert_pool

        if use_sbert:
            self.sbert = SentenceTransformer('bert-base-nli-mean-tokens')
        else:
            self.sbert = None

        self.clustering = Clusters(pca_components, dist_metric)

        self.bart_summariser = AbstractiveSummariser(bart_pretrained_model, debug)
        self.generate_args = {'num_beams': 5,
                              'min_length': 5,
                              'max_length': summary_max_len,
                              'do_sample': False,
                              'num_return_sequences': 1,
                              'repetition_penalty': 1,
                              'length_penalty': 1}

        self.debug = debug

        self.filter_sentiment = filter_sentiment
        self.softmax_threshold_neg = softmax_threshold_neg
        self.softmax_threshold_pos = softmax_threshold_pos

        self.preprocessor = SummarisationPreprocessor(
                       min_len=10,
                       keep_max=True,
                       remove_username=True,
                       keep_fullstop=True,
                       not_start_words=["but", "and", "or", "i", "i'm", "i’m",
                                        "im", "i've", "i’ve", "ive", "i'll",
                                        "i’ll" , "i'd", "i’d", "id", "my", "we",
                                        "we've", "we’ve", "weve", "we'll",
                                        "we’ll", "we'd", "we’d", "wed", "our",
                                        "you", "he", "she", "they", "this",
                                        "that", "these", "those", "why", "what",
                                        "who", "which", "when"],
                       not_contain_words=["i", "i'm", "i’m", "im", "i've",
                                          "i’ve", "ive", "i'll", "i’ll" , "i'd",
                                          "i’d", "me", "my", "we", "our", "you",
                                          "your", "fuck", "shit"]
                       )


    def get_sentiment_filtered(self, df_sentiment, col_tweet, col_sentiment, col_softmax):
        """
        Filters out tweets depending on its sentiments
        """
        df_sentiment[col_softmax] = df_sentiment[col_softmax].apply(np.max)

        return df_sentiment[~((df_sentiment[col_sentiment] == 'negative') & (df_sentiment[col_softmax] > self.softmax_threshold_neg) |
                            (df_sentiment[col_sentiment] == 'positive') & (df_sentiment[col_softmax] > self.softmax_threshold_pos)) |
                            df_sentiment[col_tweet].str.contains(r'\d', regex=True)][col_tweet].values


    def extractive_sum(self, tweets, k=5, n_sent=3):
        """
        Performs extractive summarisation
        """
        self.bert_encoder = self.bert_encoder.to(self.device)
        sentences, loader = self.bert_dataloader.encode_tweets(tweets)
        sent_embs = self.bert_encoder.get_embeddings(loader,
                                                     self.pool,
                                                     self.use_hidden)
        self.bert_encoder = self.bert_encoder.cpu()
        assert len(sentences) == len(sent_embs)

        if self.sbert:
            similarity_embs = self.sbert.encode(sentences)
            similarity_embs = np.array(similarity_embs)
        else:
            similarity_embs = None

        sent_clustered = self.clustering.get_clustered_sentences(k, n_sent,
                                        sentences, sent_embs, similarity_embs)

        torch.cuda.empty_cache()

        return sent_clustered


    def abstractive_sum(self, clustered_sentences):
        """
        Performs abstractive summarisation on each cluster of sentences
        """
        summaries = []
        for each_cluster in clustered_sentences:
            self.generate_args['min_length'] = int(len(each_cluster)/8)
            summaries += self.bart_summariser.summarise([each_cluster],
                                                        self.generate_args)
        return summaries


    def return_summary(self, df_sentiment, col_tweet='original_tweets',
               col_sentiment='sentiment', col_softmax='softmax', k=5, n_sent=3):
        """
        Generate summary on the given tweets by combining extractive and
        abstractive summarisation

        :param df_sentiment: pandas.DataFrame object of the tweets with their
                sentiments
        :param col_tweet: name of the column that contains the original tweets
        :param col_sentiemnt: name of the column that contains the sentiment
        :param col_softmax: name of the column that contains the softmax output
                of sentiment analysis
        :param k: number of clusters to create
        :param n_sent: number of sentences to use per cluster

        :return: a list of strings where each string is a bulletpoint for the
                abstractive summary of the tweets
        """
        if self.filter_sentiment:
            df_sentiment = df_sentiment[[col_tweet, col_sentiment, col_softmax]].copy()
            tweets = self.get_sentiment_filtered(df_sentiment, col_tweet,
                                                 col_sentiment, col_softmax)
            if self.debug:
                print(':INFO: %s / %s left after filtering extreme sentiments' %(len(tweets), len(df_sentiment)))
        else:
            tweets = df_sentiment[col_tweet].values

        n_before = len(tweets)
        tweets = self.preprocessor.preprocess_tweets(tweets)
        if self.debug:
            print(':INFO: %s / %s left after preprocessing' %(len(tweets), n_before))

        clustered_tweets = self.extractive_sum(tweets, k, n_sent)

        return self.abstractive_sum(clustered_tweets)


if __name__ == "__main__":
    import pandas as pd
    from main_SA import SentimentAnalysisModel

    SA = SentimentAnalysisModel(model_path=None, debug=True)
    Summariser = TweetSummariser(debug=True,
                                 bert_pretrained_model='bert-base-uncased',
                                 bert_pool='avg',
                                 bert_use_hidden=True,
                                 bert_max_len=300,
                                 bert_batch_size=16,
                                 use_sbert=True,
                                 dist_metric='cosine',
                                 pca_components=None,
                                 bart_pretrained_model='semsim',
                                 filter_sentiment=True,
                                 softmax_threshold_neg=0.98,
                                 softmax_threshold_pos=0.98,
                                 summary_max_len=70
                                 )

    df_tweets = pd.read_csv('sentiment_analysis/data/australia_fires.csv')

    df_SA, SA_str = SA.return_sentiment(df_tweets,
                                        tweet_col='text',
                                        batch_size=32,
                                        apply_softmax=True)
    print(SA_str)

    summaries = Summariser.return_summary(df_SA,
                                          col_tweet='original_tweets',
                                          col_sentiment='sentiment',
                                          col_softmax='softmax',
                                          k=6,
                                          n_sent=2)

    for s in summaries:
        print(' - ', s)
