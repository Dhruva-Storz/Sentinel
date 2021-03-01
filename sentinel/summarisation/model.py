import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertModel, BartForConditionalGeneration, BartTokenizer

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class BERTEncoder(nn.Module):
    """
    BERT Encoder that encodes contextual embedding for each sentence in tweets.
    For now, process each tweet separately since the advantage of processing
    tweets sequentially is not obvious.
    """
    def __init__(self, pretrained_model='bert-base-uncased', state_dict=None):
        """
        :param pretrained_model: name of the pretrained model of BERT. Default
            is 'bert-base-uncased'
        :param bert_state_dict: custom state dictionary of BERT. If None, just
            use the default pretrained BERT weights
        """
        super(BERTEncoder, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + '0' if use_cuda else "cpu")

        self.bert = BertModel.from_pretrained(pretrained_model,
                                              output_hidden_states=True)

        if state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)

        self.bert.to(device=self.device).eval()


    def forward(self, input_ids, attention_mask, token_type_ids, pool,
                use_hidden):
        """
        :param pool: max, avg or last
        :param use_hidden: if True, use the 2nd last hidden states of BERT
        """
        final_state, _, hidden_states = self.bert(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids)
        if use_hidden:
            out = hidden_states[-2]
        else: # -> (batch_size, sequence_length, hidden_size)
            out = final_state

        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()
        token_type_ids = token_type_ids.cpu()
        sent_embs = []
        for idx, sample in enumerate(out.cpu().detach()):
            sep_indices = np.where(input_ids[idx] == 102)[0]
            if pool == 'last':
                for each in sample[sep_indices]: # (n_sentences, hidden_size)
                    sent_embs.append(each)
            else:
                for i in range(len(sep_indices)):
                    if pool == 'avg':
                        sent_embs.append(sample.mean(dim=0))
                    elif pool == 'max':
                        sent_embs.append(sample.max(each, dim=0))
                # sep_indices += 1
                # for i in range(len(sep_indices)):
                #     if i == 0:
                #         each = sample[:sep_indices[i]] # (n_tokens, hidden_size)
                #     else:
                #         each = sample[sep_indices[i-1]:sep_indices[i]]
                #     if pool == 'avg':
                #         sent_embs.append(each.mean(dim=0))
                #     elif pool == 'max':
                #         sent_embs.append(torch.max(each, dim=0))
        return sent_embs


    def get_embeddings(self, loader, pool='last', use_hidden=False):
        emb_matrix = []
        for batch in loader:
            batch = tuple(i.to(device=self.device) for i in batch)
            emb_matrix += self.forward(*batch, pool=pool, use_hidden=use_hidden)
        return torch.stack(emb_matrix)


class Clusters(object):
    """
    Clusters sentences using their sentence embeddings by performing K-Mean
    """
    def __init__(self, pca_components=None, dist_metric='cosine'):
        """
        :param cluster_model: name of the model used for clustering. Either
                'kmeans' or 'kmedoids'
        :param pca_components: float between 0 and 1. Peforms dimensionality
                reduction using PCA on sentence embeddings before performing
                clustering. Use None if you don't want to use PCA
        :param n_sent: number of sentences to choose from each cluster
        """
        self.model = KMeans

        if pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=pca_components)

        self.dist_metric = dist_metric


    def perform_cluster(self, k, sentences_emb):
        """
        :param k: number of clusters to form
        :param sentences_emb: list of embeddings of the sentences
        """
        model = self.model(n_clusters=k, n_init=100, n_jobs=-1)
        if self.pca: # Perfrom PCA for dimensionality reduction
            sentneces_emb = self.pca.fit_transform(sentences_emb)
        distances = model.fit_transform(sentences_emb) # (n_sentences, k)

        # Order sentences in order of the size of their clusters
        counter = np.array([len(np.where(model.labels_==i)[0]) for i in range(k)])
        order = counter.argsort()[::-1].argsort()

        return np.argmin(distances, axis=0), order


    def get_nearest_neighbours(self, query_idx, corpus_embeddings, n_sent):
        if self.pca: # Perfrom PCA for dimensionality reduction
            corpus_embeddings = self.pca.fit_transform(corpus_embeddings)
        query_embeddings = corpus_embeddings[query_idx]
        distances = cdist(query_embeddings, corpus_embeddings, self.dist_metric)
        return np.argsort(distances, axis=1)[:, :n_sent]


    def get_clustered_sentences(self, k, n_sent, sentences, sentences_emb,
         neighbour_emb=None, plot_elbow=False):
        """
        :param k: number of clusters to form
        :param sentences: list of sentences to cluster
        :param sentences_emb: list of embeddings of the sentences
        :param plot_pca: if True, plot 2D PCA plot with clustered labels
        :param plot_tsne: if True, plot 2D T-SNE plot with clustered labels
        """
        query_idx, order = self.perform_cluster(k, sentences_emb)
        if neighbour_emb is None:
            neighbour_emb = sentences_emb
        nearset_idx = self.get_nearest_neighbours(query_idx,
                                                  neighbour_emb,
                                                  n_sent)

        clustered_sentences = []
        for i in nearset_idx:
            clustered_sentences.append(' '.join(sentences[i]))

        if plot_elbow:
            self.visualise_elbow(sentences_emb)

        return [clustered_sentences[i] for i in order]


    def visualise_elbow(self, sentences_emb):
        sse = {}
        for k in range(1, 50):
            model = self.model(n_clusters=k, n_init=100, n_jobs=-1, random_state=0).fit(sentences_emb)
            sse[k] = model.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()


class AbstractiveSummariser(nn.Module):
    def __init__(self, pretrained_model='bart-large-cnn', debug=True):
        super().__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + '0' if use_cuda else "cpu")

        if 'semsim' in pretrained_model:
            start = time.time()
            if '.pt' in pretrained_model:
                state_dict = torch.load(pretrained_model)
            else:
                state_dict = torch.load('/vol/bitbucket/sww116/semsim.pt')['model']
            if debug:
                print('Time taken to load the semsim state dictionary is %.0f seconds' %(time.time()-start))
            self.bart = BartForConditionalGeneration.from_pretrained('bart-large-cnn', state_dict=state_dict)
            self.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model)
            self.tokenizer = BartTokenizer.from_pretrained(pretrained_model)

        self.bart.to(self.device)


    def forward(self):
        pass


    def tokenize(self, docs):
        if len(docs) > 1:
            pad_to_max_length = True
        else:
            pad_to_max_length = False

        return self.tokenizer.batch_encode_plus(docs, max_length=1024,
                       pad_to_max_length=pad_to_max_length, return_tensors='pt')


    def summarise(self, docs, generate_args):
        inputs = self.tokenize(docs)
        summaries = self.bart.generate(inputs['input_ids'].to(self.device),
                                        attention_mask=inputs['attention_mask'].to(self.device),
                                        **generate_args)
        inputs['input_ids'] = inputs['input_ids'].cpu()
        inputs['attention_mask'] = inputs['attention_mask'].cpu()

        return [self.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in summaries]
