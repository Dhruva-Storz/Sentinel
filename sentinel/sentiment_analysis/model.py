import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertModel, AlbertModel, RobertaModel


class SentimentClassificationModel(nn.Module):
    """
    Our newest sentiment classification model with pre-trained transformer (BERT
    or Albert), which performs context-heavy embeddings for the input sentences.
    Extra dense layers are added at the end to perform sentiment classification.
    """
    def __init__(self, seq_len, out_features=2, pooling='max', pool_over='seq',
                 dense_layer_lst=[320], dropout=0.1, hidden_layer=None,
                 pretrained_model='bert-base-uncased', **kwarg):
        """
        Initialise the sentiment classification model

        :param seq_len: Maximum length of tokens in a sentence. If the sentence
                is too short, it should be zero padded
        :param out_features: number of out features e.g. 2 if you want only
                positive and negative
        :param pooling: type of pooling to perform; 'max' or 'avg'
        :param pool_over: dimension to perform pooling; 'seq' or 'hidden'
        :param dense_layer_lst: list of number of neurons to use in each hidden
                dense layer. If list is empty, use a single dense layer with
                out_features neurons
        :param dropout: dropout probability to use
        :param hidden_layer: int for the pretrained model hidden layer to use
                for embedding. If None, use the final hidden state.
        :param pretrained_model: name of the pre-trained model to use, either
                based on BERT or ALBERT. See HuggingFace for all available
                pre-trained models. Default is 'bert-base-uncased'.
        """
        super().__init__()

        output_hidden_states = hidden_layer != None
        self.hidden_layer = hidden_layer

        # Load the pretrained model weights
        if 'albert' in pretrained_model:
            self.pretrained_model = AlbertModel.from_pretrained(pretrained_model,
                                        output_hidden_states=output_hidden_states)
        elif 'roberta' in pretrained_model:
            self.pretrained_model = RobertaModel.from_pretrained(pretrained_model,
                                        output_hidden_states=output_hidden_states)
        elif 'bert' in pretrained_model:
            self.pretrained_model = BertModel.from_pretrained(pretrained_model,
                                        output_hidden_states=output_hidden_states)

        self.dropout = nn.Dropout(dropout)

        if pooling == 'max':
            self.pool = F.max_pool1d
        elif pooling == 'avg':
            self.pool = F.avg_pool1d

        self.pool_over = pool_over == 'seq'
        if self.pool_over:
            pool_dim = self.pretrained_model.config.hidden_size
        else:
            pool_dim = seq_len

        dense_layers = []
        for i, hidden_dim in enumerate(dense_layer_lst):
            if i == 0:
                dense_layers.append(nn.Linear(pool_dim, hidden_dim))
            if i == len(dense_layer_lst) - 1:
                dense_layers.append(nn.Linear(hidden_dim, out_features))
            else:
                dense_layers.append(nn.Linear(hidden_dim, dense_layer_lst[i+1]))

        if not dense_layers: # no hidden layer
            dense_layers.append(nn.Linear(pool_dim, out_features))

        self.dense_layers = nn.ModuleList(dense_layers)

        for layer in self.dense_layers:
            nn.init.kaiming_normal_(layer.weight)


    def forward(self, input_ids, attention_mask, softmax=False):
        """
        :param input_ids: input ID of each token
        :param attention_mask: attention mask of token. 0 for pad, 1 otherwise
        """
        all_outputs = self.pretrained_model(input_ids=input_ids,
                                            attention_mask=attention_mask)

        # hidden_state -> (batch_size, sequence_length, hidden_dim)
        if self.hidden_layer:
            hidden_state = all_outputs[-1][self.hidden_layer]
        else:
            hidden_state = all_outputs[0]

        if self.pool_over: # Perform pooling over sequence length
            # hidden_state -> (batch_size, hidden_dim, sequence_length)
            hidden_state = torch.transpose(hidden_state, 1, 2)

        # Perform pooling: out -> (batch_size, sequence_length or hidden_dim)
        out = self.pool(hidden_state, hidden_state.shape[2]).squeeze(2)

        for idx, layer in enumerate(self.dense_layers):
            out = self.dropout(out)
            out = layer(out)
            if idx != len(self.dense_layers) - 1: # don't use ReLU for final
                out = F.relu(out)

        if softmax: # softmax the output
            out = F.softmax(out, dim=1)

        return out


class BertForSentimentClassificationOLD(nn.Module):
    def __init__(self, max_len, hidden_layer=320, dropout=0.1):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(max_len, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 2)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0] # -> (batch_size, sequence_length, hidden_size)
        # Perform max pooling
        # TODO: try average pooling?
        pooled = F.max_pool1d(last_hidden_state, last_hidden_state.shape[2]).squeeze(2) # -> (batch_size, sequence_length, 1) -> (batch_size, sequence_length)
        out = self.dropout(pooled)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)

        return self.fc2(out)


class BertForSentimentClassification(nn.Module):
    """
    Our sentiment classification model with pre-trained BERT or ALBERT, which
    performs context-heavy embeddings for the input sentences.
    Two extra dense layers were added at the end to perform sentiment
    classification.
    """
    def __init__(self, max_len, hidden_layer=320, dropout=0.1,
                 pretrained_model='bert', out_features=2):
        """
        Initialise the model
        :param max_len: Maximum length of tokens in a sentence. If the sentence
                is too short, it should be zero padded
        :param hidden_layer: number of neurons to use in hidden dense layer
        :param dropout: dropout probability to use
        :param pretrained_model: type of the pre-trained model to use. Accepted
                strings are 'bert' and 'albert'
        :param out_features: number of out features e.g. 2 if you want only
                positive and negative
        """
        super().__init__()

        if pretrained_model == 'bert':
            self.pretrained_model = BertModel.from_pretrained('bert-base-uncased')
        elif pretrained_model == 'albert':
            self.pretrained_model = AlbertModel.from_pretrained('albert-base-v2')

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(max_len, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_features)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)[0] # -> (batch_size, sequence_length, hidden_size)
        # Perform max pooling
        # TODO: try average pooling?
        pooled = F.max_pool1d(last_hidden_state, last_hidden_state.shape[2]).squeeze(2) # -> (batch_size, sequence_length, 1) -> (batch_size, sequence_length)
        out = self.dropout(pooled)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)

        return self.fc2(out)


if __name__ == "__main__":
    model = BertForSentimentClassification(200)
