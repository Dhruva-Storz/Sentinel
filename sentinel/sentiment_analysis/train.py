# import psutil
import humanize
import os
import GPUtil as GPU

import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import io

from transformers import AdamW

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, sampler

from sklearn.model_selection import StratifiedKFold


def check_GPU_availability():
    """
    Checks if GPU is available

    :return: avilable device name
    """
    if torch.cuda.is_available(): # Check we're using GPU
        torch.backends.cudnn.deterministic = True
        device = 'cuda:0'
    else:
        device = 'cpu'

    return device


def fix_seed(seed=234):
    """
    Fix seed for torch and numpy to produce consistent results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_GPU_usage():
    """
    Checks and prints the amount of GPU available and used currently
    """
    gpu = GPU.getGPUs()[0]
    gpu_stats = [gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal]

    # process = psutil.Process(os.getpid())
    # ram_free = humanize.naturalsize(psutil.virtual_memory().available)
    # proc_size = humanize.naturalsize(process.memory_info().rss)
    #
    # print("Gen RAM Free: %s | Proc size: %s" %(ram_free, proc_size))
    print("GPU RAM Free: {:.0f}MB | Used: {:.0f}MB | Util {:.0f}% | Total {:.0f}MB".format(*gpu_stats))


def get_sentiment(model, loader, device, softmax=True):
    """
    Obtain the sentiment from the loader using the given ModuleList

    :param model: torch.nn.Module object for the trained sentiment model
    :param loader: torch.utils.data.DataLoader object for the samples
    :param device: device to run the model on. 'cpu' or 'cuda:0'
    :param softmax: if False, returns the raw logit vectors. If True, returns
            the softmax applied to the logit outputs

    :return output: argmax output of the model
    :return all_softmax: softmax output of the model
    :return y: returned if the loader has y (the gold label)
    """
    model.eval()

    all_outs = None
    y = None
    with torch.no_grad():
        for data in loader:
            x, a = data[0], data[1]
            x = x.to(device=device)  # move to device, e.g. GPU
            a = a.to(device=device)
            out = model(input_ids=x,
                                attention_mask=a,
                                softmax=softmax).detach().cpu().numpy()
            if all_outs is None:
                all_outs = out
            else:
                all_outs = np.concatenate([all_outs, out], axis=0)
            if len(data) == 3:
                if y is None:
                    y = data[2].numpy()
                else:
                    y = np.concatenate([y, data[2]], axis=0)
            x = x.cpu()  # move to device, e.g. GPU
            a = a.cpu()
    pred = np.argmax(all_outs, axis=1).flatten()

    if y is not None:
        return pred, all_outs, y

    return pred, all_outs


def accuracy(prediction, target):
    """
    Calculates the accuracy of the predictions for classification

    :param prediction: softmax or argmax predictions of the sample
    :param target: target of the sample

    :return: accuracy of the prediction
    """
    if len(prediction.shape) != 1:
        prediction = np.argmax(prediction, axis=1).flatten()
    if len(target.shape) != 1:
        target = target.flatten()
    return len(prediction[prediction == target])/len(prediction)


def check_accuracy(loader, model, device, loss_fn=None):
    """
    Calculates the accuracy of the given dataset

    :param loader: torch.utils.data.DataLoader for the samples
    :param model: torch.nn.Module model to predict the sample outputs
    :param device: device to train the model on e.g. 'cpu' or 'cuda:0'
    :param loss_fn: loss function to caluate the loss on the samples. If None,
            no loss will be calculated

    :return: the average model accuracy on the loader
    """
    loss = 0
    acc = 0
    n_loader = 0

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, a, y in loader:
            n_loader += 1
            x = x.to(device=device)  # move to device, e.g. GPU
            a = a.to(device=device)
            y = y.to(device=device, dtype=torch.long).view(-1)
            logits = model(input_ids=x, attention_mask=a).detach()
            if loss_fn:
                loss += loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1)).item()
            acc += accuracy(logits.cpu().numpy(), y.cpu().numpy())

    if loss_fn:
        return acc / n_loader, loss / n_loader

    return acc / n_loader


def emoji_voting(model, loader, device, emoji_scores):
    """
    Uses emojis in each tweet to perform majority voting on the sentiment of
    each tweet. If the gold label is given in the loader, it also calculates the
    new accuracy.

    :param model: torch.nn.Module model to predict the sample outputs
    :param loader: torch.utils.data.DataLoader for the samples
    :param device: device to train the model on e.g. 'cpu' or 'cuda:0'
    :param emoji_scores: dict object that contains emoji scores of the tweets

    :return: if the gold label is given, it returns the accuracy of the emoji
            voting model. Else, return the new predictions based on majority
            voting
    """
    results = get_sentiment(model, loader, device)
    if len(results) == 3:
        pred, softmax, y = results
    else:
        pred, softmax = results
        y = None
    neutral = len(softmax[0]) == 3
    majority_vote = majority_voting(pred, emoji_scores, neutral)
    if y is not None:
        return accuracy(majority_vote, y)


def majority_voting(predictions, emoji_scores, neutral):
    """
    Performs majority voting using the given prediction and emoji scores of each
    tweet.

    :param predictions: prediction output of the model
    :param emoji_scores: dict object that contains emoji scores of the tweets
    :param neutral: True if the prediction includes neutral sentiment

    :return: the new prediction obtained by majority voting
    """
    if not neutral: # 0 for neg, 1 for pos
        predictions *= 2
        predictions -= 1 # -1 for neg, 1 for pos
    else:
        predictions -= 1 # -1, 0, +1 for neg, neu, pos

    n_samples = len(predictions)
    majority_vote = predictions.copy()
    for i in range(n_samples):
        majority_vote[i] += emoji_scores[i]['positive']
        majority_vote[i] -= emoji_scores[i]['negative']
        if majority_vote[i]:
            majority_vote[i] = majority_vote[i] / abs(majority_vote[i])
        elif not neutral:
            majority_vote[i] = predictions[i]

    if not neutral: # 0 for neg, 1 for pos
        majority_vote += 1
        majority_vote = majority_vote / 2
    else:
        majority_vote += 1

    return majority_vote.astype(np.int)


def train(model, loader_train, loader_val, device, optimizer, loss_fn,
          scheduler=None, epochs=1, log_iteration=100):
    """
    Train the model using the given train set and optimizer for given epochs.
    It also evaluates the model using the validation set

    :param model: torch.nn.Module giving the model to train
    :param loader_train: torch.utils.data.DataLoader for train set
    :param loader_val: torch.utils.data.DataLoader for validation set
    :param device: device to train the model on e.g. 'cpu' or 'cuda:0'
    :param optimizer: optimizer to use for training
    :param loss_fn: loss function to calculate the train loss
    :param scheduler: schedule the learning rate at every epoch if given
    :param epoch: number of epochs to train

    :return: None
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print('========== Epoch {:} / {:} =========='.format(e + 1, epochs))

        total_loss = 0
        n_loader = 0
        for t, (x, a, y) in enumerate(loader_train):
            n_loader += 1
            model.train()  # put model to training mode

            x = x.to(device=device)  # move to device, e.g. GPU
            a = a.to(device=device)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(input_ids=x, attention_mask=a)

            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            total_loss += loss.item()
            loss.backward()

            # Prevent gradient from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if t % log_iteration == 0:
                acc_val, loss_val = check_accuracy(loader_val, model, device, loss_fn)
                print('    Iteration {}, loss = {:.4f}, lr = {:.5f}, val_acc = {:.4f}, val_loss = {:.4f}'.format(t, loss.item(), optimizer.param_groups[0]["lr"], acc_val, loss_val))

            # elif t % 100 == 0:
            #     print('    Iteration {}, loss = {:.4f}, lr = {:.5f}'.format(t, loss.item(), optimizer.param_groups[0]["lr"]))

        if scheduler is not None: # Adjust the learning rate
            scheduler.step()

        acc_val = check_accuracy(loader_val, model, device)
        print("    - Final accuracy on validation set = {0:.4f}".format(acc_val))

        av_loss = total_loss / len(loader_train)
        print("    - Final average training loss = {0:.4f}".format(av_loss))


def cross_validation_train(model_class, dataset, hyperparam_dict, device,
                           emoji_scores=None):
    """
    Trains the given model with k-fold cross validation. It saves each model
    separately.

    :param model_class: nn.Module class to train
    :param dataset: torch.utils.data.TensorDataset for the data before train and
            test split
    :param hyperparam_dict: dict object that contains hyper-parameters to
            initialise the model and training
    :param device: device to train on e.g. 'cpu' or 'cuda:0'
    :param emoji_scores: dict object that contains the emoji score of each tweet

    :return: None
    """
    if hyperparam_dict['n_splits'] == 1:
        model_type = 'final'
        split_idx = int((len(dataset) - 1) * 0.9)
        cross_val_split = [(range(split_idx), range(split_idx, len(dataset)-1))]
    else:
        model_type = 'cv'
        # Use StratifiedKFold to ensure each class is balanced when splitting
        skf = StratifiedKFold(n_splits=hyperparam_dict['n_splits'])
        cross_val_split = skf.split(dataset, dataset.tensors[-1])

    av_accuracy = 0
    av_emoji_accuracy = 0
    for fold, (train_idx, test_idx) in enumerate(cross_val_split):
        print('========== Cross-Validation Fold {:} / {:} =========='.format(fold + 1, hyperparam_dict['n_splits']))

        dataset_train = torch.utils.data.Subset(dataset, train_idx)
        dataset_test = torch.utils.data.Subset(dataset, test_idx)
        loader_train = DataLoader(dataset_train, batch_size=hyperparam_dict['batch_size'], shuffle=True)
        loader_test = DataLoader(dataset_test, batch_size=hyperparam_dict['batch_size'])

        model_save_path = 'sentiment_analysis/saved_model_new/%s_%s_%s.pt' %(hyperparam_dict['pretrained_model'], model_type, fold + 1)
        if os.path.isfile(model_save_path):
            print('Loading the trained model...')
            hyperparam_dict = torch.load(model_save_path,
                                         map_location=lambda storage,
                                         loc: storage.cuda(device))
            model = model_class(**hyperparam_dict)
            model.load_state_dict(hyperparam_dict['state_dict'])
            model.to(device)
        else:
            model = model_class(**hyperparam_dict)

            for layer in hyperparam_dict['freeze_layers']:
                for param in model.pretrained_model.encoder.layer[layer].parameters():
                    param.requires_grad = False

            optimizer = AdamW(model.parameters(),
                              lr=hyperparam_dict['lr'],
                              eps=hyperparam_dict['eps'])
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                              step_size=hyperparam_dict['step_size'],
                              gamma=hyperparam_dict['gamma'])
            loss_fn = nn.CrossEntropyLoss()

            train(model, loader_train, loader_test, device, optimizer, loss_fn,
                  scheduler=scheduler, epochs=hyperparam_dict['epochs'])

            # Make sure to free up GPU after training
            torch.cuda.empty_cache()

        test_accuracy = check_accuracy(loader_test, model, device)
        print('Final accuracy of the model = %.4f' %(test_accuracy))
        av_accuracy += test_accuracy

        if emoji_scores is not None:
            emoji_accuracy = emoji_voting(model, loader_test, device, emoji_scores)
            print('Final accuracy of the model with emoji majority voting = %.4f' %(emoji_accuracy))
            av_emoji_accuracy += emoji_accuracy
            hyperparam_dict['test_accuracy_emoji_voting'] = emoji_accuracy

        hyperparam_dict['state_dict'] = model.state_dict()
        hyperparam_dict['test_accuracy'] = test_accuracy

        model_save_path = 'sentiment_analysis/saved_model_new/%s_%s_%s.pt' %(hyperparam_dict['pretrained_model'], model_type, fold + 1)

        torch.save(hyperparam_dict, model_save_path)
        print('Model saved in %s' %(model_save_path))

    print('Average accuracy over %s-fold cross-validation = %s' %(hyperparam_dict['n_splits'], av_accuracy / hyperparam_dict['n_splits']))
    if emoji_scores is not None:
        print('Average emoji majority voting accuracy over %s-fold cross-validation = %s' %(hyperparam_dict['n_splits'], av_emoji_accuracy / hyperparam_dict['n_splits']))
