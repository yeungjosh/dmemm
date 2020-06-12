# script for cs577 hw2 notebook mlp

import numpy as np
import string
import argparse


import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import matplotlib.pyplot as plt
# parser = argparse.ArgumentParser()
# parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
# parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
# parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
# args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('device: ',device)

# read the dataset
with open('train_set.pkl', 'rb') as f:
    train_set = pickle.load(f)
# with open('test_set.pkl', 'rb') as f:
#     test_set = pickle.load(f)


def print_traning_data(first_data):
    print(first_data)
    print('type: ',type(first_data))
    print('list length: ',len(first_data))
    print('keys: ',first_data.keys())

# now, you must parse the dataset
print("first data")
first_data = train_set[0]
print_traning_data(first_data)

# print('42 data')
# not_first_data = train_set[42]
# print_traning_data(not_first_data)
# Create a list of tuples (list of words, list of tags) for a sentence
training_data=[]
for x_i in train_set:
    word_tag_tuple = (x_i['words'], x_i['ts_raw_tags'])
    training_data.append(word_tag_tuple)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

sentiment_tags = ["T-POS", "T-NEG", "T-NEU", "O"]
one_hot_label = {}
one_hot_label['T-POS'] = [1, 0, 0, 0, 0]
one_hot_label['T-NEG'] = [0, 1, 0, 0, 0]
one_hot_label['T-NEU'] = [0, 0, 1, 0, 0]
one_hot_label['O'] = [0, 0, 0, 1, 0]
one_hot_label[START_TAG] = [0, 0, 0, 0, 1]
# do we need a start/stop label? Don't think so bc our n-grams start at the first word
word_to_ix = {}
def create_word_to_ix(training_data):
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix
tag_to_ix = {"T-POS": 0, "T-NEG": 1, "T-NEU": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}
word_to_ix=create_word_to_ix(training_data)
# every word in training data has an index
word_to_ix
# type(word_to_ix.keys())
vocab_list = list(word_to_ix.keys())
vocab_list[:10]
len(vocab_list)

# in general, n-grams have n-1 context words
# 3-grams have two context words
# BUT since now our data is of the form (['w1', 'w2', 't1'], 't2') where t2 is target,
# context is a list of 3 things
CONTEXT_SIZE = 3
EMBEDDING_DIM = 15

# This creates a list of (['w1', 'w2', 't1'], 't2') where t2 is target
def make_bigrams_with_tags(words_and_tags):
    sentence_words, tags = words_and_tags
#     print('sentence_words ',sentence_words)
#     print('tags ',tags)
    # build a list of tuples, where each tuple is ([ word_i-2, word_i-1 ], target word)
    bigrams_with_tags = [([sentence_words[i], sentence_words[i + 1], tags[i]], tags[i + 1])
    for i in range(len(sentence_words) - 1)]

    start_bigram = ([sentence_words[0], START_TAG], tags[0])

    bigrams_with_tags = [start_bigram] + bigrams_with_tags

    return bigrams_with_tags

# test_bigrams_with_tags = make_bigrams_with_tags(training_data[0])
# print('test_bigrams_with_tags: ',test_bigrams_with_tags[:3])

sentence_bigrams_with_tags =[]
for data in tqdm(training_data):
    two_grams_with_tags=make_bigrams_with_tags(data)
    sentence_bigrams_with_tags.append(two_grams_with_tags)
sentence_bigrams_with_tags_flat_list = [item for sublist in sentence_bigrams_with_tags for item in sublist] #works
sentence_bigrams_with_tags_flat_list[:10]
# TODO: Helper functions to make the code more readable
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
import random
import sklearn.model_selection
flattened = sentence_bigrams_with_tags_flat_list

random.shuffle(flattened)
train_size= int(len(flattened)*0.8)
test_size=len(flattened)-train_size
train_dataset, val_dataset = sklearn.model_selection.train_test_split(flattened, train_size=train_size, test_size=test_size)
print('len(val_dataset) ', len(val_dataset))
print('len(train_dataset) ' ,len(train_dataset))
# n-gram context_size is n
# create a class to define our Neural Net
class MLP(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, context_size):
        super(MLP, self).__init__()
        # define all the layers, parameters, etc.
        self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        #one extra embedding for all words not in training set
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)

#         self.fc1 = nn.Linear(X.shape[1], 10) tutorial
        # n-gram input
        self.fc1 = nn.Linear((context_size-1) * embedding_dim + 5, 128)
        self.act1 = nn.ReLU()
#         self.fc2 = nn.Linear(10, 1)
        num_sentiments=len(sentiment_tags)
        self.out = nn.Linear(128, num_sentiments)
#         self.act2 = nn.Sigmoid()

    def forward(self, input_words, input_tags):
        # input_words: input word index
        # your model will have two inputs: the current word and the previous tag
        # For the option of training your own embedding,
        # you will need to embed the word and the tag using an embedding layer for each
        # look into nn.Embedding for how to do this
        # you can then concatenate them and feed them through the rest of the network to predict the current tag
#         print('input_words: ',input_words)
#         print('input_tags: ',input_tags)
#         print('self.embedding_dim: ',self.embedding_dim)
#         print(list(input_words.size()))

#         print('input_.shape: ', inputs.shape)
#         print('type(input_): ',type(inputs))

#         print('self.embeddings: ',self.embeddings)
#         print('self.embeddings(input_): ',self.embeddings(inputs))
        if (list(input_words.size()) == [1]):

            zeros = torch.zeros([1, self.embedding_dim])
#             print('zeros: ',zeros)
            catted = torch.cat((zeros, self.embeddings(input_words).view((1, -1))), dim=1)
#             print('catted: ',catted)
#             print('self.embeddings(input_words).view((1, -1)) ', self.embeddings(input_words).view((1, -1)))

            embeds=catted.view((1, -1))
#             print('catted shape: ',embeds.shape)
        else:
            embeds = self.embeddings(input_words).view((1, -1))
#         print('embeds: ', embeds)
#         print('embeds shape: ', embeds.shape)
        combined = torch.cat((embeds.view(1, -1),
                          input_tags.float().view(1, -1)), dim=1)
#         print('combined: ', combined)
#         print('combined shape: ', combined.shape)
#         poop_test = F.relu(self.fc1(combined.float()))
#         print('poop_test: ', poop_test)
        a1 = self.fc1(combined)
        h1 = self.act1(a1)
        a2 = self.out(h1)
        log_probs = F.log_softmax(a2, dim=1) # y
        return log_probs
losses_over_epochs = []
loss_function = nn.NLLLoss()

# vocab = set(test_sentence_words)
# test_word_to_ix = {word: i for i, word in enumerate(vocab)}

model = MLP(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# for epoch in tqdm(range(2)):
def train_epoch(model, data, optimizer, loss_function, batch_size=1):
#     for sentence_trigram in tqdm(sentence_trigrams):
    losses = []
    total_loss = 0
    for context, target in tqdm(data):
#         print('contex: ',context)
#         print('target: ',target)
#         print('target shape: ',target)
        context_tag = context[-1] # tag i-1
        context_words = context[:-1] # word i, word i-1
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_word_idxs = torch.tensor([word_to_ix[w] if w in word_to_ix else max(word_to_ix, key=word_to_ix.get) for w in context_words], dtype=torch.long)
#         print('context_word_idxs: ',context_word_idxs)
#         print('context_word_idxs shape: ',context_word_idxs.shape)
        context_tag_idxs = torch.tensor(one_hot_label[context_tag], dtype=torch.long)
#         print('context_tag_idxs: ',context_tag_idxs)
#         print('context_tag_idxs shape: ',context_tag_idxs.shape)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_word_idxs, context_tag_idxs)
#         print('log_probs: ',log_probs)
#         print('log_probs shape: ',log_probs.shape)
#         print('ix for word: ',word_to_ix[target])
#         print('ix tensor for word: ',torch.tensor([word_to_ix[target]], dtype=torch.long))
        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
#         print("one_hot_label[target] ",one_hot_label[target])
#         print("[one_hot_label[target]] ",[one_hot_label[target]])
#         print("[tag_to_ix[target]] ",[tag_to_ix[target]])
#         loss = loss_function(log_probs, torch.tensor(one_hot_label[target], dtype=torch.long))
        loss = loss_function(log_probs, torch.tensor([tag_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()

        losses.append(total_loss)
#     print("len(losses): ",len(losses))
    return [sum(losses)/float(len(losses))]
# print('losses: ',losses)  # The loss decreased every iteration over the training data!
# plot the losses to see that the model is learning
epoch_losses = []
num_epochs = 15
for e in range(num_epochs):
    epoch_losses += train_epoch(model, sentence_bigrams_with_tags_flat_list, optimizer, loss_function)
def find_TP_FP_TF_FN(true, pred):
    if (true == pred and true != 'O'):
        return 1,0,0,0
    if true == 'O':
        if pred != 'O':
            return 0,1,0,0 #FP
    if pred == 'O':
        if true != 'O':
            return 0,0,0,1 #FN
    if (true == 'T-NEG' and pred in ['T-POS','T-NEU']):
        return 0,1,0,1
    if (true == 'T-POS' and pred in ['T-NEU','T-NEG']):
        return 0,1,0,1
    if (true == 'T-NEU' and pred in ['T-POS','T-NEG']):
        return 0,1,0,1
    return 0,0,0,0

def precision(tp, fp):
    return tp/(tp+fp)

def recall(tp, fn):
    return tp/(tp+fn)

def f1_score(prec, rec):
    return 2*prec*rec/(prec+rec)
# Check predictions before training

match=0
no_match=0
tp,fp,tf,fn=0,0,0,0

with torch.no_grad():
    for context, target in tqdm(train_dataset):
#         print('contex: ',context)
#         print('target: ',target)
#         print('target shape: ',target)
        context_tag = context[-1] # tag i-1
        context_words = context[:-1] # word i, word i-1
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_word_idxs = torch.tensor([word_to_ix[w] if w in word_to_ix else max(word_to_ix, key=word_to_ix.get) for w in context_words], dtype=torch.long)
#         print('context_word_idxs: ',context_word_idxs)
#         print('context_word_idxs shape: ',context_word_idxs.shape)
        context_tag_idxs = torch.tensor(one_hot_label[context_tag], dtype=torch.long)
#         print('context_tag_idxs: ',context_tag_idxs)
#         print('context_tag_idxs shape: ',context_tag_idxs.shape)

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_word_idxs, context_tag_idxs)
#         print('log_probs: ',log_probs)
#         print('argmax(log_probs): ',argmax(log_probs))

        pred=argmax(log_probs)
#         if pred != 3:
#             print('yes: ', pred)
        tp_i,fp_i,tf_i,fn_i = find_TP_FP_TF_FN(pred, target)
        tp+=tp_i
        fp+=fp_i
        tf+=tf_i
        fn+=fn_i

prec=precision(tp,fp)
rec=recall(tp,fn)
f1 = f1_score(prec,rec)
#     if argmax(log_probs) == tag_to_ix[target]:
#             match+=1
#         else:
#             no_match+=1
#         print('tag_to_ix[target] ',tag_to_ix[target])


print("tp,fp,tf,fn: ",tp,fp,tf,fn)
print('prec: ',prec)
print('rec: ',rec)
print('f1: ',f1)
