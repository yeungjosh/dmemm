import numpy as np
import string
import argparse

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
import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
# parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
# parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
# args = parser.parse_args()

# read the dataset
with open('train_set.pkl', 'rb') as f:
    train_set = pickle.load(f)
with open('test_set.pkl', 'rb') as f:
    test_set = pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=int, default=0, help='Option to run 1 or 0 ')
args = parser.parse_args()

LOAD = args.load_model
print('LOAD: ', LOAD)
# purdue cs servers
path = "/homes/yeungj/cs577-nlp/hw2-bilstm.pt"

def print_traning_data(first_data):
    print(first_data)
    print('type: ',type(first_data))
    print('list length: ',len(first_data))
    print('keys: ',first_data.keys())

# Create a list of tuples (list of words, list of tags) for a sentence
training_data=[]
for x_i in train_set:
    word_tag_tuple = (x_i['words'], x_i['ts_raw_tags'])
    training_data.append(word_tag_tuple)
testing_data=[]
for x_i in test_set:
    word_tag_tuple = (x_i['words'], x_i ['ts_raw_tags'])
    testing_data.append(word_tag_tuple)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

sentiment_tags = ["T-POS", "T-NEG", "T-NEU", "O"]
one_hot_label = {}
one_hot_label['T-POS'] = [1, 0, 0, 0]
one_hot_label['T-NEG'] = [0, 1, 0, 0]
one_hot_label['T-NEU'] = [0, 0, 1, 0]
one_hot_label['O'] = [0, 0, 0, 1]
# do we need a start/stop label? Don't think so bc our n-grams start at the first word
word_to_ix = {}

def create_word_to_ix(data):
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

tag_to_ix = {"T-POS": 0, "T-NEG": 1, "T-NEU": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}
num_tags = len(tag_to_ix)

word_to_ix=create_word_to_ix(training_data + testing_data)

# every word in training data has an index
# type(word_to_ix.keys())
vocab_list = list(word_to_ix.keys())
vocab_list[:10]
len(vocab_list)
# in general, n-grams have n-1 context words
# 3-grams have two context words
# BUT since now our data is of the form (['w1', 'w2', 't1'], 't2') where t2 is target,
# context is a list of 3 things
EMBEDDING_DIM = 15


# TODO: Helper functions to make the code more readable
def sentence_to_ix(sentence, w_to_ix):
    indexes = [w_to_ix[word] for word in sentence]
    return torch.tensor(indexes, dtype=torch.long)

class BiLSTM_MEMM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(BiLSTM_MEMM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Map output of the BiLSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # Parameter is a multi dimensional matrix from tag i to tag i+1. Entry i,j: transitioning j -> i
        self.conditional_probs = nn.Parameter(torch.randn(num_tags, num_tags))

        # Make sure we never go TO start tag or FROM stop tag
        self.conditional_probs.data[tag_to_ix[START_TAG], :] = -100000
        self.conditional_probs.data[:, tag_to_ix[STOP_TAG]] = -100000
#         self.hidden = (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def get_bilstm_features(self, sentence):
        num_features = len(sentence)
        hidden = (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

        embeddings_reshaped = self.embeddings(sentence).view(num_features, 1, -1)
        bilstm_feats, hidden = self.lstm(embeddings_reshaped, hidden)

        bilstm_feats = bilstm_feats.view(len(sentence), self.hidden_dim)
        bilstm_feats = self.hidden2tag(bilstm_feats)
        return bilstm_feats

    def generate_scalar_tensor(self, n):
            return torch.tensor([n], dtype=torch.float)

    def memm(self, features, target_tags):
#         print("features: ",features)
        total_score=self.generate_scalar_tensor(0)
        start_ix = tag_to_ix[START_TAG]
#         # Set start tag score to be 0

        target_tags = torch.cat([torch.tensor([start_ix], dtype=torch.long), target_tags])
#         print('target_tags ',target_tags)
#         tags = targets
        curr_tag = target_tags[0]
#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = total_score

        # Iterate through the sentence
        for tag_index, feature in enumerate(features):
            feature_score=self.generate_scalar_tensor(0)
            for next_tag in range(num_tags):
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                tag_i_feature_score=torch.exp(feature[next_tag] + self.conditional_probs[next_tag, curr_tag])
                feature_score += tag_i_feature_score
            curr_tag = target_tags[tag_index]
            total_score += torch.log(feature_score)

        stop_tag_score=self.generate_scalar_tensor(0)
        for last_tag in range(num_tags):
            stop_tag_score+=torch.exp(self.conditional_probs[last_tag, curr_tag])
        return total_score + torch.log(stop_tag_score)

#     Given a list of features (representing the words in a sentence) and their corr. tags
    # score the sentence
    # DONE
    def calculate_sentence_score(self, features, tags):
        # Gives the score of a provided tag sequence
#         print("features: ",features)
#         print("tags: ",tags)
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([tag_to_ix[START_TAG]], dtype=torch.long), tags])
        feature_dict = enumerate(features)
#         print(feature_dict)
        for i, feature in feature_dict:
#             print("i: ",i)
#             print("tags[i + 1]: ",tags[i + 1])
#             print("feat[tags[i + 1]: ",feature[tags[i + 1]])
            score += self.conditional_probs[tags[i + 1], tags[i]] + feature[tags[i + 1]]
        score += self.conditional_probs[tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def argmax(self,x):
    # returns argmax of x as int
        _, idx = torch.max(x, 1)
        if not type(idx.item()) == int:
            raise TypeError('Wrong Type')
        return idx.item()

    def viterbi_backtrace(self, optimal_tag_idx, backpointers_end_to_start):
        optimal_path = [optimal_tag_idx]
        for backptrs in backpointers_end_to_start:
            best_tag_id = backptrs[optimal_tag_idx]
            optimal_path.append(best_tag_id)

        # Remove start tag
        start = optimal_path.pop()
#         assert start == tag_to_ix[START_TAG]
#         print('start: ',start)

        optimal_path.reverse()
#         print('finished viterbi')
#         print('final viterbi path: ',optimal_path)
        return optimal_path

    def viterbi_dp(self, features):
#         print('starting viterbi')
        backpointers_matrix = []
        # Init start scores in log space
        init_first_col = torch.full((1, num_tags), -15000.)
        init_first_col[0][tag_to_ix[START_TAG]] = 0
        # at step i, viterbi_matrix holds viterbi vars for step i-1
        viterbi_matrix = init_first_col # pi, or forward_var in recurrence relation

        for feature in tqdm(features): # for word in sentence
            # print('feat: ',feat)
            step_i_backpointers = []  # holds the backpointers for this step
            step_i_vvars = []  # holds the viterbi variables for this step

            for possible_tag in range(num_tags): # for each possible tag the word feature could be
                possible_tag_var = viterbi_matrix + self.conditional_probs[possible_tag]
#                 print('bwooohahhhahahha ',self.conditional_probs[possible_tag])
                best_tag_idx = self.argmax(possible_tag_var)
                step_i_backpointers.append(best_tag_idx)
                step_i_vvars.append(possible_tag_var[0][best_tag_idx].view(1))

            step_i_viterbi_col = torch.cat(step_i_vvars)
            viterbi_matrix = (step_i_viterbi_col + feature).view(1, -1)
            backpointers_matrix.append(step_i_backpointers)
        # Calculate score to stop tag
        last_col = viterbi_matrix + self.conditional_probs[tag_to_ix[STOP_TAG]]
        best_tag_idx = self.argmax(last_col)
        optimal_score = last_col[0][best_tag_idx]
#         print('last_col: ',last_col)

        def viterbi_backtrace(best_tag_idx, backpointers_end_to_start):
            optimal_path = [optimal_tag_idx]
            for backptrs in backpointers_end_to_start:
                best_tag_id = backptrs[optimal_tag_idx]
                optimal_path.append(best_tag_id)
            # Remove start tag
            start = optimal_path.pop()
            assert start == tag_to_ix[START_TAG]
#             print('start: ',start)
            optimal_path.reverse()
            return optimal_path

        best_path = self.viterbi_backtrace(best_tag_idx, reversed(backpointers_matrix))
        return best_path, optimal_score

    def get_NLL(self, sentence, tags):
        sentence_features = self.get_bilstm_features(sentence)
        gold = self.calculate_sentence_score(sentence_features, tags)
        memm_score = self.memm(sentence_features, tags)
        return memm_score - gold

    def forward(self, sentence):
        lstm_features = self.get_bilstm_features(sentence)

        # Find the best path, given the features.
        tag_seq, score  = self.viterbi_dp(lstm_features)
        return score, tag_seq

    def predict(self,sentence):
        lstm_features=self.get_bilstm_features(sentence)
        print('viterbi input: lstm_features',lstm_features)
        tag_seq, score  = self.viterbi_dp(lstm_features)
        return score, tag_seq
#         path=self.viterbi_dp(lstm_features)
#         return path
START_TAG = "<START>"
STOP_TAG = "<STOP>"

EMBEDDING_DIM = 15
HIDDEN_DIM = 10
# TRAIN = True # add this last

bilstm_test_training_data = training_data
import random
import sklearn.model_selection
bilstm_sentences = bilstm_test_training_data

random.shuffle(bilstm_sentences)
train_size= int(len(bilstm_sentences)*0.8)
test_size=len(bilstm_sentences)-train_size
train_dataset, val_dataset = sklearn.model_selection.train_test_split(bilstm_sentences, train_size=train_size, test_size=test_size)
print('len(val_dataset) ', len(val_dataset))
print('len(train_dataset) ' ,len(train_dataset))

if LOAD == 1:
    model = torch.load(path)
    model.eval()
    print(model)
else:
    model = BiLSTM_MEMM(len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    # doesn't take in data
def train_epoch(model, optimizer, batch_size=1):
    model.train()
    losses = []
    for sentence, tags in tqdm(train_dataset):
        model.zero_grad()
        # Get our inputs ready for the network; turn them into Tensors of word indices
        sentence_in = sentence_to_ix(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        # print('targets: ',targets)
        # Run forward pass.
        loss = model.get_NLL(sentence_in, targets)
        # print('loss: ',loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())
    return [sum(losses)/float(len(losses))]

# plot the losses to see that the model is learning
e_losses = []
num_epochs = 1
for e in range(num_epochs):
    e_losses += train_epoch(model, optimizer)
print('e_losses: ',e_losses)
# Check predictions before training
with torch.no_grad():
    precheck_sent = sentence_to_ix(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
# import time
# start = time.process_time()
#
# print("TRAINING")
# # ...
# print("DONE TRAINING")
# print('Time taken: ',time.process_time() - start)

# plt.plot(e_losses)
def find_TP_FP_TF_FN(pred,true):
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
    if (tp+fp==0):
        return "N/A"
    return tp/(tp+fp)

def recall(tp, fn):
    if (tp+fp==0):
        return "N/A"
    return tp/(tp+fn)

def f1_score(tp,fp,tf,fn):
    print('tp,fp,tf,fn: ',tp,fp,tf,fn)
    if (tp+fp) ==0:
        return "Precision N/A"
    if (tp+fn) ==0:
        return "Recall N/A"
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    return 2*prec*rec/(prec+rec)


# torch.save(model, path)

match=0
no_match=0
tp,fp,tf,fn=0,0,0,0


with torch.no_grad():
    print("first sentence!")
    for sentence in testing_data[:15]:
        # need to create new word_to_ix for testing data
        sentence_in = sentence_to_ix(sentence[0], word_to_ix)
        sentence_in
        score,viterbi_path = model.predict(sentence_in)
        print('viterbi_path,',viterbi_path )
        print('sentence,', sentence[1])
        assert(len(viterbi_path)==len(sentence[1]))
        viterbi_preds = [sentiment_tags[item] for item in viterbi_path]
        y=sentence[1]
        for i in range(len(viterbi_preds)):
            tp_i,fp_i,tf_i,fn_i = find_TP_FP_TF_FN(viterbi_preds[i], y[i])
            tp+=tp_i
            fp+=fp_i
            tf+=tf_i
            fn+=fn_i

print('tp,fp,tf,fn: ',tp,fp,tf,fn)
f1 = f1_score(precision(tp, fp), recall(tp, fn))
print('f1: ',f1)
