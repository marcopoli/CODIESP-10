# @title Preparation
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

from collections import OrderedDict

import tensorflow as tf
import keras
from keras_radam import RAdam
# from keras_bert import get_custom_objects
import numpy as np
from tqdm import tqdm
# from keras_bert import Tokenizer
import pandas as pd
import tensorflow.keras.backend as K
import sys
from sklearn.metrics import classification_report

# from google.colab import drive

# @title Constants
np.random.seed(42)
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5

# @title Environment
import os

pretrained_path = 'pretrained/'
config_path = os.path.join(pretrained_path, 'config.json')
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-2000000')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# @title Load Basic Model

# import python modules defined by BERT
# from run_classifier import *
# import modeling
# import optimization
from bert import tokenization

import codecs
from keras_bert import load_trained_model_from_checkpoint

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# @title Load Data
# ## !pip install category_encoders==1.3.0
import joblib
import pandas as pd
from keras import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import random
from keras.layers import Input
import keras
from keras.layers import Conv1D, Embedding
from keras.layers import Dropout
from keras.layers import MaxPool1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPool1D
from keras.layers import Bidirectional
import category_encoders as ce

from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention


def remove_symbol(s):
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.replace(";", "")
    s = s.replace(":", "")
    s = s.replace("_", "")
    s = s.replace("+", "")
    s = s.replace("ª", "")
    s = s.replace("-", "")
    s = s.replace("<", "")
    s = s.replace(">", "")
    s = s.replace("!", "")
    s = s.replace("?", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("'", "")
    s = s.replace("0", "")
    s = s.replace("1", "")
    s = s.replace("2", "")
    s = s.replace("3", "")
    s = s.replace("4", "")
    s = s.replace("5", "")
    s = s.replace("6", "")
    s = s.replace("7", "")
    s = s.replace("8", "")
    s = s.replace("9", "")
    s = s.replace("%", "")
    s = s.strip()
    s = s.lower()
    return s


# Rimozione delle stopword
def clar_text(text):
    t = remove_symbol(str(text).strip().lower())
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))

    tt = ""
    for it in tokens:
        tt = tt + " " + it
    return tt


def _pad(input_ids, max_seq_len):
    x = []
    input_ids = input_ids[:min(len(input_ids), max_seq_len - 2)]
    input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
    return np.array(input_ids)


# LOADING DATASET
df = pd.read_csv('Training/Task1/Train_with_emptyclass.csv')
# df = pd.read_csv('Training/Task1/Train.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
# print(df.head(10))
# print(df.shape)

df.index = range(df.shape[0])
print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
df['Desc'] = df['Desc'].apply(remove_symbol)
# print(df.head(10))

# Acquisizione delle stop word
file_stopw = open("support/stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)
df['Desc'] = df['Desc'].apply(clar_text)

# suddivisione train_test
# train = df
train, test = train_test_split(df, test_size=0.2, random_state=42)

# df1 = pd.read_csv('left_TestSet.csv')
# df1 = df1[['id', 'Desc']]
# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
# df1['Desc'] = df1['Desc'].apply(remove_symbol)
# df1['Desc'] = df1['Desc'].apply(clar_text)
# test = df1

# prepare class encoder
le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
labels = le.fit(list(df['Code']))
# mapa = [0,1]
mapa = le.category_mapping[0]['mapping']
print(le.category_mapping)
labels_map = []
i = 0
labels_map = list(mapa.keys())
print(labels_map)

# Tokenization
# Inizialize the tokenizer
from bert import bert_tokenization
tokenizer = bert_tokenization.FullTokenizer(vocab_path, do_lower_case=True)
indices_train = []
indices_test = []

for text in train['Desc']:
    tk = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tk + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = _pad(token_ids, SEQ_LEN)
    indices_train.append(token_ids)
# print(tk)
print(indices_train[0])
for text in test['Desc']:
    tk = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tk + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = _pad(token_ids, SEQ_LEN)
    indices_test.append(token_ids)

indices_train = [indices_train, np.zeros_like(indices_train)]
indices_test = [indices_test, np.zeros_like(indices_test)]

train_labels = train['Code']
train_labes_indexes = []
for label in train_labels:
    train_labes_indexes.append(labels_map.index(label))

test_ids = test['Code']

# test_labes_indexes = []
# for label in test_labels:
#  test_labes_indexes.append(labels_map.index(label))

# print(test_labels)

len(labels_map)

bert = load_trained_model_from_checkpoint(
    config_file=config_path,
    checkpoint_file=checkpoint_path,
    training=True,
    trainable=True,
    seq_len=128
)

# @title Build Custom Model

inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
dense1 = keras.layers.Dense(units=1000, activation='tanh')(dense)
outputs = keras.layers.Dense(units=len(labels_map), activation='softmax')(dense1)

modelk = keras.models.Model(inputs, outputs)
modelk.compile(
    RAdam(lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

# @title Fit

# filepath = "/content/drive/My Drive/codiesp/bert_with_empty.{epoch:05d}-{val_loss:.5f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True,
#                              mode='max')

# callbacks_list = [
#     checkpoint
# ]
modelk.fit(
    indices_train,
    train_labes_indexes,
    epochs=20,
    batch_size=32,
    validation_split=0.10
    # callbacks=callbacks_list
)

res = modelk.predict(indices_test, verbose=True)

res_encoded = []
for a in res:
    val = a.argmax()
    res_encoded.append(labels_map[val])

print(res_encoded[0])


from sklearn.metrics import classification_report

print(classification_report(test_ids, res_encoded, digits=5))
