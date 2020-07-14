# @title Preparation
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

from collections import OrderedDict

import numpy as np
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
df = pd.read_csv('TestSet.csv')
df = df[['id', 'Desc']]
# df = df[pd.notnull(df['desc'])]
# print(df.head(10))
print(df.shape)

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
test = df

# , test = train_test_split(df, test_size=0.3, random_state=42)

# prepare class encoder
le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
# labels = le.fit(list(df['id']))
mapa = [0, 1]

labels_map = [0, 1]
# i = 0
# for a in mapa:
#    labels_map.append(a)
# print(labels_map)

# Tokenization
# Inizialize the tokenizer
from bert import bert_tokenization

tokenizer = bert_tokenization.FullTokenizer(vocab_path, do_lower_case=True)
# tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)
# indices_train = []
indices_test = []

# for text in train['Desc']:
#  tk = tokenizer.tokenize(text)
#  tokens = ["[CLS]"] + tk + ["[SEP]"]
#  token_ids = tokenizer.convert_tokens_to_ids(tokens)
#  token_ids = _pad(token_ids,SEQ_LEN)
#  indices_train.append(token_ids)

for text in test['Desc']:
    tk = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tk + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = _pad(token_ids, SEQ_LEN)
    indices_test.append(token_ids)

# indices_train = [indices_train, np.zeros_like(indices_train)]
indices_test = [indices_test, np.zeros_like(indices_test)]
ids = test['id']
print(ids)
# train_labels = train['Code']
# train_labes_indexes = []
# for label in train_labels:
#  if(label =='emp'):
#    train_labes_indexes.append(0)
#  else:
#    train_labes_indexes.append(1)


bert = load_trained_model_from_checkpoint(
    config_file=config_path,
    checkpoint_file=checkpoint_path,
    training=True,
    trainable=True,
    seq_len=128
)

inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
dense1 = keras.layers.Dense(units=1000, activation='tanh')(dense)
outputs = keras.layers.Dense(units=2, activation='softmax')(dense1)
modelk = keras.models.Model(inputs, outputs)
modelk.compile(
    RAdam(lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

# @title Initialize Variables
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)

modelk.load_weights('bert_only_01.00003-0.14749.hdf5')

predicts = modelk.predict(indices_test, verbose=True)
print(predicts[0])

res_encoded = []
for a in predicts:
    val = a.argmax()
    res_encoded.append(labels_map[val])

print(res_encoded)

# Estraggo elementi di cui si crede ci sia la patologia
left_to_classify = []
# left_to_classify_labels = []
left_to_classify_ids = []
res_encoded_original = res_encoded.copy()

print(test.head(10))
test_desc = test['Desc']
test_bal = test['id']

print(test_bal[0])
i = 0
for item in res_encoded:
    if item == 1:
        left_to_classify.append(test_desc[i])
        left_to_classify_ids.append(test_bal[i])
    i = i + 1

print(len(left_to_classify))
print(len(left_to_classify_ids))

# -------
# Tokenization
# Inizialize the tokenizer
from bert import bert_tokenization

tokenizer = bert_tokenization.FullTokenizer(vocab_path, do_lower_case=True)
indices_test = []
for text in left_to_classify:
    tk = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tk + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = _pad(token_ids, SEQ_LEN)
    indices_test.append(token_ids)
indices_test = [indices_test, np.zeros_like(indices_test)]

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
outputs = keras.layers.Dense(units=1789, activation='softmax')(dense1)

modelk = keras.models.Model(inputs, outputs)
modelk.compile(
    RAdam(lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

modelk.load_weights('bert_model_alltrain_senza_emp.hdf5')

res = modelk.predict(indices_test, verbose=True)
print(res[0])
print(len(res))

from collections import OrderedDict

max_vals = {}
extracted_res = []
extractede_ids = []

df = pd.read_csv('Training/Task1/Train.csv')
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
train = df
le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
labels = le.fit(list(df['Code']))
# mapa = [0,1]
map = le.category_mapping[0]['mapping']

labels_map = list(map.keys())

j = 0
previous_id = left_to_classify_ids[0]
for result in res:
    id = left_to_classify_ids[j]

    if id != previous_id:

        od = OrderedDict(max_vals)
        # print(str(id)+' '+str(od))

        for key, value in od.items():
            extracted_res.append(value.lower())
            extractede_ids.append(previous_id)
        max_vals = {}
        previous_id = id

    i = 0
    for a in result:
        if a > 0.10:
            max_vals[str(a)] = labels_map[i]
            i = i + 1
    if len(max_vals) == 0:
        v = result.max()
        index = result.argmax()
        max_vals[str(v)] = labels_map[index]

    j = j + 1

# Last item
od = OrderedDict(max_vals)

for key, value in od.items():
    extracted_res.append(value.lower())
    extractede_ids.append(previous_id)

print(extracted_res)
print(extractede_ids)

file = open('prediction_lstm.tsv', 'w+')
for i in range(0, len(extracted_res)):
    file.write(str(extractede_ids[i]).replace('.txt', '') + '\t' + str(extracted_res[i]) + '\n')
    file.flush()

file.close()
