import joblib
import pandas as pd
from keras import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
# from keras_multi_head import MultiHead
import random
from keras.layers import Input, Flatten, TimeDistributed
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

# Acquisizione delle stop word
# file_stopw = open("support/stop_word.pck", "rb")
# stop_word = pickle.load(file_stopw)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(df['Desc'])
joblib.dump(t, 'binary_tokenizer.vec')

vocab_size = len(t.word_index) + 1

# prepare class encoder
labels = [0, 1]
print(labels)


# integer encode the documents
# encoded_train = t.texts_to_sequences(train['Desc'])

# max_length = 64
# padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
# print(padded_train)

# test_ids = df1['id']
test_ids = test['id']

# train_labels = train['Code']
# print(train_labels)
# encoded_train_labels = list(train_labels)

# max_length = 64
# padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
# print(padded_train)

# test_ids = df1['id']
# test_ids = test['Code']

# train_labels = train['Code']
# print(train_labels)
# encoded_train_labels = train_labels
# le.inverse_transform(encoded_train_labels)

# integer encode the documents

encoded_test = t.texts_to_sequences(test['Desc'])
max_length = 64
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
# print(padded_test)

# test_labels = test['Code']
# print(test_labels)

# padded_test = joblib.load('padded_test.vec')
# test_labels = joblib.load('test_labels.vec')
# padded_train = joblib.load('padded_train.vec')
# encoded_train_labels = joblib.load('encoded_train_labels.vec')


# le = joblib.load('label_encoder_le.vec')

# LOAD WORDEMBEDDING

print('load_embeddings...')
# get the vectors

file = open('embeddings-l-model.vec', encoding='utf8')

# create a weight matrix for words in training docs
count = 0
embedding_matrix = np.zeros((vocab_size, 300))
vocab_and_vectors = {}
arrValues = []
z = 0
for line in file:
    if (z != 0):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        vocab_and_vectors[word] = vector
        arrValues.append(vector)
    else:
        z = z + 1
for word, i in t.word_index.items():
    try:
        embedding_vector = vocab_and_vectors.get(word)
        if embedding_vector is None:
            count = count + 1
            # max = len(google_300.vocab.keys()) - 1
            index = random.randint(0, 1000)
            # word = google_300.index2word[index]
            embedding_vector = arrValues[index]
    except:
        # keep a random embedidng
        count = count + 1
        # max = len(google_300.vocab.keys()) - 1
        index = random.randint(0, 1000)
        # word = google_300.index2word[index]
        embedding_vector = arrValues[index]
        # google_300.word_vec(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(count)
# joblib.dump(embedding_matrix, 'binary_emb_matrix.vec')
# embedding_matrix = joblib.load('binary_emb_matrix.vec')


# joblib.dump(padded_test,'lstm_fasttext__padded_test.vec')
# joblib.dump(test_labels,'lstm_fasttext__test_labels.vec')
# joblib.dump(padded_train,'lstm_fasttext__padded_train.vec')
# joblib.dump(encoded_train_labels,'lstm_fasttext__encoded_train_labels.vec')
# joblib.dump(le,'lstm_fasttext__label_encoder_le_Task2.vec')

# embedding_matrix = joblib.load('embedding_matrix.vec')
# padded_test = joblib.load('padded_test.vec')
# test_labels = joblib.load('test_labels.vec')
# padded_train = joblib.load('padded_train.vec')
# encoded_train_labels = joblib.load('encoded_train_labels.vec')
# le = joblib.load('label_encoder_le_task2.vec')

# define the model
input = Input(shape=(64,))

# - BiLSTM + CNN + attention
from keras_self_attention import SeqSelfAttention

m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False)(input)
bi = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, dropout=0.3))(m)
aa = SeqSelfAttention(attention_activation='tanh') (bi)
aa = Conv1D(128, 5, activation='relu')(aa)
aa = MaxPool1D(2)(aa)
aa = Dropout(0.2)(aa)
added = keras.layers.Concatenate(axis=1)([aa, bi])
ff = GlobalMaxPool1D()(added)
ff = Dense(4000)(ff)
ff = Dropout(0.3)(ff)
ff = Dense(2, activation='softmax')(ff)

model = keras.models.Model(inputs=[input], outputs=[ff])

model.summary(line_length=100)

from keras.callbacks import CSVLogger

# filepath="LSTM_CNN_ATT_Fasttext_03052020weights.{epoch:05d}-{val_loss:.5f}"+'_07_'+".hdf5"
filepath = "LSTM_prova"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [
    checkpoint
]

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('biLSTM_CNN_Attention_binario.hdf5')

res = model.predict(padded_test)
for i in range(0, 10):
    print(res[i])

# labels_map = []
labels_map = [0, 1]
i = 0
res_encoded = []
for a in res:
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


# Tokenizzazione e rimozione delle stopword
def tokenize_text(text):
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens


t = joblib.load("lstm_fasttext__tokenizer.vec")

encoded_left_to_classify = t.texts_to_sequences(left_to_classify)

max_length = 64
padded_left_to_classify = pad_sequences(encoded_left_to_classify, maxlen=max_length, padding='post')
print(padded_left_to_classify[6])

le = joblib.load("lstm_fasttext__label_encoder_le.vec")

embedding_matrix = joblib.load('lstm_fasttext__embedding_matrix_medical.vec')

vocab_size = len(t.word_index) + 1
input = Input(shape=(64,))
m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False) (input)
bi = Bidirectional(LSTM(64, activation ='tanh', return_sequences = True, dropout=0.3)) (m)

aa = SeqSelfAttention(attention_activation='tanh') (bi)
aa = Conv1D(128,5, activation ='relu' ) (aa)
aa = MaxPool1D(2) (aa)
aa = Dropout(0.2) (aa)

added = keras.layers.Concatenate(axis=1)([aa,bi])

ff = GlobalMaxPool1D() (added)
ff = Dense(4000)(ff)
ff = Dropout(0.3) (ff)
ff =Dense(1788, activation='softmax') (ff)

model = keras.models.Model(inputs=[input], outputs=[ff])

model.summary(line_length=100)
model.compile (loss='categorical_crossentropy' , optimizer='adam' , metrics=[ 'accuracy'] )

model.load_weights('LSTM_CNN_ATT_Fasttext_final_03052020.h5')

res = model.predict(padded_left_to_classify)
print(res[0])

from collections import OrderedDict

max_vals = {}
extracted_res = []
extractede_ids = []
map = le.category_mapping[0]['mapping']
labels_map = []
i = 0
for a, b in map:
    labels_map.append(a)

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
  file.write(str(extractede_ids[i]).replace('.txt', '')+'\t'+str(extracted_res[i])+'\n')
  file.flush()

file.close()

