import joblib
import pandas as pd
from keras import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, f1_score
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


# from keras_self_attention import SeqSelfAttention


# rimozione dei simboli inutili da una stringa
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


# Tokenizzazione e rimozione delle stopword
def tokenize_text(text):
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens


# LOADING DATASET
df = pd.read_csv('Training/Task1/Train_with_emptyclass.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
print(df.head(10))
print(df.shape)

df.index = range(df.shape[0])
print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
df['Desc'] = df['Desc'].apply(remove_symbol)
print(df.head(10))

train, test = train_test_split(df, test_size=0.2, random_state=42)

# df1 = pd.read_csv('TestSet.csv')
# df1 = df1[['id', 'Desc']]
# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
# df1['Desc'] = df1['Desc'].apply(remove_symbol)
# test = df1

# Acquisizione delle stop word
file_stopw = open("support/stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(df['Desc'])

vocab_size = len(t.word_index) + 1

# prepare class encoder
le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
labels = le.fit(list(df['Code']))
print(labels)

# integer encode the documents
encoded_train = t.texts_to_sequences(train['Desc'])

max_length = 64
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
print(padded_train)

# test_ids = df1['id']
test_ids = test['Code']

train_labels = train['Code']
# print(train_labels)
encoded_train_labels = le.transform(list(train_labels))
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

joblib.dump(t, 't.vec')
joblib.dump(le, 'le.vec')

# le = joblib.load('label_encoder_le.vec')

# LOAD WORDEMBEDDING

print('load_embeddings...')
# get the vectors

file = open('embeddings-l-model.vec', encoding='utf8')
# file = open('cc.es.300.vec', encoding='utf8')
# file = open('glove-sbwc.i25.vec', encoding='utf8')

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
joblib.dump(embedding_matrix, 'e.vec')
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

# - BiLSTM
# m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False)(input)
# bi = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, dropout=0.3))(m)
# bi = Flatten()(bi)
# ff = Dense(4000)(bi)
# ff = Dropout(0.3)(ff)
# ff = Dense(1789, activation='softmax')(ff)

# - BiLSTM + CNN
# m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False)(input)
# bi = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, dropout=0.3))(m)
# aa = Conv1D(128, 5, activation='relu')(bi)
# aa = MaxPool1D(2)(aa)
# aa = Dropout(0.2)(aa)
# added = keras.layers.Concatenate(axis=1)([aa, bi])
# ff = GlobalMaxPool1D()(added)
# ff = Dense(4000)(ff)
# ff = Dropout(0.3)(ff)
# ff = Dense(1789, activation='softmax')(ff)

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
ff = Dense(1789, activation='softmax')(ff)

# - CNN
# m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False)(input)
# aa = Conv1D(128, 5, activation='relu')(m)
# aa = MaxPool1D(2)(aa)
# aa = Dropout(0.2)(aa)
# added = keras.layers.Concatenate(axis=1)([aa, bi])
# ff = GlobalMaxPool1D()(aa)
# ff = Dense(4000)(ff)
# ff = Dropout(0.3)(ff)
# ff = Dense(1789, activation='softmax')(ff)

# - CNN + Attention
# m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=64, trainable=False)(input)
# aa = SeqSelfAttention(attention_activation='tanh')(m)
# aa = Conv1D(128, 5, activation='relu')(aa)
# aa = MaxPool1D(2)(aa)
# aa = Dropout(0.2)(aa)
# added = keras.layers.Concatenate(axis=1)([aa, bi])
# ff = GlobalMaxPool1D()(aa)
# ff = Dense(4000)(ff)
# ff = Dropout(0.3)(ff)
# ff = Dense(1789, activation='softmax')(ff)

model = keras.models.Model(inputs=[input], outputs=[ff])

model.summary(line_length=100)

from keras.callbacks import CSVLogger

# filepath="LSTM_CNN_ATT_Fasttext_03052020weights.{epoch:05d}-{val_loss:.5f}"+'_07_'+".hdf5"
filepath = "LSTM_prova"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [
    checkpoint
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(padded_train, encoded_train_labels, 512, 120,
                    validation_split=0.10,
                    callbacks=callbacks_list,
                    verbose=1)

model.save('BiLSTM_CNN_Attention_multiclass_train.hdf5')

print('salvato')
res = model.predict(padded_test)
map = le.category_mapping[0]['mapping']

# labels_map = []
labels_map = list(map.keys())
i = 0

res_encoded = []
for a in res:  # ,b
    val = a.argmax()
    res_encoded.append(labels_map[val])

print(res_encoded)
print('Testing accuracy %s' % accuracy_score(test_ids, res_encoded))
print('Testing F1 score: {}'.format(f1_score(test_ids, res_encoded, average='weighted')))

from sklearn.metrics import classification_report

print(classification_report(test_ids, res_encoded, digits=5))
