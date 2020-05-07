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
from keras.layers import Conv1D , Embedding
from keras.layers import Dropout
from keras.layers import MaxPool1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPool1D
from keras.layers import Bidirectional
import category_encoders as ce

from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention

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

#LOADING DATASET
df = pd.read_csv('Train.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
print(df.head(10))
print(df.shape)

df.index = range(df.shape[0])
print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
df['Desc'] = df['Desc'].apply(remove_symbol)
print(df.head(10))

train, test = train_test_split(df, test_size=0.3, random_state=42)

# Acquisizione delle stop word
file_stopw = open("stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(df['Desc'])
vocab_size = len(t.word_index) + 1

#prepare class encoder
le = ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
labels = le.fit(list(df['Code']))
print(labels)

print(le.category_mapping)
print(len(le.category_mapping))


# integer encode the documents
encoded_train = t.texts_to_sequences(train['Desc'])

max_length = 256
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
print(padded_train)

train_labels = train['Code']
print(train_labels)
encoded_train_labels = le.transform(list(train_labels))
#le.inverse_transform(encoded_train_labels)

# integer encode the documents
encoded_test = t.texts_to_sequences(test['Desc'])

max_length = 256
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_test)

test_labels = test['Code']
print(test_labels)


#LOAD WORDEMBEDDING
import gensim
#google_300 = gensim.models.KeyedVectors.load_word2vec_format("cc.es.300.vec")

# create a weight matrix for words in training docs
#count = 0
#embedding_matrix = np.zeros((vocab_size, 300))
#for word, i in t.word_index.items():
#    try:
#        embedding_vector = google_300.word_vec(word)
#    except:
#        #keep a random embedidng
#        count = count+1
#        max = len(google_300.vocab.keys()) - 1
#        index = random.randint(0, max)
#        word = google_300.index2word[index]
#        embedding_vector = google_300.word_vec(word)

#    if embedding_vector is not None:
#        embedding_matrix[i] = embedding_vector

#joblib.dump(embedding_matrix,'embedding_matrix.vec')
#joblib.dump(padded_test,'padded_test.vec')
#joblib.dump(test_labels,'test_labels.vec')

#joblib.dump(padded_train,'padded_train.vec')
#joblib.dump(encoded_train_labels,'encoded_train_labels.vec')

#joblib.dump(le,'label_encoder_le.vec')

embedding_matrix = joblib.load('embedding_matrix.vec')
padded_test = joblib.load('padded_test.vec')
test_labels = joblib.load('test_labels.vec')
padded_train = joblib.load('padded_train.vec')
encoded_train_labels = joblib.load('encoded_train_labels.vec')
le = joblib.load('label_encoder_le.vec')

# define the model
input = Input(shape=(256,))
m = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=256, trainable=False) (input)


bi = Bidirectional(LSTM(256, activation ='tanh', return_sequences = True, dropout=0.3)) (m)

aa = SeqSelfAttention(attention_activation='tanh') (bi)
aa = Conv1D(512,5, activation ='relu' ) (aa)
aa = MaxPool1D(2) (aa)
aa = Dropout(0.2) (aa)

added = keras.layers.Concatenate(axis=1)([aa,bi])

ff = GlobalMaxPool1D() (added)
ff = Dense(2000)(ff)
ff = Dropout(0.3) (ff)
ff =Dense(1788, activation='softmax') (ff)

model = keras.models.Model(inputs=[input], outputs=[ff])

model.summary(line_length=100)

from keras.callbacks import CSVLogger
filepath="28042020weights.{epoch:05d}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max',period=2)

callbacks_list = [
    checkpoint
]

model.compile (loss='categorical_crossentropy' , optimizer='adam' , metrics=[ 'accuracy'] )
#print(padded_train)
#history = model.fit(padded_train,encoded_train_labels,128,50,
#                      validation_split = 0.10,
#                      callbacks=callbacks_list ,
#                      verbose=1)
#model.load_weights('LSTM_CNN_ATTENTION_28042020weights.00048-6.76415_0.2118.hdf5')
#model.save('28042020model.h5')

#res = model.predict(padded_test)
#joblib.dump(res,'results_prediction.vec')

res = joblib.load('results_prediction.vec')

map = le.category_mapping[0]['mapping']

labels_map = []
i = 0
for a,b in map:
    labels_map.append(a)


res_encoded = []
for a in res:
    val = a.argmax()
    res_encoded.append(labels_map[val])


#print(encoded_train_labels)
print(res_encoded)
#res_labels = le.inverse_transform(DataFrame.res_encoded)

print('Testing accuracy %s' % accuracy_score(test_labels, res_encoded))
print('Testing F1 score: {}'.format(f1_score(test_labels, res_encoded, average='weighted')))

#Testing accuracy 0.1836734693877551
#Testing F1 score: 0.16066030988784247