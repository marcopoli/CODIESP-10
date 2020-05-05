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
vers =0
# Tokenizzazione e rimozione delle stopword
def tokenize_text(text):
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens

#LOADING DATASET
df = pd.read_csv('Training/Caps/Cap'+str(vers)+'Train.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
print(df.head(10))
print(df.shape)

df.index = range(df.shape[0])

# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
df['Desc'] = df['Desc'].apply(remove_symbol)
train = df

#Acquisizione delle stop word
file_stopw = open("support/stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)
t = joblib.load('lstm_fasttext__tokenizer_total.vec')
vocab_size = len(t.word_index) + 1

#prepare class encoder
le = ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
labels = le.fit(list(df['Code']))
joblib.dump(le,str(vers)+'_le.joblib')

print(labels)
print(le.category_mapping)
print(len(le.category_mapping))


# integer encode the documents
encoded_train = t.texts_to_sequences(train['Desc'])
max_length = 64
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
print(padded_train)

train_labels = train['Code']
print(train_labels)
print(len(le.category_mapping[0]['mapping']))
encoded_train_labels = le.transform(list(train_labels))

embedding_matrix = joblib.load('lstm_fasttext__embedding_matrix_medical_total.vec')

# define the model
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
ff =Dense(len(le.category_mapping[0]['mapping']), activation='softmax') (ff)

model = keras.models.Model(inputs=[input], outputs=[ff])

model.summary(line_length=100)

from keras.callbacks import CSVLogger
filepath="LSTM_CNN_ATT_Fasttext_03052020weights.{epoch:05d}-{val_loss:.5f}"+'_07_'+".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [
    checkpoint
]

model.compile (loss='categorical_crossentropy' , optimizer='adam' , metrics=[ 'accuracy'] )
history = model.fit(padded_train,encoded_train_labels,512,300,
                      validation_split = 0.10,
                      callbacks=callbacks_list ,
                      verbose=1)

model.save('LSTM_CNN_ATT_Fasttext_final_03052020'+'_07_'+'.h5')

res = model.predict(padded_train)
map = le.category_mapping[0]['mapping']

labels_map = []
i = 0
for a,b in map:
    labels_map.append(a)


res_encoded = []
for a in res:
    val = a.argmax()
    res_encoded.append(labels_map[val])

print(res_encoded)
print('Testing accuracy %s' % accuracy_score(train_labels, res_encoded))
print('Testing F1 score: {}'.format(f1_score(train_labels, res_encoded, average='weighted')))

from sklearn.metrics import classification_report
print(classification_report(train_labels, res_encoded,digits=5))