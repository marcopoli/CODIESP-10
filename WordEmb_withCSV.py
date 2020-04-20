import pandas as pd
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import pickle
import multiprocessing
from sklearn.metrics import accuracy_score, f1_score
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import numpy as np
from tqdm import tqdm  # barra di progresso

tqdm.pandas(desc="progress-bar")


def print_code(index):
    example = df[df.index == index][['Desc', 'Code']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Code:', example[1])


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


# Creazione del vettore finale
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


df = pd.read_csv('Train.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
print(df.head(10))
print(df.shape)

df.index = range(df.shape[0])
print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

print_code(1000)  # visualizza codice e frase in posizione 1000

# rimozione SOLO dei simboli (nessuno stemming e nessuna rimozione delle stopword)
df['Desc'] = df['Desc'].apply(remove_symbol)
print(df.head(10))

train, test = train_test_split(df, test_size=0.3, random_state=42)

# Acquisizione delle stop word
file_stopw = open("stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)

# Creazione dei train e test, tokenizzazione e rimozione delle stopword (Nessuna rimozione delle parole con lung < 2)
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)

print(train_tagged.values[30])

# 1° MODELLO
# Distributed Memory (DM)
model_dmm = Doc2Vec(dm=1, dm_mean=1, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                    min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

# scale dei vettori
for i in range(0, model_dmm.docvecs.count):
    slot = model_dmm.docvecs.int_index(i, model_dmm.docvecs.doctags, model_dmm.docvecs.max_rawint)
    model_dmm.docvecs.vectors_docs[slot] = preprocessing.scale(model_dmm[i])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                    epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)

X_test = preprocessing.scale(np.array(X_test))
X_train = preprocessing.scale(np.array(X_train))

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# todo: problema: da questo messaggio di warning:
# /Library/Python/3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to
# converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.11891679748822606 con 3 epoche
# Testing F1 score: 0.11396692610730397

print(logreg.predict([model_dmm.infer_vector(tokenize_text('fractura proximal'), steps=20)]))
# h05.20

# ------------------------------

# 2° MODELLO
# Distributed Bag of Words (DBOW)
cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, workers=cores)

# creazione del vocabolario
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
# scale dei vettori
for i in range(0, model_dbow.docvecs.count):
    slot = model_dbow.docvecs.int_index(i, model_dbow.docvecs.doctags, model_dbow.docvecs.max_rawint)
    model_dbow.docvecs.vectors_docs[slot] = preprocessing.scale(model_dbow[i])

# training per 30 epochs
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                     epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# Train del classificatore di regressione logistica
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

X_test = preprocessing.scale(np.array(X_test))
X_train = preprocessing.scale(np.array(X_train))

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.22648361381753763
# Testing F1 score: 0.1613008616293813

print(logreg.predict([model_dbow.infer_vector(tokenize_text('fractura proximal'), steps=20)]))


# ---------------
# Puliamo la RAM
model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

# Model paring
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])  # Concatenazione dei due modelli


def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


# training regressione logistica
y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)

X_test = preprocessing.scale(np.array(X_test))
X_train = preprocessing.scale(np.array(X_train))

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.388503100088574
# Testing F1 score: 0.29748627519545125

print(logreg.predict([model_dbow.infer_vector(tokenize_text('fractura proximal'), steps=20)]))

