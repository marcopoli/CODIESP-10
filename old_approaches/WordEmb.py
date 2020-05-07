import pandas as pd
from tqdm import tqdm  # barra di progresso
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import pickle
import multiprocessing
from sklearn.metrics import accuracy_score, f1_score
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models import doc2vec
from sklearn import preprocessing  # per lo scale dei x e y train


def print_code(index):
    example = df[df.index == index][['Code', 'Desc']].values[0]
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


# Creazione del vettore finale
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


# Seguiremo questo:
# https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
df = pd.read_csv('Train.csv')
df = df[['Code', 'Desc']]
# df = df[pd.notnull(df['desc'])]
print(df.head(10))
print(df.shape)

df.index = range(df.shape[0])
print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

print_code(1000)  # visualizza codice e frase in posizione 1000

# cnt_pro = df['code'].value_counts()
# plt.figure(figsize=(12, 4))
# sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('code', fontsize=12)
# plt.xticks(rotation=90)
# plt.show()

df['Desc'] = df['Desc'].apply(remove_symbol)
print(df.head(10))
train, test = train_test_split(df, test_size=0.3, random_state=42)
# Acquisizione delle stop word
file_stopw = open("stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)


def tokenize_text(text):
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens


train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)

print(train_tagged.values[30])

# Distributed Bag of Words (DBOW)
cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, workers=cores)

# creazione del vocabolario
tqdm.pandas(desc="progress-bar")
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

# scale dei vettori
for i in range(0, model_dbow.docvecs.count):
    slot = model_dbow.docvecs.int_index(i, model_dbow.docvecs.doctags, model_dbow.docvecs.max_rawint)
    model_dbow.docvecs.vectors_docs[slot] = preprocessing.scale(model_dbow[i])

# training per 10 epochs
for epoch in range(10):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                     epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# Train del classificatore di regressione logistica
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.1565934065934066
# Testing F1 score: 0.14939370761635604

print(logreg.predict([model_dbow.infer_vector(tokenize_text('fractura proximal'), steps=20)]))
# ['n48.89']

# Distributed Memory (DM)
model_dmm = Doc2Vec(dm=1, dm_mean=1, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                    min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

# scale dei vettori
for i in range(0, model_dmm.docvecs.count):
    slot = model_dmm.docvecs.int_index(i, model_dmm.docvecs.doctags, model_dmm.docvecs.max_rawint)
    model_dmm.docvecs.vectors_docs[slot] = preprocessing.scale(model_dmm[i])

for epoch in range(10):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                    epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.11734693877551021
# Testing F1 score: 0.10642747269276188

print(logreg.predict([model_dmm.infer_vector(tokenize_text('fractura proximal'), steps=20)]))
# ['s52.102']

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
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
# Testing accuracy 0.13304552590266877
# Testing F1 score: 0.1199050728244342

# print(logreg.predict([model_dmm.infer_vector(tokenize_text('fractura proximal'), steps=20)])) non funziona
# Errore: ValueError: X has 100 features per sample; expecting 200
