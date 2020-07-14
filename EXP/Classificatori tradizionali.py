import pandas as pd
from tqdm import tqdm  # barra di progresso
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from gensim.models.doc2vec import TaggedDocument
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score
# import multiprocessing
# from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
# from sklearn import preprocessing  # per lo scale dei x e y train
from sklearn.metrics import classification_report

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


df = pd.read_csv('Training/Task1/Train_with_emptyclass.csv')
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
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Acquisizione delle stop word
file_stopw = open('support/stop_word.pck', "rb")
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

model_dmm = Doc2Vec(dm=1, dm_mean=1, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                    min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])


# for epoch in range(30):
#     model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
#                     epochs=1)
#     model_dmm.alpha -= 0.002
#     model_dmm.min_alpha = model_dmm.alpha

print("Load modello doc2vec")
model_dmm = Doc2Vec.load("0Model_Dmm.bin")

y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)

print("Creazione modello")

# model = DecisionTreeClassifier()

model = LogisticRegression(n_jobs=1, C=1e5, max_iter=3000)

# model = SVC(kernel='rbf', C=0.1, verbose=True, probability=True)

# model = RandomForestClassifier()

# model = AdaBoostClassifier()

print("fit del modello")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing recall %s' % recall_score(y_test, y_pred, average='weighted'))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print(classification_report(y_test, y_pred, digits=5))
