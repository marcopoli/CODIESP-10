# I file che servono sono tutti i cap_Train.csv e stop_word.pck , entrambi presenti in
# https://drive.google.com/drive/folders/1g0luJ9T0pzvjfDYejHMBtXAAb7HZXtLn?usp=sharing
import warnings

from sklearn.svm import SVC

warnings.filterwarnings("ignore")

from boto import sns
from pandas import read_csv
#from tqdm import tqdm  # barra di progresso
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing  # per lo scale dei x e y train
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


def tokenize_text(text):
    # Acquisizione delle stop word
    file_stopw = open("support/stop_word.pck", "rb")
    stop_word = pickle.load(file_stopw)
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens


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


# Cap è il capitolo che si vuole analizzare
def classifier():
    # Seguiremo questo:
    # https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
    df = read_csv('Training/Task2/Train_with_emptyclass_Task2.csv')
    df = df[['Code', 'Desc']]
    # df = df[pd.notnull(df['desc'])]
    print(df.head(10))
    print(df.shape)

    df.index = range(df.shape[0])
    print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 211456 parole

    cnt_pro = df['Code'].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Code', fontsize=8)
    plt.xticks(rotation=90)
    # plt.show()    # se lo metto si potrebbe impallare, quindi non lo metto così esce alla fine dell'esecuzione
    #plt.savefig('Diagram_Cap' + str(cap) + '.png')

    df['Desc'] = df['Desc'].apply(remove_symbol)
    print(df.head(10))
    train = df
        #, test = train_test_split(df, test_size=0.3, random_state=42)
    df = read_csv('TestSet.csv')
    test = df[['id', 'Desc']]

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.id]), axis=1)

    # Distributed Memory (DM)
    model_dmm = Doc2Vec(dm=1, dm_mean=1, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                        min_alpha=0.065)
    model_dmm.build_vocab([x for x in train_tagged.values])

    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in train_tagged.values]), total_examples=len(train_tagged.values),
                        epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha

    y_train, X_train = vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = vec_for_learning(model_dmm, test_tagged)
    logreg = SVC(C=0.01,kernel ='rbf', verbose=True,probability=True)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    probs = logreg.predict_log_proba(X_test)
    import joblib
    joblib.dump(y_pred,'y_pred.vec')
    joblib.dump(probs, 'probs.vec')
    joblib.dump(y_test, 'ids.vec')




#classifier()
import joblib
y_pred=joblib.load('y_pred.vec')
probs=joblib.load('probs.vec')
ids =joblib.load('ids.vec')

print(ids)
print(y_pred)
print(probs)
