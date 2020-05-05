# Obiettivo: creare un classificatore binario che usi training frasi che fanno parte di quella classe e quelle non
# (lavora a livello di capitoli) e tirare fuori anche le probabilità (tipo logreg.predict_proba())

from boto import sns
from pandas import read_csv
from sklearn.svm import SVR, SVC
from tqdm import tqdm  # barra di progresso
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing  # per lo scale dei x e y train
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import csv
import pickle


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


dic_caps = pickle.load(open("support/dic_caps.pck", "rb"))


class code_desc:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc
        # self.cap = dic_caps[str(code.split(".")[0]).upper()]

    def __str__(self):
        return str(self.code) + " -> " + str(self.desc)
        # return str(self.cap) + " " + str(self.code) + " -> " + str(self.desc)


def createAllCSV4BinClassifier(cap):
    l_caps = []
    # Inseriamo le tuple che appartengono al capitolo cap
    with open('cap' + str(cap) + 'Train.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        first = True
        for riga in csv_reader:
            if first:
                # saltiamo la prima riga (quella di intestazione)
                first = False
                continue
            if str(riga["Code"]).split(".")[0].upper() in dic_caps:  # Controllo inutile...
                l_caps.append(code_desc("1", str(riga["Desc"])))

    # Inseriamo le tuple che appartengono agli altri capitoli
    for i in range(0, 21):
        if i != cap:
            with open('cap' + str(i) + 'Train.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                first = True
                for riga in csv_reader:
                    if first:
                        # saltiamo la prima riga (quella di intestazione)
                        first = False
                        continue
                    if str(riga["Code"]).split(".")[0].upper() in dic_caps:  # Controllo inutile...
                        l_caps.append(code_desc("0", str(riga["Desc"])))

    # Creazione del file di training csv del tipo BinClass_cap0Train.csv
    with open('BinClass_cap' + str(cap) + 'Train.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # scriviamo prima la riga di intestazione
        csv_writer.writerow(['Code', 'Desc'])
        # aggiungiamo ora i dati
        for i in range(0, len(l_caps)):
            csv_writer.writerow([str(l_caps[i].code), str(l_caps[i].desc)])

prec = []
def BinClassifer(cap):
    df = read_csv('Training/BinClass/BinClass_cap' + str(cap) + 'Train.csv')
    df = df[['Code', 'Desc']]
    # df = df[pd.notnull(df['desc'])]
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head(10))
    print(df.shape)

    df.index = range(df.shape[0])
    print("Parole: " + str(df['Desc'].apply(lambda x: len(x.split(' '))).sum()))  # ci sono circa 210535 parole

    cnt_pro = df['Code'].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Code', fontsize=12)
    plt.xticks(rotation=90)
    # plt.show()    # se lo metto si potrebbe impallare, quindi non lo metto così esce alla fine dell'esecuzione
    # plt.savefig('Diagram_Cap' + str(cap) + '.png')

    df['Desc'] = df['Desc'].apply(remove_symbol)
    print(df.head(10))
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Desc']), tags=[r.Code]), axis=1)

    # Distributed Memory (DM)
    model_dmm = Doc2Vec(dm=1, dm_mean=1, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                        min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                        epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha

    model_dmm.save('Model_DMM_BinClassCap' + str(cap) + '.bin') # salvo il modello

    y_train, X_train = vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = vec_for_learning(model_dmm, test_tagged)
    model = SVC(kernel='rbf',C=0.1, verbose=True, probability=True)
    model.fit(X_train, y_train)
    model.predict_log_proba
    pickle.dump(model, open('Model_SVM_BinClassCap' + str(cap) + '.bin', 'wb'))

    y_pred = model.predict(X_test)
    # y_pred = logreg.predict_proba(X_test)
    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
    prec.append(accuracy_score(y_test, y_pred))
    prec.append(f1_score(y_test, y_pred, average='weighted'))

    # Testing accuracy 0.8443656422379827   per cap = 0
    # Testing F1 score: 0.8599951797112411
    # print(logreg.predict([model_dmm.infer_vector(tokenize_text('En cabeza y cuello no se palpan adenopatías, ni bocio'
    #                                                         ' ni ingurgitación de vena yugular, con pulsos carotídeos'
    #                                                         'simétricos'), steps=20)]))
    # ['e04.9']


# for i in range(0, 21):
#     createAllCSV4BinClassifier(i)

for i in range(0, 21):
    print('Run numero ' + str(i+1))
    BinClassifer(i)

i = 0
while i < len(prec):
    print('Accuracy del Cap' + str(i) + '= ' + str(prec[i]))
    i += 1
    print('F1 del Cap' + str(i-1) + '= ' + str(prec[i]))
    i += 1
