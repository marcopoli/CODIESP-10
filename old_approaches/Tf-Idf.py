import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from math import sqrt
import csv
# nltk.download('wordnet')                      <- serve solo la prima volta
from nltk.tokenize import TweetTokenizer
# from nltk.stem.snowball import SnowballStemmer
import pickle  # per la memorizzazione della lista


# https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76 per tfidf

def cossim(a, b):
    num = 0.0
    for i in range(0, len(a)):
        num = num + (a[i] * b[i])
    den = 0.0
    temp1 = 0.0
    temp2 = 0.0
    for i in range(0, len(a)):
        temp1 = temp1 + (a[i] * a[i])
        temp2 = temp2 + (b[i] * b[i])
    den = sqrt(temp1) * sqrt(temp2)
    return num / den


# rimozione dei simboli inutili da una stringa
def remove_symbol(s):
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.replace(";", "")
    s = s.replace("_", "")
    s = s.replace(":", "")
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
    return s


# Restituisce la stringa del testo contenuto nel file con il nome "name" dato in input dopo aver fatto lo stemming
# e rimosso parole e simboli inutili
def analyze_txt(name):
    # Acquisizione e Tokenizzazione e "lemmatizzazione" del testo libero scritto in txt
    txtSentence = open(name, "r").readlines()  # lista che contene le frasi che stanno nel txt
    txtWord = []  # lista di parole del file txt
    for i in range(0, len(txtSentence)):
        temp = txtSentence[i].split(" ")  # lista di parole che stanno nel testo
        for j in range(0, len(temp)):
            txtWord.append(temp[j].lower())

    # Rimozione di simboli inutili da txtWord
    for i in range(0, len(txtWord)):
        txtWord[i] = remove_symbol(txtWord[i])
        txtWord[i] = txtWord[i].replace("\n", "")

    # Rimozione delle word vuote
    i = 0
    try:
        while True:
            if txtWord[i] == '':
                txtWord.pop(i)
            else:
                i += 1
    except:
        print("")

    # Rimozione delle stopword dal testo
    z = 0
    while z < len(stop_word):
        if stop_word[z] in txtWord:
            txtWord.pop(txtWord.index(stop_word[z]))
        else:
            z += 1

    # Lemmatizzazione del testo libero (ossia di txtWord) e trasformazione in minuscolo
    # stemmer = SnowballStemmer("spanish")
    temp = ''
    for i in range(0, len(txtWord)):
        if len(txtWord[i]) > 3:
            txtWord[i] = str(txtWord[i].lower())  # stemmer.stem(str(txtWord[i].lower()))
            temp += txtWord[i]
            temp += ' '
    return temp[0:len(temp) - 1]


class ICD10CM:
    def __init__(self, pre_code, post_code, esp, eng):
        self.pre_code = pre_code  # codice prima del punto
        self.post_code = post_code  # codice dopo il punto
        self.esp = esp
        self.eng = str(eng)


class code_score:
    def __init__(self, name_code, score):
        self.name_code = name_code
        self.score = score

    def __str__(self):
        return "Codice: " + str(self.name_code) + "(+1) - Score: " + str(self.score)


# Fase di valutazione   NON SERVE
# oggetto che indica la valutazione
class PRF:
    def __init__(self, precision, recall, f1_measure):
        self.precision = precision
        self.recall = recall
        self.f1_measure = f1_measure


# Acquisizione della lista l_word_caps
file_pck = open("l_word_caps.pck", "rb")
l_word_caps = pickle.load(file_pck)

# Acquisizione del dizionario
file_dic = open("dic_caps.pck", "rb")
dic_caps = pickle.load(file_dic)

# Acquisizione delle stop word
file_stopw = open("stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)

# Creazione di una lista che contiene tutti nomi dei testi clinici txt
all_txt = open("All_txt.txt", "r").readlines()
for i in range(0, len(all_txt)):
    all_txt[i] = str(all_txt[i]).replace("\n", "")

# TODO: non possiamo usare la precision e recall perchè in questo caso non avrebbero senso, sarebbero sempre basse
# Usando tutti i nomi dei file txt (presenti in all_txt) calcoliamo le precision@k
precisions = []
for text in range(0, len(all_txt)):
    # Inserimento della parole del testo clinico in l_word_caps
    l_word_caps.append(analyze_txt(str(all_txt[text]) + '.txt'))

    # calcola il tf-idf e retituisce il dataframe
    vectorizer = TfidfVectorizer(smooth_idf=False)
    vectors = vectorizer.fit_transform(l_word_caps)
    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    dense = vectors.todense()
    denselist = dense.tolist()
    print(denselist)
    print(len(denselist[21]))  # sarebbe la Q
    df = pd.DataFrame(denselist, columns=feature_names)
    # print(df) # mostra la matrice

    l_cossim = []
    for i in range(0, len(denselist) - 1):
        l_cossim.append(code_score(i, cossim(denselist[i], denselist[21])))
        # print(l_cossim[i])

    # ordinamento decrescente della lista l_cossim con il bubble sort
    n = len(l_cossim)
    for i in range(0, n):
        for j in range(0, n - i - 1):
            if l_cossim[j].score < l_cossim[j + 1].score:
                temp = l_cossim[j]
                l_cossim[j] = l_cossim[j + 1]
                l_cossim[j + 1] = temp

    # visualizzazione
    print("l_cossim")
    for i in range(0, len(l_cossim)):
        print(l_cossim[i])

    # acquisizione delle istanze dal file chiamato file_train_name
    csvfile = open("train_annotations_codiespD_processed.tsv")
    dialect = csv.Sniffer().sniff(
        csvfile.read(1024))  # serve per capire il dialect del tsv (nel read si può mettere 1024)
    csvfile.seek(0)
    readerCsv = csv.reader(csvfile, dialect)

    # lettura delle istanze ed inserimento in una lista
    l_txt_code = []
    txt_name = all_txt[text]
    for ROW in readerCsv:
        if ROW[0] == txt_name:
            if str(ROW[2]).find(".") > -1:
                l_txt_code.append(dic_caps[str(ROW[2]).upper().split(".")[0]])  # inserisco le categorie di ogni istanza
            else:
                l_txt_code.append(dic_caps[str(ROW[2]).upper()])  # inserisco le categorie di ogni istanza

    # inserimento nella lista l_score la coppia (capitolo,score)
    l_score = []
    for i in range(0, len(l_txt_code)):
        l_score.append(code_score(l_txt_code[i], len(l_txt_code) - i))

    # accorpamento degli score per capitoli
    for i in range(0, len(l_score) - 1):
        for j in range(i + 1, len(l_score)):
            if l_score[i].name_code == l_score[j].name_code:
                l_score[i].score += l_score[j].score
                l_score[j].score = 0

    # ordiniamo la lista l_score
    n = len(l_score)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if l_score[j].score < l_score[j + 1].score:
                temp = l_score[j]
                l_score[j] = l_score[j + 1]
                l_score[j + 1] = temp

    # rimozione delle istanze inutili
    i = 0
    while i < len(l_score):
        if l_score[i].score == 0:
            break
        else:
            i += 1

    try:
        while True:
            l_score.pop(i)
    except:
        print("La lista è stata elaborata con successo")

    # visualizziamo l_score
    print("l_score")
    for i in range(0, len(l_score)):
        print(l_score[i])

    # calcoliamo la precision@k dove k = len(l_score)
    prec = 0.0
    for i in range(0, len(l_score)):
        for j in range(0, len(l_score)):
            if l_cossim[i].name_code == l_score[j].name_code:
                prec += 1
    prec = prec / len(l_score)
    print("prec@" + str(len(l_score)) + "= " + str(prec))

    precisions.append(prec)

    # rimozione da l_word_caps le parole che fanno riferimento a testo clinico (ossia quella con indice 21
    l_word_caps.pop()  # non mettere niente è uguale a togliere l'ultima istanza

# provare sia con solo word che con sia word che abstract
# provare sia con smooth idf che senza
# provare sia con stemming che senza
somma = 0
for i in range(0, len(precisions)):
    print(precisions[i])
    somma += precisions[i]
print("La media è: " + str(somma / len(precisions)))
