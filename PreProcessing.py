import csv
# nltk.download('wordnet')                      <- serve solo la prima volta
from nltk.tokenize import TweetTokenizer
# from nltk.stem.snowball import SnowballStemmer
import pickle  # per la memorizzazione della lista
import json


class ICD10CM:
    def __init__(self, pre_code, post_code, esp, eng):
        self.pre_code = pre_code  # codice prima del punto
        self.post_code = post_code  # codice dopo il punto
        self.esp = esp
        self.eng = str(eng)


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
    return s


# inserire l'abstract e le word nel capitolo giusto di l_word_caps grazie al dizionario dic_caps
def getJson(l_word_caps, dic_caps, stop_word):
    data = json.load(open("abstractsWithCIE10_v2.json"))
    for a in range(0, len(data['articles'])):  # len(data['articles']) !!!!!!!!!!!!
        print(str(a) + "/" + str(len(data['articles'])))
        for b in range(0, len(data['articles'][a])):
            try:
                for c in range(0, len(data['articles'][a]['Mesh'][b])):
                    try:
                        for d in range(0, len(data['articles'][a]['Mesh'][b]['CIE'])):
                            # prendo il pre_code del codice
                            pre_c = str(data['articles'][a]['Mesh'][b]['CIE'][d]).split(".")[0]  # pre_code
                            if pre_c in dic_caps:
                                abstract = str(data['articles'][a]['abstractText'])
                                # stemmer = SnowballStemmer("spanish")
                                tokenizer = TweetTokenizer()
                                abstract = remove_symbol(abstract)
                                list_token_sentence = tokenizer.tokenize(str(abstract).lower())
                                # list_token_sentence e' una lista di parole della frase

                                # rimozione delle stopword
                                for z in range(0, len(stop_word)):
                                    if stop_word[z] in list_token_sentence:
                                        while stop_word[z] in list_token_sentence:
                                            list_token_sentence.remove(str(stop_word[z]))

                                # creazione stringa escludendo le parole con un numero di lettere < di 3
                                temp = ''
                                for k in range(0, len(list_token_sentence)):
                                    if len(list_token_sentence[k]) > 3:
                                        temp += str(list_token_sentence[k])  # stemmer.stem(str(list_token_sentence[k]))
                                        temp += ' '
                                abstract = temp[0:len(abstract) - 1]

                                list_word = tokenizer.tokenize(
                                    remove_symbol(str(data['articles'][a]['Mesh'][b]['Word']).lower()))
                                word = ''
                                for i in range(0, len(list_word)):
                                    word += str(list_word[i])  # stemmer.stem(str(list_word[i]))
                                    word += ' '
                                word = word[0:len(word) - 1]  # per togliere l'ultimo spazio

                                l_word_caps[dic_caps[pre_c]] = str(
                                    l_word_caps[dic_caps[pre_c]]) + " " + str(word) + " " + str(abstract)
                    except:
                        a = a  # non fare niente
            except:
                a = a  # non fare niente
    return l_word_caps


# Fase di acquisizione dal file tsv che contiene l'oggetto di cui sopra
csvfile = open("codiesp-D_codes.tsv")
dialect = csv.Sniffer().sniff(csvfile.read())  # serve per capire il dialect del tsv (nel read prima mettevo 1024)
csvfile.seek(0)
readerCsv = csv.reader(csvfile, dialect)

# Creazione degli oggetti ed inserimento all'interno di una lista
CM_Codes = []
for row in readerCsv:
    complete_code = []
    if str(row[0]).find(".") > -1:
        complete_code = row[0].split(".")
        CM_Codes.append(ICD10CM(complete_code[0], complete_code[1], row[1], row[2]))
    else:
        CM_Codes.append(ICD10CM(row[0], "", row[1], row[2]))

# Inserimento delle stopword in una lista
# stemmer = SnowballStemmer("spanish")
stop_word = open("Stop_Word_Esp.txt", "r").readlines()  # lista che contiene le stopword in inglese
for i in range(0, len(stop_word)):
    stop_word[i] = str(stop_word[i].replace("\n", ""))  # stemmer.stem(str(stop_word[i].replace("\n", "")))

# Fase di tokenizzazione, rimozione delle stop word delle frasi inglesi e lemmatizzazione di queste.
# Inserimento nella lista dei cap i termini per ogni capitolo
print("Inizio fase 1")
l_word_caps = []  # lista che conterrà le stringhe dei termini per ogni capitolo
dic_caps = {}  # dizionario che conterrà la coppia pre_code - capitolo
terms_CM_Codes = []  # lista di termini
tokenizer = TweetTokenizer()
cap = 0
for i in range(0, len(CM_Codes)):
    if CM_Codes[i].pre_code != 'Cap':
        # inserimento della coppia pre_code-capitolo e lettera-capitolo nei dizionarii
        dic_caps[CM_Codes[i].pre_code] = cap

        # eliminazione della punteggiatura
        CM_Codes[i].esp = remove_symbol(CM_Codes[i].esp)

        list_token_sentence = tokenizer.tokenize(str(CM_Codes[i].esp).lower())  # e' una lista di parole della frase
        # rimozione delle stopword
        for z in range(0, len(stop_word)):
            if stop_word[z] in list_token_sentence:
                while stop_word[z] in list_token_sentence:
                    list_token_sentence.remove(str(stop_word[z]))

        # creazione stringa
        temp = ''
        for k in range(0, len(list_token_sentence)):
            if len(list_token_sentence[k]) > 3:
                temp += str(list_token_sentence[k])  # stemmer.stem(str(list_token_sentence[k]))
                temp += ' '
        temp = temp[0:len(temp) - 1]
        CM_Codes[i].esp = temp
        try:
            l_word_caps[cap] = l_word_caps[cap] + temp
        except:
            l_word_caps.append(temp)

        # for j in range(0, len(list_token_sentence)):
        #   terms_CM_Codes.append(str(list_token_sentence[j]))
    else:
        cap += 1

# Visualizzazione delle descrizioni in spagnolo
for i in range(0, len(CM_Codes)):
    print(CM_Codes[i].esp)

# Acquisizione dal file json ed aggiunta di contenuti in l_word_caps
print("Inizio fase 2")
# print(len(l_word_caps[0]))
# print(len(l_word_caps[1]))
# print(len(l_word_caps[2]))

l_word_caps = getJson(l_word_caps, dic_caps, stop_word)

# print(len(l_word_caps[0]))
# print(len(l_word_caps[1]))
# print(len(l_word_caps[2]))

# Memorizziamo l_word_caps in modo che non si rifaccia il pre-processing
file_pck = open("l_word_caps.pck", "wb")
pickle.dump(l_word_caps, file_pck)
print("File memorizzato con successo")

# Memorizziamo dic_caps
file_dic = open("dic_caps.pck", "wb")
pickle.dump(dic_caps, file_dic)
print("Dizionario salvato con successo")

# Memorizziamo stop_word
file_stopw = open("stop_word.pck", "wb")
pickle.dump(stop_word, file_stopw)
print("Stop word salvate con successo")
