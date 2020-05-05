# Serve per creare i file CSV per ogni capitolo
# Il file che si useranno sono il Train.csv e dic_caps.pck presenti in
# https://drive.google.com/drive/folders/1g0luJ9T0pzvjfDYejHMBtXAAb7HZXtLn?usp=sharing

import csv
import pickle

dic_caps = pickle.load(open("support/dic_caps.pck", "rb"))


class code_desc:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc
        self.cap = dic_caps[str(code.split(".")[0]).upper()]

    def __str__(self):
        return str(self.cap) + " " + str(self.code) + " -> " + str(self.desc)


l_caps = []
with open('Train.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    first = True
    for riga in csv_reader:
        if first:
            # saltiamo la prima riga, quella di intestazione
            first = False
            continue
        if str(riga["Code"]).split(".")[0].upper() in dic_caps:
            l_caps.append(code_desc(str(riga["Code"]), str(riga["Desc"])))

for i in range(len(l_caps)):
    print(l_caps[i])
print(len(l_caps))


for capitolo in range(0, 21):
    with open('cap' + str(capitolo) + 'Train.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # scriviamo prima la riga di intestazione
        csv_writer.writerow(['Code', 'Desc'])
        # aggiungiamo ora i dati
        for j in range(len(l_caps)):
            if str(l_caps[j].cap) == str(capitolo):
                csv_writer.writerow([str(l_caps[j].code), str(l_caps[j].desc)])
