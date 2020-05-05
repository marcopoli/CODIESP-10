import csv
import pickle

dic_caps_Task2 = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                  'B': 10, 'C': 11, 'D': 12, 'F': 13, 'G': 14, 'H': 15}


class code_desc:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc
        self.cap = dic_caps_Task2[str(code[0]).upper()]

    def __str__(self):
        return str(self.cap) + " " + str(self.code) + " -> " + str(self.desc)


l_caps = []
with open('Train_Task2.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    first = True
    for riga in csv_reader:
        if first:
            # saltiamo la prima riga, quella di intestazione
            first = False
            continue
        if str(riga["Code"][0]).upper() in dic_caps_Task2:
            l_caps.append(code_desc(str(riga["Code"]), str(riga["Desc"])))

for i in range(len(l_caps)):
    print(l_caps[i])
print(len(l_caps))

for capitolo in range(0, 16):
    with open('cap' + str(capitolo) + 'Train_Task2.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # scriviamo prima la riga di intestazione
        csv_writer.writerow(['Code', 'Desc'])
        # aggiungiamo ora i dati
        for j in range(len(l_caps)):
            if str(l_caps[j].cap) == str(capitolo):
                csv_writer.writerow([str(l_caps[j].code), str(l_caps[j].desc)])
