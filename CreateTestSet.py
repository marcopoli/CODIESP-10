import csv
import os

class code_desc:
    def __init__(self, id_diagnosi,desc):
        self.id_diagnosi = id_diagnosi
        self.desc = desc

    def __str__(self):
        return str(self.code) + " -> " + str(self.desc)


def TestSet():
    all_txt = os.listdir("/Users/vincenzosuriano/Desktop/Tesi/final_dataset_v3_to_publish/test/text_files/")
    all_row = []
    for i in range(0, len(all_txt)):
        text = open('/Users/vincenzosuriano/Desktop/Tesi/final_dataset_v3_to_publish/test/text_files/' +
                    str(all_txt[i]), "r").read()
        # text è la stringa che contiene il testo clinico

        text = text.split('\n')
        text = "".join(text)
        while text.find(".") > 0:
            # if keeptext.find('.') != -1:
            all_row.append(code_desc(all_txt[i], text[:text.find('.')]))
            text = text[text.find('.') + 1:]
            print("Lunghezza di keeptext: " + str(len(text)))
            print(all_txt[i])
    print('OKAY')
    print(all_row)
    # Scrittura in un file csv di all_row
    with open('TestSet.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Scriviamo prima la riga di intestazione
        csv_writer.writerow(['Id_Diagnosi', 'Frase'])
        # aggiungiamo ora i dati
        for i in range(0, len(all_row)):
            csv_writer.writerow([str(all_row[i].id_diagnosi), str(all_row[i].desc)])

TestSet()
