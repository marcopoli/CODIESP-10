import csv


class code_desc:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc

    def __str__(self):
        return str(self.code) + " -> " + str(self.desc)


def TrainCSV():
    csvfile = open("train_annotations_codiespX_processed.tsv")
    dialect = csv.Sniffer().sniff(csvfile.read())  # serve per capire il dialect del tsv (nel read prima mettevo 1024)
    csvfile.seek(0)
    readerCsv = csv.reader(csvfile, dialect)

    all_row = []
    for row in readerCsv:
        if row[1] == 'DIAGNOSTICO':
            print("Analizzo...")
            # dobbiamo cercare la porzione di testo nel file txt
            text = open("/Users/vincenzosuriano/Desktop/Tesi/train_dev/train/text_files/"
                        + str(row[0]) + ".txt", "r").read()
            # text è la stringa che contiene il testo clinico

            rif = str(row[4]).replace(";", " ").split(" ")
            if len(rif) == 2:
                pos1 = int(rif[0])
                pos2 = int(rif[1])
                # bisogna prendere da punto a punto. quindi bisogna riconoscere ". ", è importante lo spazio perchè ci
                #  sono casi in cui c'è una formula e quindi non si avranno gli effetti voluti
                b = pos1  # per andare indietro (before)
                flag = False
                while b != 0 and flag is False:
                    if text[b] == " ":
                        if text[b - 1] == ".":
                            pos1 = b + 1
                            flag = True
                        else:
                            b -= 1
                    else:
                        b -= 1
                    if text[b] == '\n':
                        pos1 = b + 1
                        flag = True
                if b == 0:
                    pos1 = b

                a = pos2  # per andare avanti (after)
                flag = False
                while a != len(text) and flag is False:
                    if text[a] == ".":
                        if text[a + 1] == " ":
                            pos2 = a - 1
                            flag = True
                        else:
                            a += 1
                    else:
                        a += 1
                    if text[a] == '\n':
                        pos2 = a - 1
                        flag = True

                all_row.append(code_desc(str(row[2]), text[pos1:pos2 + 1]))
            else:
                if len(rif) == 4:
                    pos1 = int(rif[0])
                    pos2 = int(rif[1])
                    pos3 = int(rif[2])
                    pos4 = int(rif[3])
                    b = pos1  # per andare indietro (before)
                    flag = False
                    while b != 0 and flag is False:
                        if text[b] == " ":
                            if text[b - 1] == ".":
                                pos1 = b + 1
                                flag = True
                            else:
                                b -= 1
                        else:
                            b -= 1
                        if text[b] == '\n':
                            pos1 = b + 1
                            flag = True
                    if b == 0:
                        pos1 = b

                    a = pos2  # per andare avanti (after)
                    flag = False
                    while a != len(text) and flag is False:
                        if text[a] == ".":
                            if text[a + 1] == " ":
                                pos2 = a - 1
                                flag = True
                            else:
                                a += 1
                        else:
                            a += 1
                        if text[a] == '\n':
                            pos2 = a - 1
                            flag = True

                    all_row.append(code_desc(str(row[2]), text[pos1:pos2 + 1]))  # aggiungo la prima parte

                    b = pos3  # per andare indietro (before)
                    flag = False
                    while b != 0 and flag is False:
                        if text[b] == " ":
                            if text[b - 1] == ".":
                                pos3 = b + 1
                                flag = True
                            else:
                                b -= 1
                        else:
                            b -= 1
                        if text[b] == '\n':
                            pos3 = b + 1
                            flag = True
                    if b == 0:
                        pos3 = b

                    a = pos4  # per andare avanti (after)
                    flag = False
                    while a != len(text) and flag is False:
                        if text[a] == ".":
                            if text[a + 1] == " ":
                                pos4 = a - 1
                                flag = True
                            else:
                                a += 1
                        else:
                            a += 1
                        if text[a] == '\n':
                            pos2 = a - 1
                            flag = True

                    all_row.append(code_desc(str(row[2]), text[pos3:pos4 + 1]))  # aggiungo la seconda parte

    for i in range(0, len(all_row)):
        print(all_row[i])

    # Scrittura in un file csv di all_row
    with open('Train.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Scriviamo prima la riga di intestazione
        csv_writer.writerow(['Code', 'Desc'])
        # aggiungiamo ora i dati
        for i in range(0, len(all_row)):
            csv_writer.writerow([str(all_row[i].code), str(all_row[i].desc)])


def TrainCSV_with_emptyclass():
    csvfile = open("train_annotations_codiespX_processed.tsv")
    dialect = csv.Sniffer().sniff(csvfile.read())  # serve per capire il dialect del tsv (nel read prima mettevo 1024)
    csvfile.seek(0)
    readerCsv = csv.reader(csvfile, dialect)

    tempfile = ''
    keeptext = ''
    all_row = []
    for row in readerCsv:
        if row[1] == 'DIAGNOSTICO':
            print("Analizzo...")
            # dobbiamo cercare la porzione di testo nel file txt
            text = open("/Users/vincenzosuriano/Desktop/Tesi/train_dev/train/text_files/"
                        + str(row[0]) + ".txt", "r").read()
            # text è la stringa che contiene il testo clinico

            if tempfile != str(row[0]):
                if keeptext != '':
                    # keeptext.replace('\n', '')
                    keeptext = keeptext.split('\n')
                    keeptext = "".join(keeptext)
                    while keeptext.find('.') > 0:
                        #if keeptext.find('.') != -1:
                        all_row.append(code_desc('empty', keeptext[:keeptext.find('.')]))
                        keeptext = keeptext[keeptext.find('.') + 1:]
                        print("Lunghezza di keeptext: " + str(len(keeptext)))
                        print(str(row[0]))

                print('OKAY')
                keeptext = text
                tempfile = str(row[0])

            rif = str(row[4]).replace(";", " ").split(" ")
            if len(rif) == 2:
                pos1 = int(rif[0])
                pos2 = int(rif[1])
                # bisogna prendere da punto a punto. quindi bisogna riconoscere ". ", è importante lo spazio perchè
                #  ci sono casi in cui c'è una formula e quindi non si avranno gli effetti voluti
                b = pos1  # per andare indietro (before)
                flag = False
                while b != 0 and flag is False:
                    if text[b] == " ":
                        if text[b - 1] == ".":
                            pos1 = b + 1
                            flag = True
                        else:
                            b -= 1
                    else:
                        b -= 1
                    if text[b] == '\n':
                        pos1 = b + 1
                        flag = True
                if b == 0:
                    pos1 = b

                a = pos2  # per andare avanti (after)
                flag = False
                while a != len(text) and flag is False:
                    if text[a] == ".":
                        if text[a + 1] == " ":
                            pos2 = a - 1
                            flag = True
                        else:
                            a += 1
                    else:
                        a += 1
                    if text[a] == '\n':
                        pos2 = a - 1
                        flag = True

                all_row.append(code_desc(str(row[2]), text[pos1:pos2 + 1]))
                keeptext = keeptext[:pos1 - 1] + keeptext[pos2 + 1:]
                #  print(keeptext)

            else:
                if len(rif) == 4:
                    pos1 = int(rif[0])
                    pos2 = int(rif[1])
                    pos3 = int(rif[2])
                    pos4 = int(rif[3])
                    b = pos1  # per andare indietro (before)
                    flag = False
                    while b != 0 and flag is False:
                        if text[b] == " ":
                            if text[b - 1] == ".":
                                pos1 = b + 1
                                flag = True
                            else:
                                b -= 1
                        else:
                            b -= 1
                        if text[b] == '\n':
                            pos1 = b + 1
                            flag = True
                    if b == 0:
                        pos1 = b

                    a = pos2  # per andare avanti (after)
                    flag = False
                    while a != len(text) and flag is False:
                        if text[a] == ".":
                            if text[a + 1] == " ":
                                pos2 = a - 1
                                flag = True
                            else:
                                a += 1
                        else:
                            a += 1
                        if text[a] == '\n':
                            pos2 = a - 1
                            flag = True

                    all_row.append(code_desc(str(row[2]), text[pos1:pos2 + 1]))  # aggiungo la prima parte
                    # keeptext = keeptext[:pos1 - 1] + keeptext[pos2 + 1:]
                    # print(keeptext)

                    b = pos3  # per andare indietro (before)
                    flag = False
                    while b != 0 and flag is False:
                        if text[b] == " ":
                            if text[b - 1] == ".":
                                pos3 = b + 1
                                flag = True
                            else:
                                b -= 1
                        else:
                            b -= 1
                        if text[b] == '\n':
                            pos3 = b + 1
                            flag = True
                    if b == 0:
                        pos3 = b

                    a = pos4  # per andare avanti (after)
                    flag = False
                    while a != len(text) and flag is False:
                        if text[a] == ".":
                            if text[a + 1] == " ":
                                pos4 = a - 1
                                flag = True
                            else:
                                a += 1
                        else:
                            a += 1
                        if text[a] == '\n':
                            pos4 = a - 1
                            flag = True

                    all_row.append(code_desc(str(row[2]), text[pos3:pos4 + 1]))  # aggiungo la seconda parte

                    if pos1 > pos3:
                        keeptext = keeptext[:pos1 - 1] + keeptext[pos2 + 1:]
                        keeptext = keeptext[:pos3 - 1] + keeptext[pos4 + 1:]
                    else:
                        keeptext = keeptext[:pos3 - 1] + keeptext[pos4 + 1:]
                        keeptext = keeptext[:pos1 - 1] + keeptext[pos2 + 1:]
                    # print(keeptext)

    for i in range(0, len(all_row)):
        print(all_row[i])

    # Scrittura in un file csv di all_row
    with open('Train_with_emptyclass.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Scriviamo prima la riga di intestazione
        csv_writer.writerow(['Code', 'Desc'])
        # aggiungiamo ora i dati
        for i in range(0, len(all_row)):
            csv_writer.writerow([str(all_row[i].code), str(all_row[i].desc)])


# TrainCSV()
TrainCSV_with_emptyclass()
