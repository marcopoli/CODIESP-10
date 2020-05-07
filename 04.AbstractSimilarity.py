import pickle
import numpy as np
import pandas as pd
import sister
from sister.word_embedders import FasttextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

embedder = sister.MeanEmbedding(lang="es", word_embedder = FasttextEmbedding('es'))

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
    s = s.strip()
    s = s.lower()
    return s

# Rimozione delle stopword
file_stopw = open("support/stop_word.pck", "rb")
stop_word = pickle.load(file_stopw)

def clar_text(text):
    t = remove_symbol(str(text).strip().lower())
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    tt = ""
    for it in tokens:
      tt = tt +" "+it
    return tt

def _pad(input_ids, max_seq_len):
    x = []
    input_ids = input_ids[:min(len(input_ids), max_seq_len - 2)]
    input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
    return np.array(input_ids)

def tokenize_text(text):
    tokens = list(str(text).lower().split(" "))
    for z in range(0, len(stop_word)):
        if stop_word[z] in tokens:
            while stop_word[z] in tokens:
                tokens.remove(str(stop_word[z]))
    return tokens

df3 = pd.read_csv('Training/Task2/Train_with_only_abstract_Task2.csv')
df3 = df3[['Code', 'Desc']]
df3['Desc'] = df3['Desc'].apply(remove_symbol)

codes = df3['Code']
descrs = df3['Desc']
print(descrs)

final_codes = []

dic = []
i = 0
for row in descrs:
    tokens = tokenize_text(row)
    r = ''
    for t in tokens:
        r = r +' '+ str(t)
    vector = embedder(r)
    try:
        if len(vector) == 300:
            dic.append(vector)
            final_codes.append(codes[i])
    except:
        print('except')

    i = i+1
    if i%1000 == 0:
        print(i)

import joblib
dic = np.array(dic)
#print(dic.shape)
#joblib.dump(dic,'description_embeddings.vec')
#description_embeddings = joblib.load('description_embeddings.vec')
#print(len(description_embeddings[1]))
#description_embeddings = np.array(description_embeddings)

print('finish')

#Take it from
left_to_classify_ids = joblib.load('left_to_classify_ids.vec')
left_to_classify_vector = joblib.load('left_to_classify_vector.vec')
left_to_classify_vector = np.array(left_to_classify_vector)

print(left_to_classify_vector.shape)

res = cosine_similarity(left_to_classify_vector,dic)
print(res[0])
#print(description_embeddings)
#for v in left_to_classify_vector:
#    values = []
#    for key, value in description_embeddings.items():
#        #print(key)
#        item = value
#        val = cosine_similarity([v], [item])
        #print(val[0][0])
#        values.append(val[0][0])
#    res.append(values)
    #print(values)

#print(res)
#joblib.dump(res,'similarity_res.vec')

file = open('task2_res_similiarity.tsv','w+')

extracted = [[i.argmax()] for i in res]

i = 0
for item in extracted:
    id = left_to_classify_ids[i]
    code = final_codes[item[0]]
    #count = 0
    #for a in range(0,len(item)):
    #    if item[a] > (item.max()-(item.max()*0.005)):
    #        count = count+1
    #        code = final_codes[a]
    #        file.write(str(id)+'\t'+str(code).lower()+'\n')
    #        file.flush()
    #if count == 0:
    #    index = item.argmax()
    #    code = final_codes[index]
    #    file.write(str(id) + '\t' + str(code).lower() + '\n')
    #    file.flush()
    file.write(str(id) + '\t' + str(code).lower() + '\n')
    file.flush()

    i =i+1

