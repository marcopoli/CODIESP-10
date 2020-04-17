import gensim
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import KeyedVectors

# info: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# spiega anche glove

# https://zenodo.org/record/2542722#.XpXm0y1aaYU <- embeddings lingua medica spagnola

# dati di training
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'frase'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

# modello di training
model = Word2Vec(sentences, min_count=1)
# - size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent
# each token (word).
# - window: (default 5) The maximum distance between a target word and words around the target word.
# - min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence
# - less than this count will be ignored. workers: (default 3) The number of threads to use while training. sg: (
# - default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).


# visualizziazione del modello
print(model)

# visualizziazione del vocabolario
words = list(model.wv.vocab)
print(words)

# accedo al vettore usando la parola 'sentence'
print(model['sentence'])

# salvo il modello
model.save('model.bin')

# carico il modello in lingua spagnola
new_model = gensim.models.Word2Vec.load('es.bin') # https://github.com/Kyubyong/wordvectors pre-trained
print(new_model)

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

# calculate: (king - man) + woman = ? (il risultato sarà queen, reina in spagnolo)
result = new_model.wv.most_similar(positive=['mujer', 'rey'], negative=['hombre'], topn=3)
print(result)


