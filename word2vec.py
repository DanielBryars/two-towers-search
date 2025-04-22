from gensim.models import KeyedVectors
import numpy as np
#https://code.google.com/archive/p/word2vec/

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

print(model.most_similar('king'))
print(model['king'])  # Vector for 'king'

sentence = "This is my query"
words = sentence.lower().split()

word_vectors = [model[word] for word in words if word in model]

sentence_vector = np.mean(word_vectors, axis=0)

print(sentence_vector.shape)  # Should be (300,)

