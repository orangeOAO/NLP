from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
import os

matplotlib.use('Agg')
model = Word2Vec.load("my_word2vec_model.model")

words = ['king', 'queen', 'man', 'woman', 'paris', 'france', 'berlin', 'germany', 'car', 'bus', 'orange', 'girl']
word_vectors = np.array([model.wv[word] for word in words if word in model.wv])

pca = PCA(n_components=2)
X = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], edgecolors='k', c='r')

for i, word in enumerate(words):
    plt.text(X[i, 0]+0.05, X[i, 1]+0.05, word, fontsize=12)

current_directory = os.getcwd()
save_path = os.path.join(current_directory, 'world_vector.png')
plt.savefig(save_path)

def estimate_similarity(word1, word2):
    similarity = model.wv.similarity(word1, word2)
    return similarity

def analogy(word1, word2, word3):
    result = model.wv.most_similar(positive=[word3, word2], negative=[word1], topn=1)
    return result[0][0]

if __name__ == "__main__":

    sim_score = estimate_similarity('king', 'queen')
    print(f"Similarity between 'king' and 'queen': {sim_score:.4f}")
    analogy_result = analogy('king', 'fat', 'queen')
    print(f"Word completing the analogy king:fat :: queen:{analogy_result}")

