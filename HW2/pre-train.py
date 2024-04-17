from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

import pandas as pd

pd.set_option('display.width', 1000)

dataset = pd.read_csv("Sentiment Analysis Dataset.csv", on_bad_lines='skip')
print(dataset.head())
texts = dataset['SentimentText']

processed_data = [simple_preprocess(sentence) for sentence in texts]
my_model = Word2Vec(sentences=processed_data, vector_size=300, window=5, min_count=1, workers=4)
my_model.save("my_word2vec_model.model")
