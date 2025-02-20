import pandas as pd
from svd_utils import word_embeddings_svd

data = pd.read_csv('data/train.csv')
data = data['Description']

data = data[:20000]
svd = word_embeddings_svd(context_window=4,data = data)
svd.save_embeddings('word_embeddings_svd_4.pt')
svd = word_embeddings_svd(context_window=5,data = data)
svd.save_embeddings('word_embeddings_svd_5.pt')
svd.save_word2idx()
