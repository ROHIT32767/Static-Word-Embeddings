import numpy as np
import pandas as pd
from skip_gram_utils import preprocessing_data, create_vocab, train_skip_gram_neg
import torch

# pre-processing the text
data = pd.read_csv('data/train.csv')
data = data['Description']
sentences = preprocessing_data(data)

# creating the vocabulary
vocab = create_vocab(sentences)

# look-up tables
word2idx = {word:idx for idx, word in enumerate(vocab)}
idx2word = {idx:word for word, idx in word2idx.items()}

embedding_dim = 300
n_epochs = 5
lr = 0.001
batch_size = 512
k = 5
ns_exponent = 0.75
ns_array_len = 5000000

model, losses = train_skip_gram_neg(vocab, word2idx, sentences, embedding_dim, n_epochs, lr, batch_size, k, ns_exponent, ns_array_len, window_size=4)
embeddings = model.in_embedding.weight.data.cpu().numpy()
torch.save(embeddings, 'word_embeddings_skip_gram_4.pt')

model, losses = train_skip_gram_neg(vocab, word2idx, sentences, embedding_dim, n_epochs, lr, batch_size, k, ns_exponent, ns_array_len, window_size=5)
embeddings = model.in_embedding.weight.data.cpu().numpy()
torch.save(embeddings, 'word_embeddings_skip_gram_5.pt')

torch.save(word2idx, 'word2idx_skip_gram.pt')
torch.save(idx2word, 'idx2word_skip_gram.pt')
