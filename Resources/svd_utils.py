import numpy as np
from collections import Counter
import contractions
import string
import re
import torch

class word_embeddings_svd:
    def __init__(self, context_window, embedding_dim = 300, data = None):
        self.embedding_dim = embedding_dim
        self.data = data
        self.sentences = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.co_occurrence_matrix = None
        self.U = None
        self.S = None
        self.V = None
        self.word_embeddings = None
        self.context_window = context_window

        self.sentences = self.preprocess_sentences()
        self.vocab = self.create_vocab()
        self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
        self.co_occurrence_matrix = self.create_co_occurrence_matrix()
        self.svd_func()
        self.get_word_embeddings()

    def preprocess_sentences(self):
        sentences = [contractions.fix(sentence) for sentence in self.data]
        sentences = [sentence.lower() for sentence in sentences]
        sentences = [re.sub(r'http\S+', 'URL', sentence) for sentence in sentences]
        sentences = [re.sub(r'www\S+', 'URL', sentence) for sentence in sentences]
        sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
        sentences = [(sentence.split()) for sentence in sentences]
        sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]
        print("preprocessing done")
        return sentences
    
    def create_vocab(self):
        words = [word for sentence in self.sentences for word in sentence]
        word_count = Counter(words)
        vocab = {k:v for k,v in word_count.items() if v >= 3}
        vocab['<UNK>'] = sum(word_count[k] for k in word_count if word_count[k] < 3)
        print("vocab created")
        return vocab
    
    def create_co_occurrence_matrix(self):
        co_occurrence_matrix = np.zeros((len(self.vocab), len(self.vocab)))
        for sentence in self.sentences:
            for i, word in enumerate(sentence):
                if word in self.word2idx:
                    word_idx = self.word2idx[word]
                else:
                    word_idx = self.word2idx['<UNK>']
                start = max(0, i - self.context_window)
                end = min(len(sentence), i + self.context_window)
                for j in range(start, end):
                    if i != j:
                        if sentence[j] in self.word2idx:
                            context_idx = self.word2idx[sentence[j]]
                        else:
                            context_idx = self.word2idx['<UNK>']
                        co_occurrence_matrix[word_idx][context_idx] += 1
        print("co-occurrence matrix created")
        return co_occurrence_matrix
        
    def svd_func(self):
        print("SVD started")
        U, S, V = np.linalg.svd(self.co_occurrence_matrix)
        self.U = U
        self.S = S
        self.V = V
        print("SVD done")
        return U, S, V
    
    def get_word_embeddings(self):
        self.word_embeddings = self.U[:, :self.embedding_dim]
        return self.word_embeddings
    
    def get_word_embedding(self, word):
        if word in self.word2idx:
            idx = self.word2idx[word]
        else:
            idx = self.word2idx['<UNK>']
        return self.word_embeddings[idx]
    
    def save_embeddings(self, path):
        torch.save(self.word_embeddings, path)
    
    def save_word2idx(self):
        torch.save(self.word2idx, "word2idx_svd.pt")
        torch.save(self.idx2word, "idx2word_svd.pt")