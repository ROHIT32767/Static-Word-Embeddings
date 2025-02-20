import numpy as np
from collections import Counter
import contractions
import string
import re
import torch
import torch.nn as nn
import torch.optim as optim

def preprocessing_data(data):
    sentences = [contractions.fix(sentence) for sentence in data]
    sentences = [sentence.lower() for sentence in sentences]
    sentences = [re.sub(r'http\S+', 'URL', sentence) for sentence in sentences]
    sentences = [re.sub(r'www\S+', 'URL', sentence) for sentence in sentences]
    sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
    sentences = [(sentence.split()) for sentence in sentences]
    sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]
    return sentences

def create_vocab(sentences):
    words = [word for sentence in sentences for word in sentence]
    word_count = Counter(words)
    vocab = {k:v for k,v in word_count.items() if v >= 3}
    vocab['<UNK>'] = sum(word_count[k] for k in word_count if word_count[k] < 3)
    return vocab

def subsampling_words(word2idx, vocab, threshold = 1e-5):
    total_words = sum(vocab.values())
    freqs = {word:count/total_words for word, count in vocab.items()}
    p_drop = {word:1 - np.sqrt(threshold/freqs[word]+threshold) for word in word2idx}
    subsampled_words = {word2idx[word] for word in word2idx if np.random.rand() < (1 - p_drop[word])} 
    return subsampled_words

def get_contexts(index, sentence, window_size):
    contexts = []
    for j in range(max(0, index-window_size), min(len(sentence), index+window_size+1)):
        if index != j:
            contexts.append(sentence[j])
    return contexts

def get_target_context_pairs(sentences, word2idx, vocab, window_size):
    X, Y = [], []
    sub_sampled_words = subsampling_words(word2idx, vocab)
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word in word2idx:
                idx = word2idx[word]
            else:
                idx = word2idx['<UNK>']
            if idx in sub_sampled_words:
                contexts = get_contexts(i, sentence, window_size)
                for context in contexts:
                    if context in word2idx:
                        c_idx = word2idx[context]
                    else:
                        c_idx = word2idx['<UNK>']
                    if c_idx in sub_sampled_words:
                        X.append(idx)
                        Y.append(c_idx)
    return X, Y

def get_negative_samples(n_samples, k , ns_exponent, ns_array_len, vocab, word2idx):
    freq_array = {word:count**ns_exponent for word, count in vocab.items()}
    total_freq = sum(freq_array.values())
    freq_scaled = {word2idx[word]:max(1,int((count/total_freq)*ns_array_len)) for word, count in freq_array.items()}
    ns_array = np.array([word for word, count in freq_scaled.items() for _ in range(count)])
    sample = np.random.choice(ns_array, size=(n_samples, k))
    return sample

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNeg, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, input, context):
        input_embedding = self.in_embedding(input)
        n1 = input_embedding.shape[0]
        n2 = input_embedding.shape[1]
        input_embedding = input_embedding.view(n1, 1, n2)
        context_embedding = self.out_embedding(context)
        scores = torch.bmm(input_embedding, context_embedding.permute(0, 2, 1))
        scores = scores.view(scores.shape[0], scores.shape[2])
        return scores
    
def train_skip_gram_neg(vocab, word2idx, sentences, embedding_dim, n_epochs, lr, batch_size, k, ns_exponent, ns_array_len, window_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGramNeg(len(vocab), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    losses = []
    X, Y = get_target_context_pairs(sentences, word2idx, vocab, window_size)
    for epoch in range(n_epochs):
        running_loss = []
        for i in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(Y[i:i+batch_size]).to(device)
            negative_samples = get_negative_samples(len(X_batch), k, ns_exponent, ns_array_len, vocab, word2idx)
            negative_samples = torch.tensor(negative_samples).to(device)
            context = torch.cat([Y_batch.view(Y_batch.shape[0], 1), negative_samples], dim = 1)
            pos_labels = torch.ones(Y_batch.shape[0], 1).to(device)
            neg_labels = torch.zeros(Y_batch.shape[0], k).to(device)
            labels = torch.cat([pos_labels, neg_labels], dim = 1)
            optimizer.zero_grad()
            output = model(X_batch, context)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        print(f'Epoch: {epoch+1}, Loss: {np.mean(running_loss)}')
        losses.append(np.mean(running_loss))

    return model, losses