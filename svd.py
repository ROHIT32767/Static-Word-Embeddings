import os
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import brown

# Download the Brown Corpus if not already downloaded
nltk.download('brown')

class tokenization:
    def __init__(self):
        pass

    def replaceHashtags(self, txt):
        return re.sub('\#[a-zA-Z]\w+', '', txt)

    def replace_email(self, corpus):
        return re.sub(r'\S*@\S*\s?', r'', corpus)

    def replaceURL(self, txt):
        return re.sub(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r'', txt)

    def replaceMentions(self, txt):
        return re.sub(r'@\w+', r'', txt)

    def replaceDateTime(self, txt):
        txt = re.sub(
            r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '', txt)
        return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'', txt)

    def upperToLower(self, txt): return txt.lower()

    def replacePunctuation(self, txt): return re.sub(
        r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|—|’|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~{1,})', r'', txt)

    def replaceMobileNumber(self, txt):
        return re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'', txt)

    def replaceNumericals(self, txt):
        return re.sub(r'\d+', r'', txt)

def tokenize(corpus):
    token = tokenization()
    corpus = token.replaceHashtags(corpus)
    corpus = token.replace_email(corpus)
    corpus = token.replaceURL(corpus)
    corpus = token.replaceMentions(corpus)
    corpus = token.upperToLower(corpus)
    corpus = token.replaceDateTime(corpus)
    corpus = token.replacePunctuation(corpus)
    corpus = token.replaceMobileNumber(corpus)
    corpus = token.replaceNumericals(corpus)
    return corpus.split()

# Load the Brown Corpus
sentences = brown.sents()
sentences_tokenized = [tokenize(' '.join(sentence)) for sentence in sentences]

# Define a defaultdict to store the co-occurrence counts
co_occurrence = defaultdict(int)
window_size = 2  # You can adjust the window size

# Create a vocabulary
vocab = list(set([word for sentence in sentences_tokenized for word in sentence]))

# Initialize the co-occurrence matrix
matrix = np.zeros((len(vocab), len(vocab)))
word_to_id = {word: i for i, word in enumerate(vocab)}

# Build the co-occurrence matrix
for sentence in sentences_tokenized:
    for i, word in enumerate(sentence):
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i != j:
                matrix[word_to_id[word], word_to_id[sentence[j]]] += 1

# Perform SVD on the co-occurrence matrix
svd = TruncatedSVD(n_components=100)  # You can adjust the number of dimensions
print("Performing SVD on the co-occurrence matrix...")
word_vectors_svd = svd.fit_transform(matrix)
print("SVD performed successfully!")

# Normalize the word vectors
norms = np.linalg.norm(word_vectors_svd, axis=1, keepdims=True)
norms[norms == 0] = 1e-8  # replace any zero values with a small value
word_vectors_normalized = word_vectors_svd / norms

# Create a dictionary of word vectors
word_vectors_dict = {word: word_vectors_normalized[i] for i, word in enumerate(vocab)}

# Save the embeddings as .pt files
torch.save(word_vectors_dict, 'word_embeddings.pt')
print("Embeddings saved successfully!")

# Print all the embeddings into a file for further use using dict
with open('embeddings.txt', 'w') as file:
    for key, value in word_vectors_dict.items():
        file.write(f"{key} {' '.join(map(str, value))}\n")

print("Embeddings written to embeddings.txt!")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(word_vectors_normalized)
labels = vocab

print("Plotting the embeddings...")
# Define the word for which to find similar words
word = "government"

# Find the 10 most similar words to the given word
most_similar = [vocab[i] for i in np.argsort(np.linalg.norm(word_vectors_normalized - word_vectors_normalized[vocab.index(word)], axis=1))[1:11]]

# Plot the embeddings, highlighting the given word and its 10 most similar words
plt.figure(figsize=(14, 14))
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    if label == word:
        plt.scatter(x, y, c='r')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    elif label in most_similar:
        plt.scatter(x, y, c='b')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        
plt.show()