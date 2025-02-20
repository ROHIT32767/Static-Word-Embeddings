import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import brown

nltk.download('brown')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EMBEDDING_DIM = 100
CONTEXT_SIZE = 2  
NUM_NEGATIVE_SAMPLES = 5  
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 32

# Preprocess the corpus
class Corpus:
    def __init__(self, minimum_freq=1):
        self.corpus = brown.sents()
        self.words = [word.lower() for sentence in self.corpus for word in sentence]
        self.vocab, self.word_to_idx, self.idx_to_word = self.build_vocab(minimum_freq)

    def build_vocab(self, minimum_freq):
        word_counts = Counter(self.words)
        vocab = [word for word, count in word_counts.items() if count >= minimum_freq]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return vocab, word_to_idx, idx_to_word

    def generate_training_data(self, context_size):
        data = []
        for i in range(context_size, len(self.words) - context_size):
            target_word = self.word_to_idx.get(self.words[i], None)
            if target_word is None:
                continue  # Skip words not in vocabulary
            context_words = [self.word_to_idx.get(self.words[i - j - 1], None) for j in range(context_size)]
            context_words += [self.word_to_idx.get(self.words[i + j + 1], None) for j in range(context_size)]
            context_words = [word for word in context_words if word is not None]
            for context_word in context_words:
                data.append((target_word, context_word))
        return data

# Custom Dataset class
class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Negative Sampling
def get_negative_samples(target, num_negative_samples, vocab_size):
    neg_samples = []
    while len(neg_samples) < num_negative_samples:
        neg_sample = np.random.randint(0, vocab_size)
        if neg_sample != target:
            neg_samples.append(neg_sample)
    return neg_samples

# Skip-gram Model with Negative Sampling
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, target, context, negative_samples):
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        negative_embeddings = self.context_embeddings(negative_samples)
        
        positive_score = self.log_sigmoid(torch.sum(target_embedding * context_embedding, dim=1))
        negative_score = self.log_sigmoid(-torch.bmm(negative_embeddings, target_embedding.unsqueeze(2)).squeeze(2)).sum(1)
        
        loss = - (positive_score + negative_score).mean()
        return loss

# Training the model
def train_model(model, dataloader, optimizer, num_epochs, vocab_size, num_negative_samples):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for target, context in dataloader:
            target = target.long().to(device)
            context = context.long().to(device)
            negative_samples = torch.LongTensor([get_negative_samples(t.item(), num_negative_samples, vocab_size) for t in target]).to(device)

            optimizer.zero_grad()
            loss = model(target, context, negative_samples)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Function to get similar words
def get_similar_words(word, embeddings, word_to_idx, idx_to_word, top_n=5):
    idx = word_to_idx.get(word, None)
    if idx is None:
        return []  # Word not in vocabulary
    word_embedding = embeddings[idx]
    similarities = np.dot(embeddings, word_embedding)
    closest_idxs = (-similarities).argsort()[1:top_n+1]
    return [idx_to_word[idx] for idx in closest_idxs]

# Main function
if __name__ == "__main__":
    # Load and preprocess the Brown Corpus
    corpus = Corpus(minimum_freq=5)
    training_data = corpus.generate_training_data(CONTEXT_SIZE)
    dataset = Word2VecDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model
    vocab_size = len(corpus.vocab)
    model = SkipGramNegSampling(vocab_size, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, dataloader, optimizer, NUM_EPOCHS, vocab_size, NUM_NEGATIVE_SAMPLES)

    # Save the model and embeddings
    torch.save(model.state_dict(), "skipgram_model.pt")
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    with open("embeddings.txt", "w") as f:
        for word, idx in corpus.word_to_idx.items():
            f.write(f"{word} {' '.join(map(str, embeddings[idx]))}\n")

    # Example usage: Get similar words
    similar_words = get_similar_words("do", embeddings, corpus.word_to_idx, corpus.idx_to_word, top_n=5)
    print("Similar words to 'do':", similar_words)
