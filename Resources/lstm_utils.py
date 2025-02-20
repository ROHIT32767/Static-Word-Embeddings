import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skip_gram_utils import preprocessing_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def get_training_data(train_data, word_embeddings, word2idx, test_flag = False):

    sentences = train_data["Description"]
    processed_sentences = preprocessing_data(sentences)

    sentence_lengths = [len(sentence) for sentence in processed_sentences]
    max_len = int(np.percentile(sentence_lengths, 90))

    print(f'Maximum sentence length: {max_len}')

    for i in range(len(processed_sentences)):
        if len(processed_sentences[i]) < max_len:
            processed_sentences[i] += ['<PAD>'] * int(max_len - len(processed_sentences[i]))
        else:
            processed_sentences[i] = processed_sentences[i][:max_len]

    X_train = []
    for sentence in processed_sentences:
        sentence_vector = []
        for word in sentence:
            if word in word2idx:
                sentence_vector.append(word_embeddings[word2idx[word]])
            else:
                sentence_vector.append(word_embeddings[word2idx['<UNK>']])
        X_train.append(sentence_vector)

    labels = train_data["Class Index"]
    n_classes = len(labels.unique())

    Y_train = torch.zeros(len(labels), n_classes)
    for i, label in enumerate(labels):
        Y_train[i][label-1] = 1

    X_train = np.array(X_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)

    Y_train = np.array(Y_train)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    if test_flag:
        return X_train, Y_train, n_classes
    
    train_size = int(0.8 * len(X_train))
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]
    X_train = X_train[:train_size]
    Y_train = Y_train[:train_size]

    return X_train, Y_train, X_val, Y_val, n_classes

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, device, activation='relu'):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
def train_lstm(train_loader, val_loader, model, device, n_epochs, lr):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    return train_losses, val_losses, model

def get_metrics(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    cm = confusion_matrix(ground_truth, predictions)
    return accuracy, f1, precision, recall, cm


def get_predictions(data_loader, model, device):
    predictions = []
    ground_truth = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        ground_truth.extend(targets.argmax(dim=1).cpu().numpy())
    return predictions, ground_truth