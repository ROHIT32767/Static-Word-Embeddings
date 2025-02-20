import torch
import pandas as pd
from lstm_utils import get_training_data, LSTMClassifier, train_lstm, get_predictions, get_metrics
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

word_embeddings = torch.load('word_embeddings_svd_best.pt')
word2idx = torch.load('word2idx_svd.pt')
idx2word = torch.load('idx2word_svd.pt')

word_embeddings = torch.tensor(word_embeddings)
word_embeddings = torch.cat((word_embeddings, torch.zeros(1, 300)), dim=0)

word2idx['<PAD>'] = len(word2idx)
idx2word[len(idx2word)] = '<PAD>'

X_train, Y_train, X_val, Y_val, n_classes = get_training_data(train_data, word_embeddings, word2idx)
X_test, Y_test, n_classes = get_training_data(test_data, word_embeddings, word2idx, test_flag=True)

train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

input_dim = 300
hidden_dim = 128
output_dim = n_classes
n_layers = 2
bidirectional = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers, bidirectional, device)

n_epochs = 10
lr = 0.001

train_losses, val_losses, model_trained = train_lstm(train_loader, val_loader, model, device, n_epochs, lr)

model_trained.eval()

train_predictions, train_ground_truth = get_predictions(train_loader, model_trained, device)
val_predictions, val_ground_truth = get_predictions(val_loader, model_trained, device)
test_predictions, test_ground_truth = get_predictions(test_loader, model_trained, device)

train_accuracy, train_f1, train_precision, train_recall, train_cm = get_metrics(train_predictions, train_ground_truth)
val_accuracy, val_f1, val_precision, val_recall, val_cm = get_metrics(val_predictions, val_ground_truth)
test_accuracy, test_f1, test_precision, test_recall, test_cm = get_metrics(test_predictions, test_ground_truth)

print(f'Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
print(f'Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

print('Train Confusion Matrix:')
print(train_cm)
print('Val Confusion Matrix:')
print(val_cm)
print('Test Confusion Matrix:')
print(test_cm)
