import numpy as np

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

##todays agenda : -
##1. softmax
##2. text features - with padding,embedding,tokenization,output category
##3. vector dimensions explain and understand the usage

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y,y_cap):
    return -np.mean(np.sum(y * np.log(y_cap + 1e-9), axis=1))


def feedforward(x,weights,bias):
    y = np.matmul(x,weights)+bias
    return softmax(y)

def gradient(x, y, y_cap, weights, bias):
    # Gradient calculation (to be implemented)
    grad_weights = np.matmul(x.T, (y_cap - y)) / x.shape[0]
    grad_bias = np.mean(y_cap - y, axis=0, keepdims=True)
    return grad_weights, grad_bias

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_Rate):
    weights -= learning_Rate*grad_weights
    bias -= learning_Rate*grad_bias
    return weights,bias

def initialize_weights_and_biases(vocab_size, embedding_dim, max_length, num_classes):
    embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
    output_weights = np.random.randn(embedding_dim * max_length, num_classes) * 0.01
    output_bias = np.zeros((1, num_classes))
    return embedding_matrix, output_weights, output_bias

def read_data_from_file(filename):
    sentences = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            sentence, label = line.strip().split(',')
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels



def one_hot_encoding(encoded_labels, num_classes):
    one_hot = np.zeros((len(encoded_labels), num_classes))
    one_hot[np.arange(len(encoded_labels)), encoded_labels] = 1
    return one_hot

def encode_labels(labels):
    unique_labels = list(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])
    return encoded_labels, unique_labels

def preprocess_text(sentences):
    words = [sentence.split() for sentence in sentences]
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    word_counts = Counter(word for sentence in tokenized_sentences for word in sentence)
    ## with this tokenized sentences and word count for all the words in text's
    ##now build the vocabolary
    vocab = {word: i + 1 for i, word in enumerate(word_counts)}
    numerical_sequences = [[vocab.get(word, 0) for word in sentence] for sentence in tokenized_sentences]
    print(numerical_sequences)
    max_len = max(len(seq) for seq in numerical_sequences)
    print(max_len)
    padded_sequences = np.array([seq + [0] * (max_len - len(seq)) for seq in numerical_sequences])
    print(padded_sequences)
    return padded_sequences,vocab,max_len

def create_embedding_matrix(vocab, embedding_dim):
    vocab_size = len(vocab) + 1  # +1 for padding token
    embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
    return embedding_matrix

def embed_sequences(padded_sequences, embedding_matrix):
    embedded_sequences = embedding_matrix[padded_sequences]
    return embedded_sequences


def main():
    filename = 'file6.txt'
    sentences, labels = read_data_from_file(filename)
    padded_sequences, vocab,max_len = preprocess_text(sentences)
    encoded_labels, unique_labels = encode_labels(labels)
    one_hot_labels = one_hot_encoding(encoded_labels, len(unique_labels))
    # one_hot_labels = to_one_hot(encoded_labels, len(unique_labels))

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2,
                                                        random_state=42)
    vocab_size = len(vocab)
    embedding_dim = 16
    num_classes = len(unique_labels)
    embedding_matrix = create_embedding_matrix(vocab, embedding_dim)
    embedded_train = embed_sequences(X_train, embedding_matrix).reshape(X_train.shape[0], embedding_dim * max_len)
    embedded_test = embed_sequences(X_test, embedding_matrix).reshape(X_test.shape[0], embedding_dim * max_len)

    max_length = X_train.shape[1]
    # Initialize weights and biases (no hidden layer)
    embedding_matrix, output_weights, output_bias = initialize_weights_and_biases(
        vocab_size, embedding_dim, max_length, num_classes
    )

    embedded_train = embedding_matrix[X_train].reshape(X_train.shape[0], embedding_dim * max_len)
    embedded_test = embedding_matrix[X_test].reshape(X_test.shape[0], embedding_dim * max_len)

    learning_rate = 0.00001
    epochs = 10000

    for epoch in range(epochs):
        y_cap = feedforward(embedded_train, output_weights, output_bias)
        loss = cross_entropy_loss(y_train, y_cap)
        grad_weights, grad_bias = gradient(embedded_train, y_train, y_cap, output_weights, output_bias)
        output_weights, output_bias = updated_parameters(output_weights, output_bias, grad_weights, grad_bias,
                                                         learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    y_cap_test = feedforward(embedded_test, output_weights, output_bias)
    predictions = np.argmax(y_cap_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    # print(f"Test Accuracy: {accuracy}")
    # Display output and predicted output
    for i in range(len(predictions)):
        predicted_label = unique_labels[predictions[i]]
        true_label = unique_labels[true_labels[i]]
        print(f"Sample {i + 1}: Predicted: {predicted_label}, True: {true_label}")

if __name__=="__main__":
    main()