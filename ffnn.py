import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.W1(input_vector)
        hidden_layer = self.activation(hidden_layer)

        # [to fill] obtain output layer representation
        output_layer = self.W2(hidden_layer)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # list to store results for each epoch    
    results = {
        "epoch": [],
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        train_loss = None
        train_correct = 0
        train_total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            train_loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                train_correct += int(predicted_label == gold_label)
                train_total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if train_loss is None:
                    train_loss = example_loss
                else:
                    train_loss += example_loss
            train_loss = train_loss / minibatch_size
            train_loss.backward()
            optimizer.step()
        train_acc = train_correct / train_total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_correct / train_total))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        val_loss = None
        val_correct = 0
        val_total = 0
        val_preds, val_gold = [], []

        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            val_loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector).item()

                val_preds.append(predicted_label)
                val_gold.append(gold_label)

                val_correct += int(predicted_label == gold_label)
                val_total += 1

                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if val_loss is None:
                    val_loss = example_loss
                else:
                    val_loss += example_loss
            val_loss = val_loss / minibatch_size

        val_acc = val_correct / val_total

        # Compute precision, recall, F1 (macro avg)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_gold, val_preds, average='macro', zero_division=0
        )

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy: {:.4f}".format(val_acc))
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(val_precision, val_recall, val_f1))
        print("Validation time for this epoch: {:.2f}s".format(time.time() - start_time))

        # Save results
        results["epoch"].append(epoch + 1)
        results["val_acc"].append(val_acc)
        results["val_loss"].append(val_loss.item())
        results["val_precision"].append(val_precision)
        results["val_recall"].append(val_recall)
        results["val_f1"].append(val_f1)
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss.item())


    print("\n========== Training Summary ==========")
    print("{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}".format("Epoch", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"))
    for i in range(len(results["epoch"])):
        print("{:<10d}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}".format(
            results["epoch"][i],
            results["train_acc"][i],
            results["val_acc"][i],
            results["train_loss"][i],
            results["val_loss"][i]
        ))

    best_val_acc = max(results["val_acc"])
    final_val_acc = results["val_acc"][-1]
    print("BEST VALIDATION ACCURACY:", best_val_acc)
    print("FINAL VALIDATION ACCURACY:", final_val_acc)


    plt.figure(figsize=(18, 10))  # Bigger canvas for more subplots

    # Accuracy plot
    plt.subplot(2, 3, 1)
    plt.plot(results["epoch"], results["val_acc"], label='Validation Accuracy', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Loss plot
    plt.subplot(2, 3, 2)
    plt.plot(results["epoch"], results["val_loss"], label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Precision plot
    plt.subplot(2, 3, 3)
    plt.plot(results["epoch"], results["val_precision"], label='Validation Precision', marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Validation Precision")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Recall plot
    plt.subplot(2, 3, 4)
    plt.plot(results["epoch"], results["val_recall"], label='Validation Recall', marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Validation Recall")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # F1 Score plot
    plt.subplot(2, 3, 5)
    plt.plot(results["epoch"], results["val_f1"], label='Validation F1 Score', marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("ffnn_metrics_dim{}.png".format(args.hidden_dim))
    plt.show()

    # write out to results/test.out
    