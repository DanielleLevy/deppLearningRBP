import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import product
import os
import argparse
import random
import keras

import numpy as np


# הגדרת מודל CNN חדש

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=256, kernel_size=8, activation='relu', input_shape=input_shape))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# הגדרת רצפים אפשריים
import numpy as np

CHARS = ['A', 'C', 'G', 'T']

def one_hot_encode(seq, maxlen=41):
    """
    מבצע One Hot Encoding לרצפי DNA, כולל התייחסות לתו 'N'.
    """
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = np.zeros((maxlen, len(CHARS)))

    for i, char in enumerate(seq):
        if i < maxlen:
            encoded[i] = mapping.get(char, [0, 0, 0, 0])

    # Padding אם הרצף קצר מהאורך המקסימלי
    if len(seq) < maxlen:
        padding_length = maxlen - len(seq)
        encoded[len(seq):] = [0.25, 0.25, 0.25, 0.25]  # Padding עם ערך של 0.25 בכל מקום

    return encoded





# פונקציה לקביעת התוויות על פי הסייקל
def suffle(sequence):
    """
    מבצע ערבוב רגיל של רצף.

    Args:
        sequence (str): הרצף המקורי.

    Returns:
        str: הרצף המעורבל.
    """
    sequence_list = list(sequence)
    random.shuffle(sequence_list)
    return ''.join(sequence_list)


# Assign labels based on cycle
def assign_labels_based_on_cycle(cycle_files):
    sequence_labels = {}
    highest_cycle = len(cycle_files)

    for cycle_index, cycle_file in enumerate(cycle_files):
        with open(cycle_file, 'r') as file:
            for line in file:
                sequence, count = line.strip().split(',')
                if cycle_index + 1 == highest_cycle:
                    sequence_labels[sequence] = 1
                elif cycle_index == 0 and sequence not in sequence_labels:
                    sequence_labels[sequence] = 0

    if len(cycle_files) == 1:
        for sequence in list(sequence_labels.keys()):
            if sequence_labels[sequence] == 0:
                shuffled_sequence = suffle(sequence)
                if shuffled_sequence not in sequence_labels:
                    sequence_labels[shuffled_sequence] = 0

    return sequence_labels


# Prepare training data
def prepare_data_for_training(rbp_name, cycle_files):
    positive_cycle = cycle_files[-1]
    negative_cycle = cycle_files[0]

    positive_sequences = []
    negative_sequences = []

    with open(positive_cycle, 'r') as pos_file:
        for line in pos_file:
            sequence = line.strip().split(',')[0]
            positive_sequences.append(sequence)
            if len(positive_sequences) == 50000:
                break

    with open(negative_cycle, 'r') as neg_file:
        for line in neg_file:
            sequence = line.strip().split(',')[0]
            if sequence not in positive_sequences:
                negative_sequences.append(sequence)
            if len(negative_sequences) == 50000:
                break

    if len(cycle_files) == 1:
        negative_sequences = [suffle(seq) for seq in positive_sequences]

    positive_labels = [1] * len(positive_sequences)
    negative_labels = [0] * len(negative_sequences)

    X_train = positive_sequences + negative_sequences
    y_train = positive_labels + negative_labels

    X_train = np.array([one_hot_encode(seq) for seq in X_train])

    return X_train, y_train


# Train model and predict on RNAcompete data
def train_and_predict(rbp_name, cycle_files, rncmpt_path):
    X_train, y_train = prepare_data_for_training(rbp_name, cycle_files)

    shuffled_indices = np.arange(len(X_train))
    np.random.shuffle(shuffled_indices)
    X_train = X_train[shuffled_indices]
    y_train = np.array(y_train)[shuffled_indices]

    model = build_model(input_shape=(41, 4))

    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.3, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ])

    # Now predict on the RNAcompete data
    with open(rncmpt_path, 'r') as file:
        rncmpt_sequences = [line.strip() for line in file]

    X_test = np.array([one_hot_encode(seq) for seq in rncmpt_sequences])
    predictions = model.predict(X_test)

    np.savetxt(f"{rbp_name}_predictions.txt", predictions)

    return predictions


def load_rbp_cycle_info(file_path):
    rbp_files = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_rbp = None
        for line in lines:
            line = line.strip()
            if line.startswith("RBP"):
                current_rbp = line.split(':')[0]
                rbp_files[current_rbp] = []
            elif line.startswith("htr-selex"):
                rbp_files[current_rbp].append(line)
    return rbp_files
