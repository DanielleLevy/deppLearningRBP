import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import product
import os
import argparse
import random
import keras



# הגדרת מודל CNN חדש
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=256, kernel_size=8, activation='relu', input_shape=input_shape))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    Adam = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# הגדרת רצפים אפשריים
CHARS = ['A', 'C', 'G', 'T']


# פונקציה להמרת רצפי RNA לייצוג one-hot
def one_hot_encode(seq, maxlen=41):
    mapping = {char: idx for idx, char in enumerate(CHARS)}
    encoded = np.zeros((maxlen, len(CHARS)))
    seq_len = len(seq)

    for i, char in enumerate(seq):
        if i < maxlen:
            encoded[i, mapping[char]] = 1

    if seq_len < maxlen:
        encoded[seq_len:] = 0.25

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

def assign_labels_based_on_cycle(cycle_files):
    sequence_labels = {}
    highest_cycle = len(cycle_files)

    # תחילה, נעבור על כל הסייקלים ונקבע תוויות
    for cycle_index, cycle_file in enumerate(cycle_files):
        with open(cycle_file, 'r') as file:
            for line in file:
                sequence, count = line.strip().split(',')
                if cycle_index + 1 == highest_cycle:
                    sequence_labels[sequence] = 1  # הסייקל הגבוה ביותר מקבל תווית 1
                elif cycle_index == 0 and sequence not in sequence_labels:
                    sequence_labels[sequence] = 0  # סייקל 1 שלא נמצא בסייקל גבוה יותר

    # בדיקה אם יש רק סייקל 1
    if len(cycle_files) == 1:
        # נוסיף רצפים ערובבים (DISUFFLE) עבור סייקל 1
        for sequence in list(sequence_labels.keys()):
            if sequence_labels[sequence] == 0:
                shuffled_sequence = suffle(sequence)
                if shuffled_sequence not in sequence_labels:
                    sequence_labels[shuffled_sequence] = 0

    return sequence_labels

# הכנת נתונים לאימון המודל
def prepare_data_for_training(rbp_name, cycle_files, rncmpt_path):
    # קבלת תוויות על בסיס הסייקל
    sequence_labels = assign_labels_based_on_cycle(cycle_files)

    rncmpt_sequences = []
    y = []

    with open(rncmpt_path, 'r') as file:
        for line in file:
            sequence = line.strip().split()[0]
            rncmpt_sequences.append(one_hot_encode(sequence))
            label = sequence_labels.get(sequence, 0)  # אם הרצף לא נמצא, תווית 0
            y.append(label)

    X = np.array(rncmpt_sequences)
    y = np.array(y)

    return X, y


# אימון המודל וחיזוי
def train_and_predict(rbp_name, cycle_files, rncmpt_path):
    X, y = prepare_data_for_training(rbp_name, cycle_files, rncmpt_path)

    # ערבול הנתונים
    shuffled_indices = np.arange(len(X))
    np.random.shuffle(shuffled_indices)
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    model = build_model(input_shape=(41, len(CHARS)))

    model.fit(X, y, epochs=30, batch_size=64, validation_split=0.3, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ])

    predictions = model.predict(X)
    np.savetxt(f"{rbp_name}_predictions.txt", predictions)

    return predictions


# קריאת קבצי SELEX והכנת תוויות
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


