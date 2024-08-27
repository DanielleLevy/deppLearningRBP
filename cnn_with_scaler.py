import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from itertools import product
import random
import os


CHARS = ['A', 'C', 'G', 'T']

# Function to perform One Hot Encoding for DNA sequences
def one_hot_encode(seq, maxlen=41):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25]}
    encoded = np.zeros((maxlen, len(CHARS)))

    for i, char in enumerate(seq):
        if i < maxlen:
            encoded[i] = mapping.get(char, [0, 0, 0, 0])

    if len(seq) < maxlen:
        padding_length = maxlen - len(seq)
        encoded[len(seq):] = [0.25, 0.25, 0.25, 0.25]

    return encoded

# Function to shuffle a sequence
def suffle(sequence):
    sequence_list = list(sequence)
    random.shuffle(sequence_list)
    return ''.join(sequence_list)

# Function to generate all possible combinations of DNA sequences of a given length
def get_all_combinations(length):
    return [''.join(combination) for combination in product(CHARS, repeat=length)]

# Function to initialize a dictionary for sequence to score mapping
def initialize_seq_to_score_dict(seqs):
    return {seq: 0 for seq in seqs}

# Function to compute scores for each sequence based on SELEX cycles and weights
def get_seqs_with_scores(cycle_files, weights, length=5):
    combinations = get_all_combinations(length)
    seq_to_counts = [initialize_seq_to_score_dict(combinations) for _ in cycle_files]

    for idx, cycle_file in enumerate(cycle_files):
        with open(cycle_file, 'r') as file:
            for line in file:
                sequence, count = line.strip().split(',')
                count = int(count)
                for i in range(len(sequence) - length + 1):
                    seq = sequence[i:i + length]
                    if seq in seq_to_counts[idx]:
                        seq_to_counts[idx][seq] += count * weights[idx + 1]
                    else:
                        if "N" in seq:
                            char_to_replace_idx = np.random.randint(0, 4)
                            char_to_replace = CHARS[char_to_replace_idx]
                            seq = seq.replace("N", char_to_replace)
                            seq_to_counts[idx][seq] += count * weights[idx + 1]
        print(f"Finished with cycle file {idx + 1}")

    combined_scores = initialize_seq_to_score_dict(combinations)
    total_weight = sum(abs(w) for w in weights.values())
    for seq in combined_scores:
        weighted_sum = 0
        for idx in range(len(cycle_files)):
            weighted_sum += seq_to_counts[idx][seq]
        combined_scores[seq] = weighted_sum / total_weight

    return combined_scores

# Function to precompute scalar values for all possible 5-mers across cycles
def precompute_scalars(cycle_files, weights, length=5):
    seq_to_score = get_seqs_with_scores(cycle_files, weights, length=length)
    return seq_to_score

# Function to compute scalar for a given sequence using precomputed scores
def compute_scalar_with_precomputed(sequence, precomputed_scores, length=5):
    scalar_value = 0
    for i in range(len(sequence) - length + 1):
        sub_seq = sequence[i:i + length]
        scalar_value += precomputed_scores.get(sub_seq, 0)
    return scalar_value

# Function to prepare training data for the model with precomputed scalar values
def prepare_data_for_training_with_precomputed(rbp_name, cycle_files):
    weights = {1: 10, 2: 100, 3: 1000, 4: 10000}
    precomputed_scores = precompute_scalars(cycle_files, weights, length=5)

    sequences, scalar_values, y_train = [], [], []

    positive_cycle = cycle_files[-1]
    negative_cycle = cycle_files[0]

    with open(positive_cycle, 'r') as pos_file:
        for line in pos_file:
            sequence = line.strip().split(',')[0]
            sequences.append(sequence)
            scalar_values.append(compute_scalar_with_precomputed(sequence, precomputed_scores))
            y_train.append(1)
            if len(y_train) == 50000:
                break

    with open(negative_cycle, 'r') as neg_file:
        for line in neg_file:
            sequence = line.strip().split(',')[0]
            if sequence not in sequences:
                sequences.append(sequence)
                scalar_values.append(compute_scalar_with_precomputed(sequence, precomputed_scores))
                y_train.append(0)
            if len(y_train) == 100000:
                break

    if len(cycle_files) == 1:
        negative_sequences = [suffle(seq) for seq in sequences[:50000]]
        for sequence in negative_sequences:
            sequences.append(sequence)
            scalar_values.append(compute_scalar_with_precomputed(sequence, precomputed_scores))
            y_train.append(0)

    X_train = np.array([one_hot_encode(seq) for seq in sequences])
    y_train = np.array(y_train)
    scalar_values = np.array(scalar_values).reshape(-1, 1)

    return X_train, scalar_values, y_train

# Function to build the CNN model
def build_model(input_shape, scalar_input_shape):
    # קלט רצף ה-DNA
    sequence_input = Input(shape=input_shape)

    # קלט הסקלאר (למשל מספר החמישיות)
    scalar_input = Input(shape=scalar_input_shape)

    # השכבה הקונבולוציונית הראשונה בגודל קרנל 5
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(sequence_input)

    # השכבה הקונבולוציונית השנייה בגודל קרנל 5
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)

    # שכבת Global Max Pooling להקטנת הממדיות
    x = layers.GlobalMaxPooling1D()(x)

    # ביצוע איחוד בין הפלט של השכבות הקונבולוציוניות לבין קלט הסקלאר
    concatenated = layers.concatenate([x, scalar_input])

    # שכבות Fully Connected
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dense(32, activation='relu')(x)

    # שכבת היציאה עם נוירון אחד והפעלה מסוג sigmoid לסיווג בינארי
    output = layers.Dense(1, activation='sigmoid')(x)

    # בניית המודל עם שני קלטים ופלט אחד
    model = models.Model(inputs=[sequence_input, scalar_input], outputs=output)

    # קומפילציה של המודל
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model
# Train model and predict on RNAcompete data
def train_and_predict_with_precomputed(rbp_name, cycle_files, rncmpt_path):
    X_train, scalar_values, y_train = prepare_data_for_training_with_precomputed(rbp_name, cycle_files)
    # ביצוע ערבול לפני האימון
    shuffled_indices = np.arange(len(X_train))
    np.random.shuffle(shuffled_indices)

    X_train = X_train[shuffled_indices]
    scalar_values = scalar_values[shuffled_indices]
    y_train = y_train[shuffled_indices]

    model = build_model(input_shape=(41, 4), scalar_input_shape=(1,))

    model.fit([X_train, scalar_values], y_train, epochs=30, batch_size=64, validation_split=0.3, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ])

    # Predict on RNAcompete data
    precomputed_scores = precompute_scalars(cycle_files, weights={1: 10, 2: 100, 3: 1000, 4: 10000}, length=5)
    with open(rncmpt_path, 'r') as file:
        rncmpt_sequences = [line.strip() for line in file]

    X_test = np.array([one_hot_encode(seq) for seq in rncmpt_sequences])
    scalar_test_values = np.array([compute_scalar_with_precomputed(seq, precomputed_scores) for seq in rncmpt_sequences]).reshape(-1, 1)

    predictions = model.predict([X_test, scalar_test_values])

    np.savetxt(f"{rbp_name}_predictions.txt", predictions)

    return predictions


