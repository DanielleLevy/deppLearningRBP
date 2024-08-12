import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import product
import numpy as np

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



CHARS = ['A', 'C', 'G', 'T']


def one_hot_encode(seq, maxlen=41):
    mapping = {char: idx for idx, char in enumerate(CHARS)}
    encoded = np.zeros((maxlen, len(CHARS)))
    seq_len = len(seq)

    # Fill the array with the one-hot encoded sequence
    for i, char in enumerate(seq):
        if i < maxlen:
            encoded[i, mapping[char]] = 1

    # Apply padding with 0.25 across all nucleotides for remaining positions
    if seq_len < maxlen:
        encoded[seq_len:] = 0.25

    return encoded


def get_all_six_len_combinations():
    combinations = [''.join(combination) for combination in product(CHARS, repeat=6)]
    return combinations


def initialize_seq_to_score_dict(seqs):
    return {seq: 0 for seq in seqs}


def get_six_len_seqs_with_scores(cycle_files):
    combinations = get_all_six_len_combinations()
    seq_to_counts = [initialize_seq_to_score_dict(combinations) for _ in cycle_files]

    # משקלות לכל סייקל
    weights = {1: 1, 2: 10, 3: 100, 4: 1000}

    # קריאת הנתונים מכל קובץ סייקל
    for idx, cycle_file in enumerate(cycle_files):
        with open(cycle_file, 'r') as file:
            for line in file:
                sequence, count = line.strip().split(',')
                count = int(count)
                for i in range(len(sequence) - 5):
                    six_length_sequence = sequence[i:i + 6]
                    if six_length_sequence in seq_to_counts[idx]:
                        seq_to_counts[idx][six_length_sequence] += count
                    else:
                        if "N" in six_length_sequence:
                            char_to_replace_idx = np.random.randint(0, 4)
                            char_to_replace = CHARS[char_to_replace_idx]
                            six_length_sequence = six_length_sequence.replace("N", char_to_replace)
                            seq_to_counts[idx][six_length_sequence] += count
        print(f"finished with cycle file {idx + 1}")

    # חישוב תוצאות משוקללות
    combined_scores = initialize_seq_to_score_dict(combinations)
    total_weights = sum(weights.values())
    for seq in combined_scores:
        weighted_sum = 0
        for idx in range(len(cycle_files)):
            weighted_sum += seq_to_counts[idx][seq] * weights[idx + 1]
        combined_scores[seq] = weighted_sum / total_weights

    return combined_scores


def prepare_data_for_training(rbp_name, cycle_files, rncmpt_path):
    seq_to_score = get_six_len_seqs_with_scores(cycle_files)

    rncmpt_sequences = []
    with open(rncmpt_path, 'r') as file:
        for line in file:
            sequence = line.strip().split()[0]
            rncmpt_sequences.append(one_hot_encode(sequence))

    X = np.array(rncmpt_sequences)
    y = np.array([seq_to_score.get(''.join([CHARS[np.argmax(n)] for n in seq]), 0) for seq in rncmpt_sequences])

    return X, y


def train_and_predict(rbp_name, cycle_files, rncmpt_path):
    X, y = prepare_data_for_training(rbp_name, cycle_files, rncmpt_path)

    model = build_model(input_shape=(41, len(CHARS)))

    model.fit(X, y, epochs=10, batch_size=32)

    predictions = model.predict(X)
    np.savetxt(f"{rbp_name}_predictions.txt", predictions)

    return predictions
