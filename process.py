import numpy as np
from itertools import product
import os
from scipy.stats import pearsonr

CHARS = ['A', 'C', 'G', 'T']

def get_all_six_len_combinations():
    """
    יוצר את כל הקומבינציות האפשריות של רצפים באורך 7 מתוך ארבעה בסיסים (A, C, G, T).
    """
    combinations = [''.join(combination) for combination in product(CHARS, repeat=8)]
    return combinations

def initialize_seq_to_score_dict(seqs):
    """
    מאתחל מילון שבו כל רצף באורך 7 הוא מפתח והערך ההתחלתי של כל מפתח הוא 0.
    """
    return {seq: 0 for seq in seqs}

def get_six_len_seqs_with_scores(cycle_files, weights):
    """
    מעבד את כל קבצי ה-SELEX לסייקלים ומחזיר את המידע המאוחד, תוך נרמול התוצאות ושקלול לפי חשיבות הסייקלים.
    """
    combinations = get_all_six_len_combinations()
    seq_to_counts = [initialize_seq_to_score_dict(combinations) for _ in cycle_files]

    # קריאת הנתונים מכל קובץ סייקל
    for idx, cycle_file in enumerate(cycle_files):
        with open(cycle_file, 'r') as file:
            for line in file:
                sequence, count = line.strip().split(',')
                count = int(count)
                for i in range(len(sequence) - 7):
                    six_length_sequence = sequence[i:i+8]
                    if six_length_sequence in seq_to_counts[idx]:
                        seq_to_counts[idx][six_length_sequence] += count * weights[idx + 1]  # שקלול לפי משקל הסייקל
                    else:
                        if "N" in six_length_sequence:
                            char_to_replace_idx = np.random.randint(0, 4)
                            char_to_replace = CHARS[char_to_replace_idx]
                            six_length_sequence = six_length_sequence.replace("N", char_to_replace)
                            seq_to_counts[idx][six_length_sequence] += count * weights[idx + 1]
        print(f"finished with cycle file {idx + 1}")

    # חישוב תוצאות משוקללות
    combined_scores = initialize_seq_to_score_dict(combinations)
    total_weight = sum(abs(w) for w in weights.values())  # שימוש בערך מוחלט של משקלות
    for seq in combined_scores:
        weighted_sum = 0
        for idx in range(len(cycle_files)):
            weighted_sum += seq_to_counts[idx][seq]
        combined_scores[seq] = weighted_sum / total_weight

    return combined_scores

def get_rncmpt_scores(cycle_files, rncmpt_path, weights):
    """
    משווה בין תדירויות הרצפים בקבצי SELEX ומחשב את עוצמת הקישור עבור קובץ RNAcompete עבור סט משקלות מסוים.
    """
    seq_to_score = get_six_len_seqs_with_scores(cycle_files, weights)
    final_scores = []

    with open(rncmpt_path, 'r') as file:
        for line in file:
            sequence = line.split()[0]
            seq_scores = []
            for i in range(len(sequence) - 7):
                six_length_sequence = sequence[i:i+8]
                if six_length_sequence in seq_to_score:
                    seq_scores.append(seq_to_score[six_length_sequence])
                else:
                    if "U" in six_length_sequence:
                        six_length_sequence = six_length_sequence.replace("U", "T")
                        seq_scores.append(seq_to_score[six_length_sequence])

            final_seq_score = np.mean(seq_scores)
            final_scores.append(final_seq_score)

    print("finished with rncmpt")
    return final_scores

def load_rbp_cycle_info(file_path):
    """
    טוען את המידע על קבצי ה-RBP מתוך קובץ ה-Text ושומר אותו במילון.
    המפתחות הם שמות ה-RBP והערכים הם רשימות של קבצי SELEX עבור כל RBP.
    """
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



