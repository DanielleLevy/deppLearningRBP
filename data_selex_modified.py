
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.stats import pearsonr

PROTEIN = "RBP_HTR_SELEX"
BASE_PATH = "bio_deep/data"
CHARS = ['A', 'C', 'G', 'T']

def get_all_six_len_combinations():
    combinations = [''.join(combination) for combination in product(CHARS, repeat=6)]
    return combinations

def initialize_seq_to_score_dict(seqs):
    return {seq: 0 for seq in seqs}

def get_selex_data(path):
    seq_to_counts = {}
    with open(path, 'r') as file:
        for line in file:
            sequence, count = line.strip().split(',')
            seq_to_counts[sequence] = int(count)
    return seq_to_counts

def get_selex_data_and_scores(path_to_input, path_to_rbns):
    seq_to_counts_input = get_selex_data(path_to_input)
    seq_to_counts_rbns = get_selex_data(path_to_rbns)
    
    # Process SELEX files
    for seq in seq_to_counts_input:
        if seq in seq_to_counts_rbns:
            seq_to_counts_rbns[seq] += seq_to_counts_input[seq]
    
    return seq_to_counts_rbns

