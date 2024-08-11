
import sys
import numpy as np
from data_selex_modified import get_selex_data_and_scores

def main():
    rncmpt_path = sys.argv[1]
    input_path = sys.argv[2]
    rbns_path = sys.argv[3]

    predicted_scores = get_selex_data_and_scores(input_path, rbns_path)
    np.savetxt("bindings_intensities.txt", predicted_scores)

if __name__ == "__main__":
    main()
