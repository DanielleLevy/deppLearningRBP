import sys
import numpy as np
from process import get_rncmpt_scores, load_rbp_cycle_info
import os

def main():
    rbp_info_path = sys.argv[1]  # הנתיב לקובץ המידע על ה-RBP והסייקלים
    base_path = sys.argv[2]  # בסיס הנתיבים, למשל התיקייה שבה נמצאים קבצי ה-RBP וה-RNAcompete

    # טוען את המידע על קבצי הסייקלים לכל RBP
    rbp_files = load_rbp_cycle_info(rbp_info_path)

    rncmpt_path = os.path.join(base_path, "RNAcompete_sequences_rc.txt")

    for rbp, cycle_files in rbp_files.items():
        print(f"Processing {rbp} with {len(cycle_files)} cycles")
        predicted_scores = get_rncmpt_scores(cycle_files, rncmpt_path)
        output_file = f"{rbp}_bindings_intensities.txt"
        np.savetxt(output_file, predicted_scores)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
