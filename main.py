import sys
import numpy as np
from process import get_rncmpt_scores, load_rbp_cycle_info
import os
from cnn import  train_and_predict

def main_naive():
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




def main_cnn():
    rbp_info_path = sys.argv[1]
    base_path = sys.argv[2]

    rbp_files = load_rbp_cycle_info(rbp_info_path)

    rncmpt_path = os.path.join(base_path, "RNAcompete_sequences_rc.txt")

    for rbp, cycle_files in rbp_files.items():
        print(f"Processing {rbp} with {len(cycle_files)} cycles")

        predictions = train_and_predict(rbp, cycle_files, rncmpt_path)

        print(f"Results for {rbp} saved.")


def main():
    rbp_info_path = sys.argv[1]  # הנתיב לקובץ המידע על ה-RBP והסייקלים
    base_path = sys.argv[2]  # בסיס הנתיבים, למשל התיקייה שבה נמצאים קבצי ה-RBP וה-RNAcompete

    # סטים שונים של משקלות לבדיקה, כולל משקלות שליליים עבור cycle 1
    weight_sets = [
        {1: 1, 2: 2, 3: 5, 4: 10},
        {1:10,2:100,3:1000,4:10000},
        {1: 10, 2: 100, 3: 1000, 4: 100000},
        {1: -1, 2: 10, 3: 100, 4: 1000},
        {1: 1, 2: 1, 3: 3, 4: 5},
        {1: -1, 2: 3, 3: 3, 4: 7},  # משקל שלילי ל-cycle 1
        {1: -2, 2: 4, 3: 6, 4: 8} ,  # משקל שלילי חזק יותר ל-cycle 1
        {1: -2, 2: 0, 3: 0, 4: 8},  # משקל שלילי חזק יותר ל-cycle 1
        {1: 1, 2: 2, 3: 3, 4: 4},  # ליניארי
        {1: 1, 2: 4, 3: 16, 4: 64},  # גאומטרי
        {1: -1, 2: 1, 3: 2, 4: 4},  # הפוך עם משקל שלילי ל-cycle 1
        {1: 0.5, 2: 1, 3: 5, 4: 20},  # משקלות דינמיים עם השפעה חזקה ל-cycles האחרונים
        {1: np.log(1 + 1), 2: np.log(2 + 1), 3: np.log(3 + 1), 4: np.log(4 + 1)},  # לוגריתמי
        {1: 1, 2: 2, 3: 10, 4: 15},  # לפי תוצאות ידועות
        {1: 1, 2: 4, 3: 9, 4: 16},  # ריבועי
        {1: 1, 2: 1.5, 3: 3, 4: 6},  # משקלות גמישים לפי בדיקה
        {1: -2, 2: 0, 3: 2, 4: 8}  # משקל שלילי חזק יותר ל-cycle 1
    ]

    # טוען את המידע על קבצי הסייקלים לכל RBP
    rbp_files = load_rbp_cycle_info(rbp_info_path)

    rncmpt_path = os.path.join(base_path, "RNAcompete_sequences_rc.txt")

    for rbp, cycle_files in rbp_files.items():
        print(f"Processing {rbp} with {len(cycle_files)} cycles")
        results = []
        for weights in weight_sets:
            predicted_scores = get_rncmpt_scores(cycle_files, rncmpt_path, weights)
            results.append(predicted_scores)

        # שילוב התוצאות לעמודות שונות בקובץ פלט אחד
        results = np.array(results).T  # להפוך את התוצאות כך שכל סט משקלות יהיה בעמודה
        output_file = f"./naive_8/{rbp}_bindings_intensities_combined.txt"
        np.savetxt(output_file, results, delimiter="\t", header="\t".join([f"Weights {list(weights.values())}" for weights in weight_sets]))
        print(f"Results saved to {output_file}")
if __name__ == "__main__":
    main_cnn()