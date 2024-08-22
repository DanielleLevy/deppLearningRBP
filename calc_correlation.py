import os
import numpy as np
from scipy.stats import pearsonr

def calculate_pearson_correlation(rnacompete_dir, output_dir):
    weight_correlations = {}  # מילון לשמירת המתאמים לכל סט משקולות
    results = []

    for i in range(1, 39):  # לולאה מ-1 עד 38 כולל
        rbp_name = f"RBP{i}"
        rnacompete_file = os.path.join(rnacompete_dir, f"{rbp_name}.txt")
        output_file = os.path.join(output_dir, f"{rbp_name}_bindings_intensities_combined.txt")

        # בדיקה אם קובץ ה-RNAcompete וקובץ הפלט קיימים
        if not os.path.exists(rnacompete_file):
            print(f"File {rnacompete_file} not found, skipping.")
            continue

        if not os.path.exists(output_file):
            print(f"File {output_file} not found, skipping.")
            continue

        # טוען את הנתונים מקובץ RNAcompete
        rnacompete_data = np.loadtxt(rnacompete_file)

        # טוען את הנתונים מקובץ הפלט ומקבל את שמות העמודות (המשקולות)
        output_data = np.loadtxt(output_file, skiprows=1)
        with open(output_file, 'r') as f:
            header = f.readline().strip().split('\t')

        # חישוב מתאם פירסון עבור כל עמודה לפי המשקולות
        for idx, header_name in enumerate(header):
            weight_str = header_name.replace("Weights ", "").strip('# ')
            weights = eval(weight_str)  # הפיכת המחרוזת לרשימה
            correlation, _ = pearsonr(rnacompete_data, output_data[:, idx])

            # שמירת המתאמים במילון לפי המשקולות
            if tuple(weights) not in weight_correlations:
                weight_correlations[tuple(weights)] = []
            weight_correlations[tuple(weights)].append(correlation)

    # חישוב ממוצעים לפי המשקולות ושמירת התוצאות
    for weights, correlations in weight_correlations.items():
        average_correlation = np.mean(correlations)
        results.append(f"Average correlation for weights {weights}: {average_correlation:.4f}")

    # שמירת התוצאות לקובץ
    with open("naive_8/pearson_correlations_by_weights.txt", 'w') as result_file:
        result_file.write("\n".join(results))

    print("Pearson correlations calculated and saved to 'pearson_correlations_by_weights.txt'.")

# דוגמה לשימוש
rnacompete_dir = "./RNAcompete_intensities"  # נתיב לתיקיית ה-RNAcompete
output_dir = "./naive_8"  # נתיב לתיקיית הפלט

calculate_pearson_correlation(rnacompete_dir, output_dir)
"""

import os
import numpy as np
from scipy.stats import pearsonr

def calculate_pearson_correlation(rnacompete_dir, output_dir):
    correlations = []
    results = []

    for file_name in os.listdir(rnacompete_dir):
        if file_name.endswith(".txt"):
            rbp_name = file_name.split('.')[0]
            rnacompete_file = os.path.join(rnacompete_dir, file_name)
            output_file = os.path.join(output_dir, f"{rbp_name}_predictions.txt")

            # טוען את הנתונים משני הקבצים
            rnacompete_data = np.loadtxt(rnacompete_file)
            output_data = np.loadtxt(output_file)

            # חישוב מתאם פירסון
            correlation, _ = pearsonr(rnacompete_data, output_data)
            correlations.append(correlation)

            # שמירת התוצאה
            results.append(f"{rbp_name}: {correlation:.4f}")

    # חישוב מתאם פירסון ממוצע
    average_correlation = np.mean(correlations)
    results.append(f"Average correlation: {average_correlation:.4f}")

    # שמירת התוצאות לקובץ
    with open("pearson_correlations.txt", 'w') as result_file:
        result_file.write("\n".join(results))

    print("Pearson correlations calculated and saved to 'pearson_correlations.txt'.")

# דוגמה לשימוש
rnacompete_dir = "./RNAcompete_intensities"  # נתיב לתיקיית ה-RNAcompete
output_dir = "."  # נתיב לתיקיית הפלט
calculate_pearson_correlation(rnacompete_dir, output_dir)
"""