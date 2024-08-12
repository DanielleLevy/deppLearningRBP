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
            output_file = os.path.join(output_dir, f"{rbp_name}_bindings_intensities.txt")

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
    with open("pearson_correlations_improve.txt", 'w') as result_file:
        result_file.write("\n".join(results))

    print("Pearson correlations calculated and saved to 'pearson_correlations.txt'.")

# דוגמה לשימוש
rnacompete_dir = "./RNAcompete_intensities"  # נתיב לתיקיית ה-RNAcompete
output_dir = "."  # נתיב לתיקיית הפלט

calculate_pearson_correlation(rnacompete_dir, output_dir)
