import os


def get_rbp_cycle_files(base_path):
    """
    מחזיר מילון שבו המפתחות הם שמות ה-RBP והערכים הם רשימות של קבצים לכל RBP.
    """
    rbp_files = {}
    for file_name in os.listdir(base_path):
        if file_name.endswith(".txt"):
            rbp_name = file_name.split('_')[0]
            if rbp_name not in rbp_files:
                rbp_files[rbp_name] = []
            rbp_files[rbp_name].append(os.path.join(base_path, file_name))

    return rbp_files


def save_rbp_cycle_info(rbp_files, output_file):
    """
    שומר את המידע על מספר הסייקלים לכל RBP בקובץ טקסט.
    """
    with open(output_file, 'w') as file:
        for rbp, files in rbp_files.items():
            file.write(f"{rbp}: {len(files)} cycles\n")
            for cycle_file in files:
                file.write(f"  {cycle_file}\n")


# דוגמה לשימוש:
base_path = "htr-selex"
rbp_files = get_rbp_cycle_files(base_path)
output_file = "rbp_cycle_info.txt"
save_rbp_cycle_info(rbp_files, output_file)
print(f"RBP cycle information saved to {output_file}")
