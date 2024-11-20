import os

# Path to the folder containing the files
folder_path =r"C:\Users\sarat\OneDrive\Documents\SEM 5\Capstone\judgements\judgements\raw_judge"
files = os.listdir(folder_path)
files.sort()
for index, file in enumerate(files, start=1):
    
    new_name = f'document{index}{os.path.splitext(file)[1]}'
    
    
    old_file = os.path.join(folder_path, file)
    new_file = os.path.join(folder_path, new_name)
    
    
    os.rename(old_file, new_file)

print('Files renamed successfully.')



