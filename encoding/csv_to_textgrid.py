import os
import csv
from os.path import join

def csv_to_textgrid(data_dir, textgrid_dir):
    # Ensure the TextGrid directory exists
    os.makedirs(textgrid_dir, exist_ok=True)

    # List all CSV files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            print(f"Converting {filename} to TextGrid")
            csv_file_path = join(data_dir, filename)
            textgrid_file_path = join(textgrid_dir, filename.replace('.csv', '.TextGrid'))

            with open(csv_file_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                # Initialize variables to find min and max times
                min_time = float('inf')
                max_time = float('-inf')

                # Read through the file to determine min and max times
                intervals = list(reader)
                for row in intervals:
                    if len(row) < 3 or not row[1].strip() or not row[2].strip():
                        print(f"Skipping malformed or incomplete row in {filename}: {row}")
                        continue
                    start, end = float(row[1]), float(row[2])
                    if start < min_time:
                        min_time = start
                    if end > max_time:
                        max_time = end

                # Write the TextGrid file with the computed xmin and xmax
                with open(textgrid_file_path, 'w') as tgfile:
                    tgfile.write('File type = "ooTextFile"\nObject class = "TextGrid"\n')
                    tgfile.write(f'xmin = {min_time}\nxmax = {max_time}\n')
                    tgfile.write('tiers? <exists>\nsize = 1\nitem []:\n')
                    tgfile.write('    item [1]:\n        class = "IntervalTier"\n')
                    tgfile.write(f'        name = "words"\n        xmin = {min_time}\n        xmax = {max_time}\n')
                    tgfile.write(f'        intervals: size = {len(intervals)}\n')
                    for index, (word, start, end) in enumerate(intervals, start=1):
                        tgfile.write(f'        intervals [{index}]:\n')
                        tgfile.write(f'            xmin = {start}\n')
                        tgfile.write(f'            xmax = {end}\n')
                        tgfile.write(f'            text = "{word}"\n')

# Example usage
DATA_DIR = "C:/Users/owner/PycharmProjects/deep-fMRI-dataset/em_data"
# TEXTGRID_DIR = join(DATA_DIR, "TextGrids")
# CSV_DIR = join(DATA_DIR, "timing")
# csv_to_textgrid(CSV_DIR, TEXTGRID_DIR)


import h5py

def print_structure(hdf5_file, indent=0):
    """
    Recursively prints the groups and datasets in an HDF5 file.
    """
    for key in hdf5_file.keys():
        item = hdf5_file[key]
        print('    ' * indent + key)
        if isinstance(item, h5py.Dataset):  # If it's a Dataset, print its shape
            print('    ' * (indent + 1) + f"Dataset: {item.shape}, dtype: {item.dtype}")
        elif isinstance(item, h5py.Group):  # If it's a Group, do a recursive call
            print_structure(item, indent + 1)

# Path to your HDF5 file
file_path = 'C:/Users/owner/PycharmProjects/deep-fMRI-dataset/em_data/english1000sm.hf5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    print("Structure of the HDF5 file:")
    print_structure(file)
