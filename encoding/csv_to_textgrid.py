import os
import csv
from os.path import join

def csv_to_textgrid(data_dir, textgrid_dir):
    # Ensure the TextGrid directory exists
    os.makedirs(textgrid_dir, exist_ok=True)

    # List all CSV files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            new_filename = filename.replace("wk", "week").replace("lec", "lecture").replace('.csv', '.TextGrid')
            print(f"Converting {filename} to {new_filename}")
            csv_file_path = join(data_dir, filename)
            textgrid_file_path = join(textgrid_dir, new_filename)

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
                    tgfile.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
                    tgfile.write(f'xmin = {min_time}\nxmax = {max_time}\n')
                    tgfile.write('tiers? <exists>\nsize = 1\nitem []:\n')
                    tgfile.write('    item [1]:\n        class = "IntervalTier"\n')
                    tgfile.write(f'        name = "words"\n        xmin = {min_time}\n        xmax = {max_time}\n')
                    tgfile.write(f'        intervals: size = {len(intervals)}\n')
                    for index, row in enumerate(intervals, start=1):
                        if len(row) >= 3:
                            word = row[0]
                            start = row[1]
                            end = row[2]
                            # Handle optional 'correct' value if present
                            correct = row[3] if len(row) > 3 else "undefined"
                            tgfile.write(f'        intervals [{index}]:\n')
                            tgfile.write(f'            xmin = {start}\n')
                            tgfile.write(f'            xmax = {end}\n')
                            tgfile.write(f'            text = "{word}"\n')
                        else:
                            print(f"Skipping incomplete row: {row}")


# Example usage
DATA_DIR = "/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset/encoding/data/"
TEXTGRID_DIR = join(DATA_DIR, "TextGrids")
CSV_DIR = join(DATA_DIR, "CSV")
csv_to_textgrid(CSV_DIR, TEXTGRID_DIR)


