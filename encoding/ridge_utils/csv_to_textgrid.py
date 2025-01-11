import os
import numpy as np
from os.path import join, dirname
import csv
from config import DATA_DIR

def csv_to_textgrid(csv_file_path, textgrid_file_path):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # Initialize variables to find min and max times
        min_time = float('inf')
        max_time = float('-inf')

        # First, read through the file to determine min and max times
        intervals = list(reader)  # Read all data into memory to reuse it
        for row in intervals:
            start, end = float(row[1]), float(row[2])
            if start < min_time:
                min_time = start
            if end > max_time:
                max_time = end

        # Now write the TextGrid file with the computed xmin and xmax
        with open(textgrid_file_path, 'w') as tgfile:
            tgfile.write('File type = "ooTextFile"\nObject class = "TextGrid"\n')
            tgfile.write(f'xmin = {min_time}\nxmax = {max_time}\n')
            tgfile.write('tiers? <exists>\nsize = 1\nitem []:\n')
            tgfile.write('    item [1]:\n        class = "IntervalTier"\n')
            tgfile.write(f'        name = "words"\n        xmin = {min_time}\n        xmax = {max_time}\n')
            tgfile.write(f'        intervals: size = {len(intervals)}\n')
            for index, row in enumerate(intervals, start=1):
                word, start, end = row[0], row[1], row[2]
                tgfile.write(f'        intervals [{index}]:\n')
                tgfile.write(f'            xmin = {start}\n')
                tgfile.write(f'            xmax = {end}\n')
                tgfile.write(f'            text = "{word}"\n')

csv_to_textgrid(join(DATA_DIR, "timing"), join(DATA_DIR, "TextGrid"))
