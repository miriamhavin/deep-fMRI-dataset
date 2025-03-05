import os
import argparse
import csv
import re


def csv_to_textgrid(csv_path, output_dir, delimiter=',', skip_empty=True, skip_header=True):
    """
    Convert a delimited CSV to TextGrid format with robust error handling.

    Args:
        csv_path (str): Full path to the CSV file
        output_dir (str): Output directory for TextGrid files
        delimiter (str): Delimiter character in the CSV
        skip_empty (bool): Skip rows with empty fields
        skip_header (bool): Skip the first row if it appears to be a header
    """
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.TextGrid")

    # Read all rows from the CSV
    all_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 3:  # We need at least 3 columns
                all_rows.append(row)

    if not all_rows:
        raise ValueError(f"No valid rows found in {csv_path}")

    # Determine if first row is a header (if it has non-numeric data in time columns)
    first_row_is_header = False
    if skip_header and len(all_rows) > 1:
        try:
            float(all_rows[0][1])  # Try to convert second column to float
            float(all_rows[0][2])  # Try to convert third column to float
        except (ValueError, IndexError):
            first_row_is_header = True

    # Skip header if detected
    data = all_rows[1:] if first_row_is_header else all_rows

    # Filter out rows with empty fields
    valid_rows = []
    for row in data:
        try:
            word = row[0]
            start_time = float(row[1] if row[1] else 0)  # Default to 0 if empty
            end_time = float(row[2] if row[2] else 0)  # Default to 0 if empty

            # Only include rows where times are valid and start < end
            if not (start_time == 0 and end_time == 0) and start_time < end_time:
                valid_rows.append((word, start_time, end_time))
        except (ValueError, IndexError) as e:
            # Skip rows that cause conversion errors if skip_empty is True
            if not skip_empty:
                raise ValueError(f"Error parsing row {row}: {str(e)}")

    if not valid_rows:
        raise ValueError(f"No valid data rows found in {csv_path} after filtering")

    # Calculate min and max time
    min_time = min(row[1] for row in valid_rows)
    max_time = max(row[2] for row in valid_rows)

    # Create TextGrid file content
    textgrid_content = f"""File type = "ooTextFile"
Object class = "TextGrid"
xmin = {min_time}
xmax = {max_time}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = {min_time}
        xmax = {max_time}
        intervals: size = {len(valid_rows)}
"""

    # Add intervals
    for i, (word, start_time, end_time) in enumerate(valid_rows, 1):
        textgrid_content += f"""        intervals [{i}]:
            xmin = {start_time}
            xmax = {end_time}
            text = "{word}"
"""

    # Write the TextGrid to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(textgrid_content)

    return output_file, len(valid_rows), len(all_rows)


def main():
    parser = argparse.ArgumentParser(description='Robust CSV to TextGrid converter')
    parser.add_argument('--input_dir', required=True, help='Input directory containing CSV files')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for TextGrid files')
    parser.add_argument('--delimiter', '-d', default=',', help='CSV delimiter (default: comma)')
    parser.add_argument('--include-problematic', '-p', action='store_true',
                        help='Try to include rows with empty fields by filling with defaults')
    parser.add_argument('--keep-header', '-k', action='store_true',
                        help='Do not skip the header row')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Keep track of statistics
    total_files = 0
    successful_files = 0
    total_rows_processed = 0
    total_rows_output = 0

    # Process all files
    for csv_filename in os.listdir(args.input_dir):
        if csv_filename.endswith('.csv'):
            total_files += 1
            csv_path = os.path.join(args.input_dir, csv_filename)

            try:
                output_file, rows_output, rows_total = csv_to_textgrid(
                    csv_path,
                    args.output_dir,
                    args.delimiter,
                    not args.include_problematic,
                    not args.keep_header
                )

                print(f"Converted: {csv_filename} → {os.path.basename(output_file)} "
                      f"({rows_output}/{rows_total} rows)")

                successful_files += 1
                total_rows_processed += rows_total
                total_rows_output += rows_output

            except Exception as e:
                print(f"Error processing {csv_filename}: {str(e)}")

    # Print summary
    print("\nConversion Summary:")
    print(f"Files processed: {total_files}")
    print(f"Files successfully converted: {successful_files}")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Total rows included in output: {total_rows_output}")
    print(f"TextGrid files saved in: {args.output_dir}")


if __name__ == "__main__":
    # Directory where your TextGrid files are stored
    directory = "/TextGrids"

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Match files with the pattern "wk{A}_lec{B}"
        match = re.match(r"wk(\d+)_lec(\d+).TextGrid$", filename)
        if match:
            # Extract parts A and B from the filename
            week_number, lecture_number = match.groups()

            # Construct new filename with "vid" instead of "lec"
            new_filename = f"wk{week_number}_vid{lecture_number}.TextGrid"

            # Full path for old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")

