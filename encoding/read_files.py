import numpy as np
import argparse

# Define the file path
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True)
args = parser.parse_args()
subject = args.subject
file_path = f'/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset-miriam/results/contextual/{subject}/random_split/'

# Load each file
bscorrs = np.load(file_path + 'bscorrs.npz')
corrs = np.load(file_path + 'corrs.npz')
valinds = np.load(file_path + 'valinds.npz')
valphas = np.load(file_path + 'valphas.npz')
pvals = np.load(file_path + 'pvals.npz')

# Files and their arrays
files = {
    'bscorrs': bscorrs,
    'corrs': corrs,
    'valinds': valinds,
    'valphas': valphas,
    'pvals': pvals
}

# Examine all files and their contents
print(f"\n===== EXPLORING DATA FOR SUBJECT: {subject} =====\n")

for file_name, file_data in files.items():
    print(f"\n----- File: {file_name}.npz -----")

    # List all arrays in the file
    array_keys = list(file_data.keys())
    print(f"Contains {len(array_keys)} arrays: {array_keys}")

    # Examine each array in detail
    for key in array_keys:
        array = file_data[key]
        print(f"\n  Array: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")

        # Get array statistics
        if array.size > 0:  # Only if array is not empty
            flat_array = array.flatten()
            print(flat_array)

            # Count special values
            nan_count = np.isnan(flat_array).sum()
            neginf_count = np.isneginf(flat_array).sum()
            posinf_count = np.isposinf(flat_array).sum()

            print(f"  Special values:")
            print(f"    NaN: {nan_count} ({nan_count / flat_array.size:.2%} of elements)")
            print(f"    -Inf: {neginf_count} ({neginf_count / flat_array.size:.2%} of elements)")
            print(f"    +Inf: {posinf_count} ({posinf_count / flat_array.size:.2%} of elements)")

            # Filter out non-finite values for statistics
            valid_values = flat_array[np.isfinite(flat_array)]
            if valid_values.size > 0:
                print(f"  Statistics (excluding non-finite values):")
                print(f"    Min: {np.min(valid_values)}")
                print(f"    Max: {np.max(valid_values)}")
                print(f"    Mean: {np.mean(valid_values)}")
                print(f"    Median: {np.median(valid_values)}")
                print(f"    Std Dev: {np.std(valid_values)}")

                # Show histogram-like distribution summary
                if valid_values.size > 10:  # Only if we have enough values
                    percentiles = [0, 10, 25, 50, 75, 90, 100]
                    percentile_values = np.percentile(valid_values, percentiles)
                    print(f"  Value distribution:")
                    for p, v in zip(percentiles, percentile_values):
                        print(f"    {p}th percentile: {v}")

            # Show a few sample values
            sample_size = min(5, flat_array.size)
            if sample_size > 0:
                print(f"  Sample values (first {sample_size}):")
                print(f"    {flat_array[:sample_size]}")

print("\n===== EXPLORATION COMPLETE =====")