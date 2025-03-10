import numpy as np
import matplotlib.pyplot as plt

# Replace with your actual save location
save_location = "/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset/results/contextual/103"  # Example path

# Load the saved mean correlations
mean_corrs_file = np.load(f"{save_location}/mean_corrs.npz")
mean_corrs = mean_corrs_file['arr_0']  # NPZ files typically store arrays with default names like 'arr_0'

# Print basic statistics about the mean correlations
print(f"Shape of mean correlations: {mean_corrs.shape}")
print(f"Mean of mean correlations: {np.mean(mean_corrs)}")
print(f"Median of mean correlations: {np.median(mean_corrs)}")
print(f"Max mean correlation: {np.max(mean_corrs)}")
print(f"Min mean correlation: {np.min(mean_corrs)}")

# Optional: create a histogram to visualize the distribution of mean correlations
plt.figure(figsize=(10, 6))
plt.hist(mean_corrs, bins=50)
plt.title('Distribution of Mean Correlation Values')
plt.xlabel('Mean Correlation')
plt.ylabel('Count')
plt.axvline(x=0, color='r', linestyle='--')  # Add a line at x=0 for reference
plt.savefig(f"{save_location}/mean_corrs_histogram.png")
print(f"Histogram saved to {save_location}/mean_corrs_histogram.png")