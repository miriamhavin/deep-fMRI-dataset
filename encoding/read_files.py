import numpy as np

# Define the file path
file_path = "/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset/results/contextual/102/"

# Load each file
bscorrs = np.load(file_path + "bscorrs.npz")
corrs = np.load(file_path + "corrs.npz")
valinds = np.load(file_path + "valinds.npz")
valphas = np.load(file_path + "valphas.npz")

# To see what arrays are stored in each file
print("bscorrs contains:", list(bscorrs.keys()))
print("corrs contains:", list(corrs.keys()))
print("valinds contains:", list(valinds.keys()))
print("valphas contains:", list(valphas.keys()))

# To access a specific array from a file (using 'arr_0' as an example, adjust based on actual keys)
# This shows the shape and first few elements
if 'arr_0' in bscorrs:
    print("bscorrs shape:", bscorrs['arr_0'].shape)
    print("bscorrs first few values:", bscorrs['arr_0'].flatten()[:5])

if 'arr_0' in corrs:
    print("corrs shape:", corrs['arr_0'].shape)
    print("corrs first few values:", corrs['arr_0'].flatten()[:5])

# Calculate some basic statistics
if 'arr_0' in corrs:
    corr_values = corrs['arr_0']
    print("Mean correlation:", np.nanmean(corr_values))
    print("Median correlation:", np.nanmedian(corr_values))
    print("Max correlation:", np.nanmax(corr_values))
    print("Min correlation:", np.nanmin(corr_values))
    print("Number of NaN values:", np.isnan(corr_values).sum())