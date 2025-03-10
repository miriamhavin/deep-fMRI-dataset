import numpy as np

# Define the file path
file_path = '/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset/results/contextual/105/'

# Load each file
bscorrs = np.load(file_path + 'bscorrs.npz')
corrs = np.load(file_path + 'corrs.npz')
valinds = np.load(file_path + 'valinds.npz')
valphas = np.load(file_path + 'valphas.npz')

# To see what arrays are stored in each file
print('bscorrs contains:', list(bscorrs.keys()))
print('corrs contains:', list(corrs.keys()))
print('valinds contains:', list(valinds.keys()))
print('valphas contains:', list(valphas.keys()))

# To access a specific array from a file (using 'arr_0' as an example, adjust based on actual keys)
# This shows the shape and first few elements
if 'arr_0' in bscorrs:
    print('bscorrs shape:', bscorrs['arr_0'].shape)
    print('bscorrs first few values:', bscorrs['arr_0'].flatten()[:5])

if 'arr_0' in corrs:
    print('corrs shape:', corrs['arr_0'].shape)
    print('corrs first few values:', corrs['arr_0'].flatten()[:5])

# Check for NaN and infinite values
if 'arr_0' in corrs:
    corr_values = corrs['arr_0']
    print('Number of NaN values in corrs:', np.isnan(corr_values).sum())
    print('Number of -inf values in corrs:', np.isneginf(corr_values).sum())
    print('Number of +inf values in corrs:', np.isposinf(corr_values).sum())

    # Handle NaN and infinite values
    corr_values = np.nan_to_num(corr_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate some basic statistics
    print('Mean correlation:', np.mean(corr_values))
    print('Median correlation:', np.median(corr_values))
    print('Max correlation:', np.max(corr_values))
    print('Min correlation:', np.min(cor