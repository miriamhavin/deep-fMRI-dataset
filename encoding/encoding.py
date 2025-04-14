import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging
from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.ridge import bootstrap_ridge, ridge, ridge_corr, ridge_corr_pred
from config import REPO_DIR, EM_DATA_DIR
from encoding_utils import get_week_lecture
from significance_testing import model_pvalue, permutation_test, fdr_correct
from ridge_utils.npp import mcorr
import numpy as np
import h5py
import os
from os.path import join
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, zscore
import scipy.stats as stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=1)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=50)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_components", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("-use_corr", action="store_true")
    parser.add_argument("-single_alpha", action="store_true")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    globals().update(args.__dict__)

    fs = " ".join(_FEATURE_CONFIG.keys())
    assert feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
    assert np.amax(sessions) <= 6 and np.amin(sessions) >= 1, "1 <= session <= 5"

    sessions = list(map(str, sessions))
    with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    # Instead of splitting by stories, collect all stories and all data first
    all_stories = []
    dir_path = "/sci/labs/arielgoldstein/miriam1234/6motion_students"
    for sess in sessions:
        stories = sess_to_story[sess][0] + [sess_to_story[sess][1]]  # Include test story too
        valid_stories = [story for story in stories if story in cut_stories([story], subject)]
        all_stories.extend(valid_stories)

    # Remove duplicates while preserving order
    all_stories = list(dict.fromkeys(all_stories))

    save_location = join(REPO_DIR, "results", feature, subject, "random_split")
    print("Saving encoding model & results to:", save_location)
    os.makedirs(save_location, exist_ok=True)

    # Get feature space for all stories
    print(f"Getting feature space for {len(all_stories)} stories")
    downsampled_feat = get_feature_space(feature, all_stories)

    def fisher_z(r):
        """Convert Pearson r to Fisher Z"""
        return 0.5 * np.log((1 + np.array(r)) / (1 - np.array(r)))


    def inverse_fisher_z(z):
        """Convert Fisher Z back to Pearson r"""
        z_array = np.array(z)
        return (np.exp(2 * z_array) - 1) / (np.exp(2 * z_array) + 1)


    def train_neural_encoder(subject, feature, n_components=30, n_folds=5):

        # Get all stimulus and response data for the story
        all_stim = apply_zscore_and_hrf(all_stories, downsampled_feat, trim, ndelays)
        all_resp = get_response(all_stories, subject)

        # Filter to include only stimulus-locked voxels (use non_constant_voxels as a proxy)
        voxel_std = np.std(all_resp, axis=0)
        non_constant_voxels = voxel_std > 1e-10
        all_resp_trimmed = all_resp[:, non_constant_voxels]

        print(f"Original number of voxels: {len(voxel_std)}")
        print(f"Number of non-constant voxels: {np.sum(non_constant_voxels)}")

        # Apply PCA to reduce dimensionality
        print(f"Applying PCA to reduce feature dimensions to {n_components}...")
        pca = PCA(n_components=n_components)
        all_stim_pca = pca.fit_transform(all_stim)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

        # Set up 5-fold cross-validation without shuffling (to preserve temporal structure)
        kfold = KFold(n_splits=n_folds, shuffle=False)

        # Initialize arrays to store correlation scores for each voxel and fold
        n_voxels = all_resp_trimmed.shape[1]
        fold_corrs = np.zeros((n_folds, n_voxels))

        # Run 5-fold cross-validation
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(all_stim_pca)):
            print(f"\nProcessing fold {fold_idx + 1}/{n_folds}")

            # Split data into train and test sets
            train_stim = all_stim_pca[train_index]
            test_stim = all_stim_pca[test_index]
            train_resp = all_resp_trimmed[train_index]
            test_resp = all_resp_trimmed[test_index]

            print(f"Training on {train_stim.shape[0]} samples, testing on {test_stim.shape[0]} samples")

            # Train linear regression model (ridge with small alpha for stability)
            wt = ridge(train_stim, train_resp, alpha=1.0, singcutoff=1e-10)

            # Predict responses on test set
            test_pred = np.dot(test_stim, wt)

            # Calculate Pearson correlation between predicted and actual responses
            for vox_idx in range(n_voxels):
                fold_corrs[fold_idx, vox_idx] = np.corrcoef(test_resp[:, vox_idx],
                                                            test_pred[:, vox_idx])[0, 1]

            # Print fold results
            print(f"Fold {fold_idx + 1} mean correlation: {np.nanmean(fold_corrs[fold_idx]):.4f}")

        # Transform correlations to Fisher Z, average, then convert back to r
        # First, handle NaN values
        fold_corrs = np.nan_to_num(fold_corrs)

        # Transform to Fisher Z
        fisher_z_scores = fisher_z(fold_corrs)

        # Average Z scores across folds
        avg_z_scores = np.mean(fisher_z_scores, axis=0)

        # Convert back to correlation
        avg_corrs = inverse_fisher_z(avg_z_scores)
        correlation_threshold = 0.1
        high_correlations = np.sum(avg_corrs > correlation_threshold)
        total_correlations = len(avg_corrs)
        percentage_high_correlations = (high_correlations / total_correlations) * 100

        print(
            f"Number of correlations > {correlation_threshold}: {high_correlations}/{total_correlations} ({percentage_high_correlations:.2f}%)")

        # Print overall results
        print("\nOverall performance:")
        print(f"Mean correlation: {np.mean(avg_corrs):.4f}")
        print(f"Median correlation: {np.median(avg_corrs):.4f}")
        print(f"Max correlation: {np.max(avg_corrs):.4f}")

        # Save results
        save_location = join(REPO_DIR, "results", feature, subject, "neural_encoder")
        os.makedirs(save_location, exist_ok=True)

        np.savez(join(save_location, "fold_corrs.npz"), fold_corrs)
        np.savez(join(save_location, "avg_corrs.npz"), avg_corrs)

        # Save metadata
        metadata = {
            'n_components': n_components,
            'n_folds': n_folds,
            'n_voxels': n_voxels,
            'mean_corr': float(np.mean(avg_corrs)),
            'median_corr': float(np.median(avg_corrs)),
            'max_corr': float(np.max(avg_corrs)),
            'percentage_high_correlations': percentage_high_correlations

        }

        with open(join(save_location, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        return avg_corrs


    # Call the function
    avg_corrs = train_neural_encoder(subject, feature)