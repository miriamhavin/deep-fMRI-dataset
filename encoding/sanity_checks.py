import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
import warnings
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold


def save_fig(filename, save_dir="sanity_check_figures"):
    """Helper function to save figures in a dedicated directory"""
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# ========== 1. DATA QUALITY CHECKS ==========

def temporal_snr_analysis(neural_data):
    """
    Calculate and visualize temporal SNR across voxels

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    """
    # Calculate temporal SNR (mean/std over time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsnr = np.mean(neural_data, axis=0) / np.std(neural_data, axis=0)

    # Replace infinities with NaN
    tsnr[~np.isfinite(tsnr)] = np.nan

    # Plot histogram of tSNR values
    plt.figure(figsize=(10, 6))
    plt.hist(tsnr[~np.isnan(tsnr)], bins=50, alpha=0.7)
    plt.axvline(np.nanmedian(tsnr), color='r', linestyle='--',
                label=f'Median tSNR: {np.nanmedian(tsnr):.2f}')
    plt.xlabel('Temporal SNR')
    plt.ylabel('Number of Voxels')
    plt.title('Distribution of Temporal SNR Across Voxels')
    plt.legend()
    save_fig('temporal_snr_distribution.png')

    # Summary statistics
    print(f"Temporal SNR Summary:")
    print(f"  Median tSNR: {np.nanmedian(tsnr):.2f}")
    print(f"  Mean tSNR: {np.nanmean(tsnr):.2f}")
    print(f"  Min tSNR: {np.nanmin(tsnr):.2f}")
    print(f"  Max tSNR: {np.nanmax(tsnr):.2f}")
    print(f"  % voxels with tSNR > 20: {np.mean(tsnr > 20) * 100:.2f}%")
    print(f"  % voxels with tSNR > 50: {np.mean(tsnr > 50) * 100:.2f}%")

    return tsnr


# ========== 2. FEATURE SPACE CHECKS ==========

def feature_space_analysis(features):
    """
    Analyze properties of the feature space

    Parameters:
    -----------
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    """
    # Check for constant dimensions
    feature_std = np.std(features, axis=0)
    constant_features = np.where(feature_std == 0)[0]
    print(f"Number of constant features: {len(constant_features)}")
    if len(constant_features) > 0:
        print(f"Indices of constant features: {constant_features}")

    # Plot distribution of feature standard deviations
    plt.figure(figsize=(10, 6))
    plt.hist(feature_std, bins=50, alpha=0.7)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Standard Deviations')
    save_fig('feature_std_distribution.png')

    # Feature dimensionality analysis using PCA
    pca = PCA().fit(features)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', markersize=3)
    plt.axhline(0.9, color='r', linestyle='--',
                label='90% Explained Variance')

    # Find dimension that explains 90% variance
    dims_90pct = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0] + 1
    plt.axvline(dims_90pct, color='g', linestyle='--',
                label=f'Dims for 90% Var: {dims_90pct}')

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig('feature_pca_explained_variance.png')

    print(f"Feature Dimensionality:")
    print(f"  Original dimensions: {features.shape[1]}")
    print(f"  Dimensions explaining 90% variance: {dims_90pct}")
    print(
        f"  Dimensions explaining 95% variance: {np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1}")

    # Feature correlation structure
    if features.shape[1] <= 500:  # Only if not too many features
        plt.figure(figsize=(10, 8))
        feature_corr = np.corrcoef(features.T)
        sns.heatmap(feature_corr, cmap='coolwarm', center=0,
                    xticklabels=False, yticklabels=False)
        plt.title('Feature Correlation Matrix')
        save_fig('feature_correlation_matrix.png')

    return feature_std


# ========== 3. MODEL VALIDATION CHECKS ==========

def null_distribution_test(neural_data, features, n_permutations=100, n_voxels=1000):
    """
    Compare real correlations to null distribution from permuted data

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    n_permutations : int
        Number of permutations to create null distribution
    n_voxels : int
        Number of voxels to sample for computational efficiency
    """
    n_timepoints = neural_data.shape[0]

    # Sample voxels if needed
    if neural_data.shape[1] > n_voxels:
        sampled_voxels = np.random.choice(neural_data.shape[1], n_voxels, replace=False)
        neural_subset = neural_data[:, sampled_voxels]
    else:
        neural_subset = neural_data
        n_voxels = neural_data.shape[1]

    # Calculate real correlations (max correlation across features)
    real_corrs = np.zeros(n_voxels)
    for v in range(n_voxels):
        if v % 100 == 0:
            print(f"Processing real voxel {v}/{n_voxels}")

        v_corrs = []
        for f in range(features.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    corr, _ = pearsonr(neural_subset[:, v], features[:, f])
                    if not np.isnan(corr):
                        v_corrs.append(abs(corr))
                except:
                    continue

        if v_corrs:
            real_corrs[v] = max(v_corrs)

    # Calculate null distribution by permuting timepoints
    null_corrs = np.zeros((n_permutations, n_voxels))

    for p in range(n_permutations):
        print(f"Running permutation {p + 1}/{n_permutations}")

        # Permute the neural data timepoints
        perm_indices = np.random.permutation(n_timepoints)
        permuted_neural = neural_subset[perm_indices, :]

        # Calculate correlations with permuted data
        for v in range(n_voxels):
            v_corrs = []
            for f in range(min(5, features.shape[1])):  # Limit features for speed
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        corr, _ = pearsonr(permuted_neural[:, v], features[:, f])
                        if not np.isnan(corr):
                            v_corrs.append(abs(corr))
                    except:
                        continue

            if v_corrs:
                null_corrs[p, v] = max(v_corrs)

    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(real_corrs, bins=30, alpha=0.7, label='Real Correlations')
    plt.hist(null_corrs.flatten(), bins=30, alpha=0.5, label='Null Distribution')
    plt.xlabel('Maximum Absolute Correlation')
    plt.ylabel('Count')
    plt.title('Real vs. Null Correlation Distributions')
    plt.legend()
    save_fig('real_vs_null_correlations.png')

    # Statistical comparison
    mean_real = np.mean(real_corrs)
    mean_null = np.mean(null_corrs)
    tstat, pval = ttest_1samp(real_corrs, np.mean(null_corrs))

    print(f"Null Distribution Test:")
    print(f"  Mean real correlation: {mean_real:.4f}")
    print(f"  Mean null correlation: {mean_null:.4f}")
    print(f"  Difference: {mean_real - mean_null:.4f}")
    print(f"  T-statistic: {tstat:.4f}")
    print(f"  P-value: {pval:.8f}")

    return real_corrs, null_corrs


def noise_ceiling_estimation(neural_data, n_splits=5):
    """
    Estimate noise ceiling by calculating reliability between data splits

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    n_splits : int
        Number of splits for cross-validation
    """
    n_timepoints = neural_data.shape[0]
    n_voxels = neural_data.shape[1]

    # Use K-fold cross-validation to create splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Calculate split-half reliability for each voxel
    reliabilities = np.zeros((n_splits, n_voxels))

    for i, (idx1, idx2) in enumerate(kf.split(range(n_timepoints))):
        data1 = neural_data[idx1, :]
        data2 = neural_data[idx2, :]

        # Calculate temporal mean for each split
        mean1 = np.mean(data1, axis=0)
        mean2 = np.mean(data2, axis=0)

        # Calculate correlation between splits for each voxel
        for v in range(n_voxels):
            if v % 1000 == 0:
                print(f"Processing voxel {v}/{n_voxels} for split {i + 1}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    corr, _ = pearsonr(data1[:, v], data2[:, v])
                    reliabilities[i, v] = corr if not np.isnan(corr) else 0
                except:
                    reliabilities[i, v] = 0

    # Average reliability across splits
    mean_reliability = np.mean(reliabilities, axis=0)

    # Plot reliability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mean_reliability, bins=50, alpha=0.7)
    plt.axvline(np.mean(mean_reliability), color='r', linestyle='--',
                label=f'Mean: {np.mean(mean_reliability):.3f}')
    plt.axvline(np.median(mean_reliability), color='g', linestyle='--',
                label=f'Median: {np.median(mean_reliability):.3f}')
    plt.xlabel('Split-Half Reliability (Correlation)')
    plt.ylabel('Number of Voxels')
    plt.title('Distribution of Voxel Reliability')
    plt.legend()
    save_fig('voxel_reliability_distribution.png')

    # Calculate noise ceiling (sqrt of reliability)
    noise_ceiling = np.sqrt(np.abs(mean_reliability))

    # Plot noise ceiling distribution
    plt.figure(figsize=(10, 6))
    plt.hist(noise_ceiling, bins=50, alpha=0.7)
    plt.axvline(np.mean(noise_ceiling), color='r', linestyle='--',
                label=f'Mean: {np.mean(noise_ceiling):.3f}')
    plt.axvline(np.median(noise_ceiling), color='g', linestyle='--',
                label=f'Median: {np.median(noise_ceiling):.3f}')
    plt.xlabel('Noise Ceiling (sqrt of Reliability)')
    plt.ylabel('Number of Voxels')
    plt.title('Distribution of Voxel Noise Ceiling')
    plt.legend()
    save_fig('noise_ceiling_distribution.png')

    print(f"Noise Ceiling Estimation:")
    print(f"  Mean voxel reliability: {np.mean(mean_reliability):.4f}")
    print(f"  Median voxel reliability: {np.median(mean_reliability):.4f}")
    print(f"  Mean noise ceiling: {np.mean(noise_ceiling):.4f}")
    print(f"  Median noise ceiling: {np.median(noise_ceiling):.4f}")
    print(f"  % voxels with reliability > 0.2: {np.mean(mean_reliability > 0.2) * 100:.2f}%")

    return mean_reliability, noise_ceiling


# ========== 4. TEMPORAL STRUCTURE CHECKS ==========

def check_temporal_structure(neural_data, features, n_voxels=100):
    """
    Check temporal structure and autocorrelation in data and features

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    n_voxels : int
        Number of voxels to sample
    """
    n_timepoints = neural_data.shape[0]

    # Sample voxels
    sampled_voxels = np.random.choice(neural_data.shape[1],
                                      min(n_voxels, neural_data.shape[1]),
                                      replace=False)

    # Calculate autocorrelation
    max_lag = min(50, n_timepoints // 4)
    neural_autocorr = np.zeros((max_lag, n_voxels))

    for v, voxel_idx in enumerate(sampled_voxels):
        voxel_data = neural_data[:, voxel_idx]
        voxel_data = (voxel_data - np.mean(voxel_data)) / np.std(voxel_data)

        for lag in range(max_lag):
            neural_autocorr[lag, v] = np.corrcoef(voxel_data[lag:], voxel_data[:-lag if lag > 0 else None])[0, 1]

    # Average across voxels
    mean_neural_autocorr = np.mean(neural_autocorr, axis=1)

    # Calculate feature autocorrelation for a few features
    n_features_to_sample = min(10, features.shape[1])
    sampled_features = np.random.choice(features.shape[1], n_features_to_sample, replace=False)
    feature_autocorr = np.zeros((max_lag, n_features_to_sample))

    for f, feature_idx in enumerate(sampled_features):
        feature_data = features[:, feature_idx]
        feature_data = (feature_data - np.mean(feature_data)) / np.std(feature_data)

        for lag in range(max_lag):
            feature_autocorr[lag, f] = np.corrcoef(feature_data[lag:], feature_data[:-lag if lag > 0 else None])[0, 1]

    # Average across features
    mean_feature_autocorr = np.mean(feature_autocorr, axis=1)

    # Plot autocorrelations
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_lag), mean_neural_autocorr, 'b-', label='Neural Data')
    plt.plot(range(max_lag), mean_feature_autocorr, 'r-', label='Features')
    plt.xlabel('Lag (time points)')
    plt.ylabel('Autocorrelation')
    plt.title('Average Autocorrelation in Neural Data and Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig('autocorrelation.png')

    # Plot time series of a single voxel and feature
    plt.figure(figsize=(14, 8))

    # First 200 timepoints of a single voxel
    display_points = min(200, n_timepoints)
    plt.subplot(2, 1, 1)
    example_voxel = neural_data[:display_points, sampled_voxels[0]]
    example_voxel = (example_voxel - np.mean(example_voxel)) / np.std(example_voxel)
    plt.plot(range(display_points), example_voxel, 'b-')
    plt.plot(range(display_points), gaussian_filter1d(example_voxel, sigma=3), 'r-', alpha=0.7)
    plt.title('Example Voxel Time Series')
    plt.ylabel('Z-scored BOLD')
    plt.grid(True, alpha=0.3)

    # First 200 timepoints of a single feature
    plt.subplot(2, 1, 2)
    example_feature = features[:display_points, sampled_features[0]]
    example_feature = (example_feature - np.mean(example_feature)) / np.std(example_feature)
    plt.plot(range(display_points), example_feature, 'g-')
    plt.plot(range(display_points), gaussian_filter1d(example_feature, sigma=3), 'r-', alpha=0.7)
    plt.title('Example Feature Time Series')
    plt.xlabel('Time Points')
    plt.ylabel('Z-scored Feature Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('timeseries_examples.png')

    print(f"Temporal Structure Analysis:")
    print(f"  First-lag neural autocorrelation: {mean_neural_autocorr[1]:.4f}")
    print(f"  First-lag feature autocorrelation: {mean_feature_autocorr[1]:.4f}")

    return mean_neural_autocorr, mean_feature_autocorr


# ========== 5. STIMULUS ALIGNMENT CHECKS ==========

def stimulus_alignment_check(neural_data, features, events_timing=None, expected_delay=5, max_lag=10, n_voxels=100,
                             n_features=10):
    """
    Check alignment between neural data and stimulus features

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    events_timing : list or numpy.ndarray, optional
        Timing of stimulus events (e.g., word onsets)
    expected_delay : int
        Expected hemodynamic delay in time points (typically 4-6s)
    max_lag : int
        Maximum lag to consider in both directions
    n_voxels : int
        Number of voxels to sample
    n_features : int
        Number of features to sample
    """
    # First, run the time-lagged correlation analysis
    lag_corrs, peak_lag = time_lagged_correlation(neural_data, features, max_lag, n_voxels, n_features)

    # Check if peak lag is within expected range for hemodynamic response
    is_lag_expected = abs(peak_lag - expected_delay) <= 2

    print(f"Stimulus Alignment Check:")
    print(f"  Expected peak lag: ~{expected_delay} TRs")
    print(f"  Actual peak lag: {peak_lag} TRs")
    print(f"  Alignment appears {'GOOD' if is_lag_expected else 'POTENTIALLY PROBLEMATIC'}")

    # If event timings are provided, we can do more detailed checks
    if events_timing is not None and len(events_timing) > 0:
        # Plot neural response locked to events
        plt.figure(figsize=(10, 6))

        # Sample a few voxels for visualization
        sampled_voxels = np.random.choice(neural_data.shape[1], min(5, neural_data.shape[1]), replace=False)

        # Define window around events
        window_before = 3  # TRs before event
        window_after = 12  # TRs after event
        window_size = window_before + window_after + 1

        # Extract time windows around events
        event_responses = []
        for event_time in events_timing:
            event_time = int(event_time)  # Ensure integer
            if event_time - window_before >= 0 and event_time + window_after < neural_data.shape[0]:
                response_window = neural_data[event_time - window_before:event_time + window_after + 1, sampled_voxels]
                event_responses.append(response_window)

        if len(event_responses) > 0:
            # Stack and average responses
            avg_response = np.mean(np.stack(event_responses), axis=0)

            # Plot average response for each sampled voxel
            for i in range(len(sampled_voxels)):
                plt.plot(range(-window_before, window_after + 1),
                         avg_response[:, i],
                         marker='o', markersize=3, alpha=0.7,
                         label=f'Voxel {sampled_voxels[i]}')

            plt.axvline(x=0, color='r', linestyle='--', label='Event Onset')
            plt.axvline(x=expected_delay, color='g', linestyle='--', label=f'Expected Peak ({expected_delay} TRs)')

            plt.xlabel('Time from Event (TRs)')
            plt.ylabel('BOLD Response')
            plt.title('Event-Locked Neural Response')
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_fig('event_locked_response.png')

            # Check for expected hemodynamic response function shape
            # Peak should occur around the expected delay
            peak_times = np.argmax(avg_response, axis=0) - window_before

            print(f"  Average peak time relative to event: {np.mean(peak_times):.2f} TRs")
            print(f"  Percentage of voxels peaking within expected range: "
                  f"{np.mean((peak_times >= expected_delay - 2) & (peak_times <= expected_delay + 2)) * 100:.2f}%")

    return lag_corrs, peak_lag, is_lag_expected


# ========== TIME-LAGGED CORRELATION ANALYSIS ==========

def time_lagged_correlation(neural_data, features, max_lag=10, n_voxels=100, n_features=10):
    """
    Compute correlation at different lags between neural data and features

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    max_lag : int
        Maximum lag to consider in both directions
    n_voxels : int
        Number of voxels to sample
    n_features : int
        Number of features to sample
    """
    n_timepoints = neural_data.shape[0]
    n_total_lags = 2 * max_lag + 1  # include negative lags, zero lag, and positive lags

    # Sample voxels and features
    sampled_voxels = np.random.choice(neural_data.shape[1],
                                      min(n_voxels, neural_data.shape[1]),
                                      replace=False)
    sampled_features = np.random.choice(features.shape[1],
                                        min(n_features, features.shape[1]),
                                        replace=False)

    # Initialize lag correlation matrix
    lag_corrs = np.zeros((n_total_lags, n_voxels, n_features))

    # Compute correlations at different lags
    for v, voxel_idx in enumerate(sampled_voxels):
        if v % 10 == 0:
            print(f"Processing voxel {v + 1}/{n_voxels}")

        voxel_data = neural_data[:, voxel_idx]

        for f, feature_idx in enumerate(sampled_features):
            feature_data = features[:, feature_idx]

            # Calculate correlation at each lag
            for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
                if lag < 0:
                    # Neural data shifted backward (features lead)
                    n1 = voxel_data[-lag:]
                    f1 = feature_data[:lag]
                elif lag > 0:
                    # Neural data shifted forward (neural leads)
                    n1 = voxel_data[:-lag]
                    f1 = feature_data[lag:]
                else:
                    # No lag
                    n1 = voxel_data
                    f1 = feature_data

                # Skip if data after alignment is too short
                if len(n1) < 10:  # minimum length for correlation
                    lag_corrs[lag_idx, v, f] = 0
                    continue

                # Calculate correlation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        corr, _ = pearsonr(n1, f1)
                        lag_corrs[lag_idx, v, f] = corr if not np.isnan(corr) else 0
                    except:
                        lag_corrs[lag_idx, v, f] = 0

    # Average across voxels and features
    mean_lag_corrs = np.mean(np.mean(lag_corrs, axis=1), axis=1)

    # Plot average lag correlation
    plt.figure(figsize=(12, 6))
    plt.plot(range(-max_lag, max_lag + 1), mean_lag_corrs, 'o-', markersize=5)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Lag')

    # Find peak correlation lag
    peak_lag = range(-max_lag, max_lag + 1)[np.argmax(np.abs(mean_lag_corrs))]
    plt.axvline(x=peak_lag, color='g', linestyle='--',
                label=f'Peak Lag: {peak_lag}')

    plt.xlabel('Lag (Neural relative to Features)')
    plt.ylabel('Average Correlation')
    plt.title('Time-Lagged Correlation between Neural Data and Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig('time_lagged_correlation.png')

    print(f"Time-Lagged Correlation Analysis:")
    print(f"  Peak lag: {peak_lag}")
    print(f"  Correlation at peak lag: {mean_lag_corrs[max_lag + peak_lag]:.4f}")
    print(f"  Correlation at zero lag: {mean_lag_corrs[max_lag]:.4f}")

    return mean_lag_corrs, peak_lag


# ========== 6. FULL SANITY CHECK PIPELINE ==========

def run_sanity_checks(neural_data, features, subject_id=None, run_heavy_checks=False, events_timing=None, tr=2.0):
    """
    Run all sanity checks on the data

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    features : numpy.ndarray
        Feature data array of shape (time_points, features)
    subject_id : str
        Subject identifier for saving results
    run_heavy_checks : bool
        Whether to run computationally intensive checks
    events_timing : list or numpy.ndarray, optional
        Timing of stimulus events (in TRs)
    tr : float
        TR (repetition time) in seconds
    """
    print("=" * 50)
    print(f"RUNNING SANITY CHECKS")
    if subject_id:
        print(f"Subject: {subject_id}")
    print(f"Neural data shape: {neural_data.shape}")
    print(f"Feature data shape: {features.shape}")
    print("=" * 50)

    # Create results directory with subject name if provided
    results_dir = f"sanity_check_results_{subject_id}" if subject_id else "sanity_check_results"
    os.makedirs(results_dir, exist_ok=True)

    # Basic checks
    print("\n1. DATA QUALITY CHECKS")
    tsnr = temporal_snr_analysis(neural_data)

    print("\n2. FEATURE SPACE CHECKS")
    feature_std = feature_space_analysis(features)

    print("\n3. TEMPORAL STRUCTURE CHECKS")
    autocorr_neural, autocorr_features = check_temporal_structure(neural_data, features)

    print("\n4. STIMULUS ALIGNMENT CHECKS")
    expected_delay_trs = int(round(5.0 / tr))  # Default 5 second HRF delay converted to TRs
    lag_corrs, peak_lag, is_aligned = stimulus_alignment_check(
        neural_data, features,
        events_timing=events_timing,
        expected_delay=expected_delay_trs)

    # Heavier checks if requested
    if run_heavy_checks:
        print("\n5. NOISE CEILING ESTIMATION")
        reliability, noise_ceiling = noise_ceiling_estimation(neural_data)

        print("\n6. NULL DISTRIBUTION TEST")
        real_corrs, null_corrs = null_distribution_test(neural_data, features, n_permutations=10)

    # Save a summary report
    with open(f"{results_dir}/sanity_check_summary.txt", "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"SANITY CHECK SUMMARY\n")
        if subject_id:
            f.write(f"Subject: {subject_id}\n")
        f.write(f"Neural data shape: {neural_data.shape}\n")
        f.write(f"Feature data shape: {features.shape}\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. DATA QUALITY\n")
        f.write(f"  Median tSNR: {np.nanmedian(tsnr):.2f}\n")
        f.write(f"  % voxels with tSNR > 20: {np.mean(tsnr > 20) * 100:.2f}%\n\n")

        f.write("2. FEATURE SPACE\n")
        f.write(f"  Number of constant features: {np.sum(feature_std == 0)}\n")
        f.write(f"  Median feature std: {np.median(feature_std):.4f}\n\n")

        f.write("3. TEMPORAL STRUCTURE\n")
        f.write(f"  First-lag neural autocorrelation: {autocorr_neural[1]:.4f}\n")
        f.write(f"  First-lag feature autocorrelation: {autocorr_features[1]:.4f}\n\n")

        f.write("4. STIMULUS ALIGNMENT\n")
        f.write(f"  Expected hemodynamic delay: ~{expected_delay_trs} TRs\n")
        f.write(f"  Peak lag: {peak_lag} TRs\n")
        f.write(f"  Alignment assessment: {'GOOD' if is_aligned else 'POTENTIALLY PROBLEMATIC'}\n")
        f.write(f"  Correlation at peak lag: {lag_corrs[max_lag + peak_lag]:.4f}\n")
        f.write(f"  Correlation at zero lag: {lag_corrs[max_lag]:.4f}\n\n")

        if run_heavy_checks:
            f.write("5. NOISE CEILING\n")
            f.write(f"  Mean voxel reliability: {np.mean(reliability):.4f}\n")
            f.write(f"  Mean noise ceiling: {np.mean(noise_ceiling):.4f}\n\n")

            f.write("6. NULL DISTRIBUTION TEST\n")
            f.write(f"  Mean real correlation: {np.mean(real_corrs):.4f}\n")
            f.write(f"  Mean null correlation: {np.mean(null_corrs):.4f}\n")
            f.write(f"  Difference: {np.mean(real_corrs) - np.mean(null_corrs):.4f}\n")

    print("\nSanity checks complete. Results saved to:", results_dir)
    return results_dir

# Example usage:
# run_sanity_checks(zPresp_trimmed, delPstim, subject_id="sub01")