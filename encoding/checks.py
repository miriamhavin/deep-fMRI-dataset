import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import glob
from scipy.stats import zscore, sem
import seaborn as sns

# Add your project directory to the path
sys.path.append('/sci/labs/arielgoldstein/miriam1234/deep-fMRI-dataset/encoding')

# Import your utility functions
from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from config import REPO_DIR, EM_DATA_DIR


def load_timing_data_from_folder(timing_folder, sessions, tr=2.0):
    """
    Load timing data from multiple CSV files in a folder.
    Each CSV corresponds to a session and contains word, start_time, end_time columns.
    """
    all_timing_data = []

    for session in sessions:
        # Look for CSV files matching this session
        pattern = os.path.join(timing_folder, f"*{session}*.csv")
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Warning: No timing file found for session {session}")
            continue

        # Use the first matching file
        session_file = matching_files[0]
        print(f"Loading timing data for session {session} from {os.path.basename(session_file)}")

        try:
            # Load the CSV file without headers
            session_df = pd.read_csv(session_file, header=None)
            print(f"Loaded without headers. {len(session_df.columns)} columns found.")

            # Simply assign the first column as 'word' and second as 'start_time'
            session_df['word'] = session_df.iloc[:, 0]
            session_df['start_time'] = pd.to_numeric(session_df.iloc[:, 1], errors='coerce')

            # If there's a third column, assign it as 'end_time'
            if session_df.shape[1] >= 3:
                session_df['end_time'] = pd.to_numeric(session_df.iloc[:, 2], errors='coerce')

            # Add session info
            session_df['session'] = session

            # Convert times to TR if needed
            if 'start_time' in session_df.columns:
                session_df['start_time_tr'] = (session_df['start_time'] / tr).round().astype(int)

            all_timing_data.append(session_df)

        except Exception as e:
            print(f"Error loading {session_file}: {e}")

    # Combine all data
    if all_timing_data:
        combined_df = pd.concat(all_timing_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} words from {len(all_timing_data)} sessions")
        return combined_df
    else:
        print("No timing data loaded!")
        return pd.DataFrame()


def visualize_extended_time_window(neural_data, word_timings_df, output_dir, tr=2.0, window_before=60, window_after=60):
    """
    Visualize neural responses time-locked to word presentations,
    averaging across all voxels, with extended time window (1 minute before and after).

    Parameters:
    -----------
    neural_data : numpy.ndarray
        Neural data array of shape (time_points, voxels)
    word_timings_df : pandas.DataFrame
        DataFrame containing word timing information
    output_dir : str
        Directory to save output plots
    tr : float
        TR (repetition time) in seconds
    window_before : int
        Number of TRs before event onset to include
    window_after : int
        Number of TRs after event onset to include
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get word onset times in TR units
    if 'start_time_tr' in word_timings_df.columns:
        word_times_tr = word_timings_df['start_time_tr'].values
    elif 'start_time' in word_timings_df.columns:
        # Convert to TR units
        word_times_tr = (word_timings_df['start_time'] / tr).round().astype(int)
    else:
        print("ERROR: No timing information available after loading CSVs")
        return

    # Calculate TRs needed for 1 minute before and after
    trs_1min = int(60 / tr)
    window_before = min(trs_1min, window_before)  # Ensure we don't exceed 1 minute
    window_after = min(trs_1min, window_after)  # Ensure we don't exceed 1 minute

    print(f"Using time window of {window_before * tr} seconds before to {window_after * tr} seconds after word onset")

    # Filter to keep only events with enough data before and after
    valid_words = (word_times_tr >= window_before) & (word_times_tr < neural_data.shape[0] - window_after)
    word_times_valid = word_times_tr[valid_words]
    valid_words_df = word_timings_df.iloc[np.where(valid_words)[0]]

    print(f"Analyzing responses to {len(word_times_valid)} words (filtered to ensure sufficient data)")

    # Average response across all voxels
    # First, normalize each voxel time series
    #normalized_data = zscore(neural_data)

    # Extract time windows around words for all voxels
    window_size = window_before + window_after + 1
    n_voxels = neural_data.shape[1]

    # Initialize array to store responses
    print(f"Extracting extended time windows ({window_size} TRs) for each word...")
    all_voxel_responses = np.zeros((len(word_times_valid), window_size, n_voxels))

    # Extract neural response for each word
    for i, word_time in enumerate(word_times_valid):
        if i % 100 == 0:
            print(f"Processing word {i}/{len(word_times_valid)}")

        window_start = word_time - window_before
        window_end = word_time + window_after + 1

        # Get neural data for this time window across all voxels
        all_voxel_responses[i, :, :] = neural_data[window_start:window_end, :]

    # Average across all voxels for each word and time point
    avg_response_per_word = np.mean(all_voxel_responses, axis=2)

    # Then average across all words
    avg_response = np.mean(avg_response_per_word, axis=0)
    sem_response = sem(avg_response_per_word, axis=0)

    # Plot average response with error bands
    plt.figure(figsize=(14, 8))
    time_axis = np.arange(-window_before, window_after + 1) * tr  # Convert to seconds

    plt.plot(time_axis, avg_response, 'b-', linewidth=2, label='Mean BOLD Response')
    plt.fill_between(time_axis,
                     avg_response - sem_response,
                     avg_response + sem_response,
                     alpha=0.3, color='blue', label='SEM')

    # Add vertical lines for important time points
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Word Onset')
    plt.axvline(x=5, color='g', linestyle='--', linewidth=2, label='Expected HRF Peak (5s)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add text labels for time regions
    plt.text(-50, max(avg_response) * 0.9, "Pre-stimulus", fontsize=12, ha='center')
    plt.text(50, max(avg_response) * 0.9, "Post-stimulus", fontsize=12, ha='center')

    # Improve plot aesthetics
    plt.xlabel('Time from Word Onset (seconds)', fontsize=14)
    plt.ylabel('Average BOLD Response (z-scored)', fontsize=14)
    plt.title('Extended Time Window: Neural Response to Words (All Voxels)', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add minor grid lines
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='x', alpha=0.2)

    # Add x-ticks every 10 seconds for readability
    plt.xticks(np.arange(-60, 70, 10))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extended_time_window_response.png'), dpi=300)
    print(f"Saved extended time window plot to {output_dir}/extended_time_window_response.png")

    # Smooth the response for trend analysis
    from scipy.ndimage import gaussian_filter1d
    smoothed_response = gaussian_filter1d(avg_response, sigma=2)

    # Plot smoothed response
    plt.figure(figsize=(14, 8))
    plt.plot(time_axis, avg_response, 'b-', linewidth=1, alpha=0.5, label='Raw Response')
    plt.plot(time_axis, smoothed_response, 'r-', linewidth=2, label='Smoothed Response')

    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Word Onset')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.xlabel('Time from Word Onset (seconds)', fontsize=14)
    plt.ylabel('Average BOLD Response (z-scored)', fontsize=14)
    plt.title('Extended Time Window: Smoothed Neural Response', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(-60, 70, 10))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extended_time_window_smoothed.png'), dpi=300)
    print(f"Saved smoothed time window plot to {output_dir}/extended_time_window_smoothed.png")

    # Create a spectrogram to look for periodicities
    from scipy import signal

    plt.figure(figsize=(14, 8))

    # Calculate power spectral density
    freqs, psd = signal.welch(avg_response, fs=1 / tr, nperseg=min(64, len(avg_response)))

    # Convert to period in seconds
    periods = 1 / freqs[1:]  # Skip the DC component (0 Hz)
    power = psd[1:]

    plt.plot(periods, power, 'b-', linewidth=2)
    plt.xlabel('Period (seconds)', fontsize=14)
    plt.ylabel('Power Spectral Density', fontsize=14)
    plt.title('Frequency Analysis of Neural Response', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)  # Focus on periods up to 60 seconds

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_analysis.png'), dpi=300)
    print(f"Saved frequency analysis to {output_dir}/frequency_analysis.png")

    # Analyze slow drift by fitting a line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(time_axis, avg_response)

    plt.figure(figsize=(14, 8))
    plt.plot(time_axis, avg_response, 'b-', linewidth=2, label='Neural Response')
    plt.plot(time_axis, intercept + slope * time_axis, 'r--', linewidth=2,
             label=f'Linear Trend (slope={slope:.2e}, p={p_value:.4f})')

    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Word Onset')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.xlabel('Time from Word Onset (seconds)', fontsize=14)
    plt.ylabel('Average BOLD Response (z-scored)', fontsize=14)
    plt.title('Linear Trend Analysis of Extended Time Window', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(-60, 70, 10))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_trend_analysis.png'), dpi=300)
    print(f"Saved linear trend analysis to {output_dir}/linear_trend_analysis.png")

    # Save data for future analysis
    np.savez(os.path.join(output_dir, 'extended_time_window_data.npz'),
             time_axis=time_axis,
             avg_response=avg_response,
             sem_response=sem_response,
             smoothed_response=smoothed_response)

    return avg_response, time_axis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize extended time window neural response')
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--feature", type=str, required=True, help="Feature space to use")
    parser.add_argument("--sessions", nargs='+', type=int, default=[1], help="Session numbers to include")
    parser.add_argument("--tr", type=float, default=2.0, help="TR (repetition time) in seconds")
    parser.add_argument("--timing_folder", type=str, required=True, help="Path to folder containing timing CSV files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = f"extended_analysis_{args.subject}"
    else:
        output_dir = args.output_dir

    # Convert sessions to strings for loading stories
    sessions_str = list(map(str, args.sessions))

    # Load session to story mapping
    with open(os.path.join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    # Get test stories
    train_stories, test_stories = [], []
    dir_path = "/sci/labs/arielgoldstein/miriam1234/6motion_students"
    for sess in sessions_str:
        stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
        cstories = cut_stories(stories, args.subject)
        ctstory = cut_stories([tstory], args.subject)[0] if cut_stories([tstory], args.subject) and \
                                                       cut_stories([tstory], args.subject)[0] is not None else None
        train_stories.extend(cstories)
        if ctstory is not None and ctstory not in test_stories:
            test_stories.append(ctstory)
    assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
    allstories = list(set(train_stories) | set(test_stories))

    print(f"Test stories: {test_stories}")

    # Get neural responses
    print(f"Loading neural responses for {args.subject}...")
    neural_data = get_response(test_stories, args.subject)

    # Z-score the data
    print("Z-scoring data...")
    z_neural_data = zscore(neural_data)

    # Load word timing data from folder
    timing_df = load_timing_data_from_folder(args.timing_folder, args.sessions, args.tr)

    if timing_df.empty:
        print("Error: No valid timing data found. Cannot visualize word responses.")
        sys.exit(1)

    # Calculate TRs for 1 minute
    trs_1min = int(60 / args.tr)
    print(f"1 minute = {trs_1min} TRs at TR={args.tr}s")

    # Visualize extended time window
    visualize_extended_time_window(
        z_neural_data,
        timing_df,
        output_dir,
        tr=args.tr,
        window_before=trs_1min,  # 1 minute before
        window_after=trs_1min  # 1 minute after
    )

    print(f"Extended time window analysis complete. Results saved to {output_dir}/")