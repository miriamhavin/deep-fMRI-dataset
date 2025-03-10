import numpy as np
import time
import pathlib
import os
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname
import nibabel as nib
from ridge_utils.npp import zscore, mcorr
from ridge_utils.utils import make_delayed
from config import DATA_DIR
import re

def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

	Args:
		stories: List of stimuli stories.

	Variables:
		downsampled_feat (dict): Downsampled feature vectors for all stories.
		trim: Trim downsampled stimulus matrix.
		delays: List of delays for Finite Impulse Response (FIR) model.

	Returns:
		delstim: <float32>[TRs, features * ndelays]
	"""
	stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
	stim = np.vstack(stim)
	delays = range(1, ndelays+1)
	delstim = make_delayed(stim, delays)
	return delstim

def get_week_lecture(text):
    matches = re.findall(r'\d+', text)
    return matches[0], matches[1] if len(matches) > 1 else None

def cut_stories(stories, subject):
	cstories = []
	dir_path = "/sci/labs/arielgoldstein/miriam1234/6motion_students"
	for story in stories:
		week_num, lecture_num = get_week_lecture(story)
		resp_path = os.path.join(dir_path, f"s{subject}_wk{week_num}_vid{lecture_num}_6motion_mni.nii.gz")
		# Check if the file exists before attempting to load
		if not os.path.exists(resp_path):
			print(f"Warning: File not found subject {subject} week {week_num} lecture {lecture_num}")
			continue  # Skip to the next iteration if the file does not exist
		cstories.append(story)
	return cstories


def get_response(stories, subject):
   """Get the subject's fMRI response for stories, skipping files that do not exist."""
   dir_path = "/sci/labs/arielgoldstein/miriam1234/6motion_students"
   responses = []
   for story in stories:
      week_num, lecture_num = get_week_lecture(story)
      resp_path = os.path.join(dir_path, f"s{subject}_wk{week_num}_vid{lecture_num}_6motion_mni.nii.gz")
      img = nib.load(resp_path)
      data = img.get_fdata()
      flat_data = flatten_data(data)
      trimmed_data = flat_data[10:-10, :]
      responses.extend(trimmed_data)

   stacked_data = np.vstack(responses)
   means = np.mean(stacked_data, axis=0)
   stds = np.std(stacked_data, axis=0, ddof=1)

   # Avoid division by zero
   stds[stds == 0] = 1.0

   # Z-score calculation
   z_scored_data = (stacked_data - means) / stds

   return z_scored_data


def flatten_data(data):
	"""
    Reshape 4D fMRI data to 2D (time points × voxels) format for each session separately.

    Parameters:
    -----------
    data : list of numpy.ndarray
        List of 4D fMRI data arrays, each with shape (x, y, z, time)

    Returns:
    --------
    list of numpy.ndarray
        List of 2D arrays, each with shape (time_points, voxels)
    """
	session_data = np.transpose(data, (3, 0, 1, 2))
	n_timepoints = session_data.shape[0]
	session_data_2d = session_data.reshape(n_timepoints, -1)
	return session_data_2d

def get_permuted_corrs(true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return corrs

def permutation_test(true, pred, blocklen, nperms):
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(
		lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(zPresp, pred, blocklen, nperms, mode='', thres=0.001):
	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return
