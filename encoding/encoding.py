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
from ridge_utils.ridge import bootstrap_ridge
from config import REPO_DIR, EM_DATA_DIR
from encoding_utils import get_week_lecture
from sklearn.model_selection import KFold

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature", type=str, required=True)
	parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
	parser.add_argument("--trim", type=int, default=5)
	parser.add_argument("--ndelays", type=int, default=4)
	parser.add_argument("--nboots", type=int, default=50)
	parser.add_argument("--chunklen", type=int, default=40)
	parser.add_argument("--nchunks", type=int, default=50)
	parser.add_argument("--singcutoff", type=float, default=1e-10)
	parser.add_argument("-use_corr", action="store_true")
	parser.add_argument("-single_alpha", action="store_true")
	logging.basicConfig(level=logging.INFO)


	args = parser.parse_args()
	globals().update(args.__dict__)

	fs = " ".join(_FEATURE_CONFIG.keys())
	assert feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert np.amax(sessions) <= 6 and np.amin(sessions) >=1, "1 <= session <= 5"

	sessions = list(map(str, sessions))
	with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
		sess_to_story = json.load(f)
	train_stories, test_stories = [], []
	dir_path = "/sci/labs/arielgoldstein/miriam1234/6motion_students"
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		cstories = cut_stories(stories, subject)
		ctstory = cut_stories([tstory], subject)[0] if cut_stories([tstory], subject) and \
													   cut_stories([tstory], subject)[0] is not None else None
		train_stories.extend(cstories)
		if ctstory is not None and ctstory not in test_stories:
			test_stories.append(ctstory)
	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
	allstories = list(set(train_stories) | set(test_stories))
	save_location = join(REPO_DIR, "results", feature, subject)
	print("Saving encoding model & results to:", save_location)
	os.makedirs(save_location, exist_ok=True)

	downsampled_feat = get_feature_space(feature, allstories)
	print("Stimulus & Response parameters:")
	print("trim: %d, ndelays: %d" % (trim, ndelays))

	# Delayed stimulus
	delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, trim, ndelays)
	print("delRstim: ", delRstim.shape)
	delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat, trim, ndelays)
	print("delPstim: ", delPstim.shape)

	# Response
	zRresp = get_response(train_stories, subject)
	print("zRresp: ", zRresp.shape)
	zPresp = get_response(test_stories, subject)
	print("zPresp: ", zPresp.shape)


	# Filter constant voxels
	print("Filtering constant voxels...")
	voxel_std = np.std(zRresp, axis=0)
	non_constant_voxels = voxel_std > 1e-10
	zRresp_trimmed = zRresp[:, non_constant_voxels]
	zPresp_trimmed = zPresp[:, non_constant_voxels]

	print(f"Original number of voxels: {len(voxel_std)}")
	print(f"Number of non-constant voxels: {np.sum(non_constant_voxels)}")
	print(f"Filtered data shapes - zRresp_trimmed: {zRresp_trimmed.shape}, zPresp_trimmed: {zPresp_trimmed.shape}")

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print("Ridge parameters:")
	print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
		nboots, chunklen, nchunks, single_alpha, use_corr))

	if zRresp_trimmed.size == 0 or zPresp_trimmed.size == 0:
		print("Error: One of the response arrays is empty.")
		sys.exit(1)

	#wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
	#	delRstim, zRresp_trimmed, delPstim, zPresp_trimmed, alphas, nboots, chunklen,
	#	nchunks, singcutoff=singcutoff, single_alpha=single_alpha,
	#	use_corr=use_corr)

	# Save regression results.
	#np.savez("%s/weights" % save_location, wt)
	#np.savez("%s/corrs" % save_location, corrs)
	#np.savez("%s/valphas" % save_location, valphas)
	#np.savez("%s/bscorrs" % save_location, bscorrs)
	#np.savez("%s/valinds" % save_location, np.array(valinds))
	#print("Total r2: %d" % sum(corrs * np.abs(corrs)))

	from significance_testing import model_pvalue, permutation_test, mrsq, fdr_correct, fdr_qvalues


	# Load your saved weights and correlations
	weights = np.load(os.path.join(save_location, "weights.npz"))['arr_0']
	corrs = np.load(os.path.join(save_location, "corrs.npz"))['arr_0']

	# Load or reference your test stimuli and responses from the encoding script
	# Assuming delPstim and zPresp_trimmed are available from your encoding script
	test_stim = delPstim
	test_resp = zPresp_trimmed

	print(f"Testing {test_resp.shape[1]} voxels")

	# Get predicted responses
	predicted_resp = np.dot(test_stim, weights)
	print(f"Predicted response shape: {predicted_resp.shape}")

	# Fix the blocklen to be divisible by the number of time points
	n_timepoints = test_resp.shape[0]  # 1057 in your case
	print(f"Number of time points: {n_timepoints}")


	# Choose a blocklen that divides evenly into n_timepoints
	# Find the largest divisor of n_timepoints that is less than or equal to your desired blocklen
	def find_largest_divisor(n, max_val=40):
		for i in range(max_val, 0, -1):
			if n % i == 0:
				return i
		return 1


	blocklen = find_largest_divisor(n_timepoints, 40)
	print(f"Using blocklen: {blocklen}")

	# Now run the permutation test with the adjusted blocklen
	pvals, perm_rsqs, real_rsqs = permutation_test(
		test_resp, predicted_resp,
		metric=lambda a, b: mrsq(a, b, corr_units=True),
		blocklen=blocklen, nperms=1000)

	# Correct for multiple comparisons
	pID, pN = fdr_correct(pvals, thres=0.05)
	print(f"FDR-corrected p-value threshold: {pID}")

	# Find significant voxels
	significant_voxels = np.where(pvals < pID)[0]
	print(f"Number of significant voxels: {len(significant_voxels)}")
	print(f"Percentage of significant voxels: {len(significant_voxels) / len(pvals) * 100:.2f}%")

	# Option 2: Apply model_pvalue to each voxel (more computationally intensive)
	# This calculates a separate bootstrap for each voxel
	"""
	bootstrap_pvals = []
	parametric_pvals = []

	# For computational efficiency, you may want to sample a subset of voxels
	n_voxels = test_resp.shape[1]
	for voxel_idx in range(n_voxels):
	    if voxel_idx % 100 == 0:
	        print(f"Testing voxel {voxel_idx}/{n_voxels}")

	    voxel_weights = weights[:, voxel_idx]
	    voxel_resp = test_resp[:, voxel_idx]

	    bs_pval, p_pval = model_pvalue(voxel_weights, test_stim, voxel_resp, nboot=1000)
	    bootstrap_pvals.append(bs_pval)
	    parametric_pvals.append(p_pval)

	bootstrap_pvals = np.array(bootstrap_pvals)
	parametric_pvals = np.array(parametric_pvals)
	"""

	# Calculate q-values (alternative to FDR correction)
	qvals = fdr_qvalues(pvals)
	significant_qvals = np.where(qvals < 0.05)[0]
	print(f"Number of significant voxels (q-values): {len(significant_qvals)}")

	# Save results
	np.savez(os.path.join(save_location, "pvals.npz"), pvals)
	np.savez(os.path.join(save_location, "qvals.npz"), qvals)
	np.savez(os.path.join(save_location, "significant_voxels.npz"), significant_voxels)

	# Optionally, create a summary dictionary and save as JSON
	import json

	summary = {
		"total_voxels": len(pvals),
		"significant_voxels": len(significant_voxels),
		"percentage_significant": len(significant_voxels) / len(pvals) * 100,
		"fdr_threshold": float(pID),
		"mean_correlation": float(np.mean(corrs)),
		"max_correlation": float(np.max(corrs))
	}

	with open(os.path.join(save_location, "statistical_summary.json"), "w") as f:
		json.dump(summary, f, indent=4)

	print("Statistical testing complete. Results saved to:", save_location)

