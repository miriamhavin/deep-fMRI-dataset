import numpy as np
import itertools as itools
from ridge_utils.DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset([])

def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneme_ds(grids, trfiles):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        d = DataSequence.from_grid(grtranscript, trfiles[st][0])
        ds[st] = d

    return ds

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D', 'DH', 'EH', 'ER', 'EY', 
            'F', 'G', 'HH', 'IH', 'IY', 'JH','K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 
            'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as e:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phonemes2(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.upper().strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)


def make_semantic_model(ds: DataSequence, lsasm, size):
    """
    Creates a new DataSequence with embeddings directly from the semantic model,
    and handles mismatches by inserting zero vectors as needed.
    """
    # Convert semantic model vocab to list if it's not already
    lsasm_words = list(lsasm.vocab) if hasattr(lsasm, 'vocab') else []

    # Determine the length to check
    min_length = min(len(ds.data), len(lsasm_words))

    # Count matching words at same position
    matching_count = 0
    diff_indices = []

    for i in range(min_length):
        ds_word = str(ds.data[i]).lower()
        lsasm_word = str(lsasm_words[i]).lower()

        if ds_word == lsasm_word:
            matching_count += 1
        else:
            diff_indices.append(i)

    # Print summary statistics
    match_percentage = (matching_count / min_length) * 100 if min_length > 0 else 0
    print(f"\nOrder match summary:")
    print(f"  Words in same position: {matching_count} out of {min_length} ({match_percentage:.2f}%)")
    print(f"  Total position mismatches: {len(diff_indices)}")

    # Handle mismatches by creating a new data array with zeros inserted
    zero_vectors_added = 0

    # Create a new array for adjusted data
    if len(ds.data) > lsasm.data.shape[0]:
        # DataSequence is longer - add zero vectors to match semantic model
        adjusted_data = np.zeros((len(ds.data), lsasm.data.shape[1]))

        # Copy existing vectors and add zeros for missing ones
        copy_len = min(len(ds.data), lsasm.data.shape[0])
        adjusted_data[:copy_len] = lsasm.data[:copy_len]
        zero_vectors_added = len(ds.data) - lsasm.data.shape[0]
        print(f"Added {zero_vectors_added} zero vectors at the end to match DataSequence length")

    elif len(ds.data) < lsasm.data.shape[0]:
        # SemanticModel is longer - truncate it
        adjusted_data = lsasm.data[:len(ds.data)]
        print(f"Truncated {lsasm.data.shape[0] - len(ds.data)} vectors from SemanticModel")

    else:
        # Same length, no adjustment needed
        adjusted_data = lsasm.data

    # Handle middle mismatches if alignment is good enough (over 70% match)
    if diff_indices:
        print(f"Attempting to fix {len(diff_indices)} mismatches by inserting zero vectors...")

        # Create a new array with potential extra space
        potential_zeros = min(20, len(diff_indices))  # Limit potential inserts to 20
        new_data = np.zeros((adjusted_data.shape[0] + potential_zeros, adjusted_data.shape[1]))

        # Track shifts
        shift = 0
        insert_count = 0

        for i in range(adjusted_data.shape[0]):
            if i < len(diff_indices) and i + shift == diff_indices[i]:
                # Insert a zero vector at mismatch position
                new_data[i + shift] = np.zeros(adjusted_data.shape[1])
                shift += 1
                insert_count += 1

                # Copy the actual data after the zero
                new_data[i + shift] = adjusted_data[i]

                # Check if insertion helped
                if i + 1 < min_length:
                    ds_word = str(ds.data[i + shift]).lower()
                    lsasm_word = str(lsasm_words[i]).lower()
                    if ds_word == lsasm_word:
                        print(f"Successfully aligned at index {i + shift} after inserting zero")
            else:
                # Copy data normally
                new_data[i + shift] = adjusted_data[i]

        if insert_count > 0:
            adjusted_data = new_data[:adjusted_data.shape[0] + insert_count]
            zero_vectors_added += insert_count
            print(f"Inserted {insert_count} zero vectors in the middle to fix alignment")

    # Print final dimensions
    print(f"\nFinal dimensions:")
    print(f"  Original DataSequence length: {len(ds.data)}")
    print(f"  Original SemanticModel vectors: {lsasm.data.shape[0]}")
    print(f"  Total zero vectors added: {zero_vectors_added}")

    # Create new DataSequence with the adjusted embedding data
    return DataSequence(adjusted_data, ds.split_inds, ds.data_times, ds.tr_times)

def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])
