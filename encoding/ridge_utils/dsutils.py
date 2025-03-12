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
    and performs detailed comparison of word differences.
    """
    # Print basic dimensions
    print(f"DataSequence length: {len(ds.data)}")
    print(f"Semantic model vectors: {lsasm.data.shape[0]}")
    print(f"Vector dimension: {lsasm.data.shape[1]}")

    # Check if dimensions align
    if len(ds.data) != lsasm.data.shape[0]:
        print(
            f"WARNING: Mismatch between DataSequence length ({len(ds.data)}) and semantic model vectors ({lsasm.data.shape[0]})")

        # Compare the actual word lists to find differences
        print("\nComparing word sequences...")

        # Convert semantic model vocab to list if it's not already
        lsasm_words = list(lsasm.vocab) if hasattr(lsasm, 'vocab') else []

        # Create sets of words for comparison
        ds_words_set = set(str(w).lower() for w in ds.data)
        lsasm_words_set = set(str(w).lower() for w in lsasm_words)

        # Find words that are only in one set
        only_in_ds = ds_words_set - lsasm_words_set
        only_in_lsasm = lsasm_words_set - ds_words_set

        # Print summary statistics
        print(f"\nWords only in DataSequence: {len(only_in_ds)}")
        print(f"Words only in SemanticModel: {len(only_in_lsasm)}")

        # Show some examples from each
        if only_in_ds:
            print("\nSample words only in DataSequence (max 10):")
            for w in list(only_in_ds)[:10]:
                print(f"  '{w}'")

        if only_in_lsasm:
            print("\nSample words only in SemanticModel (max 10):")
            for w in list(only_in_lsasm)[:10]:
                print(f"  '{w}'")

        # Determine the minimum length for sequence comparison
        min_length = min(len(ds.data), len(lsasm_words))

        # Print the first few position-specific differences
        diff_count = 0
        print("\nPosition-specific differences (same index, different word):")
        for i in range(min_length):
            if str(ds.data[i]).lower() != str(lsasm_words[i]).lower():
                print(f"  Index {i}: DataSequence='{ds.data[i]}', SemanticModel='{lsasm_words[i]}'")
                diff_count += 1
                if diff_count >= 10:  # Limit to first 10 differences
                    print("  (Showing only first 10 differences)")
                    break

        if diff_count == 0:
            print("  No position-specific differences found in the overlapping portion")

        # Check for length differences
        if len(ds.data) > len(lsasm_words):
            print(f"\nExtra positions in DataSequence (showing first 5):")
            for i in range(len(lsasm_words), min(len(ds.data), len(lsasm_words) + 5)):
                print(f"  Index {i}: '{ds.data[i]}'")
        elif len(lsasm_words) > len(ds.data):
            print(f"\nExtra positions in SemanticModel (showing first 5):")
            for i in range(len(ds.data), min(len(lsasm_words), len(ds.data) + 5)):
                print(f"  Index {i}: '{lsasm_words[i]}'")

        # Create adjusted data array
        min_length = min(len(ds.data), lsasm.data.shape[0])
        adjusted_data = lsasm.data[:min_length]
        print(f"\nTruncating to {min_length} vectors to match")
    else:
        adjusted_data = lsasm.data

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
