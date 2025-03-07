import numpy as np
import itertools as itools
from ridge_utils.DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

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


def make_semantic_model(ds: DataSequence, lsasms, sizes):
    """
    ds
        datasequence to operate on
    lsasms
        list of semantic models to use
    sizes
        list of sizes of resulting vectors from each semantic model
    """
    newdata = []
    num_lsasms = len(lsasms)

    # Print some debug info
    for i in range(num_lsasms):
        lsasm = lsasms[i]
        print(f"Model {i} has data shape: {lsasm.data.shape}")
        print(f"Model {i} has {len(lsasm.vocab)} words in vocabulary")
        print(f"First 5 words in vocab: {lsasm.vocab[:5]}")

        # Check a few sample words to see if they're in the vocabulary
        sample_words = ds.data[:5]  # First 5 words from the dataset
        for word in sample_words:
            encoded_word = str.encode(word.lower())
            if encoded_word in lsasm.vindex:
                print(f"Word '{word}' is in vocabulary at index {lsasm.vindex[encoded_word]}")
            else:
                print(f"Word '{word}' is NOT in vocabulary")

    # Now try to get vectors
    for w in ds.data:
        v = []
        for i in range(num_lsasms):
            lsasm = lsasms[i]
            size = sizes[i]

            try:
                # Try multiple encoding formats
                word_to_lookup = w.lower()
                encoded_word = str.encode(word_to_lookup)

                # First try direct lookup
                if encoded_word in lsasm.vindex:
                    word_idx = lsasm.vindex[encoded_word]
                    vector = lsasm.data[:, word_idx]
                    v = np.concatenate((v, vector))
                else:
                    # Try np.str_ format
                    np_str_word = np.str_(word_to_lookup)
                    if np_str_word in lsasm.vindex:
                        word_idx = lsasm.vindex[np_str_word]
                        vector = lsasm.data[:, word_idx]
                        v = np.concatenate((v, vector))
                    else:
                        # Word not found, use zeros
                        v = np.concatenate((v, np.zeros(size)))
            except (KeyError, IndexError, ValueError) as e:
                # Log the error
                print(f"Error accessing word '{w}': {str(e)}")
                v = np.concatenate((v, np.zeros(size)))

        newdata.append(v)

    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

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
