import numpy as np
import itertools as itools
from ridge_utils.DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset(["[noise]", "<unk>"])

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


def make_contextual_vector_model(ds: DataSequence, lsasms: list, sizes: list):
    """
    Creates contextual vectors by maintaining word order and processing them sequentially

    Parameters:
    -----------
    ds : DataSequence
        datasequence to operate on
    lsasms : list
        semantic models to use
    sizes : list
        sizes of resulting vectors from each semantic model

    Returns:
    --------
    DataSequence
        A new DataSequence with contextual vectors
    """
    print("Starting make_contextual_vector_model")
    print(f"Number of words in DataSequence: {len(ds.data)}")
    print(f"Number of semantic models: {len(lsasms)}")
    print(f"Sizes for each model: {sizes}")
    print(f"Total vector size: {sum(sizes)}")

    # Debug: Check what's in the first semantic model
    print("Checking semantic model content:")
    if lsasms and len(lsasms) > 0:
        first_model = lsasms[0]
        if hasattr(first_model, 'keys'):
            # Try to get a few keys to see what's in there
            try:
                keys = list(first_model.keys())[:5]
            except Exception as e:
                print(f"Error getting keys: {e}")
        else:
            print("Model doesn't have keys method")
            print(f"Model type: {type(first_model)}")

    newdata = []
    num_lsasms = len(lsasms)

    # Verify that we have the same number of models as sizes
    if num_lsasms != len(sizes):
        print("[ERROR] Number of models does not match number of sizes!")
        raise ValueError("Number of semantic models must match number of sizes")

    # Process words in order
    missing_words_count = 0
    for i in range(len(ds.data)):
        word = ds.data[i]
        context_vector = np.zeros(sum(sizes))  # Initialize with zeros for the total size
        current_pos = 0
        for j in range(num_lsasms):
            lsasm = lsasms[j]
            size = sizes[j]

            try:
                # Try different word encoding formats to find a match
                word_lower = word.lower()

                # Debug additional word format attempts
                if i % 100 == 0:
                    formats_to_try = [
                        word_lower,  # plain lowercase
                        str.encode(word_lower),  # encoded bytes
                        word_lower.encode('utf-8'),  # explicitly utf-8 encoded
                        word,  # original case
                        str.encode(word)  # original case encoded
                    ]

                # Try different formats
                vector_found = False
                for word_format in [word_lower, str.encode(word_lower), word_lower.encode('utf-8'), word,
                                    str.encode(word)]:
                    try:
                        word_vector = lsasm[word_format]
                        vector_found = True
                        # Place it at the correct position in the context vector
                        context_vector[current_pos:current_pos + size] = word_vector
                        if i % 100 == 0:
                            break
                    except (KeyError, TypeError):
                        continue

                # If no format worked
                if not vector_found:
                    raise KeyError(f"Word not found in any format: {word}")

            except KeyError:
                # If word not found, leave zeros in that position
                missing_words_count += 1
            current_pos += size

        newdata.append(context_vector)

    print(f"Processing complete. Total words: {len(ds.data)}")
    print(f"Words missing from at least one model: {missing_words_count}")
    print(f"Final data shape: {np.array(newdata).shape}")

    # Create new DataSequence with the contextual vectors
    result = DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

    print("Created new DataSequence")
    print(f"New DataSequence shape: {result.data.shape}")
    print(f"Split indices: {result.split_inds}")
    print(f"Data times length: {len(result.data_times) if result.data_times is not None else 'None'}")
    print(f"TR times length: {len(result.tr_times) if result.tr_times is not None else 'None'}")

    return result

def make_semantic_model(ds: DataSequence, lsasms, sizes):
    """
    ds
        datasequence to operate on
    lsasms
        list of semantic models to use
    sizes
        list of sizes of resulting vectors from each semantic model
    """
    # Validate inputs
    assert len(lsasms) == len(sizes), "Number of semantic models must match number of sizes"

    newdata = []
    num_lsasms = len(lsasms)

    # Check expected dimensions for each semantic model
    for j, lsasm in enumerate(lsasms):
        expected_size = sizes[j]
        actual_size = lsasm.data.shape[1] if hasattr(lsasm, 'data') and hasattr(lsasm.data, 'shape') else None
        if actual_size and actual_size != expected_size:
            print(f"Warning: Model {j} has dimension {actual_size} but expected {expected_size}")

    for i in range(len(ds.data)):
        # Initialize v as numpy array with zeros
        v = np.array([], dtype=float)

        for j in range(num_lsasms):
            lsasm = lsasms[j]
            size = sizes[j]

            try:
                # Verify we're not exceeding bounds
                if hasattr(lsasm, 'data') and i < lsasm.data.shape[0]:
                    vector = lsasm.data[i]
                    # Verify vector dimension matches expected size
                    if len(vector) != size:
                        print(f"Warning: Vector from model {j} has {len(vector)} dimensions, expected {size}")
                        # Resize if needed (pad or truncate)
                        if len(vector) < size:
                            vector = np.pad(vector, (0, size - len(vector)))
                        else:
                            vector = vector[:size]
                else:
                    vector = np.zeros(size)
            except (IndexError, ValueError, AttributeError) as e:
                print(f"Error accessing model {j} at index {i}: {e}")
                vector = np.zeros(size)

            # Concatenate to v
            v = np.concatenate((v, vector))

        newdata.append(v)

    # Verify final dimensions
    result = np.array(newdata)
    expected_feature_dim = sum(sizes)
    if result.shape[1] != expected_feature_dim:
        print(f"Warning: Final vectors have {result.shape[1]} dimensions, expected {expected_feature_dim}")

    return DataSequence(result, ds.split_inds, ds.data_times, ds.tr_times)


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
