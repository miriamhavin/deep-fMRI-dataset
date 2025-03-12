import h5py
import numpy as np
import os
import argparse
import json
from pprint import pprint


def inspect_hdf5(filename, args=None):
    """
    Thoroughly inspects an HDF5 file, examining structure, dimensions, and content.

    Parameters:
    -----------
    filename : str
        Path to the HDF5 file to inspect
    args : argparse.Namespace, optional
        Command line arguments
    """
    print(f"\n{'=' * 80}")
    print(f"INSPECTING HDF5 FILE: {filename}")
    print(f"{'=' * 80}")

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return

    try:
        with h5py.File(filename, 'r') as f:
            # Get basic file info
            print(f"\n[FILE INFO]")
            print(f"File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

            # List all top-level groups/datasets
            print(f"\n[TOP-LEVEL STRUCTURE]")
            top_level = list(f.keys())
            print(f"Top-level keys: {top_level}")

            # Examine each dataset and group
            metadata = {}
            for key in top_level:
                print(f"\n[EXAMINING '{key}']")
                if isinstance(f[key], h5py.Dataset):
                    dataset = f[key]
                    info = {
                        'type': 'dataset',
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'chunks': dataset.chunks,
                        'compression': dataset.compression,
                        'size_mb': np.prod(dataset.shape) * dataset.dtype.itemsize / (1024 * 1024)
                    }
                    metadata[key] = info

                    print(f"  - Type: Dataset")
                    print(f"  - Shape: {info['shape']}")
                    print(f"  - Data type: {info['dtype']}")
                    print(f"  - Chunking: {info['chunks']}")
                    print(f"  - Compression: {info['compression']}")
                    print(f"  - Size in memory: {info['size_mb']:.2f} MB")

                    # Sample data if numeric
                    if dataset.dtype.kind in ('i', 'u', 'f', 'c'):  # integer, unsigned, float, complex
                        if len(dataset.shape) == 1:
                            if len(dataset) > 0:
                                sample = dataset[:min(5, len(dataset))]
                                print(f"  - First {len(sample)} elements: {sample}")
                        elif len(dataset.shape) == 2:
                            if dataset.shape[0] > 0 and dataset.shape[1] > 0:
                                # Sample both first row and first column
                                first_row = dataset[0, :min(5, dataset.shape[1])]
                                first_col = dataset[:min(5, dataset.shape[0]), 0]
                                print(f"  - First row (first 5 elements): {first_row}")
                                print(f"  - First column (first 5 elements): {first_col}")

                                # Get stats for numeric data
                                if dataset.dtype.kind in ('i', 'u', 'f'):
                                    try:
                                        # Sample a subset for large datasets
                                        if np.prod(dataset.shape) > 1000000:
                                            if len(dataset.shape) == 1:
                                                sample_indices = np.random.choice(dataset.shape[0],
                                                                                  size=1000, replace=False)
                                                sample_data = dataset[sample_indices]
                                            else:
                                                row_indices = np.random.choice(dataset.shape[0],
                                                                               size=min(1000, dataset.shape[0]),
                                                                               replace=False)
                                                col_indices = np.random.choice(dataset.shape[1],
                                                                               size=min(1000, dataset.shape[1]),
                                                                               replace=False)
                                                sample_data = dataset[row_indices[:, np.newaxis], col_indices]
                                        else:
                                            sample_data = dataset[:]

                                        # Compute stats
                                        stats = {
                                            'min': float(np.min(sample_data)),
                                            'max': float(np.max(sample_data)),
                                            'mean': float(np.mean(sample_data)),
                                            'std': float(np.std(sample_data)),
                                            'num_zeros': int(np.sum(sample_data == 0)),
                                            'percent_zeros': float(np.mean(sample_data == 0) * 100)
                                        }
                                        print(f"  - Stats (from sample): {stats}")
                                    except Exception as e:
                                        print(f"  - Error computing stats: {e}")

                    # Handle string/bytes data specifically
                    elif dataset.dtype.kind in ('S', 'U', 'O'):  # string, unicode, object
                        if len(dataset.shape) == 1 and len(dataset) > 0:
                            try:
                                sample = dataset[:min(5, len(dataset))]
                                print(
                                    f"  - First {len(sample)} elements: {[s.decode('utf-8') if isinstance(s, bytes) else s for s in sample]}")

                                # Count unique values
                                if len(dataset) < 100000:
                                    unique_count = len(np.unique(dataset))
                                    print(f"  - Number of unique values: {unique_count}")
                            except Exception as e:
                                print(f"  - Error processing string data: {e}")

                elif isinstance(f[key], h5py.Group):
                    group = f[key]
                    subkeys = list(group.keys())
                    info = {
                        'type': 'group',
                        'subkeys': subkeys,
                        'num_items': len(subkeys)
                    }
                    metadata[key] = info

                    print(f"  - Type: Group")
                    print(f"  - Number of items: {info['num_items']}")
                    print(f"  - Sub-keys: {', '.join(subkeys)}")

                    # Recursively examine first few items in the group
                    for i, subkey in enumerate(subkeys[:3]):
                        subitem = group[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(
                                f"  - Subitem {i + 1}: '{subkey}' (Dataset, shape={subitem.shape}, dtype={subitem.dtype})")
                        elif isinstance(subitem, h5py.Group):
                            sub_subkeys = list(subitem.keys())
                            print(f"  - Subitem {i + 1}: '{subkey}' (Group, {len(sub_subkeys)} items)")

                    if len(subkeys) > 3:
                        print(f"  - ... and {len(subkeys) - 3} more items")

            # Check for semantic model specific structure
            print(f"\n[SEMANTIC MODEL ANALYSIS]")

            has_embeddings = 'embeddings' in f
            has_words = 'words' in f

            if has_embeddings and has_words:
                print("File appears to be a semantic model with embeddings and vocabulary.")

                embeddings = f['embeddings']
                words = f['words']

                emb_shape = embeddings.shape
                vocab_size = len(words)

                print(f"Embeddings shape: {emb_shape}")
                print(f"Vocabulary size: {vocab_size}")

                # Check if dimensions match expectations for a semantic model
                if len(emb_shape) == 2:
                    if emb_shape[1] == vocab_size:
                        print("✓ Matrix dimensions match expected format: features × vocabulary")
                        print(f"  - Each word is represented by a {emb_shape[0]}-dimensional vector")
                    elif emb_shape[0] == vocab_size:
                        print("⚠️ Matrix appears to be transposed: vocabulary × features")
                        print(f"  - Each word is represented by a {emb_shape[1]}-dimensional vector")
                    else:
                        print("⚠️ Matrix dimensions don't match vocabulary size")
                        print(f"  - Embeddings: {emb_shape}")
                        print(f"  - Vocabulary: {vocab_size} words")

                # Sample words and their vectors
                if vocab_size > 0:
                    try:
                        sample_indices = np.random.choice(vocab_size, size=min(5, vocab_size), replace=False)
                        print("\nSample word vectors:")

                        for idx in sample_indices:
                            word = words[idx]
                            if isinstance(word, bytes):
                                word = word.decode('utf-8')

                            if len(emb_shape) == 2:
                                if emb_shape[1] == vocab_size:  # features × vocabulary
                                    vector = embeddings[:, idx]
                                else:  # vocabulary × features
                                    vector = embeddings[idx, :]

                            print(f"  - '{word}': shape={vector.shape}, first 5 elements={vector[:5]}")

                        # Check if we should print all words
                        if args.print_words:
                            print("\nComplete list of all words:")
                            for i, word in enumerate(words):
                                if isinstance(word, bytes):
                                    word = word.decode('utf-8')
                                print(f"{i + 1}. {word}")
                    except Exception as e:
                        print(f"Error sampling word vectors: {e}")
            else:
                if not has_embeddings:
                    print("⚠️ No 'embeddings' dataset found")
                if not has_words:
                    print("⚠️ No 'words' dataset found")

            # Summary
            print(f"\n[SUMMARY]")
            print(f"File contains {len(top_level)} top-level items")
            print(f"Total datasets: {sum(1 for k, v in metadata.items() if v['type'] == 'dataset')}")
            print(f"Total groups: {sum(1 for k, v in metadata.items() if v['type'] == 'group')}")

            # Return a dictionary with all the gathered metadata
            return metadata

    except Exception as e:
        print(f"Error inspecting file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Inspect HDF5 files in detail')
    parser.add_argument('files', nargs='+', help='HDF5 files to inspect')
    parser.add_argument('--save', help='Save inspection results to a JSON file')
    parser.add_argument('--print-words', action='store_true', help='Print all words in the vocabulary')
    parser.add_argument('--show-dimensions', action='store_true',
                        help='Print detailed dimension information for all datasets')

    args = parser.parse_args()

    all_metadata = {}
    for filename in args.files:
        metadata = inspect_hdf5(filename, args)
        if metadata:
            all_metadata[filename] = metadata

        # If show-dimensions flag is specified, print detailed dimension info
        if args.show_dimensions:
            try:
                with h5py.File(filename, 'r') as f:
                    print(f"\nDetailed dimensions for {filename}:")

                    def print_dimensions(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

                    f.visititems(print_dimensions)
            except Exception as e:
                print(f"Error showing dimensions: {e}")

    if args.save and all_metadata:
        with open(args.save, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"\nInspection results saved to {args.save}")


if __name__ == "__main__":
    main()