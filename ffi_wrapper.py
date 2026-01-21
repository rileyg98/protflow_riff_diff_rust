import numpy as np
import riffdiff_rust_library  # Import the PyO3-based library

def run_validator_to_file(set_lengths, compat_entries):
    """
    Generates valid rotamer combinations and writes them to a binary file.

    Args:
        set_lengths (list[int]): A list where each element is the number of rotamers for a set.
        compat_entries (list[tuple]): A list of compatible rotamer pairs.
                                      Each tuple is (set1, set2, idx1, idx2).
    
    Returns:
        str: The path to the output binary file.
    """
    output_path = "./validator.bin"
    
    # Convert inputs to NumPy arrays with the correct dtype for the Rust function
    compat_array = np.array(compat_entries, dtype=np.uint32)
    lengths_array = np.array(set_lengths, dtype=np.uint32)

    # Call the PyO3-based Rust function
    riffdiff_rust_library.generate_valid_combinations_to_file(
        compat_array,
        lengths_array,
        output_path
    )
    return output_path


def score_files(combo_path, rotamer_paths, n_combos, n_sets, top_n):
    """
    Scores the combinations from the binary file and returns the top N.

    Args:
        combo_path (str): Path to the binary file of combinations.
        rotamer_paths (list[str]): List of paths to the score CSV files.
        n_combos (int): Total number of combinations in the combo_file.
        n_sets (int): The number of residue sets (i.e., columns in the data).
        top_n (int): The number of top combinations to return.

    Returns:
        np.ndarray: A NumPy array of shape (top_n, n_sets) containing the indices
                    of the best rotamer combinations.
    """
    # rotamer_paths is actually 
    # Transform rotamer_paths to fragment_info paths
    fragment_info_paths = []
    for path in rotamer_paths:
        # Extract rotamer_id from the clash_check path format
        # e.g., 'outputs/motif_library_assembly/clash_check/{rotamer_id}_rechained.json'
        filename = path.split('/')[-1]  # Get '{rotamer_id}_rechained.json'
        rotamer_id = filename.replace('_rechained.json', '') # Get '{rotamer_id}'
        
        # Construct the new fragment_info path
        new_path = f"outputs/fragments/fragment_info/fragments_{rotamer_id}.csv"
        fragment_info_paths.append(new_path)

    best_combos_arr = riffdiff_rust_library.find_top_combos(
        combo_path,
        fragment_info_paths, # Pass the transformed paths
        n_combos,
        n_sets,
        top_n
    )
    return best_combos_arr
