from cffi import FFI
import os
import json
ffi = FFI()
import numpy as np
ffi.cdef("""
    typedef struct {
        unsigned int set1;
        unsigned int set2;
        unsigned int idx1;
        unsigned int idx2;
    } CompatEntry;

    void generate_valid_combinations_to_file(
        const CompatEntry* compat_ptr,
        size_t compat_len,
        const unsigned int* set_lengths_ptr,
        size_t set_lengths_len,
        const char* output_path
    );
    typedef struct {
        uint16_t* combos_ptr;
        size_t    num_combos;
        size_t    n_sets;
    } FfiComboResult;

    FfiComboResult* find_top_combos_ffi(
        const char* combo_file,
        const char** score_files,
        size_t n_score_files,
        size_t n_combos,
        size_t n_sets,
        size_t top_n
    );

    void free_combo_result(FfiComboResult* ptr);
""")

# Load your compiled Rust library
lib = ffi.dlopen("/mnt/riffdiff/riff_diff_protflow/libcombo_validator.so")  # adjust path as needed

def run_validator_to_file(set_lengths, compat_entries):
    output_path = "./validator.bin"
    # Prepare CompatEntry C array
    entry_array = [
        {"set1": e[0], "set2": e[1], "idx1": e[2], "idx2": e[3]}
        for e in compat_entries
    ]
    entry_c_array = ffi.new("CompatEntry[]", entry_array)

    # Prepare set lengths C array
    lengths_c_array = ffi.new("unsigned int[]", set_lengths)

    # Prepare output file path
    output_path_bytes = output_path.encode("utf-8")
    output_path_c = ffi.new("char[]", output_path_bytes)
    print("Set lengths: " + str(len(set_lengths)))
    # Call Rust function
    lib.generate_valid_combinations_to_file(
        entry_c_array, len(entry_array),
        lengths_c_array, len(set_lengths),
        output_path_c
    )
    return output_path


def score_files(combo_path, rotamer_paths, n_combos, n_sets, top_n):
    c_paths = [ffi.new("char[]", path.encode("utf-8")) for path in rotamer_paths]
    c_score_files = ffi.new("const char*[]", c_paths)
    combo_path_bytes = output_path.encode("utf-8")
    combo_path_c = ffi.new("char[]", combo_path_bytes)
    result_ptr = lib.find_top_combos_ffi(combo_path_c, c_score_files, len(rotamer_paths), n_combos, n_sets, top_n)
    n = res.num_combos
    k = res.n_sets
    flat = ffi.buffer(res.combos_ptr, n * k * 2)
    arr = np.frombuffer(flat, dtype=np.uint16).reshape((n, k)).copy()
    lib.free_combo_result(result_ptr)
    return arr
