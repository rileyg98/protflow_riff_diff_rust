use anyhow::Context;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods, ToPyArray};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompatEntry {
    pub set1: u32,
    pub set2: u32,
    pub idx1: u32,
    pub idx2: u32,
}

#[pyfunction]
fn find_top_combos(
    py: Python,
    combo_file: String,
    score_files: Vec<String>,
    n_combos: usize,
    n_sets: usize,
    top_n: usize,
) -> PyResult<Py<PyArray2<u16>>> {
    let combo_path = Path::new(&combo_file);
    let score_paths: Vec<PathBuf> = score_files.into_iter().map(PathBuf::from).collect();

    // Load combo data
    let combo_data =
        load_u16_mmap(combo_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let combos = ValidComboMatrix {
        n_combos,
        n_sets,
        data: combo_data,
    };

    // Load scores
    let mut scores = Vec::new();
    for path in &score_paths {
        match load_f32_score_array_from_csv(path) {
            Ok(s) => scores.push(s),
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
        }
    }
    let score_set = RotamerScoreSet { scores };

    // Score and find the best
    let best = score_combinations(&combos, &score_set, top_n);

    // Flatten combo data into u16 array
    let mut flat_result: Vec<u16> = Vec::with_capacity(best.len() * n_sets);
    for ScoredCombo { index, .. } in &best {
        let row = combos.get_combo(*index);
        flat_result.extend_from_slice(row);
    }

    // Create a NumPy array and reshape it
    let array = flat_result
        .to_pyarray(py)
        .reshape([best.len(), n_sets])
        .map_err(|e: pyo3::PyErr| e)?;
    Ok(array.to_owned().into())
}

#[pymodule]
fn riffdiff_rust_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_top_combos, m)?)?;
    Ok(())
}


#[unsafe(no_mangle)]
pub extern "C" fn generate_valid_combinations_to_file(
    compat_ptr: *const CompatEntry,
    compat_len: usize,
    set_lengths_ptr: *const u32,
    set_lengths_len: usize,
    output_path_ptr: *const libc::c_char,
) {
    let compat_slice = unsafe { std::slice::from_raw_parts(compat_ptr, compat_len) };
    let set_lengths = unsafe { std::slice::from_raw_parts(set_lengths_ptr, set_lengths_len) };
    let output_path = unsafe { std::ffi::CStr::from_ptr(output_path_ptr).to_str().unwrap() };

    let n_sets = set_lengths.len();
    let max_set_size = *set_lengths.iter().max().unwrap_or(&0) as usize;

    let mut compat_map = vec![vec![vec![vec![false; max_set_size]; max_set_size]; n_sets]; n_sets];
    for i in 0..n_sets {
        for j in 0..n_sets {
            compat_map[i][j] = vec![vec![false; max_set_size]; max_set_size];
        }
    }

    for entry in compat_slice {
        let (i, j, a, b) = (
            entry.set1 as usize,
            entry.set2 as usize,
            entry.idx1 as usize,
            entry.idx2 as usize,
        );
        compat_map[i][j][a][b] = true;
        compat_map[j][i][b][a] = true;
    }


    let writer = Arc::new(Mutex::new(BinWriter::new(output_path).expect("Unable to create BinWriter")));

    let compat_map = Arc::new(compat_map);
    let set_lengths = Arc::new(set_lengths.to_vec());

    // Top-level parallelism: depth = 0
    (0..set_lengths[0]).into_par_iter().for_each(|first_idx| {
        let mut combo = vec![first_idx];
        recurse_write(
            1,
            &set_lengths,
            &compat_map,
            &mut combo,
            &writer,
        );
    });
    writer.lock().unwrap()
        .close_with_metadata(format!("{}.meta", output_path), n_sets)
        .expect("Unable to write metadata");

}

fn recurse_write(
    depth: usize,
    set_lengths: &[u32],
    compat_map: &Vec<Vec<Vec<Vec<bool>>>>,
    current_combo: &mut Vec<u32>,
    writer: &Arc<Mutex<BinWriter>>,
) {
    let n_sets = set_lengths.len();
    if depth == n_sets {
        //let json_line = json!(current_combo);
        let w = writer.lock().unwrap();
        w.write_line(current_combo).expect("Unable to write line");
        return;
    }

    for idx in 0..set_lengths[depth] {
        let mut is_valid = true;
        for prev_set in 0..depth {
            let prev_idx = current_combo[prev_set] as usize;
            if !compat_map[prev_set][depth][prev_idx][idx as usize] {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            current_combo.push(idx);
            recurse_write(depth + 1, set_lengths, compat_map, current_combo, writer);
            current_combo.pop();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn generate_valid_combinations(
    compat_ptr: *const CompatEntry,
    compat_len: usize,
    set_lengths_ptr: *const u32,
    set_lengths_len: usize,
    output_callback: extern "C" fn(*const u32, usize),
) {
    let compat_slice = unsafe { std::slice::from_raw_parts(compat_ptr, compat_len) };
    let set_lengths = unsafe { std::slice::from_raw_parts(set_lengths_ptr, set_lengths_len) };
    let n_sets = set_lengths.len();

    // Build compatibility map
    let mut compat_map = vec![vec![vec![vec![false; 0]; 0]; n_sets]; n_sets];
    let max_set_size = *set_lengths.iter().max().unwrap_or(&0) as usize;
    for i in 0..n_sets {
        for j in 0..n_sets {
            compat_map[i][j] = vec![vec![false; max_set_size]; max_set_size];
        }
    }
    for entry in compat_slice {
        let (i, j, a, b) = (
            entry.set1 as usize,
            entry.set2 as usize,
            entry.idx1 as usize,
            entry.idx2 as usize,
        );
        compat_map[i][j][a][b] = true;
        compat_map[j][i][b][a] = true; // Symmetric
    }

    // Parallelize the first level of recursion
    let first_set_size = set_lengths[0] as usize;

    (0..first_set_size).into_par_iter().for_each(|idx| {
        let mut combo = vec![idx as u32];
        backtrack(
            1,
            n_sets,
            set_lengths,
            &compat_map,
            &mut combo,
            output_callback,
        );
    });
}

fn backtrack(depth: usize, n_sets: usize, set_lengths: &[u32], compat_map: &Vec<Vec<Vec<Vec<bool>>>>, current_combo: &mut Vec<u32>, output_callback: extern "C" fn(*const u32, usize)) {
    if depth == n_sets {
        output_callback(current_combo.as_ptr(), current_combo.len());
        
        return;
    }
    for idx in 0..set_lengths[depth] {
        let mut valid = true;
        for prev_set in 0..depth {
            let prev_idx = current_combo[prev_set] as usize;
            let cur_idx = idx as usize;
            if !compat_map[prev_set][depth][prev_idx][cur_idx] {
                valid = false;
                break;
            }
        }
        if valid {
            current_combo.push(idx);
            backtrack(depth + 1, n_sets, set_lengths, compat_map, current_combo, output_callback);
            current_combo.pop();
        }
    }
}

pub struct BinWriter {
    writer: Arc<Mutex<BufWriter<File>>>,
    line_count: Arc<Mutex<usize>>,
}

impl BinWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = Arc::new(Mutex::new(BufWriter::new(file)));
        let line_count = Arc::new(Mutex::new(0));
        Ok(BinWriter { writer, line_count })
    }

    pub fn write_line(&self, line: &[u32]) -> std::io::Result<()> {
        let mut writer = self.writer.lock().unwrap();
        for &value in line {
            // Downcast to u8
            let val_u16 = value as u16;
            writer.write_all(&val_u16.to_le_bytes())?;
        }
        //writer.flush()?;
        *self.line_count.lock().unwrap() += 1;
        Ok(())
    }

    pub fn close_with_metadata<P: AsRef<Path>>(&self, metadata_path: P, row_len: usize) -> std::io::Result<()> {
        let lines = *self.line_count.lock().unwrap();
        // Explicitly flush the buffer
        self.writer.lock().unwrap().flush()?;

        // Write metadata
        let mut metadata_file = File::create(metadata_path)?;
        let metadata = format!("{{\"rows\": {}, \"cols\": {}, \"dtype\": \"u16\"}}", lines, row_len);
        metadata_file.write_all(metadata.as_bytes())?;
        Ok(())
    }
}

struct RotamerScoreSet {
    scores: Vec<Arc<[f32]>>
}

pub struct ValidComboMatrix {
    pub n_combos: usize,
    pub n_sets: usize, 
    pub data: Arc<[u16]>
}

impl ValidComboMatrix {
    fn get_combo(&self, combo_index: usize) -> &[u16] {
        let start = combo_index * self.n_sets;
        &self.data[start..start+self.n_sets]
    }
}
#[derive(Debug, PartialEq)]
struct ScoredCombo {
    score: f32,
    index: usize
}

impl Eq for ScoredCombo {}

impl PartialOrd for ScoredCombo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoredCombo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse sort for min-heap behaviour
        other.score.partial_cmp(&self.score).unwrap()
    }
}



fn score_combinations(
    combos: &ValidComboMatrix,
    scores: &RotamerScoreSet,
    top_n: usize,
) -> Vec<ScoredCombo> {
    let heap = Mutex::new(BinaryHeap::with_capacity(top_n + 1));

    (0..combos.n_combos).into_par_iter().for_each(|i| {
        let combo_indices = combos.get_combo(i);
        let score_sum: f32 = combo_indices
            .iter()
            .enumerate()
            .map(|(residue_i, &rotamer_index)| scores.scores[residue_i][rotamer_index as usize])
            .sum();

        let avg_score = score_sum / combos.n_sets as f32;
        
        let mut heap_guard = heap.lock().unwrap();
        if heap_guard.len() < top_n {
            heap_guard.push(ScoredCombo { score: avg_score, index: i });
        } else if avg_score > heap_guard.peek().unwrap().score {
            heap_guard.pop();
            heap_guard.push(ScoredCombo { score: avg_score, index: i });
        }
    });

    heap.into_inner().unwrap().into_sorted_vec()
}

fn load_u16_mmap(path: &Path) -> anyhow::Result<Arc<[u16]>> {
    let file = File::open(path).context("Failed to open combo mmap")?;
    let mmap = unsafe { Mmap::map(&file).context("Failed to mmap file")? };
    let data: &[u16] = bytemuck::cast_slice(&mmap[..]);
    Ok(Arc::from(data))
}

use std::io::{BufReader};
use anyhow::Result;
use csv::ReaderBuilder;

pub fn load_f32_score_array_from_csv(path: &Path) -> Result<Arc<[f32]>> {
    let file = File::open(path)
        .with_context(|| "Failed to open score CSV")?;
    let reader = BufReader::new(file);
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);

    let mut scores = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        let record = result.with_context(|| format!("Error parsing CSV line {}", i + 1))?;
        let score_str = record.get(28)
            .with_context(|| format!("Missing score in column 28 at line {}", i + 1))?;
        let score: f32 = score_str.parse()
            .with_context(|| format!("Invalid score '{}' at line {}", score_str, i + 1))?;
        scores.push(score);
    }

    Ok(Arc::from(scores))
}