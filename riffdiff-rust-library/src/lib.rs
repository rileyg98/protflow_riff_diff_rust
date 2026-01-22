use anyhow::Context;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods, ToPyArray, PyReadonlyArray1, PyReadonlyArray2};

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

use crossbeam_channel::{unbounded, Sender};

const BATCH_SIZE: usize = 1000000; // How many combinations a thread accumulates before sending

#[pyfunction]
fn generate_valid_combinations_to_file(
    compat_data: PyReadonlyArray2<u32>,
    set_lengths: PyReadonlyArray1<u32>,
    output_path: String,
) -> PyResult<()> {
    let compat_slice = compat_data.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let set_lengths_slice = set_lengths.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let n_sets = set_lengths_slice.len();
    let max_set_size = *set_lengths_slice.iter().max().unwrap_or(&0) as usize;

    let mut compat_map = vec![vec![vec![vec![false; max_set_size]; max_set_size]; n_sets]; n_sets];
    for row in compat_slice.chunks_exact(4) {
        let (i, j, a, b) = (row[0] as usize, row[1] as usize, row[2] as usize, row[3] as usize);
        compat_map[i][j][a][b] = true;
        compat_map[j][i][b][a] = true;
    }

    let compat_map = Arc::new(compat_map);
    let set_lengths_arc = Arc::new(set_lengths_slice.to_vec());

    // --- Producer-Consumer Implementation ---
    let (sender, receiver) = unbounded::<Vec<Vec<u32>>>();
    let output_meta_path = format!("{}.meta", output_path);

    let writer_handle = std::thread::spawn(move || {
        let mut writer = BinWriter::new(&output_path).expect("Failed to create BinWriter");
        while let Ok(batch) = receiver.recv() {
            writer.write_batch(&batch).expect("Failed to write batch");
        }
        writer.close_with_metadata(&output_meta_path, n_sets).expect("Failed to write metadata");
    });

    (0..set_lengths_slice[0]).into_par_iter().for_each_with(sender, |s, first_idx| {
        let mut local_buffer = Vec::with_capacity(BATCH_SIZE);
        let mut combo = vec![first_idx];
        
        recurse_and_send(1, &set_lengths_arc, &compat_map, &mut combo, &mut local_buffer, s);

        if !local_buffer.is_empty() {
            s.send(local_buffer).expect("Failed to send final batch");
        }
    });

    writer_handle.join().expect("Writer thread panicked");

    Ok(())
}

#[pymodule]
fn riffdiff_rust_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_top_combos, m)?)?;
    m.add_function(wrap_pyfunction!(generate_valid_combinations_to_file, m)?)?;
    Ok(())
}

fn recurse_and_send(
    depth: usize,
    set_lengths: &Arc<Vec<u32>>,
    compat_map: &Arc<Vec<Vec<Vec<Vec<bool>>>>>,
    current_combo: &mut Vec<u32>,
    local_buffer: &mut Vec<Vec<u32>>,
    sender: &Sender<Vec<Vec<u32>>>,
) {
    let n_sets = set_lengths.len();
    if depth == n_sets {
        local_buffer.push(current_combo.clone());
        if local_buffer.len() >= BATCH_SIZE {
            let batch_to_send = std::mem::replace(local_buffer, Vec::with_capacity(BATCH_SIZE));
            sender.send(batch_to_send).expect("Failed to send batch");
        }
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
            recurse_and_send(depth + 1, set_lengths, compat_map, current_combo, local_buffer, sender);
            current_combo.pop();
        }
    }
}

pub struct BinWriter {
    writer: BufWriter<File>,
    line_count: usize,
}

impl BinWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(BinWriter {
            writer: BufWriter::new(file),
            line_count: 0,
        })
    }

    pub fn write_batch(&mut self, batch: &[Vec<u32>]) -> std::io::Result<()> {
        for line in batch {
            for &value in line {
                let val_u16 = value as u16;
                self.writer.write_all(&val_u16.to_le_bytes())?;
            }
        }
        self.line_count += batch.len();
        Ok(())
    }

    pub fn close_with_metadata<P: AsRef<Path>>(mut self, metadata_path: P, row_len: usize) -> std::io::Result<()> {
        self.writer.flush()?;
        let metadata = format!("{{\"rows\": {}, \"cols\": {}, \"dtype\": \"u16\"}}", self.line_count, row_len);
        let mut metadata_file = File::create(metadata_path)?;
        metadata_file.write_all(metadata.as_bytes())?;
        Ok(())
    }
}

struct RotamerScoreSet {
    scores: Vec<Vec<f32>>
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

pub fn load_f32_score_array_from_csv(path: &Path) -> Result<Vec<f32>> {
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

    Ok(scores)
}