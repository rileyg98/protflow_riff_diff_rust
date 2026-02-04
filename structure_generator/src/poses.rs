use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseRecord {
    pub input_poses: Option<String>,
    pub poses: String,
    pub poses_description: String,

    #[serde(flatten)]
    pub extra_fields: HashMap<String, Value>,
}

#[derive(Debug, Default)]
pub struct Poses {
    pub work_dir: PathBuf,
    pub df: Vec<PoseRecord>,
}

impl Poses {
    pub fn new(work_dir: impl AsRef<Path>) -> Self {
        Self {
            work_dir: work_dir.as_ref().to_path_buf(),
            df: Vec::new(),
        }
    }

    pub fn load_from_json(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::open(path.as_ref())
            .context(format!("Failed to open poses file: {:?}", path.as_ref()))?;
        let reader = BufReader::new(file);
        let records: Vec<PoseRecord> = serde_json::from_reader(reader)?;
        self.df = records;
        Ok(())
    }

    pub fn save_to_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())
            .context(format!("Failed to create poses file: {:?}", path.as_ref()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.df)?;
        Ok(())
    }

    pub fn filter_poses_by_value(&mut self, score_col: &str, value: f64, operator: &str) {
        self.df.retain(|rec| {
            if let Some(v) = rec.extra_fields.get(score_col) {
                if let Some(num) = v.as_f64() {
                    match operator {
                        ">=" => num >= value,
                        "<=" => num <= value,
                        ">" => num > value,
                        "<" => num < value,
                        "==" => (num - value).abs() < 1e-6,
                        _ => true,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        });
    }

    pub fn filter_poses_by_rank(&mut self, n: usize, score_col: &str, ascending: bool) {
        self.df.sort_by(|a, b| {
            let va = a
                .extra_fields
                .get(score_col)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::INFINITY);
            let vb = b
                .extra_fields
                .get(score_col)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::INFINITY);
            if ascending {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        self.df.truncate(n);
    }
}
