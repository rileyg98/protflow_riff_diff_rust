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

    pub fn add_prefix(&mut self, prefix: &str) {
        // Implementation to prefix columns/keys if needed, similar to Python
        // For now, this might not be strictly necessary if we handle it in the runners
    }
}
