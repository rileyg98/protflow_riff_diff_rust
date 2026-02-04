use crate::poses::Poses;
use crate::runners::Runner;
use anyhow::Result;
use async_trait::async_trait;
use log::{info, warn};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;

pub struct ChainRemover {
    pub python_path: String,
    pub script_path: String,
    pub chains: Option<Vec<String>>,
}

impl ChainRemover {
    pub fn new(python_path: &str, script_dir: &str) -> Self {
        let script_path = Path::new(script_dir).join("remove_chains_batch.py");
        Self {
            python_path: python_path.to_string(),
            script_path: script_path.to_string_lossy().to_string(),
            chains: None,
        }
    }
}

#[async_trait]
impl Runner for ChainRemover {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        fs::create_dir_all(&work_dir).await?;
        let work_dir_canon = std::fs::canonicalize(&work_dir)?;

        let chains = self
            .chains
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Chains to remove not specified"))?;

        let mut input_dict = HashMap::new();
        for pose in &poses.df {
            let abs_path =
                std::fs::canonicalize(&pose.poses).unwrap_or_else(|_| PathBuf::from(&pose.poses));
            input_dict.insert(abs_path.to_string_lossy().to_string(), chains.clone());
        }

        let input_json_path = work_dir_canon.join("remove_chains_input.json");
        let f = std::fs::File::create(&input_json_path)?;
        serde_json::to_writer(&f, &input_dict)?;

        let cmd = format!(
            "{} {} --input_json {} --output_dir {}",
            self.python_path,
            self.script_path,
            input_json_path.to_string_lossy(),
            work_dir_canon.to_string_lossy()
        );

        info!("Running ChainRemover: {}", cmd);
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .current_dir(&work_dir_canon)
            .spawn()?;

        let status = child.wait()?;
        if !status.success() {
            anyhow::bail!("ChainRemover failed");
        }

        let mut new_records = Vec::new();
        for pose in &poses.df {
            let file_name = Path::new(&pose.poses).file_name().unwrap();
            let new_path = work_dir_canon.join(file_name);

            if new_path.exists() {
                let mut record = pose.clone();
                record.poses = new_path.to_string_lossy().to_string();
                new_records.push(record);
            } else {
                warn!("Expected output file not found: {:?}", new_path);
            }
        }
        poses.df = new_records;

        Ok(())
    }
}

pub struct ChainAdder {
    pub python_path: String,
    pub script_path: String,
    pub copy_chain: String,
    pub ref_pdb: Option<String>,
}

impl ChainAdder {
    pub fn new(python_path: &str, script_dir: &str, copy_chain: &str) -> Self {
        let script_path = Path::new(script_dir).join("add_chains_batch.py");
        Self {
            python_path: python_path.to_string(),
            script_path: script_path.to_string_lossy().to_string(),
            copy_chain: copy_chain.to_string(),
            ref_pdb: None,
        }
    }
}

#[async_trait]
impl Runner for ChainAdder {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        fs::create_dir_all(&work_dir).await?;
        let work_dir_canon = std::fs::canonicalize(&work_dir)?;

        let mut input_dict = HashMap::new();

        for pose in &poses.df {
            let abs_path =
                std::fs::canonicalize(&pose.poses).unwrap_or_else(|_| PathBuf::from(&pose.poses));
            let abs_path_str = abs_path.to_string_lossy().to_string();

            let mut opts = HashMap::new();
            opts.insert(
                "copy_chain".to_string(),
                Value::String(self.copy_chain.clone()),
            );

            let mut pose_ref_pdb = self.ref_pdb.clone();
            if let Some(val) = pose.extra_fields.get("updated_reference_frags_location") {
                if let Some(s) = val.as_str() {
                    pose_ref_pdb = Some(s.to_string());
                }
            }

            if let Some(ref_p) = pose_ref_pdb {
                let ref_abs =
                    std::fs::canonicalize(&ref_p).unwrap_or_else(|_| PathBuf::from(&ref_p));
                opts.insert(
                    "reference_pdb".to_string(),
                    Value::String(ref_abs.to_string_lossy().to_string()),
                );
            } else {
                warn!(
                    "Reference PDB not set for ChainAdder, skipping pose {}",
                    abs_path_str
                );
                continue;
            }

            input_dict.insert(abs_path_str, opts);
        }

        if input_dict.is_empty() {
            return Ok(());
        }

        let input_json_path = work_dir_canon.join("add_chains_input.json");
        let f = std::fs::File::create(&input_json_path)?;
        serde_json::to_writer(&f, &input_dict)?;

        let cmd = format!(
            "{} {} --input_json {} --output_dir {}",
            self.python_path,
            self.script_path,
            input_json_path.to_string_lossy(),
            work_dir_canon.to_string_lossy()
        );

        info!("Running ChainAdder: {}", cmd);
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .current_dir(&work_dir_canon)
            .spawn()?;

        let status = child.wait()?;
        if !status.success() {
            anyhow::bail!("ChainAdder failed");
        }

        let mut new_records = Vec::new();
        for pose in &poses.df {
            let file_name = Path::new(&pose.poses).file_name().unwrap();
            let new_path = work_dir_canon.join(file_name);

            if new_path.exists() {
                let mut record = pose.clone();
                record.poses = new_path.to_string_lossy().to_string();
                new_records.push(record);
            } else {
                warn!("Expected output file not found: {:?}", new_path);
            }
        }
        poses.df = new_records;

        Ok(())
    }
}
