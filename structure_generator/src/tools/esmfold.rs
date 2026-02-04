use crate::poses::{PoseRecord, Poses};
use crate::runners::Runner;
use anyhow::Result;
use async_trait::async_trait;
use log::info;
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;

pub struct ESMFold {
    pub python_path: String,
    pub script_path: String,
    pub options: String,
}

impl ESMFold {
    pub fn new(python_path: &str, script_dir: &str) -> Self {
        let script_path = Path::new(script_dir).join("esmfold_inference.py");
        Self {
            python_path: python_path.to_string(),
            script_path: script_path.to_string_lossy().to_string(),
            options: "".to_string(),
        }
    }

    fn prep_fastas(&self, poses: &[PoseRecord], work_dir: &Path) -> Result<Vec<PathBuf>> {
        let fasta_dir = work_dir.join("input_fastas");
        std::fs::create_dir_all(&fasta_dir)?;

        // For now, simple implementation: One big fasta or per-pose?
        // ProtFlow splits into batches.
        // Let's create one batch for now or split if needed.
        // We will just create 1 batch to keep it simple locally.

        let batch_path = fasta_dir.join("fasta_0001.fa");
        let mut batch_file = File::create(&batch_path)?;

        for pose in poses {
            let path = Path::new(&pose.poses);
            let mut content = String::new();
            File::open(path)?.read_to_string(&mut content)?;

            // Append with newline
            writeln!(batch_file, "{}", content.trim())?;
        }

        Ok(vec![batch_path])
    }
}

#[async_trait]
impl Runner for ESMFold {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        fs::create_dir_all(&work_dir).await?;
        let work_dir_canon = std::fs::canonicalize(&work_dir)?;

        info!("Running ESMFold in {:?}", work_dir_canon);

        // Prep fastas
        let fasta_files = self.prep_fastas(&poses.df, &work_dir_canon)?;

        // Output dir for predictions
        let preds_dir = work_dir_canon.join("esm_preds");
        fs::create_dir_all(&preds_dir).await?;

        // Run inference for each batch
        for fasta_file in fasta_files {
            // cmd: python esmfold_inference.py --fasta input.fa --output_dir output_dir {options}
            let cmd = format!(
                "{} {} --fasta {:?} --output_dir {:?} {}",
                self.python_path, self.script_path, fasta_file, preds_dir, self.options
            );

            info!("Executing: {}", cmd);
            let mut child = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .current_dir(&work_dir_canon)
                .spawn()?;

            let status = child.wait()?;
            if !status.success() {
                anyhow::bail!("ESMFold failed for fasta {:?}", fasta_file);
            }
        }

        // Collect scores
        // ProtFlow: esm_preds contains .json and .pdb files (maybe in subdirs?)
        // The script `esmfold_inference.py` output structure depends on the script.
        // Assuming it outputs directly to output_dir or subdirs.
        // ProtFlow glob: f"{pdb_dir}/fasta_*/*.json"

        // Let's assume flat or glob recursively.
        let mut new_records = Vec::new();
        let output_pdbs_dir = work_dir_canon.join("output_pdbs");
        fs::create_dir_all(&output_pdbs_dir).await?;

        // Walk preds_dir
        let mut read_dir = fs::read_dir(&preds_dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                // Check subdir
                let mut sub_read = fs::read_dir(&path).await?;
                while let Some(sub_entry) = sub_read.next_entry().await? {
                    self.process_file(&sub_entry.path(), &output_pdbs_dir, &mut new_records)
                        .await?;
                }
            } else {
                self.process_file(&path, &output_pdbs_dir, &mut new_records)
                    .await?;
            }
        }

        poses.df = new_records;

        Ok(())
    }
}

impl ESMFold {
    async fn process_file(
        &self,
        path: &Path,
        output_pdbs_dir: &Path,
        records: &mut Vec<PoseRecord>,
    ) -> Result<()> {
        if path.extension().map_or(false, |ext| ext == "json") {
            // Found json score file
            // Expect corresponding PDB
            let stem = path.file_stem().unwrap();
            let pdb_name = format!("{}.pdb", stem.to_string_lossy());

            // Locate PDB. It should be in the same dir?
            let pdb_path = path.with_file_name(&pdb_name);

            if pdb_path.exists() {
                // Move PDB to output_pdbs
                let new_pdb_path = output_pdbs_dir.join(&pdb_name);
                fs::copy(&pdb_path, &new_pdb_path).await?;

                let content = fs::read_to_string(path).await?;
                let data: std::collections::HashMap<String, Value> =
                    serde_json::from_str(&content)?;

                // Extract description from filename or json?
                let desc = stem.to_string_lossy().to_string();

                let record = PoseRecord {
                    input_poses: None, // Lost origin for now unless we track it in fasta headers and map back
                    poses: new_pdb_path.to_string_lossy().to_string(),
                    poses_description: desc,
                    extra_fields: data,
                };
                records.push(record);
            }
        }
        Ok(())
    }
}
