use crate::poses::{PoseRecord, Poses};
use crate::runners::{LocalJobStarter, Runner};
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::warn;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct LigandMPNN {
    pub python_path: String,
    pub script_path: String,
    pub options: String,
    pub nseq: usize,
}

impl LigandMPNN {
    pub fn new(python_path: &str, script_path: &str) -> Self {
        Self {
            python_path: python_path.to_string(),
            script_path: script_path.to_string(),
            options: String::new(),
            nseq: 1,
        }
    }

    fn write_cmd(&self, pose: &PoseRecord, output_dir: &Path) -> Result<String> {
        let output_path = output_dir.to_string_lossy();

        // pose.poses is String
        let mut cmd = format!(
            "{} {} --out_folder {} --pdb_path {} --number_of_batches={}",
            self.python_path, self.script_path, output_path, pose.poses, self.nseq
        );

        if !self.options.is_empty() {
            cmd.push_str(" ");
            cmd.push_str(&self.options);
        }

        Ok(cmd)
    }

    fn parse_fasta_header(header: &str) -> HashMap<String, Value> {
        let mut scores = HashMap::new();
        // Example header: >pdb_name, score=1.23, key=val
        let parts: Vec<&str> = header.split(", ").collect();
        if parts.len() > 1 {
            for part in &parts[1..] {
                if let Some((k, v)) = part.split_once('=') {
                    if let Ok(num) = v.parse::<f64>() {
                        scores.insert(
                            k.to_string(),
                            serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap()),
                        );
                    } else {
                        scores.insert(k.to_string(), serde_json::Value::String(v.to_string()));
                    }
                }
            }
        }
        scores
    }
}

#[async_trait]
impl Runner for LigandMPNN {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        let seq_dir = work_dir.join("seqs");
        tokio::fs::create_dir_all(&seq_dir)
            .await
            .context("Failed to create working directories")?;

        let mut new_records: Vec<PoseRecord> = Vec::new();

        for pose in &poses.df {
            // Check if we need to copy input pdb to working dir? No, --pdb_path handles it.
            let cmd = self.write_cmd(pose, &seq_dir)?;

            // Run LigandMPNN
            LocalJobStarter::run_command(&cmd, &work_dir).await?;

            let pdb_name = std::path::Path::new(&pose.poses_description)
                .file_stem()
                .unwrap()
                .to_string_lossy();
            let expected_fa = seq_dir.join(format!("{}.fa", pdb_name));

            if expected_fa.exists() {
                let content = tokio::fs::read_to_string(&expected_fa).await?;
                for line in content.lines() {
                    if line.starts_with('>') {
                        let header = &line[1..];
                        let scores = Self::parse_fasta_header(header);

                        let mut rec = pose.clone();

                        // Check if sequence threaded PDB exists
                        let seq_name = header.split(", ").next().unwrap_or(&pdb_name);
                        let expected_pdb =
                            work_dir.join("backbones").join(format!("{}.pdb", seq_name));

                        if expected_pdb.exists() {
                            rec.poses = expected_pdb.to_string_lossy().to_string();
                        } else {
                            rec.poses = expected_fa.to_string_lossy().to_string();
                        }

                        rec.extra_fields.extend(scores);
                        new_records.push(rec);
                    }
                }
            } else {
                warn!("Expected FASTA file not found: {:?}", expected_fa);
            }
        }

        poses.df = new_records;
        poses.save_to_json(poses.work_dir.join(format!("{}_scores.json", prefix)))?;
        Ok(())
    }
}
