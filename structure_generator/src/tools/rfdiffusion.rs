use crate::poses::{PoseRecord, Poses};
use crate::runners::{LocalJobStarter, Runner};
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::warn;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokio::process::Command;

pub struct RFDiffusion {
    pub python_path: String,
    pub script_path: String,
    pub options: String,
    pub num_diffusions: usize,
}

impl RFDiffusion {
    pub fn new(python_path: &str, script_path: &str) -> Self {
        Self {
            python_path: python_path.to_string(),
            script_path: script_path.to_string(),
            options: String::new(),
            num_diffusions: 1,
        }
    }

    fn write_cmd(&self, pose: &PoseRecord, output_dir: &Path, i: usize) -> Result<String> {
        let desc = format!(
            "{}_{:04}",
            std::path::Path::new(&pose.poses_description)
                .file_stem()
                .unwrap()
                .to_string_lossy(),
            i
        );
        let output_prefix = output_dir.join(&desc);

        let mut cmd = format!(
            "{} {} {} inference.output_prefix={}",
            self.python_path,
            self.script_path,
            self.options,
            output_prefix.to_string_lossy()
        );

        if !self.options.contains("inference.input_pdb") {
            cmd.push_str(&format!(" inference.input_pdb={}", pose.poses));
        }

        if !self.options.contains("inference.num_designs") {
            cmd.push_str(" inference.num_designs=1");
        }

        Ok(cmd)
    }
}

#[async_trait]
impl Runner for RFDiffusion {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        let pdb_dir = work_dir.join("output_pdbs");
        tokio::fs::create_dir_all(&pdb_dir)
            .await
            .context("Failed to create working directories")?;

        let mut new_records = Vec::new();

        for pose in &poses.df {
            for i in 0..self.num_diffusions {
                let cmd = self.write_cmd(pose, &pdb_dir, i)?;

                // Run RFDiffusion
                LocalJobStarter::run_command(&cmd, &work_dir).await?;

                // Parse output .trb
                let desc = format!(
                    "{}_{:04}",
                    std::path::Path::new(&pose.poses_description)
                        .file_stem()
                        .unwrap()
                        .to_string_lossy(),
                    i
                );
                let expected_trb = pdb_dir.join(format!("{}.trb", desc));

                if expected_trb.exists() {
                    let parse_script = "scripts/parse_trb.py"; // Assume relative path for now

                    let output = Command::new("python")
                        .arg(parse_script)
                        .arg(&expected_trb)
                        .output()
                        .await
                        .context("Failed to run parse_trb.py")?;

                    if output.status.success() {
                        let json_out: Value = serde_json::from_slice(&output.stdout)?;

                        let new_pdb = json_out["location"].as_str().unwrap_or("").to_string();
                        let new_desc = json_out["description"]
                            .as_str()
                            .unwrap_or(&desc)
                            .to_string();

                        let mut extra = HashMap::new();
                        if let Some(obj) = json_out.as_object() {
                            for (k, v) in obj {
                                if k != "location" && k != "description" {
                                    extra.insert(k.clone(), v.clone());
                                }
                            }
                        }

                        new_records.push(PoseRecord {
                            input_poses: Some(pose.poses.clone()),
                            poses: new_pdb,
                            poses_description: new_desc,
                            extra_fields: extra,
                        });
                    } else {
                        warn!("Failed to parse TRB file: {:?}", expected_trb);
                    }
                } else {
                    warn!("Expected TRB file not found: {:?}", expected_trb);
                }
            }
        }

        poses.df = new_records;
        poses.save_to_json(poses.work_dir.join(format!("{}_scores.json", prefix)))?;

        Ok(())
    }
}
