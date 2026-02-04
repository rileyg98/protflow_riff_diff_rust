use crate::poses::{PoseRecord, Poses};
use crate::runners::Runner;
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::{info, warn};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;

pub struct Rosetta {
    pub script_path: String,
    pub python_path: String, // Not used directly for binary execution but consistent with other tools
    pub nstruct: usize,
    pub application: String,
    pub options: String,
}

impl Rosetta {
    pub fn new(script_path: &str) -> Self {
        Self {
            script_path: script_path.to_string(),
            python_path: "python".to_string(),
            nstruct: 1,
            application: "rosetta_scripts.linuxgccrelease".to_string(), // Default app
            options: "".to_string(),
        }
    }

    fn construct_command(
        &self,
        pose_path: &str,
        output_dir: &Path,
        i: usize,
        pose_filename: &str,
    ) -> String {
        // -out:path:all {output_dir} -in:file:s {pose_path} -out:prefix r{i:04}_ -out:file:scorefile r{i:04}_{}_score.json -out:file:scorefile_format json
        let prefix = format!("r{:04}_", i);
        let scorefile = format!("r{:04}_{}_score.json", i, pose_filename);

        format!(
            "{} -out:path:all {:?} -in:file:s {:?} -out:prefix {} -out:file:scorefile {:?} -out:file:scorefile_format json {}",
            self.script_path,
            output_dir,
            pose_path,
            prefix,
            scorefile,
            self.options
        )
    }
}

#[async_trait]
impl Runner for Rosetta {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()> {
        let work_dir = poses.work_dir.join(prefix);
        fs::create_dir_all(&work_dir).await?;
        let work_dir_canon = std::fs::canonicalize(&work_dir)?; // Canonicalize for subprocess

        info!(
            "Running Rosetta {} in {:?}",
            self.application, work_dir_canon
        );

        let mut next_poses_records = Vec::new();

        // Iterate over input poses (using df field)
        for (i, pose) in poses.df.iter().enumerate() {
            let input_pdb = &pose.poses; // Field is 'poses' (path string)
            let pose_filename = Path::new(input_pdb).file_stem().unwrap().to_str().unwrap();

            for n in 1..=self.nstruct {
                let cmd = self.construct_command(input_pdb, &work_dir_canon, n, pose_filename);

                info!("Executing: {}", cmd);

                let mut child = Command::new("sh")
                    .arg("-c")
                    .arg(&cmd)
                    .current_dir(&work_dir_canon)
                    .spawn()?;

                let status = child.wait()?;
                if !status.success() {
                    warn!("Rosetta failed for pose {} struct {}", pose_filename, n);
                    continue;
                }

                let score_json_path =
                    work_dir_canon.join(format!("r{:04}_{}_score.json", n, pose_filename));
                if score_json_path.exists() {
                    let content = fs::read_to_string(&score_json_path).await?;
                    let v: FileScore =
                        serde_json::from_str(&content).context("Failed to parse score json")?;

                    let decoy_name = v.decoy.clone();
                    // New description: {original_desc}_{n:04}
                    let new_desc = format!("{}_{:04}", pose.poses_description, n);

                    let old_path = work_dir_canon.join(format!("{}.pdb", decoy_name));
                    let new_path = work_dir_canon.join(format!("{}.pdb", new_desc));

                    if old_path.exists() {
                        fs::rename(&old_path, &new_path).await?;

                        let mut new_record = PoseRecord {
                            input_poses: Some(input_pdb.clone()),
                            poses: new_path.to_string_lossy().to_string(),
                            poses_description: new_desc,
                            extra_fields: pose.extra_fields.clone(), // Copy existing extra fields
                        };

                        // Merge scores into extra_fields
                        // Note: v.extra contains string/number/etc.
                        for (k, val) in v.extra {
                            new_record.extra_fields.insert(k, val);
                        }

                        next_poses_records.push(new_record);
                    } else {
                        warn!("Expected output PDB not found: {:?}", old_path);
                    }
                }
            }
        }

        // Updating poses
        poses.df = next_poses_records;

        Ok(())
    }
}

#[derive(serde::Deserialize, Debug)]
struct FileScore {
    decoy: String,
    #[serde(flatten)]
    extra: std::collections::HashMap<String, Value>,
}
