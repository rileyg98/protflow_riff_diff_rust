use crate::poses::{PoseRecord, Poses};
use anyhow::{Context, Result};
use log::{error, info};
use std::path::{Path, PathBuf};
use tokio::process::Command;

use async_trait::async_trait;

#[async_trait]
pub trait Runner {
    async fn run(&self, poses: &mut Poses, prefix: &str) -> Result<()>;
}

pub struct LocalJobStarter;

impl LocalJobStarter {
    pub async fn run_command(cmd_str: &str, working_dir: &Path) -> Result<()> {
        info!("Running command: {}", cmd_str);

        let args = shell_words::split(cmd_str).context("Failed to parse command string")?;
        if args.is_empty() {
            return Err(anyhow::anyhow!("Command string is empty"));
        }

        let output = Command::new(&args[0])
            .args(&args[1..])
            .current_dir(working_dir)
            .output()
            .await
            .context("Failed to execute command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Command failed: {}\nStderr: {}", cmd_str, stderr);
            return Err(anyhow::anyhow!(
                "Command failed with status: {}. Stderr: {}",
                output.status,
                stderr
            ));
        }

        Ok(())
    }
}
