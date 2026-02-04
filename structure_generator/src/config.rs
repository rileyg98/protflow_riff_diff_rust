use anyhow::{Context, Result};
use log::info;
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub rfdiffusion_script: Option<String>,
    pub ligandmpnn_script: Option<String>,
    pub rosetta_bin: Option<String>,
    pub esmfold_python: Option<String>,
    pub protein_edits_scripts_dir: Option<String>,
    pub python_path: String,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context(format!("Failed to read config file: {:?}", path))?;
        let config: Config =
            serde_json::from_str(&content).context("Failed to parse config JSON")?;
        Ok(config)
    }

    pub fn load(file_path: Option<&PathBuf>) -> Result<Self> {
        let mut config = if let Some(path) = file_path {
            info!("Loading config from file: {:?}", path);
            Self::from_file(path)?
        } else {
            // Default empty config if no file provided - relying largely on envs or defaults
            Config {
                rfdiffusion_script: None,
                ligandmpnn_script: None,
                rosetta_bin: None,
                esmfold_python: None,
                protein_edits_scripts_dir: None,
                python_path: "python".to_string(), // Default python
            }
        };

        // Override with environment variables if set
        if let Ok(val) = env::var("RFDIFFUSION_SCRIPT") {
            config.rfdiffusion_script = Some(val);
        }
        if let Ok(val) = env::var("LIGANDMPNN_SCRIPT") {
            config.ligandmpnn_script = Some(val);
        }
        if let Ok(val) = env::var("ROSETTA_BIN_PATH") {
            config.rosetta_bin = Some(val);
        }
        if let Ok(val) = env::var("ESMFOLD_PYTHON_PATH") {
            config.esmfold_python = Some(val);
        }
        if let Ok(val) = env::var("AUXILIARY_RUNNER_SCRIPTS_DIR") {
            config.protein_edits_scripts_dir = Some(val);
        }
        if let Ok(val) = env::var("PROTFLOW_PYTHON") {
            config.python_path = val;
        }

        Ok(config)
    }
}
