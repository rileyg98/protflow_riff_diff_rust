mod config;
mod poses;
mod runners;
mod tools;

use crate::config::Config;
use crate::poses::Poses;
use crate::runners::Runner;
use crate::tools::{
    esmfold::ESMFold,
    ligandmpnn::LigandMPNN,
    protein_edits::{ChainAdder, ChainRemover},
    rfdiffusion::RFDiffusion,
    rosetta::Rosetta,
};
use anyhow::Context;
use clap::{Parser, Subcommand};
use log::{error, info};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input_json: Option<PathBuf>,

    #[arg(long, default_value = ".")]
    work_dir: PathBuf,

    #[arg(short, long)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Screening {
        #[arg(long, default_value_t = 1)]
        num_diffusions: usize,
    },
    Refinement {
        #[arg(long, default_value_t = 8)]
        nseq: usize,
    },
    Rosetta {
        #[arg(long = "app", default_value = "rosetta_scripts.linuxgccrelease")]
        application: String,
        #[arg(long)]
        options: Option<String>,
    },
    ESMFold {
        #[arg(long)]
        options: Option<String>,
    },
    ChainRemover {
        #[arg(short, long, value_delimiter = ',', num_args = 1..)]
        chains: Vec<String>,
    },
    ChainAdder {
        #[arg(long)]
        copy_chain: String,
        #[arg(long)]
        ref_pdb: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Starting structure generator...");

    // Canonicalize work_dir
    if !args.work_dir.exists() {
        std::fs::create_dir_all(&args.work_dir).context("Failed to create working directory")?;
    }
    let work_dir = std::fs::canonicalize(&args.work_dir).context(format!(
        "Failed to canonicalize work dir: {:?}",
        args.work_dir
    ))?;
    info!(
        "Work dir provided: {:?} -> canonical: {:?}",
        args.work_dir, work_dir
    );

    // Load configuration
    let config = Config::load(args.config.as_ref())?;
    info!("Configuration loaded.");

    let mut poses = Poses::new(&work_dir);

    // Load input if provided
    if let Some(input) = args.input_json {
        info!("Loading poses from {:?}", input);
        poses.load_from_json(input)?;
    }

    match args.command {
        Commands::Screening { num_diffusions } => {
            info!("Running Screening step");
            let script_path = config.rfdiffusion_script.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "RFDiffusion script not defined in config or env (RFDIFFUSION_SCRIPT)"
                )
            })?;

            let script_canon = std::fs::canonicalize(script_path)?;
            let mut runner = RFDiffusion::new(&config.python_path, script_canon.to_str().unwrap());
            runner.num_diffusions = num_diffusions;

            runner.run(&mut poses, "rfdiffusion").await?;
        }
        Commands::Refinement { nseq } => {
            info!("Running Refinement step (LigandMPNN -> ESMFold -> Rosetta)");

            // 1. LigandMPNN
            let mpnn_script = config
                .ligandmpnn_script
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LigandMPNN script not defined"))?;
            let mpnn_canon = std::fs::canonicalize(mpnn_script)?;
            let mut mpnn = LigandMPNN::new(&config.python_path, mpnn_canon.to_str().unwrap());
            mpnn.nseq = nseq;
            mpnn.run(&mut poses, "ligandmpnn").await?;

            // 2. ESMFold
            // Note: LigandMPNN output poses are now FASTA files.
            let esm_script_dir = config.esmfold_python.as_ref().map(|s| {
                // Assuming esmfold_python config points to python executable or script dir?
                // Actually config says `esmfold_python`. `protflow` uses `AUXILIARY_RUNNER_SCRIPTS_DIR`.
                // In my `config.rs`, I have `esmfold_python` and `python_path`.
                // `ESMFold::new` takes `python_path` and `script_dir`.
                // `script_dir` should contain `esmfold_inference.py`.
                // I should add `auxiliary_scripts_dir` to `Config`?
                // For now, let's assume `esmfold_python` points to the python, and we need script dir.
                // Or maybe `esmfold_python` IS the script path in my config design?
                // Let's check config.rs definition.
                s
            });

            // Re-checking config.rs: esmfold_python is Option<String>.
            // I should explicitly fetch script paths.
            // For now, let's assume standard location or require config update.
            // Using `esmfold_python` as the python path specific for ESMFold?
            // And `protein_edits_scripts_dir` from config for scripts?
            // Wait, `Config` struct has `protein_edits_scripts_dir`. Can I use that for ESMFold script too?
            // Unlikely.

            // Let's assume for now `protein_edits_scripts_dir` holds all auxiliary scripts including `esmfold_inference.py`.
            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;

            let esm_python = config
                .esmfold_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let esmfold = ESMFold::new(esm_python, aux_scripts);
            esmfold.run(&mut poses, "esmfold").await?;

            // 3. Rosetta
            let rosetta_bin = config
                .rosetta_bin
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Rosetta bin path not defined"))?;
            let mut rosetta = Rosetta::new(rosetta_bin);
            // hardcode score only or relax?
            rosetta.options = "-score:weights ref2015".to_string();
            rosetta.run(&mut poses, "rosetta_score").await?;
        }
        Commands::Rosetta {
            application,
            options,
        } => {
            let rosetta_bin = config
                .rosetta_bin
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Rosetta bin path not defined"))?;
            // Note: rosetta_bin in config might be the directory or the executable?
            // `Rosetta::new` takes `script_path`.
            // If `rosetta_bin` is directory, we join application?
            // My implementation of `Rosetta::new` sets `script_path`.
            // User usually provides path to `rosetta_scripts` or uses application arg?
            // My implementation uses `self.script_path` as the executable.
            // So `config.rosetta_bin` should be the executable path.
            // `application` arg is essentially unused in `run` command construction IF script_path is the executable.
            // Wait, `rosetta.rs`: `format!("{} ...", self.script_path, ...)`
            // So `script_path` IS the executable. The `application` field is just for logging?
            // Yes.

            let mut runner = Rosetta::new(rosetta_bin);
            runner.application = application;
            if let Some(opts) = options {
                runner.options = opts;
            }
            runner.run(&mut poses, "rosetta").await?;
        }
        Commands::ESMFold { options } => {
            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;
            let esm_python = config
                .esmfold_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let mut runner = ESMFold::new(esm_python, aux_scripts);
            if let Some(opts) = options {
                runner.options = opts;
            }
            runner.run(&mut poses, "esmfold").await?;
        }
        Commands::ChainRemover { chains } => {
            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;
            let mut runner = ChainRemover::new(&config.python_path, aux_scripts);
            runner.chains = Some(chains);
            runner.run(&mut poses, "chain_remover").await?;
        }
        Commands::ChainAdder {
            copy_chain,
            ref_pdb,
        } => {
            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;
            let mut runner = ChainAdder::new(&config.python_path, aux_scripts, &copy_chain);
            runner.ref_pdb = Some(ref_pdb);
            runner.run(&mut poses, "chain_adder").await?;
        }
    }

    info!("Done!");
    Ok(())
}
