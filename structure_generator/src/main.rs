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
use log::info;
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
        #[arg(long, default_value_t = 1)]
        cycles: usize,
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
            info!(
                "Running Screening workflow (num_diffusions: {})",
                num_diffusions
            );

            // 1. RFDiffusion
            let rfdiff_script = config
                .rfdiffusion_script
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("RFDiffusion script not defined"))?;
            let rfdiff_canon = std::fs::canonicalize(rfdiff_script)?;
            let rfdiff_python = config
                .rfdiffusion_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let mut rfdiff = RFDiffusion::new(rfdiff_python, rfdiff_canon.to_str().unwrap());
            rfdiff.num_diffusions = num_diffusions;
            rfdiff.run(&mut poses, "screening_rfdiffusion").await?;

            // 2. ChainAdder (renumber/copy ref)
            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;
            let ed_python = config
                .protein_edits_python
                .as_ref()
                .unwrap_or(&config.python_path);

            let adder = ChainAdder::new(ed_python, aux_scripts, "Z");
            adder.run(&mut poses, "screening_chain_adder").await?;

            // 3. LigandMPNN
            let mpnn_script = config
                .ligandmpnn_script
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LigandMPNN script not defined"))?;
            let mpnn_canon = std::fs::canonicalize(mpnn_script)?;
            let mpnn_python = config
                .ligandmpnn_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let mut mpnn = LigandMPNN::new(mpnn_python, mpnn_canon.to_str().unwrap());
            mpnn.nseq = 1; // Default for screening
            mpnn.run(&mut poses, "screening_mpnn").await?;

            // 4. Rosetta (bbopt)
            let rosetta_bin = config
                .rosetta_bin
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Rosetta bin path not defined"))?;
            let rosetta = Rosetta::new(rosetta_bin);
            rosetta.run(&mut poses, "screening_rosetta").await?;

            // 5. ESMFold
            let esm_python = config
                .esmfold_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let esmfold = ESMFold::new(esm_python, aux_scripts);
            esmfold.run(&mut poses, "screening_esmfold").await?;

            // 6. Filtering
            poses.filter_poses_by_value("screening_esmfold_plddt", 70.0, ">=");
            poses.filter_poses_by_rank(10, "screening_rosetta_total_score", true);
        }
        Commands::Refinement { cycles, nseq } => {
            info!(
                "Running Refinement step ({} cycles, nseq: {})",
                cycles, nseq
            );

            let mpnn_script = config
                .ligandmpnn_script
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LigandMPNN script not defined"))?;
            let mpnn_canon = std::fs::canonicalize(mpnn_script)?;
            let mpnn_python = config
                .ligandmpnn_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let mut mpnn = LigandMPNN::new(mpnn_python, mpnn_canon.to_str().unwrap());
            mpnn.nseq = nseq;

            let aux_scripts = config
                .protein_edits_scripts_dir
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Auxiliary scripts dir not defined"))?;
            let esm_python = config
                .esmfold_python
                .as_ref()
                .unwrap_or(&config.python_path);
            let esmfold = ESMFold::new(esm_python, aux_scripts);

            let rosetta_bin = config
                .rosetta_bin
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Rosetta bin path not defined"))?;
            let mut rosetta = Rosetta::new(rosetta_bin);

            for i in 1..=cycles {
                info!("--- Refinement Cycle {} ---", i);

                // 1. LigandMPNN
                info!("Cycle {}: Running LigandMPNN", i);
                mpnn.run(&mut poses, &format!("cycle_{}_mpnn", i)).await?;

                // 2. ESMFold
                info!("Cycle {}: Running ESMFold", i);
                esmfold
                    .run(&mut poses, &format!("cycle_{}_esmfold", i))
                    .await?;

                // 3. Rosetta
                info!("Cycle {}: Running Rosetta", i);
                rosetta.options = "-score:weights ref2015".to_string();
                rosetta
                    .run(&mut poses, &format!("cycle_{}_rosetta", i))
                    .await?;

                // Filter? For now just keep all.
                // poses.filter_poses_by_rank(5, &format!("cycle_{}_rosetta_total_score", i), true);
            }
        }
        Commands::Rosetta {
            application,
            options,
        } => {
            let rosetta_bin = config
                .rosetta_bin
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Rosetta bin path not defined"))?;

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
            let mut runner = ChainRemover::new(
                config
                    .protein_edits_python
                    .as_ref()
                    .unwrap_or(&config.python_path),
                aux_scripts,
            );
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
            let mut runner = ChainAdder::new(
                config
                    .protein_edits_python
                    .as_ref()
                    .unwrap_or(&config.python_path),
                aux_scripts,
                &copy_chain,
            );
            runner.ref_pdb = Some(ref_pdb);
            runner.run(&mut poses, "chain_adder").await?;
        }
    }

    info!("Done!");
    Ok(())
}
