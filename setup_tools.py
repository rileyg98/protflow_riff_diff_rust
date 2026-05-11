#!/usr/bin/env python3
import os
import subprocess
import argparse
import logging
import shutil
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TOOLS_DIR = os.path.join(os.getcwd(), "tools")

def run_cmd(cmd, cwd=None, shell=True):
    """Run a shell command and handle errors."""
    logging.info(f"Running: {cmd}")
    try:
        subprocess.run(cmd, cwd=cwd, shell=shell, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}\nError: {e}")
        sys.exit(1)

def check_requirements():
    """Verify system requirements are met."""
    requirements = ['conda', 'git', 'wget', 'curl']
    missing = []
    for req in requirements:
        if not shutil.which(req):
            missing.append(req)
    
    if missing:
        logging.error(f"Missing required tools: {', '.join(missing)}")
        sys.exit(1)
    logging.info("All system requirements met.")

def setup_rfdiffusion():
    """Setup RFDiffusion: Clone, weights, conda env, SE3Transformer."""
    logging.info("--- Setting up RFDiffusion ---")
    repo_dir = os.path.join(TOOLS_DIR, "RFdiffusion")
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/RosettaCommons/RFdiffusion.git {repo_dir}")
    
    models_dir = os.path.join(repo_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    weights = [
        "Base_ckpt.pt", "Complex_base_ckpt.pt", "Complex_Fold_base_ckpt.pt",
        "InpaintSeq_ckpt.pt", "InpaintSeq_Fold_ckpt.pt", "ActiveSite_ckpt.pt",
        "Base_epoch8_ckpt.pt", "Complex_beta_ckpt.pt", "RF_structure_prediction_weights.pt"
    ]
    base_url = "http://files.ipd.uw.edu/pub/RFdiffusion"
    hash_map = {
        "Base_ckpt.pt": "6f5902ac237024bdd0c176cb93063dc4",
        "Complex_base_ckpt.pt": "e29311f6f1bf1af907f9ef9f44b8328b",
        "Complex_Fold_base_ckpt.pt": "60f09a193fb5e5ccdc4980417708dbab",
        "InpaintSeq_ckpt.pt": "74f51cfb8b440f50d70878e05361d8f0",
        "InpaintSeq_Fold_ckpt.pt": "76d00716416567174cdb7ca96e208296",
        "ActiveSite_ckpt.pt": "5532d2e1f3a4738decd58b19d633b3c3",
        "Base_epoch8_ckpt.pt": "12fc204edeae5b57713c5ad7dcb97d39",
        "Complex_beta_ckpt.pt": "f572d396fae9206628714fb2ce00f72e",
        "RF_structure_prediction_weights.pt": "1befcb9b28e2f778f53d47f18b7597fa"
    }

    for w in weights:
        w_path = os.path.join(models_dir, w)
        if not os.path.exists(w_path):
            url = f"{base_url}/{hash_map[w]}/{w}"
            run_cmd(f"wget {url} -O {w_path}")

    # Conda environment
    run_cmd("conda env create -f env/SE3nv.yml", cwd=repo_dir)
    
    # SE3Transformer and module install
    setup_cmds = [
        "source $(conda info --base)/etc/profile.d/conda.sh && conda activate SE3nv && "
        "cd env/SE3Transformer && pip install --no-cache-dir -r requirements.txt && "
        "python setup.py install && cd ../.. && pip install -e ."
    ]
    run_cmd(setup_cmds[0], cwd=repo_dir)
    logging.info("RFDiffusion setup complete.")

def setup_esmfold():
    """Setup ESMFold: Clone one-command-install, create env."""
    logging.info("--- Setting up ESMFold ---")
    repo_dir = os.path.join(TOOLS_DIR, "One-command-install-ESMfold")
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/mabr3112/One-command-install-ESMfold.git {repo_dir}")
    
    run_cmd("conda env create -f environment.yml -n esmfold", cwd=repo_dir)
    logging.info("ESMFold setup complete.")

def setup_ligandmpnn():
    """Setup LigandMPNN: Clone, weights, conda env."""
    logging.info("--- Setting up LigandMPNN ---")
    repo_dir = os.path.join(TOOLS_DIR, "LigandMPNN")
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/dauparas/LigandMPNN.git {repo_dir}")
    
    run_cmd("bash get_model_params.sh './model_params'", cwd=repo_dir)
    run_cmd("conda create -n ligandmpnn python=3.11 -y && "
            "source $(conda info --base)/etc/profile.d/conda.sh && conda activate ligandmpnn && "
            "pip install -r requirements.txt", cwd=repo_dir)
    logging.info("LigandMPNN setup complete.")

def setup_localcolabfold():
    """Setup LocalColabFold: Clone, pixi install."""
    logging.info("--- Setting up LocalColabFold ---")
    repo_dir = os.path.join(TOOLS_DIR, "localcolabfold")
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/yoshitakamo/localcolabfold.git {repo_dir}")
    
    if not shutil.which("pixi"):
        run_cmd("curl -fsSL https://pixi.sh/install.sh | sh")
        os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.pixi/bin")

    run_cmd("pixi install && pixi run setup", cwd=repo_dir)
    logging.info("LocalColabFold setup complete.")

def setup_rosetta():
    """Setup Rosetta: Download bundle, extract."""
    logging.info("--- Setting up Rosetta 3.13 ---")
    rosetta_url = "https://downloads.rosettacommons.org/downloads/academic/3.13/rosetta_bin_linux_3.13_bundle.tgz"
    target_file = os.path.join(TOOLS_DIR, "rosetta_bin_linux_3.13_bundle.tgz")
    
    if not os.path.exists(target_file):
        run_cmd(f"wget {rosetta_url} -O {target_file}")
    
    run_cmd(f"tar -xzf {target_file} -C {TOOLS_DIR}")
    logging.info("Rosetta setup complete.")

def main():
    global TOOLS_DIR
    parser = argparse.ArgumentParser(description="Automate setup of protein design tools.")
    parser.add_argument("--tools", nargs="+", choices=["rfdiffusion", "esmfold", "ligandmpnn", "localcolabfold", "rosetta"],
                        help="Select specific tools to setup. Default is all.")
    parser.add_argument("--dir", default=TOOLS_DIR, help=f"Directory to install tools into (default: {TOOLS_DIR})")
    
    args = parser.parse_args()
    TOOLS_DIR = os.path.abspath(args.dir)
    os.makedirs(TOOLS_DIR, exist_ok=True)

    check_requirements()

    setup_map = {
        "rfdiffusion": setup_rfdiffusion,
        "esmfold": setup_esmfold,
        "ligandmpnn": setup_ligandmpnn,
        "localcolabfold": setup_localcolabfold,
        "rosetta": setup_rosetta
    }

    tools_to_setup = args.tools if args.tools else setup_map.keys()

    for tool in tools_to_setup:
        try:
            setup_map[tool]()
        except Exception as e:
            logging.error(f"Failed to setup {tool}: {e}")

if __name__ == "__main__":
    main()
