# LocalColabFold

[ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) on your local PC (or macOS). See also [ColabFold repository](https://github.com/sokrypton/ColabFold).

## What is LocalColabFold?

LocalColabFold is an installer script designed to make ColabFold functionality available on users' local machines. It supports wide range of operating systems, such as Windows 10 or later (using Windows Subsystem for Linux 2), macOS, and Linux.

> [!NOTE]
> If you only intend to predict a small number of naturally occurring proteins, I recommend using [ColabFold notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) or downloading structures from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) or [UniProt](https://www.uniprot.org/). LocalColabFold is suitable for more advanced applications, such as batch processing of structure predictions for natural complexes, non-natural proteins, or predictions with manually specified MSAs/templates.**

## Advantages of LocalColabFold

- **Structure inference and relaxation will be accelerated if your PC has Nvidia GPU and CUDA drivers.**
- **No Time out (90 minutes and 12 hours)**
- **No GPU limitations**
- **NOT necessary to prepare the large database required for native AlphaFold2**.

## Note (May 21, 2024)

- Since current GPU-supported jax > 0.4.26 requires CUDA 12.1 or later and cudnn 9, please upgrade or install your CUDA driver and cudnn. CUDA 12.4 is recommended.

## Note (Jan 30, 2024)

- ColabFold now upgrade to 1.5.5 (compatible with AlphaFold 2.3.2). Now LocalColabFold requires **CUDA 12.1 or later**. Please update your CUDA driver if you have not done so.
- Now (Local)ColabFold can predict protein structures without connecting the Internet. Use [`setup_databases.sh`](https://github.com/sokrypton/ColabFold/blob/main/setup_databases.sh) script to download and build the databases (See also [ColabFold Downloads](https://colabfold.mmseqs.com/)). An instruction to run `colabfold_search` to obtain the MSA and templates locally is written in [this comment](https://github.com/sokrypton/ColabFold/issues/563).

## New Updates

- 15Jan2026, Use [pixi](https://pixi.prefix.dev/latest) to install localcolabfold easily.
- 30Jan2024, ColabFold 1.5.5 (Compatible with AlphaFold 2.3.2). Now LocalColabFold requires **CUDA 12.1 or later**. Please update your CUDA driver.
- 30Apr2023, Updated to use python 3.10 for compatibility with Google Colaboratory.
- 09Mar2023, version 1.5.1 released. The base directory has been changed to `localcolabfold` from `colabfold_batch` to distinguish it from the execution command.
- 09Mar2023, version 1.5.0 released. See [Release v1.5.0](https://github.com/YoshitakaMo/localcolabfold/releases/tag/v1.5.0)
- 05Feb2023, version 1.5.0-pre released.
- 16Jun2022, version 1.4.0 released. See [Release v1.4.0](https://github.com/YoshitakaMo/localcolabfold/releases/tag/v1.4.0)
- 07May2022, **Updated `update_linux.sh`.** See also [How to update](#how-to-update). Please use a new option `--use-gpu-relax` if GPU relaxation is required (recommended).
- 12Apr2022, version 1.3.0 released. See [Release v1.3.0](https://github.com/YoshitakaMo/localcolabfold/releases/tag/v1.3.0)
- 09Dec2021, version 1.2.0-beta released. easy-to-use updater scripts added. See [How to update](#how-to-update).
- 04Dec2021, LocalColabFold is now compatible with the latest [pip installable ColabFold](https://github.com/sokrypton/ColabFold#running-locally). In this repository, I will provide a script to install ColabFold with some external parameter files to perform relaxation with AMBER. The weight parameters of AlphaFold and AlphaFold-Multimer will be downloaded automatically at your first run.

## Installation

### For Linux (Ubuntu 22.04 or later is recommended)

1. Make sure `curl`, `git`, and `wget` commands are already installed on your PC. If not present, you need install them at first. For Ubuntu, type `sudo apt -y install curl git wget`.
2. Make sure your Cuda compiler driver is **11.8 or later** (the latest version 12.4 is preferable). If you don't have a GPU or don't plan to use a GPU, you can skip this step:

   ```shell
    $ nvcc --version
      nvcc: NVIDIA (R) Cuda compiler driver
      Copyright (c) 2005-2022 NVIDIA Corporation
      Built on Wed_Sep_21_10:33:58_PDT_2022
      Cuda compilation tools, release 11.8, V11.8.89
      Build cuda_11.8.r11.8/compiler.31833905_0
   ```

   DO NOT use `nvidia-smi` to check the version.<br>See [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) if you haven't installed it.

3. Install `pixi` package manager by following the instructions at [pixi installation page](https://pixi.prefix.dev/latest/installation).,

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

4. Clone this repository and run the installation script:

   ```bash
    git clone https://github.com/yoshitakamo/localcolabfold.git
    cd localcolabfold
    pixi install && pixi run setup
   ```

   Localcolabfold will be installed in the `/path/to/localcolabfold/.pixi/envs/default/` directory.

5. Use `run_colabfoldbatch_sample.sh` as a sample script to run `colabfold_batch`. Make sure to set the correct PATH (`/path/to/localcolabfold/.pixi/envs/default/bin`) in the shell script.

6. Run the script to start the structure prediction:

   ```bash
   bash run_colabfoldbatch_sample.sh
   ```
