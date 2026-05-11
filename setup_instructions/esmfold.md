# One-command-install-ESMfold
Install ESMFold into a fresh conda environment with just one line and a .yml file.

First, clone this repository:
```
git clone https://github.com/mabr3112/One-command-install-ESMfold.git
```

To install the environment, run:
```
cd One-command-install-ESMfold
conda env create -f environment.yml -n your_env_name
```

To test if the installation of ESMFold was successful, you can run a little test-script that is stored in the examples.
Activate the environment, go to the examples folder and execute the run_test.sh script:

```
conda activate your_env_name
cd examples
bash run_test.sh
```

This should produce the folded structure of the test-peptide in the outputs folder.
You can also feel free to use and adapt the prediction script to your liking.