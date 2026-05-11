Getting started / installation
Thanks to Sergey Ovchinnikov,many of the features of RFdiffusion are available as a Google Colab Notebook if you would like to run it there!

There is also an official, Rosetta Commons-maintainted Docker image that you can find here. Thank you to Sergey Lyskov, Ajasja Ljubetic, and Hope Woods for creating this resource.

We strongly recommend reading this README carefully before getting started with RFdiffusion, and working through some of the examples in the Colab Notebook.

If you want to set up RFdiffusion locally, follow the steps below:

To get started using RFdiffusion, clone the repo:

```
git clone https://github.com/RosettaCommons/RFdiffusion.git
```

You'll then need to download the model weights into the RFDiffusion directory.

```
cd RFdiffusion
mkdir models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt
```

Optional:

```
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt
```

# original structure prediction weights

```
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt
```

# Conda Install SE3-Transformer
Ensure that you have either Anaconda or Miniconda installed.

You also need to install NVIDIA's implementation of SE(3)-Transformers 

Here is how to install the NVIDIA SE(3)-Transformer code:

```
conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
```

Anytime you run diffusion you should be sure to activate this conda environment by running the following command:

```
conda activate SE3nv
```

Total setup should take less than 30 minutes on a standard desktop computer. Note: Due to the variation in GPU types and drivers that users have access to, we are not able to make one environment that will run on all setups. As such, we are only providing a yml file with support for CUDA 11.1 and leaving it to each user to customize it to work on their setups. This customization will involve changing the cudatoolkit and (possibly) the PyTorch version specified in the yml file.

