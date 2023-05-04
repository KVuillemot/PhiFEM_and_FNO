# Deep physics with varying geometry : a $\phi$-FEM approach

This repository contains all codes to reproduce results of the paper "Deep physics with varying geometry : a $\phi$-FEM approach", collaboration with Michel Duprez, Vanessa Lleras, Alexei Lozinski and Vincent Vigon. 

Each folder correspond to one test case of the paper. 

- `Dirichlet_homogeneous` contains the codes for the first test case : Poisson equation with homogeneous Dirichlet boundary conditions on random parameterized ellipses,
- `Dirichlet_non_homogeneous`, the codes for the second test case : Poisson equation with non-homogeneous Dirichlet boundary conditions on random parameterized ellipses,
- `Random_shapes`, the codes the third test case : Poisson equation with homogeneous Dirichlet boundary conditions, on non-parametric random shapes. 

They are all composed of the same files : 
- `compare_methods.ipynb` : a notebook containing the codes to compare the results of a trained operator, a standard finite element method and $\phi$-FEM; 
- `generate_data.py` : a python file to generate a dataset for the corresponding test case;
- `plot_results.ipynb` : a notebook with the codes to get results on the validation sample;  
- `prepare_data.py` : some utils to generate the dataset; 
- `training.ipynb` : the notebook containing the code to perform a training of the operator for each test case;
- `utils_compare_methods.py` : file containing function to compare FNO, $\phi$-FEM and standard FEM; 
- `utils_training.py` : principal functions used during a training.
- `utils.py` : file containing severall utils functions;

Moreover, for the first test case, you will find the file `change_size_images.ipynb`, used to compare results for different sizes of input of the FNO.
Finally, for the third test case, you can find the file `domain_generator.py`, used to generate random shapes and level-set functions. 

In addition, each folder contains a dataset and pretrained operators.


To execute these codes, you will need several packages : *FEniCS*, *numpy*, *matplotlib*, *tensorflow* (with GPU support), *seaborn*, *pandas* and *vedo*. The easiest way to perform these installations is by using Anaconda. If you have Cuda 11.7 compatible drivers installed on your computer, see the result of  
```bash
nvidia-smi
```
you can install the environment `phifem.yml` with 

```bash 
conda env create -f phifem.yml
```

and then just type 
```bash 
source activate phifem
``` 

If the result of nvidia-smi is not Cuda 11.7, you can either install another driver or create a conda environment. 

For that, use the following for Cuda $<$ 12.0 (see \[[tensorflow](https://www.tensorflow.org/install/pip?hl=fr)]) : 

```bash 
conda create --name envname python=3.9 
source activate envname 
conda install -c conda-forge fenics mshr 
conda install -c conda-forge cudatoolkit=result_nvidia-smi 
pip install nvidia-cudnn-cu11=8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install tensorflow==2.12.*
pip install numpy matplotlib seaborn pandas vedo 
```

Remark : note that some of the codes need many ressources to be used. In particular, the file `./Random_shapes/compare_methods.ipynb` needs approximately 30Gb of memory and takes a long time to run.
