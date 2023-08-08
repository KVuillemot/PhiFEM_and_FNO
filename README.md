# A $\phi$-FEM approach to train a Neural Operator as a fast PDE solver for variable geometries

**Authors: Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, and Killian Vuillemot**

This repository contains all codes to reproduce results of the paper "A $\phi$-FEM approach to train a Neural Operator as a fast PDE solver for variable geometries", in collaboration with Michel Duprez, Vanessa Lleras, Alexei Lozinski, and Vincent Vigon. 

The directory `./Generate_figures/` contains two Python files that generate figures in the paper. The two other directories contain all the codes for the two numerical test cases. They solve the Poisson equation with non-homogeneous Dirichlet boundary conditions on random parameterized ellipses (`./Ellipses/`) or non-parametric random shapes (`./Random_shapes/`). 

In `./Ellipses/`, there are 4 directories: 
- `data/`: it contains all the files that compose a data set of size 1500 to perform a training of the operator. 
- `main/`: it contains the most important codes. More precisely, 
    - `prepare_data.py`: all the functions needed to generate a data set, 
    - `generate_data.py`: generation of a data set, 
    - `utils.py`,
    - `utils_training.py`: implementation of the FNO, 
    - `training.ipynb`: to train the operator on a given data set,
    - `plot_results.ipynb`: to check the performance of a training on the validation sample and a new test sample, 
    - `utils_compare_methods.py`: implementation of a finite element method and of $\phi$-FEM,
    - `compare_methods.ipynb`: to compare the performance of the technique with the ones of a standard finite element method and of $\phi$-FEM (error and computation times),
    - `change_size_images.ipynb`: to evaluate the performance of a model when changing the size of the input images.

- `change_padding/`: contains the codes to compare the results of several small training for different padding techniques. 
- `change_parameters/`: contains the codes to compare the results of several small training for different values of the hyperparameters. 

In `./Random_shapes/`: 
- `data/`: contains the files that compose a data set of size 9850 to perform a training of the operator. 
- `prepare_data.py`: all the functions needed to generate a data set, 
- `generate_domains.py`: code to generate random connected domains contained in the box $(0,1)^2$, using Fourier series,
- `generate_data.py`: generation of a data set, 
- `check_data.ipynb`: check the residues of the generated data set, and remove the atypical individuals to have a clean data set,
- `utils.py`,
- `utils_training.py`: implementation of the FNO, 
- `training.ipynb`: to train the operator on a given data set,
- `plot_results.ipynb`: to check the performance of a training on the validation sample and a new test sample, 
- `utils_compare_methods.py`: implementation of a finite element method and of $\phi$-FEM,
- `compare_methods.ipynb`: to compare the performance of the technique with the ones of a standard finite element method and of $\phi$-FEM (error and computation times).

To execute these codes, you will need several packages : 
[*FEniCS*](https://fenicsproject.org/),
[*numpy*](https://numpy.org/doc/stable/index.html),
[*matplotlib*](https://matplotlib.org/),
[*tensorflow*](https://www.tensorflow.org/?hl=fr) (with GPU support),
[*seaborn*](https://seaborn.pydata.org/),
[*pandas*](https://pandas.pydata.org/),
[*vedo*](https://vedo.embl.es/#refs),
[*Cuda*](https://developer.nvidia.com/cuda-downloads). The easiest way to perform these installations is by using Anaconda. 

First check the installed drivers:   
```bash
nvidia-smi
```
If the resulting Cuda version is 11.7, you can install the environment `phifem.yml` with 

```bash 
conda env create -f phifem.yml
```

and then just type 
```bash 
source activate phifem
conda install -c conda-forge superlu_dist=6.2.0
pip3 install mpi4py==3.0.3 --no-binary mpi4py --user --force --no-cache-dir
``` 

If the result of nvidia-smi is not Cuda 11.7, you can either install another driver or create a conda environment. 

For that, use the following for Cuda $<$ 12.0 (see \[[tensorflow](https://www.tensorflow.org/install/pip?hl=fr)]) : 

```bash 
conda create --name envname python=3.8.10
source activate envname 
conda install -c conda-forge fenics mshr 
conda install -c conda-forge superlu_dist=6.2.0
pip3 install mpi4py==3.0.3 --no-binary mpi4py --user --force --no-cache-dir
conda install -c conda-forge cudatoolkit=result_nvidia-smi 
pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install tensorflow==2.12.*
pip install numpy matplotlib seaborn pandas vedo 
```


Finally, if you want to generate new shapes using level-set functions and compute the error of the FNO prediction, you will need to perform few more installations to create standard meshes. These installations are detailed in the file `install_and_use_mmg.md`. 


Remark : note that some codes need many resources to be used. In particular, the file `./Random_shapes/compare_methods.ipynb` needs more than 20Gb of memory and takes a long time to run.
