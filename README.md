# A $\phi$-FEM approach to train a Neural Operator as a fast PDE solver for variable geometries

**Authors: Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, and Killian Vuillemot**

This repository contains all codes to reproduce results of the paper "A $\phi$-FEM approach to train a Neural Operator as a fast PDE solver for variable geometries", in collaboration with Michel Duprez, Vanessa Lleras, Alexei Lozinski, and Vincent Vigon. 

The directory `./Generate_figures/` contains two Python files that generate figures used in the paper. The two other directories contain all the codes for the two numerical test cases. They solve the Poisson equation with non-homogeneous Dirichlet boundary conditions on random parameterized ellipses (`./Ellipses/`) or complex random shapes (`./Random_shapes/`). 

In `./Ellipses/`, there are 3 directories: 
- `data/`: it contains all the files that compose a data set of size 1800 to perform a training of the operator. 
- `main/`: it contains the most important codes. More precisely, 
    - `change_size_images.ipynb`: to evaluate the performance of a model when changing the size of the input images.
    - `compare_loss_levels.ipynb`: to compare the performance of $\phi$-FEM-FNO different levels of loss function.
    - `compare_methods.ipynb`: to compare the performance of the of $\phi$-FEM-FNO approach with the ones of a standard finite element method and of $\phi$-FEM (error and computation times),
    - `generate_data.py`: generation of a data set, 
    - `losses.py`: used during training, functions to compute the loss, 
    - `plot_results.ipynb`: to check the performance of a training on the validation sample and a test sample, 
    - `prepare_data.py`: all the functions needed to generate a data set, 
    - `scheduler.py`: implementation of the learning rate scheduler, 
    - `training.py`: implementation of the FNO and of the training loop, 
    - `utils.py`,
    - `utils_compare_methods.py`: implementation of a finite element method and of $\phi$-FEM.

- `main_standard_fem/`: contains the codes to generate a dataset and train a FNO using standard-FEM solutions, interpolated on a cartesian grid of resolution $64 \times 64$. 

The directory `./Random_shapes` is composed of the same files, adapted to the test case of random complex shapes. In addition, it contains the file `generate_domains.py` used to create the complex random domains. 

The data set of size 8733, used to train the operator for the case of random shapes is available at : [https://figshare.com/articles/dataset/Data_set_/23905671](https://figshare.com/articles/dataset/Data_set_/23905671). To use it, just download all and place the files in the directory `./Random_shapes/data_8733/`.

Moreover, in the `main` folders, we provide the best state of the parameters for each model. 

To execute these codes, you will need several packages : 
[*FEniCS*](https://fenicsproject.org/),
[*numpy*](https://numpy.org/doc/stable/index.html),
[*matplotlib*](https://matplotlib.org/),
[*pythorch*](https://pytorch.org/) (with GPU support),
[*seaborn*](https://seaborn.pydata.org/),
[*pandas*](https://pandas.pydata.org/),
[*vedo*](https://vedo.embl.es/#refs),
[*Cuda*](https://developer.nvidia.com/cuda-downloads). 

The easiest way to perform these installations is by using Anaconda. 

First check the installed drivers:   
```bash
nvidia-smi
```
If the resulting Cuda version is 11.7, you can install the environment `fenics_torch.yml` with 

```bash 
conda env create -f fenics_torch.yml
```

and then just type 
```bash 
source activate fenics_torch
conda install -c conda-forge superlu_dist=6.2.0
pip3 install mpi4py==3.0.3 --no-binary mpi4py --user --force --no-cache-dir
``` 

If the result of nvidia-smi is not Cuda 11.7, you can either install another driver or create a conda environment. 

For that, use the following (see [*pytorch*](https://pytorch.org/get-started/previous-versions/)) : 

```bash 
conda create --name envname python=3.11
conda activate envname 
conda install -c conda-forge fenics mshr 
conda install -c conda-forge superlu_dist=6.2.0
sudo pip3 install mpi4py==3.0.3 --no-binary mpi4py --user --force --no-cache-dir
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX (XXX = 117, 118, ...)
pip install numpy matplotlib seaborn pandas vedo 
```


Finally, if you want to generate new shapes using level-set functions and compute the error of the FNO prediction, you will need to perform few more installations to create standard meshes. These installations are detailed in the file `install_and_use_mmg.md`. 


Remark : note that some codes need many resources to be used. Everything can be executed without a GPU but it may take mych time to run. Moreover, some of the codes need more than 10Gb of RAM to be executed. 
