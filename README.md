# $\varphi$-FEM-FNO: a new approach to train a Neural Operator as a fast PDE solver for variable geometries

**Authors: Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, and Killian Vuillemot**

This repository contains all codes to reproduce results of the paper "$\varphi$-FEM-FNO: a new approach to train a Neural Operator as a fast PDE solver for variable geometries", in collaboration with Michel Duprez, Vanessa Lleras, Alexei Lozinski, and Vincent Vigon. 

The directory `./Generate_figures/` contains three Python files that generate figures used in the paper. The other directories contain all the codes for the numerical test cases. They solve the Poisson equation with non-homogeneous Dirichlet boundary conditions on random parameterized ellipses (`./Ellipses/`) or complex random shapes defined by Gaussian functions (`./Complex_shapes/`) and non-linear elasticity equation on rectangles plates with 5 circular holes (`./Plate/`). 

In `./Ellipses/`, there are several directories: 
- `data/`, `data_convergence_fems`, `data_test`, `data_test_phi_fem_2500`: they all contain the files the data sets used in the first test case of the paper. 
- `main/`: it contains the codes to run the following methods:
    - Geo-FNO (see [https://github.com/neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO) for the original implementation and [https://arxiv.org/abs/2207.05209](https://arxiv.org/abs/2207.05209) for the reference paper)
    - $\varphi$-FEM-FNO, 
    - $\varphi$-FEM-FNO-2, 
    - $\varphi$-FEM,
    - Standard FEM,  
    - $\varphi$-FEM-UNET, 
    - Standard-FEM-FNO.

All folders are structured in the same way:
- `compare_methods.ipynb`: to run the considered method on a test data set and evaluate the performance agains other approaches
- `generate_data.py`: generation of a data set, 
- `losses.py`: used during training, functions to compute the loss, 
- `plot_results.ipynb` (only in `phi_fem_fno`): to check the performance of a training on the validation sample and a big test sample, 
- `prepare_data.py`: all the functions needed to generate a data set, 
- `scheduler.py`: implementation of the learning rate scheduler, 
- `training.ipynb`: to run a training,
- `utils_compare_methods.py`: implementation of a finite element method and of $\varphi$-FEM.
- `utils_plot.py`: some functions to help with plotting results, 
- `utils_training.py`: implementation of the model and training loop, 
- `utils.py`: some utilities functions,


The directory `./Complex_shapes` is composed of the same files except that only $\varphi$-FEM-FNO and Standard-FEM-FNO are implemented, adapted to the case of random complex shapes. In addition, it contains the file `compute_hausdorrf.py` used to illustrate the correlation between error and Hausdorff distance. 

Finally, the directory `./Plate` is composed of the same files for $\varphi$-FEM-FNO only, to implement the third test case of the paper. 

In all the `./test_case/main/METHOD` folders, we provide the best states of the parameters for each models. Once executed, the results of the codes are stored in folders `./test_case/main/results/` and the created outputs in `test_case/main/images/`. 


To execute these codes, you will need several packages : 
- [*FEniCSX*](https://fenicsproject.org/),
- [*multiphenicsx*](https://multiphenics.github.io/index.html), only needed for the third test case,
- [*numpy*](https://numpy.org/doc/stable/index.html),
- [*matplotlib*](https://matplotlib.org/),
- [*pythorch*](https://pytorch.org/) (with GPU support),
- [*seaborn*](https://seaborn.pydata.org/),
- [*pandas*](https://pandas.pydata.org/),
- [*PyVista*](https://pyvista.org/cite/index.html),
- [*GMSH*](https://gmsh.info/) and [*PyGMSH*](https://pypi.org/project/pygmsh/)
- [*Cuda*](https://developer.nvidia.com/cuda-downloads), 
- [*meshio*](https://github.com/nschloe/meshio),
- [*scipy*](https://scipy.org/). 

The easiest way to perform these installations is by using Anaconda. 

First check the installed drivers:   
```bash
nvidia-smi
```

If the result of nvidia-smi is not Cuda 11.7, you can either install another driver or create a conda environment. 

For that, use the following (see [*pytorch*](https://pytorch.org/get-started/previous-versions/)) : 

```bash 
conda create --name phi_fem_fno python=3.12.6
conda activate phi_fem_fno 
conda install -c conda-forge fenics-dolfinx==0.8.0 mpich pyvista
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia (#11.8 for example)
pip install numpy matplotlib seaborn pandas scipy
[sudo] apt install python3-gmsh
pip install pygmsh
pip install meshio[all]
```

To install multiphenicsx, following the official documentation (https://multiphenics.github.io/installing.html):

```bash
conda install -c conda-forge gxx
pip install nanobind==1.9.2
conda install scikit-build-core[pyproject]

git clone https://github.com/multiphenics/multiphenicsx.git
cd multiphenicsx
DOLFINX_VERSION=$(python3 -c 'import dolfinx; print(dolfinx.__version__)')
git checkout dolfinx-v${DOLFINX_VERSION}
if [ -f setup.cfg ]; then
    python3 -m pip install '.[tutorials]'
else
    python3 -m pip install --check-build-dependencies --no-build-isolation '.[tutorials]'
fi
```


Finally, if you want to generate new shapes using level-set functions and compute the error of the FNO prediction (for the first and second test cases), you will need to perform few more installations to create standard meshes. These installations are detailed in the file [`install_and_use_mmg.md`](https://github.com/KVuillemot/PhiFEM_and_FNO/blob/main/install_and_use_mmg.md). 


Remark : note that some codes need many resources to be used. Everything can be executed without a GPU but it may take mych time to run. Moreover, some of the codes need more than 10Gb of RAM to be executed. 
