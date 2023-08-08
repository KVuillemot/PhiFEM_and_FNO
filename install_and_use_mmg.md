# Install and use pymedit to create standard meshes from level-set functions with FEniCS

**Authors: Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, and Killian Vuillemot**

To generate meshes using level-set functions and use them with FEniCS, it is required to install: [mmg](https://www.mmgtools.org/), [Medit](https://github.com/ISCDtoolbox/Medit), [pymedit](https://gitlab.com/florian.feppon/pymedit), and [meshio](https://github.com/nschloe/meshio). All the installations are presented in the following. Moreover, two examples of use are included in this file. 

The 3 first parts are taken from: https://github.com/MmgTools/Mmg-Day-2018_TP/tree/master.

## Install mmg 
Clone the repo and build the application:
```bash
git clone https://github.com/MmgTools/mmg.git
cd mmg
mkdir build
cd build
sudo cmake ..
sudo make
sudo make install
```

If you want to add to your path, in your bashrc add: 
```bash
cd bin 
echo PATH=$PATH:$(pwd) >> ~/.bashrc
source ~/.bashrc
```

## Install Medit 

Clone the repo and build the application:

```bash
git clone https://github.com/ISCDtoolbox/Medit.git
cd Medit
mkdir build
cd build
sudo cmake ..
sudo make
sudo make install
```

If you want to add to your path, in your bashrc add: 
```bash
echo PATH=$PATH:$(pwd) >> ~/.bashrc
source ~/.bashrc
```

## Some graphic packages for linux

```bash
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libxi-dev
sudo apt-get install -y libxmu-dev
```

## Install and configure pymedit

If you use a conda env, activate it first. Then:

- Install pymedit:
```bash
pip install pymedit
```

- Configure pymedit:
    - Go to your site-packages directory, for example:

    ```bash
    cd /home/username/.local/lib/python3.XX/site-packages/pymedit
    ```
    or with conda env: 
    ```bash
    cd /home/username/anaconda3/envs/envname/lib/python3.XX/site-packages/pymedit
    ```
    
    - Open the file `abstract.py`
    - Change the line 816: modify "END" by "End" 
    - If you want to remove the details: 
        - In `abstract.py`: comment lines 138-140, 145-147, 153-156
        - In `mesh.py`: comment line 321, 
        - In `mesh3D.py`: comment line 368.

## Install meshio
Simply use: 
```bash 
pip install meshio
```


## Examples of use

### Creation of a level-set defined mesh, without pymedit

- Create a unit square mesh with FEniCS: 
```python
import dolfin as df

mesh = df.UnitSquareMesh(100, 100)
df.File('boxmesh.xml') << mesh
```

- Convert the mesh: 
```bash
meshio convert boxmesh.xml boxmesh.mesh
```

- Generate an array of the level-set values (same size as the number of vertices in your mesh) and flatten it, for example: 
```python
import numpy as np
import os

n = 101
X, Y = np.meshgrid(np.linspace(0.0, 1.0, n),
                   np.linspace(0.0, 1.0, n))
phi = (X - 0.5) ** 2 + (Y - 0.5) ** 2 - (0.3) ** 2
phi = phi.flatten()
```

- Save your function: 
```python
f = open(
    'phi.txt',
    'w',
)
f.write('MeshVersionFormatted 2 \n')
f.write('\n')
f.write('Dimension 2 \n')
f.write('\n')
f.write('SolAtVertices \n')
f.write(f'{np.shape(phi)[0]} \n')
f.write('1 1 \n')
f.write('\n')

for i in range(len(phi)):
    f.write(f'{phi[i]}\n')

f.write('\n')
f.write('End')


os.rename('phi.txt', 'phi.sol')
```

- Finally generate the mesh: 

```bash
mmg2d_O3 boxmesh -sol phi.sol -ls -nr -nsd 3 -hmax VALUE
```

### Creation of a level-set defined mesh, without pymedit

We use the same example as before. 

- Generate an array of the level-set values (same size as the number of vertices in your mesh), for example: 
```python
import dolfin as df 
import numpy as np
import os
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)

n = 101
X, Y = np.meshgrid(np.linspace(0.0, 1.0, n),
                   np.linspace(0.0, 1.0, n))
phi = (X - 0.5) ** 2 + (Y - 0.5) ** 2 - (0.3) ** 2
```

- Then, you will have to create a box and set `phi` as a $\mathbb{P}^1$ function on the box: 

```python
n = np.shape(phi)[0]
M = square(n - 1, n - 1)
M.debug = 4 

phi = phi.flatten("F")
phiP1 = P1Function(M, phi) # Setting a P1 level set function
```

- Finally, it only remains to generate the subdomain according to the level-set function, mesh it, and generate and save it in a FEniCS usable format: 


```python
newM = mmg2d(
    M,
    hmax=hmax,
    hmin=hmin,
    hgrad=None,
    sol=phiP1,
    ls=True,
    verb=0,
)

Mf = trunc(newM, 3) # Trunc the negative subdomain of the level set
Mf.save("Thf.mesh")  # Saving in binary format
command = "meshio convert Thf.mesh Thf.xml"
os.system(command) # Convert and save in xml format
mesh = df.Mesh("Thf.xml") # Read the mesh with FEniCS
```
