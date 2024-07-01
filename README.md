# dendritic-spine-shape-analysis
Dendritic spine analysis tool for dendritic spine image segmentation, dendritic spine morphologies extraction,analysis and clustering.

## System requirements
- Windows 8 or newer
- Minimum 1GB RAM
- Minimum 6 GB disk space

## Install
1. Download code
2. Unzip [CGAL files](https://github.com/pv6/cgal-swig-bindings/releases/download/python-build/CGAL.zip) next to code, e.g. `PATH_TO_CODE\CGAL\...`
3. Install [Anaconda](https://www.anaconda.com/)
4. Open Anaconda
5. Execute
```cmd
cd PATH_TO_CODE
conda create --name spine-analysis -c conda-forge --file requirements.txt -y python=3.8
```
## Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate spine-analysis
jupyter notebook
```

## Example datasets
### example_dendrite
Dataset consist of a .tif image of a dendrite, 22 polygonal
meshes of dendrite spines and a dendrite polygonal mesh computed with `dendrite-segmentation.ipynb` notebook. This 
example dataset provides a demonstration of dendrite image segmentation performance and functionality of the 
methods from the `Utilities.ipynb` notebook.
### 0.025 0.025 0.1 dataset
Dataset consists of 270 polygonal meshes of dendrite spines related to 54 dendrites and of 54
polygonal   meshes   for   dendrites computed with `dendrite-segmentation.ipynb` notebook.  A dataset subdirectory 
named "manual_classification" contains the expert markups from 8 people obtained using the `spine-manual-classification.ipynb` 
and the results of merging the classifications to obtain a consensus classification. This  example dataset provides a 
demonstration of dendrite spines classification and clustering performance and functionality of the 
methods from the `Utilities.ipynb` notebook.


## Build CGAL for other platforms

If cgal don't work, build it from sources:
in [tutorial](https://gist.github.com/BJTerry/e561b956d963a2fe4c4623fb06f49266) 
1. download zip https://github.com/pv6/cgal-swig-bindings and unpack
2. install cgal and swing libraries via terminal (brew manager for macos, apt or dnf for linux)
3. cd cgal-swig-bindings-main
4. conda activate spine-analysis
5. cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCGAL_DIR=/usr/local/opt/cgal -DBUILD_PYTHON=ON -DBUILD_JAVA=OFF -DPYTHON_LIBRARIES=~/miniconda3/envs/spine-analysis/bin/python - here specify path to cgal lib and python in your virtual env 
6. make -j 4
7. cp -r build-python/CGAL (...path)/dendritic-spine-shape-analysis/CGAL


## if ipywidgets doesn't work: 
try to run
- conda install jupyterlab
- jupyter nbextension enable --py widgetsnbextension
- jupyter labextension install @jupyter-widgets/jupyterlab-manager
- jupyter nbextension enable --py --sys-prefix pythreejs
- conda install -c conda-forge 'nodejs>=12'
- install meshplot with pip