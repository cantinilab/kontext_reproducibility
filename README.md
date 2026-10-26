<p align="center">
  <img src="assets/logo.png" alt="kontext logo" width="300"/>
</p>

<h1 align="center">Kontext</h1>

<p align="center">
  Kernel-based method bridging spatial domain identification and cell-cell communication.
</p>


# Presentation

This repository contains the reproducibility notebooks associated to the Kontext publication. 
There are three notebooks : 
- `benchmark.ipynb` : to compute the benchmark results and generate the associated figures 
- `mouse_cardiac_injury.ipynb` : to run the analysis of the temporal Xenium dataset presented in the paper. 
- `metastatic_lymph_node.ipynb` : to run the analysis of the 3D OpenST dataset presented in the paper

The functions that are note in the package and that are defined specifically for these results are stored in the `src/utils` directory. 


---

## Installation

### Requirements

- Python **>= 3.12**
- A working C/C++ compiler is recommended for `numba` / `scanpy` dependencies.

### From PyPI 

```bash
pip install kontext
```

### From github 

```bash
pip install git+https://github.com/cantinilab/kontext.git
```
### Local installation

```bash
# from the kontext directory
pip install -e .