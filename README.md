# HEAICT
This repository contains a collection of tools and scripts for catalyst design in high-entropy alloys (HEAs) and high-entropy intermetallics (HEIs).

<div align="center">
<img src="https://github.com/jchddd/heaict/blob/main/architecture.tif"><br>
</div>

## Installation
Copy the Folder `heaict` into the Lib folder under your Python directory, or run the code at the same level as Folder `heaict`.

## Requirment
|Software|Version|Requirement|
|--------|-------|-----------|
|torch|2.3.0|required|
|torch_geometric|2.5.3|required|
|pymatgen|2024.5.1|required|
|ase|3.22.1|required|
|pymoo|0.6.1.3|required|
|scikit-learn|1.4.2|required|
|periodictable|2.0.2|required|
|numpy|1.26.4|required|
|scipy|1.13.0|required|
|matplotlob|3.9.0|required|
|pandas|2.2.2|optional|
|tqdm|4.66.4|optional|
|gpflow|2.10.0|optional|
|tensorflow|2.19.0|optional|
|tensorflow-probability|0.25.0|optional|
|pygco|0.0.16|optional|
|optuna|4.0.0|optional|

Note:   
`pygco` is related to the graph cut algorithm. It can be ignored by disabling the `sparse_approx` parameter in the `ParetoDiscovery` class.   
`gpflow` and related `TensorFlow` libraries are used for Gaussian Process Regression (GPR) based on `gpflow`. If they are not installed, you may use GPR based on  the `scikit-learn` by importing the relevant models from `heaict.ml.GPR_scikit`.  
`tqdm` and `pandas` are primarily used for training data preprocessing (`heaict.data` related methods). If you perform this step using other methods or scripts, you may skip the installation of them.  
`optuna` is mainly for hyperparameter tuning in ML training, but other methods can be used as well.
# Overview
- **heaict/**[**cats**](https://github.com/jchddd/heaict/blob/main/heaict/cats):  An extended surface model that incorporates versatile sites to account for site-blocking effects. A microkinetic framework for evaluating NRR performance, which explicitly accounts for site coverage and the influence of applied potential U. And a problem class that integrates the above components, enabling the prediction of catalytic performance based on alloy composition.
- **heaict/**[**data**](https://github.com/jchddd/heaict/blob/main/heaict/data): Identifying adsorption sites, filtering anomalous structures, and building the ML training dataset.
- **heaict/**[**hea**](https://github.com/jchddd/heaict/blob/main/heaict/hea): Determining the thermodynamic stability of high entropy alloys, and analyzing the site occupation propensity in high entropy intermetallics.
- **heaict/**[**ml**](https://github.com/jchddd/heaict/blob/main/heaict/ml): Implementations of ASGCNN for adsorption energy prediction and GPR surrogate models (using either scikit-learn or gpflow) for simulating the composition–performance potential energy surface.
- **heaict/**[**mobo**](https://github.com/jchddd/heaict/blob/main/heaict/mobo): Scripts and functions related to multi-objective optimization.
- **Data**: Data and results related to the paper.
# Tutorials
- **Quick Start**: Run a bi-objective optimization for NRR HEA catalysts: [Tutorial 0 - Quick Start.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%200%20-%20Quick%20Start.ipynb)
- **Create HEA Slab**: Creating HEA slab structures through element substitution: [Tutorial 1 - Create HEA Slab.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%201%20-%20Create%20HEA%20Slab.ipynb)
- **Site Identification**: Identify adsorption configurations and detect anomalous structures: [Tutorial 2 - Site Identification.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%202%20-%20Site%20Identification.ipynb)
- **Dataset Construction**: Construct a machine learning training dataset: [Tutorial 3 - Dataset Construction.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%203%20-%20Dataset%20Construction.ipynb)
- **ML Model Training**: Train the ASGCNN model for adsorption energy prediction: [Tutorial 4 - ML Model Training.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%204%20-%20ML%20Model%20Training.ipynb)
- **Stability of HEA and HEI**: HEA property calculations and site occupancy analysis in HEI: [Tutorial 5 - Stability of HEA and HEI.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%205%20-%20Stability%20of%20HEA%20and%20HEI.ipynb)
- **Extended Surface Model**: Catalytic performance prediction based on the extended surface model, exemplified by the NRR reaction: [Tutorial 6 - Extended Surface Model.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%206%20-%20Extended%20Surface%20Model.ipynb)
- **Multi-Objective Optimization**: Additional details and considerations regarding multi-objective optimization: [Tutorial 7 - Multi-Objective Optimization.ipynb](https://github.com/jchddd/heaict/blob/main/Tutorials/Tutorial%207%20-%20Multi-Objective%20Optimization.ipynb)



# Citation
If you are interested in our work, you can read our literature, and cite us using
```
@article{10.1021/acscatal.5c09112,
    author = {Zhou, Jing and Wu, Geliang and Hu, Wangyu and Huang, Bowen and Yuan, Dingwang},
    title = {Multiobjective Computational Screening of High Entropy Alloys and Intermetallics toward High-Performance Nitrogen Reduction Catalysts},
    journal = {ACS Catalysis},
    volume = {16},
    number = {11},
    pages = {9836-9849},
    year = {2026},
    month = {05},
    abstract = {In this work, we develop a multisite approach to evaluate nitrogen reduction reaction (NRR) performance and apply it in a multiobjective optimization of high entropy alloys (HEAs) and high entropy intermetallics (HEIs) to identify candidates that simultaneously deliver high activity, strong selectivity, and enhanced structural stability. To account for competitive adsorption between N2 and H, our multisite model employs versatile sites that can host multiple adsorbates to capture site blocking effects on extended surfaces and incorporates a microkinetic model that explicitly tracks H and N2 site coverages to evaluate NRR activity and selectivity. Through a stepwise optimization, Ni8Pd35Cu7Mo35Mn15 HEA was identified as a candidate catalyst. Density functional theory (DFT) calculations reveal that its dominant active site, Mo3Mn@Pd, features a limiting potential of −0.32 V while effectively suppressing the competing hydrogen evolution reaction (HER). Electronic structure analysis reveals that the synergistic effect between Mo and Mn, coupled with the electronic modulation by Pd, is the key to its promising catalytic performance.},
    issn = {2155-5435},
    doi = {10.1021/acscatal.5c09112},
    url = {https://doi.org/10.1021/acscatal.5c09112},
    eprint = {https://pubs.acs.org/accacs/article-pdf/16/11/9836/63621781/cs-2025-09112e.pdf},
}
```
