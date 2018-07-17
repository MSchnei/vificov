# vificov
<img src="logo/vificcov_logo.png" width=512 align="center" />

**Vi** sual **Fi** eld **Cov** erage (ViFiCov) visualization in python.

This package takes as input parameters that describe a 2D Gaussian model for a set of voxels and returns projections of the voxels' visual field coverage.

## Installation

For installation, follow these steps:

### Option A: install via pip
```bash
pip install vificov
```
### Option B: install from github repository

0. (Optional) Create conda environment
```bash
conda create -n env_vificov python=3.6
source activate env_vificov
conda install pip
```

1. Clone repository
```bash
git clone https://github.com/MSchnei/vificov.git
```
2. Install vificov with pip
```bash
pip install /path/to/cloned/vificov
```

## How to use

### 1. Adjust the csv file
Adjust the information in the config_default.csv file, such that the provided information is correct.
It is recommended to make a specific copy of the csv file for every subject and project.

### 2. Run pyprf_motion
Open a terminal and run
```
vificov -config path/to/custom_config.csv
```

## References
This application is based on the following work:

* Le, R., Witthoft, N., Ben-Shachar, M., & Wandell, B. (2016). The field of view available to the cortical reading circuitry. BioRxiv, 17, 1–19. https://doi.org/https://doi.org/10.1101/069369
* Kok, P., Bains, L. J., Van Mourik, T., Norris, D. G., & De Lange, F. P. (2016). Selective activation of the deep layers of the human primary visual cortex by top-down feedback. Current Biology, 26(3), 371–376. https://doi.org/10.1016/j.cub.2015.12.038

## License
The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
