# The skylight project ![GitHub top language](https://img.shields.io/github/languages/top/evgkanias/sky) [![GitHub license](https://img.shields.io/github/license/evgkanias/sky)](https://github.com/evgkanias/sky/blob/main/LICENSE) ![GitHub last-commit](https://img.shields.io/github/last-commit/evgkanias/sky) ![Build Status](https://app.travis-ci.com/evgkanias/sky.svg?branch=main)

![vevoda-2022-teaser](https://raw.githubusercontent.com/evgkanias/sky/a3eb1e695daeca79d8e4c82b281208745a8f24ad/docs/vevoda_2022_infrared_sky_teaser.png)

This Python package implements three models that provide skylight information.

### 1. The Prague sky model

This model (proposed by [Vévoda P. et al., 2022](https://cgg.mff.cuni.cz/publications/infrared-skymodel-2022/); original C++ code [here](https://cgg.mff.cuni.cz/wp-content/uploads/2022/09/vevoda_2022_infrared_sky_implementation.zip)) interpolates the skylight properties by using stored data for a range of parameters, and it does not
include data for the angle of polarisation (this is calculated by using the analytical solution).

![prague-sky-model](https://raw.githubusercontent.com/evgkanias/sky/a3eb1e695daeca79d8e4c82b281208745a8f24ad/docs/prague_sky.png)

There are three available datasets that work with the Prague Sky Model, which can be downloaded form here:
* [Google Drive with the near-infrared (SWIR) dataset](https://drive.google.com/file/d/1ZOizQCN6tH39JEwyX8KvAj7WEdX-EqJl/view?usp=sharing)
  (omits data in different altitudes)
* [Google Drive with the full model](https://drive.google.com/drive/folders/19Iw0mB_UFTtbrFcojHmHc7mjc3PYe_AC?usp=sharing)
  (omits near infrared data)
* [Google Drive with the hemispherical model](https://drive.google.com/drive/folders/1R9dTbOhBXthY3y9BTI4H28acl9dJLIaV?usp=sharing)
  (omits polarisation data and different altitudes)

### 2. Analytic sky model

This model (proposed by [Wilkie A. et al., 2004](http://dx.doi.org/10.2312/EGWR/EGSR04/387-397); and
[Preetham A. J. et al., 1999](https://dl.acm.org/doi/pdf/10.1145/311535.311545)) includes radiance and polarisation
information.

![analytic-sky-model](https://raw.githubusercontent.com/evgkanias/sky/a3eb1e695daeca79d8e4c82b281208745a8f24ad/docs/analytical_sky.png)

### 3. Uniform model

This model provides the same information for any sky conditions and viewing directions, and it could be used as a
baseline model.

![uniform-sky-model](https://raw.githubusercontent.com/evgkanias/sky/a3eb1e695daeca79d8e4c82b281208745a8f24ad/docs/uniform_sky.png)

## Installation

You can easily install the package by using pip as:
```commandline
pip install git+https://github.com/evgkanias/sky.git
```

Alternatively you need to clone the GitHub repository, navigate to the main directory of the project, install the dependencies and finally
the package itself. Here is an example code that installs the package:

1. Clone this repo.
```commandline
mkdir ~/src
cd ~/src
git clone https://github.com/evgkanias/sky.git
cd sky
```
2. Install the required libraries. 
   1. using pip :
      ```commandline
      pip install -r requirements.txt
      ```

   2. using conda :
      ```commandline
      conda env create -f environment.yml
      conda activate sky-env
      ```
3. Install the package.
   1. using pip :
      ```commandline
      pip install .
      ```
   2. using conda :
      ```commandline
      conda install .
      ```
   
Note that the [pip](https://pypi.org/project/pip/) project might be needed for the above installation.

## Graphical User Interface (GUI)

If you prefer to use a graphical user interface for this package, there is a separate package that you can install and
use. This is named the [sky-gui](https://github.com/evgkanias/sky-gui) and it allows for interactive exploration of the
skylight properties for the different models. However, in this version, only the Prague Sky Model is supported.

![gui-teaser](https://raw.githubusercontent.com/evgkanias/sky/a3eb1e695daeca79d8e4c82b281208745a8f24ad/docs/gui-1.png)

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/evgkanias/sky/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Credits

The original (C++) code for this model was written by [Petr Vévoda](https://cgg.mff.cuni.cz/members/vevoda/) et al. from [Alexander Wilkie's](https://cgg.mff.cuni.cz/members/wilkie/) group in Charles University, which was part of their  [wide spectral range sky radiance model](https://cgg.mff.cuni.cz/publications/infrared-skymodel-2022/).

## Copyright

Copyright &copy; 2022, Evripidis Gkanias; Institute of Perception,
Action and Behaviour; School of Informatics; the University of Edinburgh.
