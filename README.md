# Skylight Simulation ![GitHub top language](https://img.shields.io/github/languages/top/evgkanias/sky) [![GitHub license](https://img.shields.io/github/license/evgkanias/sky)](https://github.com/evgkanias/sky/blob/main/LICENSE) ![GitHub last-commit](https://img.shields.io/github/last-commit/evgkanias/sky) ![Build Status](https://app.travis-ci.com/evgkanias/sky.svg?branch=main)

Python simulations of the skylight information based on the
[Vevoda et al. (2022)](https://cgg.mff.cuni.cz/publications/infrared-skymodel-2022/) Prague Sky Model.

![vevoda_2022_teaser](docs/vevoda_2022_infrared_sky_teaser.png)

## Installation

In order to install the package and reproduce the results of the manuscript you need to clone
the code, navigate to the main directory of the project, install the dependencies and finally
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

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/evgkanias/sky/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2022, Evripidis Gkanias; Institute of Perception,
Action and Behaviour; School of Informatics; the University of Edinburgh.