# Oh Bemn! Ocean Surface Wave Boundary Element Method

## Installation

1. Install dependencies: `gfortran=13` and `rust`. Can also be installed via mamba after
   you have set up and activated the environment below (it may be necessary to
   symling f95 to conda gfortran to make sure gcc and gfortran match).

2. Set up and activate the conda/mamba environment:

```sh
$ mamba env create -f environment.yml
$ conda activate ohbemn
```

3. Build and install this package into the conda/mamba environment:

```sh
$ pip install -e .
```

## Inspiration

* https://github.com/fjargsto/abem
* https://github.com/lzhw1991/AcousticBEM
* https://www.boundary-element-method.com/
* https://www.boundary-element-method.com/helmholtz/index.htm
* https://wikiwaves.org/Boundary_Element_Method_for_a_Fixed_Body_in_Finite_Depth
