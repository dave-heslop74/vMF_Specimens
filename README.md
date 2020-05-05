## Welcome to vMF_Specimens
This repository contains the *vMF_Specimens* package, which was created to accompany the publication:

D. Heslop & A. P. Roberts (2020). Uncertainty propagation in hierarchical paleomagnetic reconstructions. *Journal of Geophysical Research (Submitted)*.

The package provides Python (v3.7) code and an interactive Jupyter notebook to perform uncertainty propagation when averaging Fisher (1953) distributions as applied to paleomagnetic specimen directions. 

### Running on the Cloud
The simplest way to run the interactive Jupyter notebook is on the Binder cloud, using this button: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dave-heslop74/vMF_Specimens/master?urlpath=%2Fapps%2FvMF_Specimens.ipynb)

### Running locally
The *vMF_Specimens* package can be installed locally using the ```pip``` command:

```pip install git+https://github.com/dave-heslop74/vMF_Specimens.git```

Package dependencies can be found in the ```requirements.txt``` file included in this repository and the ```postBuild``` file provides commands to setup the interactive notebook.

An example Jupyter notebook ```vMF_Specimens.ipynb``` is contained in this repository and demonstrates how the package can be used.
