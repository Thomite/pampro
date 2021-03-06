## pampro - physical activity monitor processing

### Introduction

pampro is a software project for the systematic analysis of physical activity data collected in epidemiological studies. The ultimate goal is to provide a turn-key solution for physical activity data analysis, replicating published methodologies in a monitor-agnostic framework.

### New in version v0.4.0 [![DOI](https://zenodo.org/badge/18706328.svg)](https://zenodo.org/badge/latestdoi/18706328)

The main feature of this version is a [HDF5](https://www.hdfgroup.org/HDF5/) module, which provides functions to store pampro objects inside HDF5 containers. This has numerous advantages, the most important being incredibly fast loading times for triaxial Time Series data (<10 seconds for a week long 100 Hz file). It also provides a mechanism to automatically cache the results of various time consuming functions, such as nonwear detection and autocalibration; this means if a second analysis is performed, the results will be instantly recalled from storage. In the future, it will become a low-memory solution for pampro analyses, which will minimise the RAM footprint by writing results directly to disk. A demonstration of how to use the new format in pampro can be found [here](http://nbviewer.ipython.org/github/Thomite/pampro/blob/master/examples/example_hdf5.ipynb).

All other changes are optimisations made on under-the-hood code, either to improve speed or memory efficiency, that have no impact on how users will write their analyses.

### Features

* Import channels of time series data from a variety of common monitors:
	* ActiHeart (.txt)
	* Axivity binary (.cwa)
	* GeneActiv (.bin)
	* Actigraph (.dat)
	* activPAL & activPAL micro binary (.datx)
	* Any timestamped data (.csv)
* Output piecewise summary statistics of any data channel, over any size time window:
	* Time spent in any cutpoint.
	* Sum, mean, percentiles, min, max.
* Visualise the time series data.
* Extract bouts of activity in any cutpoint.
* Various triaxial acceleration methodologies:
	* Nonwear detection.
	* Autocalibration.


### Usage

Click [here](http://nbviewer.ipython.org/github/Thomite/pampro/blob/master/examples/pampro_introduction.ipynb) for a walkthrough of pampro's most basic features. Please note that designing an analysis currently requires extensive knowledge of the Python programming language. See [/examples](https://github.com/Thomite/pampro/tree/master/examples) for example scripts demonstrating various more advanced features. The growing [/methods](https://github.com/Thomite/pampro/tree/master/methods) section provides detailed explanations of the methods implemented in pampro, linking to the relevant literature where appropriate.


### Installation

In your terminal, navigate to the desired installation directory and enter the following:

```
git clone https://github.com/Thomite/pampro.git
git checkout tags/v0.4.0
cd pampro
ipython setup.py install
```

This will clone the repository, sync to the latest official release (v0.4.0) and run the Python script to install it on your system. This presupposes that [Git](http://git-scm.com) and [Python](https://store.continuum.io/cshop/anaconda/) are installed already.

### Citing pampro

If you use pampro in your work, it would be appreciated if you could cite it with the appropriate DOI:

[![DOI](https://zenodo.org/badge/18706328.svg)](https://zenodo.org/badge/latestdoi/18706328)
