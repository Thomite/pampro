## PAMPRO - Physical Activity Monitor Processing

### Introduction

PAMPRO is a software project for the systematic analysis of physical activity data collected in epidemiological studies. The ultimate goal is to provide a turn-key solution for physical activity data analysis, replicating published methodologies in a monitor-agnostic framework.


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

Click [here](http://nbviewer.ipython.org/github/Thomite/pampro/blob/master/examples/pampro_introduction.ipynb) for a walkthrough of PAMPRO's most basic features. Please note that designing an analysis currently requires extensive knowledge of the Python programming language. See [/examples](https://github.com/Thomite/pampro/tree/master/examples) for example scripts demonstrating various more advanced features. The growing [/methods](https://github.com/Thomite/pampro/tree/master/methods) section provides detailed explanations of the methods implemented in PAMPRO, linking to the relevant literature where appropriate. Finally, see [here](https://github.com/Thomite/pampro/tree/master/documents/Future.md) for a list of upcoming features.


### Installation

In your terminal, navigate to the desired installation directory and enter the following:

```
git clone https://github.com/Thomite/pampro.git
cd pampro
ipython setup.py install
```

This will clone the latest version of this repository and run the Python script to install it on your system. This presupposes that [Git](http://git-scm.com) and [Python](https://store.continuum.io/cshop/anaconda/) are installed already.
