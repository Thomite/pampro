## PAMPRO - Physical Activity Monitor Processing

### Introduction

PAMPRO is a software project for the systematic analysis of physical activity data collected in epidemiological studies. The ultimate goal is to provide a turn-key solution for physical activity data analysis, replicating published methodologies in a monitor-agnostic framework.


### Features

* Import channels of time series data from a variety of common monitors. 
	* ActiHeart (.txt)
	* Axivity binary (.cwa)
	* Actigraph (.dat) 
	* Any timestamped data (.csv)
* Output piecewise summary statistics for any length of time from any data channel.
* Visualise the time series data.
* Quantify bouts of activity in specified ranges.
* Nonwear detection.


### Usage

See [/examples](https://github.com/Thomite/pampro/tree/master/examples) for example scripts demonstrating various features. 


### Installation

In your terminal, navigate to the desired installation directory and enter the following:

```
git clone https://github.com/Thomite/pampro.git
cd pampro
py setup.py install
```

This will clone the latest version of this repository and run the Python script to install it on your system. This requires both [Git](http://git-scm.com) and [Python](https://store.continuum.io/cshop/anaconda/) to be installed already. 