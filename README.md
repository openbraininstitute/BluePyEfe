<img src="docs/source/logo/BluePyEfeBanner.jpg"/>

-----------------

# BluePyEfe: Blue Brain Python E-feature extraction

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/bluepyefe/">
    <img src="https://img.shields.io/pypi/v/bluepyefe.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
    <a href="https://bluepyefe.readthedocs.io/">
    <img src="https://readthedocs.org/projects/bluepyefe/badge/?version=latest" alt="latest documentation" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/BlueBrain/bluepyefe/blob/master/LICENSE.txt">
    <img src="https://img.shields.io/pypi/l/bluepyefe.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
   <a href="https://github.com/BlueBrain/BluePyEfe/actions">
    <img src="https://github.com/BlueBrain/BluePyEfe/workflows/Build/badge.svg?branch=master" alt="Actions build status" />
    </a>
  </td>
</tr>
<tr>
<tr>
  <td>DOI</td>
  <td>
    <a href="https://zenodo.org/badge/latestdoi/237923583">
    	<img src="https://zenodo.org/badge/237923583.svg" alt="DOI"/>
    </a>
  </td>
</tr>
<tr>
	<td>Gitter</td>
	<td>
		<a href="https://gitter.im/bluebrain/bluepyefe">
		<img src="https://badges.gitter.im/Join%20Chat.svg">
	</a>
	</td>
</tr>
</table>

Introduction
============

BluePyEfe aims at easing the process of reading experimental recordings and extracting 
batches of electrical features from these recordings. To do so, it combines
 trace reading
functions and features extraction functions from the [eFel library](https://github.com/BlueBrain/eFEL).

BluePyEfe outputs protocols and features files in the format used
by [BluePyOpt](https://github.com/BlueBrain/BluePyOpt) for neuron electrical
 model building.

How to cite
===========
This software is citable using a [DOI generated by Zenodo](https://zenodo.org/record/3728192).

Requirements
============

* [Python 3.6+](https://www.python.org/downloads/release/python-360/)
* [eFEL eFeature Extraction Library](https://github.com/BlueBrain/eFEL) (automatically installed by pip)
* [Numpy](http://www.numpy.org) (automatically installed by pip)
* [Scipy](https://www.scipy.org/) (automatically installed by pip)
* [Neo](https://neo.readthedocs.io/en/stable/) (automatically installed by pip)
* The instruction below are written assuming you have access to a command shell
on Linux / UNIX / MacOSX / Cygwin

Installation
============

To install BluePyEfe run:

```bash
pip install bluepyefe
```

Quick Start and Operating Principle
===========

For a hands-on introduction to BluePyEfe, have a look at the notebook [examples/example.ipynb](examples/example.ipynb)

The goal of the present package is to extract meaningful electrophysiological features (e-features) from voltage time series.
The e-features considered in the present package are the one implemented in the [eFEL python library](https://github.com/BlueBrain/eFEL). See [this pdf](https://bluebrain.github.io/eFEL/efeature-documentation.pdf) for a list of available e-features.

The present package makes one major assumption: E-features are more meaningful if they are coming from a set of traces rather than a single trace. And they are even more meaningful if these traces come from different cells of the same cellular type.
This assumption dictates the organisation of the package and has several consequences:

The efeatures extracted through the package will always be averaged over the trace considered. For example, the AP_amplitude will be an average over all the action potentials present in a trace. If you wish to work on an AP by AP basis, please consider using the eFEL library directly. 

A large part of the present software is therefore dedicated to averaging the features across set of "equivalent" recordings. To be able to average e-features across different cells in a meaningful way, an equivalence must be established between the traces coming from these different cells. It would not make sense to average the mean firing frequency obtain cell A on a 1s long step protocol with the one obtain for cell B on a ramp protocol that lasts for 500ms. We chose to define recordings as equivalent based on two criteria: (1) They have the same name and (2) they are of the same amplitude when the amplitude is expressed as a percentage of the rheobase of the cell.

A pseudo-code for the main function of the package (bluepyefe.extract.extract_efeatures) could look as follows:
```
1. Load the data to memory by reading all the files containing the traces
2. Extract the required e-features for all the traces
3. Compute the rheobases of the cells based on one or several protocols
4. Use these rheobases to associate to each protocol an amplitude expressed in % of the rheobase
5. Compute the mean and standard deviations for the e-features across traces having the same amplitude
6. Save the results and plot the traces and e-features
```
Each of these steps are parametrized by a number of settings, therefore we recommend that you read carefully the docstring of the function.

Coming from the legacy version
===============================
The legacy version (v0.4*) is moved to the legacy branch.
Changes introduced in v2.0.0 are listed in the [CHANGELOG.rst](CHANGELOG.rst). 
That is the only file you need to look at for the changes as the future changes will also be noted there.

Funding
=======
This work has been partially funded by the European Union Seventh Framework Program (FP7/2007­2013) under grant agreement no. 604102 (HBP), and by the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 720270 (Human Brain Project SGA1) and No. 785907 (Human Brain Project SGA2) and by the EBRAINS research infrastructure, funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3).
