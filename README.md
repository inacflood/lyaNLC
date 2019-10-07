# NonLinearLya
Project led by Ina Flood under the supervision of Satya Gontcho A Gontcho, on testing the robustness of current parametrization of the effect of non linearities on the power spectrum of the Lyman alpha forest

We test the robustness of the Lya power spectrum from Andreu Arinyo-i-Prats et al. (2015) using data for the 1D power from Palanque-Delabrouille et al. (2013), fitting the model to the data. Fitting is done using the Markov Chain Monte Carlo (MCMC) method, specifically using the emcee and ptemcee libraries.

## References
* This code library is adapted from Andreu Font-Ribera’s work: https://github.com/igmhub/lyaforecast

* The paper presenting the data we are using as a reference is : https://arxiv.org/pdf/1306.5896.pdf. The introduction and what relates to Figure 20 is what’s of interest to us in priority.

* The paper presenting the non linear model we test is : https://arxiv.org/pdf/1506.04519.pdf. Sections 1, 2 and 6 specifically (+ appendix B).

## Requirements

This code requires that the python libraries camb (https://camb.readthedocs.io/en/latest/#), emcee 3.0 (https://emcee.readthedocs.io/en/stable/), and ptemcee (https://github.com/willvousden/ptemcee) be installed on your device.

## Usage

Code for model fitting and analysis, along with code to produce relevant figures, is found in the py folder. The data we use is found in the NPD2013_data folder. Relevant figures, all of which can be produced using the code in py, are found in the Figures folder.

There are 4 primary methods included in this library:
* MC_NLC_P1D performs MCMC fitting for a specified redshift. Examples of usage given in the py folder README.
* MC_NLC_P1D_post continues MCMC fitting from the endpoint of a previous run of MCMC fitting. Can be used to start fitting with one method and switch part-way through. Examples of usage given in the py folder README.
* get_Walks_NLC takes the results of MCMC fitting from the previous files and outputs tools to analyze the results, includng a corner plot, plots of the walker paths, a sampling of final walker positions, best fit parameter results and the chi-sq per dof for the results. Alter the MCMC results to be used directly in the code.
* curvefit_NLC_P1D does a quick fit of the model to the data using the scipy curve_fit function. Alter redshift z and the output file name directly in the code. 