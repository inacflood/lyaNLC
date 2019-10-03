# NonLinearLya
Project led by Satya Gontcho A Gontcho and Ina Flood, on constraining nonlinear effects on the power spectrum for the Lya forest.

We test the robustness of the Lya power spectrum from Andreu Arinyo-i-Prats et al. (2015) using data for the 1D power from Palanque-Delabrouille et al. (2013), fitting the model to the data. Fitting is done using the Markov Chain Monte Carlo (MCMC) method, specifically using the emcee and ptemcee libraries.

## References
* This code library is adapted from Font-Ribera https://github.com/igmhub/lyaforecast

* The paper presenting the data we are using as a reference is : https://arxiv.org/pdf/1306.5896.pdf. The introduction and what relates to Figure 20 is what’s of interest to us in priority.

* The paper presenting the non linear model we test is : https://arxiv.org/pdf/1506.04519.pdf. Sections 1, 2 and 6 specifically (+ appendix B).

## Requirements

This code requires that the python libraries camb (https://camb.readthedocs.io/en/latest/#), emcee 3.0 (https://emcee.readthedocs.io/en/stable/), and ptemcee (https://github.com/willvousden/ptemcee) be installed on your device.

## Usage

Code for model fitting and analysis, along with code to produce relevant figures, is found in the py folder. The data we use is found in the NPD2013_data folder. Relevant figures, all of which can be produced using the code in py, are found in the Figures folder.

There are 4 primary methods included in this library:
* 
* curvefit_NLC_P1D provides a method to do a quick fit of the model to the data using the scipy curve_fit function. Alter redshift z and the output file name directly in the code. 

### Examples

Below I give some examples for running the MCMC fitting procedures:
* No parallel tempering, no convergence testing:
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500
* Parallel tempering, no convergence testing
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --multiT
* No parallel tempering, convergence testing
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --CTSwitch