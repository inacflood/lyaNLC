# NonLinearLya
Project led by Satya Gontcho A Gontcho and Ina Flood, on constraining an analytic model of the Lya power spectrum using the measurement of the 1D power from Palanque-Delabrouille et al. (2013)

Codes adapted from Font-Ribera https://github.com/igmhub/lyaforecast

Useful documentation :
* The paper presenting the data we are using as a reference is : https://arxiv.org/pdf/1306.5896.pdf. The introduction and what relates to Figure 20 is what’s of interest to us in priority.

* The paper presenting the non linear model we are planning to test is : https://arxiv.org/pdf/1506.04519.pdf. Sections 1, 2 and 6 specifically (+ appendix B).

Examples of running emcee:
* No parallel tempering, no convergence testing:
    run MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500
* Parallel tempering, no convergence testing
    run MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --multiT
* No parallel tempering, convergence testing
    run MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --CTSwitch