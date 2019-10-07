# py

## Algorithm Flow
![codeflow](https://github.com/SGontcho/NLC_Ina/Figures/NLC_codemap.pdf)


Below I give some examples for running the MCMC fitting procedures using the command line:
#### MC_NLC_P1D
* No parallel tempering, no convergence testing:
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500
* Parallel tempering, no convergence testing
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --multiT
* No parallel tempering, convergence testing
    python MC_NLC_P1D.py --z 2.4 --err 0.5 --out_dir "run21" --pos_method 2 --ndim 5 --nwalkers 200 --nsteps 500 --CTSwitch

    python MC_NLC_P1D_post.py --out_dir "run120" --in_dir "run100" --nsteps 100000 --pooling 0 --CTSwitch

    python MC_NLC_P1D_3param_pool.py --out_dir "run100" --z 2.4 --err 0.5 --pos_method 2 --ndim 3 --nwalkers 60 --nsteps 10000 --multiT