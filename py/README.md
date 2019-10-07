# py
All of this library's algorithms are contained in this folder.
## Algorithm Flow
![codeflow](https://github.com/SGontcho/NLC_Ina/blob/master/Figures/codeflow.jpeg)

## Examples
Below I give some examples for running the MCMC fitting procedures using the command line:
#### MC_NLC_P1D
* No parallel tempering, no convergence testing, no pooling, varying q1, q2, kp, and av with priors set manually:

    python MC_NLC_P1D.py --out_dir runX --z 2.4 --params 1 1 1 0 1 0 --pos-method 2 --ndim 4 --nwalkers 60 --nsteps 500 --pooling 0 
* Parallel tempering, no convergence testing, no pooling, varying av, bv, kp, kvav, and av with priors set manually:

    python MC_NLC_P1D.py --out_dir runX --z 2.4 --params 1 1 1 1 1 0 --pos-method 2 --ndim 5 --nwalkers 60 --nsteps 1000 --pooling 0 --multiT
* No parallel tempering, convergence testing, no pooling, varying q1, q2, and bv with priors set manually:

    python MC_NLC_P1D.py --out_dir runX --z 2.4 --params 1 1 0 0 0 1 --pos-method 1 --ndim 3 --nwalkers 60 --nsteps 3000 --pooling 0 --CTSwitch
* No parallel tempering, convergence testing, with pooling, varying kp, av, bv with priors set by a fixed width around the fiducial values:

    python MC_NLC_P1D.py --out_dir runX --z 2.4 --params 0 0 1 0 1 1 --pos-method 1 --ndim 3 --nwalkers 60 --nsteps 1000 --pooling 1 --CTSwitch -err 0.5
### MC_NLC_P1D_post
* No parallel tempering, convergence testing, with pooling:

    python MC_NLC_P1D.py --out_dir runY --in_dir runZ  --nsteps 650 --pooling 1 --CTSwitch