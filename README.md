# Unfolder

This software uses PyMC3 to unfold a distribution, using the Fully Bayesian Unfolding (FBU) method,
proposed in https://arxiv.org/abs/1201.4612.
There is already a software that implements this in https://github.com/gerbaudo/fbu , but it uses PyMC2.
The current software aims to extend it to PyMC3, so that one can use more efficient sampling techniques, such as
HMC and NUTS.

More information on FBU can be found in https://arxiv.org/abs/1201.4612

More information about PyMC3 can be found in https://peerj.com/articles/cs-55/ and https://pymc-devs.github.io/pymc3/notebooks/getting_started.html. Its GitHub project is in: https://github.com/pymc-devs/pymc3

More information on NUTS can be found in http://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

More information on HMC can be found in http://www.sciencedirect.com/science/article/pii/037026938791197X?via%3Dihub


# Installing pre-requisites

You can use install Numpy, Matplotlib, SciPy, Pandas, Seaborn and pyMC3 in Ubuntu as follows.

```
sudo apt-get install python-pip python-dev build-essential python-matplotlib python-numpy python-scipy
pip install pandas
pip install seaborn
pip install pymc3
```

After this, running the following should run a test unfolding:

```
./test.py
```


