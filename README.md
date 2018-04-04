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
sudo apt-get install python3 python3-pip
pip3 install --user pandas
pip3 install --user seaborn
pip3 install --user pymc3
```

After this, running the following should run a test unfolding using the toy model in toyModel/ModelChris.json:

```
./toyModel/dumpModelPlots.py
./toyModel/closureTest.py
```

It might be necessary to add the main directory in the ```PYTHONPATH```.
One can also do a test on the statistical and systematic bias, by running (this scans the
regularisation parameter alpha and does many toy experiments to test the impact of the bias,
so it can take a while):

```
./toyModel/testBias.py withSyst bias
```

To test the toy model without adding systematic uncertainties in FBU:

```
./toyModel/testBias.py withoutSyst bias
```

A comparison with TUnfold and d'Agostini can be made with:

```
./toyModel/testBiasTUnfold.py bias
./toyModel/testBiasDagostini.py
```

# Usage

Given a background histogram to be subtracted ```bkg```; a matrix ```mig``` with counts in its element ```mig[i, j]```
how many events are expected to have been renerated in particle-level bin ```i``` and reconstructed-level bin ```j```;
and an efficiency histogram ```eff```, which has in each entry ```i```
(1 - (# events in truth bin ```i``` that fails reconstruction)/(# events in truth bin ```i```));
we can build the unfolding model as follows.
Including the truth original histogram is optional, as it can be calculated from the matrix ```mig```,
however, in such cases the error bars of the truth histogram are not guaranteed to be correct (one
needs to know the correlation between the truth events and the one that pass simultaneously
truth and reco to estimate the truth histogram from the efficiency and the migration matrix).

```
from Unfolder import Unfolder
model = Unfolder.Unfolder(bkg, mig, eff, truth)  # Call the constructor to initialise the model parameters
#model.setUniformPrior()                         # Using a uniform prior is the default
#model.setGaussianPrior()                        # For a Gaussian prior with means at the truth bins
                                                 # and width in each bin given by sqrt(truth)
#model.setGaussianPrior(mean, sd)                # If vectors (with the size of the truth distribution number of bins)
                                                 # are given, they will be used as the means and widths of the Gaussians
                                                 # bin-by-bin, instead of the defaults
#model.setFirstDerivativePrior(fb = 1.0)         # Uses a first-derivative based prior with a reference distribution if fb = 1.0.
#model.setCurvaturePrior(fb = 1.0)               # Uses a curvature based prior with a reference distribution if fb = 1.0.
#model.setEntropyPrior()                         # Uses an entropy based prior.
```

The response matrix P(reco = j|truth = i)*efficiency(i) is now stored in ```model.response```.
For convenience, the same matrix without the efficiency multiplication is stored in ```model.response_noeff```.

One can add systematic uncertainties in the unfolding model as follows:
```
model.addUnfoldingUncertainty("alternative_model_B", bkg_B, mig_B, eff_B)
model.addUnfoldingUncertainty("alternative_model_C", bkg_C, mig_C, eff_C)
# etc
```

The statistical model can be prepared for a given input data as follows:

```
model.run(data)
```

Afterwards, one can draw samples to find the posterior as follows:

```
alpha = 0
model.setAlpha(alpha)                            # this sets the regularisation parameter alpha
model.sample(20000)
```

The toy ```k``` produced in the trith bin ```i``` are stored in ```model.trace.Truth[k, i]```. So you can see
the distribution of the posterior marginalized for bin ```i``` by plotting:

```
plt.figure()
sns.distplot(model.trace.Truth[:, i], kde = True, hist = True, label = "Posterior histogram marginalized in bin %d" %i)
plt.legend()
plt.show()
```

If you are only interested in the means of the posterior marginalized in each bin, you can plot it to compare it with
the truth distribution as follows:

```
model.plotUnfolded("plotUnfolded.png")
scale = 1.0
normaliseByBinWidth = False
unit_name = "Counts"
model.plotOnlyUnfolded(scale, normaliseByBinWidth, unit_name, "plotUnfoldedScaled.png") # to normalise histograms
```

To plot the correlation matrix, the skewness and kurtosis of the marginal distributions, respectively:

```
model.plotCorr("correlation.png")
model.plotSkewness("skewness.png")
model.plotKurtosis("kurtosis.png")
```

Tests with different choices of the prior distribution and a scan over possible values of the regularisation parameter
are fundamental to check what is the best procedure for the regularisation (unless one has chosen to use a uniform prior).
To scan the values of the regularisation parameter, generating toy input histograms of a given underlying model, so
that one can test the impact of the alternative model and of its possible statistical fluctuations:

```
optAlpha, optChi2, optBias, optStd, optNorm, optNormStd = model.scanAlpha(alt_bkg, alt_mig, alt_eff,  # alt. model
                                                                          1000,                       # number of toys
                                                                          np.arange(0.0, 10, 0.5),    # range of alpha to probe
                                                                          "bias.png",                 # bias mean and variance vs alpha
                                                                          "chi2.png",                 # sum of bias mean^2/variance for all bins vs alpha
                                                                          "norm.png")                 # bias mean and variance in the normalisation vs alpha
```

Where `alt_bkg`, `alt_mig`, `alt_eff` are the parameters that describe the model to use to generate toys.
In this example, 1000 toys are generated and alpha is varied from 0 to 10 in steps of 0.5. Plots of the bias as a function of
alpha are generated according to the plot file names in the last 3 arguments.

The software also comes with two histogram classes: H1D and H2D, as well as plotting functions for them, plotH1D and
plotH2D.
Internally, all information is stored using those classes to guarantee that error propagation is done correctly.
You can also send the inputs to the software using those classes. They have the advantage to also be able to read
a ROOT TH1 or TH2 histogram.

# Basic functionality

   * `model = Unfolder(bkg, mig, eff, truth)` creates an instance of the Unfolder class.

   * `model.setGaussianPrior(widths, means)` forces the usage of a Gaussian bin-by-bin uncorrelated Gaussian prior
with the means given in the `means` array and standard deviations given in the `widths` array. Both arrays
should have the same size as the number of truth bins.

   * `model.setEntropyPrior()` forces the usage of an entropy-based prior.

   * `model.setCurvaturePrior(fb, means)` forces the usage of a curvature-based prior, with a bias distribution given
by `fb * means`. If `means` is set to `None`, the truth distribution is used. If `fb` is set to zero, this is equivalent to the
uniform prior.

   * `model.setFirstDerivativePrior(fb, means)` forces the usage of a first-derivative-based prior, with the bias distribution
given by `fb * means`. If `means` is set to `None`, the truth distribution is used. If `fb` is set to zero, this is equivalent to
tje uniform prior.

   * `model.setConstrainArea()` forces the addition of an extra constrain which fits the data number of input events to constrain
the total normalisation of the unfolded distribution.
This is not done by default.

   * `model.addUncertainty(name, bkg, reco, prior)` adds a systematic uncertainty contribution at reconstruction level, by adding a
parameter that changes the reconstruction distribution from the nominal background and nominal reconstruction-level distribution
to the ones given. This can be used for detector-level uncertainties, which are not expected to have a big impact in the migration matrix.
The prior parameter is set to 'uniform' by default, so that this nuisance parameter has a uniform distribution, but it can also
be set to 'gaussian', in which case a Gaussian prior is used, favouring the nominal model over the alternative.

   * `model.addUnfoldingUncertainty(name, bkg, mig, eff, prior)` adds a systematic uncertainty contribution as a difference between the
reconstruction-level distribution with the nominal or with the alternative model given by the `bkg`, `mig` and `eff` parameters.
The prior parameter is set to 'uniform' by default, so that this nuisance parameter has a uniform distribution, but it can also
be set to 'gaussian', in which case a Gaussian prior is used, favouring the nominal model over the alternative.

   * `model.setUniformPrior()` uses a uniform prior for the truth distribution. This is the default.

   * `model.run(data)` prepares the statistical model for unfolding with the given input data. The prior and systematic uncertainties
must have been set before calling this. It does not unfold the data.

   * `model.sample(N)` throws toy experiments for the prior, rejecting samples with the probability given by the likelihood model.
After this call, the unfolded distribution mean is stored in `model.hunf` and the unfolded distribution mode
is stored in `model.hunf_mode`.
The mean and square root of the variance of the nuisance parameters can be found in `model.hnp` and `model.hnpu`.
The accepted toy sample `k` for truth bin `i` are stored in `model.trace.Truth[k, i]`, so one can plot a marginal
distribution for bin `i` using the following:

```
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.distplot(model.trace.Truth[:, i], kde = True, hist = True, label = "Posterior histogram marginalized in bin %d" %i)
plt.legend()
plt.show()
```

   * `model.plotUnfolded(fname)` plots the truth distribution of the nominal model, the unfolded mean and the unfolded mode distributions in file `fname`.

   * `model.plotOnlyUnfolded(f, normaliseByBinWidth, units, fname)` plots the truth and unfolded mean distributions multiplied by `f` if `normaliseByBinWidth` is False, or
multiplied by `f/bin width` if `normaliseByBinWidth` is True. It shows the units in the Y axis label if `units != ""`. The output file is `fname`.

   * `model.scanAlpha(bkg, mig, eff, N, listOfAlpha, filename1, filename2, filename3)` throws `N` toy experiments simulating a
model with background `bkg`, migration matrix `mig` and efficiency `eff` to estimate its impact in the unfolded mode. It performs such
a test for the list of values of the regularisation parameter in `listOfAlpha`. It outputs the mean bias and the spread of the bias for
each value of the regularisation parameter in the plot `filename1`. It estimates sum over bins of mean^2/variance of the bias for each alpha
in `filename2`. It estimates the normalisation bias for each value of alpha in `filename3`. This takes a significant time, due to the `N` toy
experiments, but it is mandatory for the selection of the regularisation parameter alpha.
The bias is defined as the difference between the unfolded distribution and the truth distribution obtained from the
projection of the migration matrix `mig`.

   * `model.setAlpha(alpha)` sets the regularisation parameter.

   * `model.setData(data)` sets the input data after a call to `model.run` has been made already. This keeps
the same statistical model from the last call to `run`, changing only the input data.

   * `model.plotMarginal(fname)` plots the marginalisation of the posterior distribution for all truth bins in file `fname`.

   * `model.plotNPMarginal(syst, fname)` plots the marginalisation of the posterior distribution for the systematic uncertainty `syst` in file `fname`.

   * `model.plotNPUMarginal(syst, fname)` plots the marginalisation of the posterior distribution for the unfolding systematic uncertainty `syst` in file `fname`.

   * `model.plotPairs(fname)` plots the marginal distributions and scatter plots for each truth bin pair in file `fname`. It can take a long time if there were many samples
generated in the call to `sample`.

   * `model.plotCov(fname)` plots the covariance matrix of the truth bins in file `fname`.

   * `model.plotCorr(fname)` plots the Pearson correlation matrix of the truth bins in file `fname`.

   * `model.plotCorrWithNP(fname)` plots the Pearson correlation matrix of the truth bins and of the systematic uncertainty nuisance parameters in file `fname`.

   * `model.plotSkewness(fname)` plots the skewness of the posterior of each truth bin in file `fname`.

   * `model.plotNP(fname)` plots the mean and width of the posterior of each systematic uncertainty nuisance parameter in file `fname`.

   * `model.plotNPU(fname)` plots the mean and width of the posterior of each unfolding systematic uncertainty nuisance parameter in file `fname`.

   * `model.plotKurtosis(fname)` plots the Fisher kurtosis of each truth posterior marginal distribution in file `fname`.

# Histogram-related auxiliary functions

One can find classes with simple implementations of the 1D or 2D histograms (H1D and H2D) in Unfolder/Histogram.py.
It also includes the functions below for plotting, which might be useful:

   * `h = H1D(array1DObj)` or `h = H2D(array2DObj)` creates an H1D or H2D with contents given by the array object. The errors are set to the square root of the contents and
the X or Y axes have centers in the integers counting from 0 to the length of the array minus one.

   * `h = H1D(root1DObj)` or `h = H2D(root2DObj)` creates an H1D or H2D with contents given by the ROOT TH1 or TH2 object.

   * `H2D.fill(x, y, weight)` fills an H2D.

   * `H1D.fill(x, weight)` fills an H1D.

   * Normal +, -, /, * operators work with H1D and H2D instances.

   * `plotH1D(histogramDictionary, xlabel, ylabel, title, fname)` plots the histograms in the `histogramDictionary` in file `fname`. The histogramDictionary
has H1D instances as the key and the value is a string to be shown in the legend.
`plotH1DLines` does the same but adds lines connecting the points.
`plotH1DWithText` does the same but allows for text to be shown in the X axis.

   * `plotH2D(h, xlabel, ylabel, title, fname)` plots a 2D histogram into file `fname`.
`plotH2DWithText` allows for text to be shown in the X and Y axes.

Other helpful functions are:

   * `H2D.project(axis)` projects an H2D instance in the x (`axis = 'x'`) or y (`axis = 'y'`) axis.

   * `H2D.T()` returns the transpose of an H2D.

   * `H1D.toROOT(name)` or `H2D.toROOT(name)` converts the H1D or H2D instance to a ROOt TH1D or TH2D instance with name given by the parameter and returns it.

   * `H1D.loadFromROOT(obj)` or `H2D.loadFromROOt(obj)` reads the ROOt object `obj` and sets the instance of H1D or H2D to have its contents.

   * `H1D.integral()` returns the integral of a histogram.

# Auxiliary functions helpful for comparison with other unfolding methods

One can find many other functions in Unfolder/ComparisonHelpers.py which can be used to compare FBU with TUnfold and the iterative Bayesian unfolding method in RooUnfold.

   * `getTUnfolder` can be used as follows to unfold using TUnfold.
```
tunfold = getTUnfolder(bkg, mig, eff, data, regMode = ROOT.TUnfold.kRegModeNone, normMode = 0)
tunfold.DoUnfold(tau)
tunfold_result = H1D(tunfold.GetOutput("rootName"))/eff
```

   * `getDAgostini` can be used as follows to unfold using the iterative Bayes method.
```
dagostini_unfolded = getDAgostini(bkg, mig, eff, data, nIter)
```

   * `getDataFromModel(bkg, mig, eff)` can be used to produce one toy experiment of data such that the background is given by `bkg`, the migration matrix is givne by `mig`
and the efficiency is given by `eff`. This can be used to simulate statistical fluctuations of this underlying model.

   * `getBiasFromToys(unfoldFunction, regularisationParameter, N, bkg, mig, eff)` generates `N` toys of a model with background given by
`bkg`, migration matrix given by `mig` and efficiency given by `eff` and it unfolds it by calling the `unfoldFunction(regularisationParameter, toyData)`. It then
returns the mean bias (averaged over the bins), the standard deviation of the bias (averaged over bins), the sum of bias mean^2/bias variance for all bins, the mean
bias in the normalisation, the standard deviation of the bias in the normalisation, and the bias if the input is given by the nominal reconstruction distribution of the
model.

   * `comparePlot(listHistograms, listLegend, f, normaliseByBinWidth, units, fname)` can be used to plot a set of histograms in `listHistograms` with legend
text given by `listLegend`, multiplied by `f` (and normalised by bin width if `normaliseByBinWidth` is True) in the file `fname`.


