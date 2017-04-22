
import copy
import pymc3 as pm
import numpy as np
import seaborn as sns 
import theano
import theano.tensor
import matplotlib.pyplot as plt
from Histogram import H1D, H2D, plotH1D, plotH2D, plotH1DWithText, plotH2DWithText, plotH1DLines
from ComparisonHelpers import getDataFromModel
from scipy import stats

theano.config.compute_test_value = 'warn'

'''
Shows reference for Full Bayesian Unfolding to encourage people to give credit for the original
work (which is not mine!).
'''
def printReferences():
  print "This software has been produced using the ideas put forth in:"
  print "Choudalakis, G., ``Fully Bayesian Unfolding'', physics.data-an:1201.4612, https://arxiv.org/abs/1201.4612"
  print "Please cite it if you plan to publish this."
  print "More information on the software itself can be found in https://github.com/daniloefl/Unfolder"
  print ""

printReferences()

'''
Class that uses Fully Bayesian Unfolding to
unfold a distribution using NUTS to sample the prior distribution.
'''
class Unfolder:
  '''
  Constructor of unfolding class
  bkg is a list or array with the contents of the background histogram
  mig is a 2D array such that mig[i, j] contains the number of events in bin i at truth level
         that are reconstructed in bin j at reco level
  eff is a list or array with the contents of the efficiency histogram, defined as:
         eff[i] = (1 - (# events in truth bin i that fail reco)/(# events in truth bin i))
  truth is the particle level histogram. If it is None, it is calculated from the migration
         matrix mig and the efficiency, but if the efficiency has been defined as 1 - (not reco & truth)/truth
         instead of (reco&truth)/truth, the error propagation will be incorrect.
  '''
  def __init__(self, bkg, mig, eff, truth = None):
    self.bkg = H1D(bkg)    # background
    self.Nr = mig.shape[1] # number of reco bins
    self.Nt = mig.shape[0] # number of truth bins
    self.mig = H2D(mig)    # joint probability
    self.eff = H1D(eff)    # efficiency
    # calculate response matrix, defined as response[i, j] = P(r = j|t = i) = P(t = i, r = j)/P(t = i)
    self.response = H2D(mig)
    self.response_noeff = H2D(mig) # response matrix if one assumes efficiency = 1
    for i in range(0, self.Nt): # for each truth bin
      rsum = 0.0
      esum = 0.0
      for j in range(0, self.Nr): # for each reco bin
        rsum += self.mig.val[i, j]    # calculate the sum of all reco bins in the same truth bin
        esum += self.mig.err[i, j]
      # rsum is now the total sum of events that has that particular truth bin
      # now, for each reco bin in truth bin i, divide that row by the total number of events in it
      # and multiply the response matrix by the efficiency
      for j in range(0, self.Nr):
        self.response.val[i, j] = self.mig.val[i, j]/rsum*self.eff.val[i]  # P(r|t) = P(t, r)/P(t) = Mtr*eff(t)/sum_k=1^Nr Mtk
        self.response.err[i, j] = 0 # FIXME
        self.response_noeff.val[i, j] = self.mig.val[i, j]/rsum
        self.response_noeff.err[i, j] = 0 # FIXME
    # if the prior is later chosen to be non-uniform, also calculate the truth distribution
    # so that the prior can be set according to it, if necessary
    # also useful to make plots
    # truth dist(i)*eff(i) = P(t=i and it is reconstructed)
    if truth == None:
      self.truth = self.mig.project('x').divideWithoutErrors(self.eff)
    else:
      self.truth = H1D(truth)
    self.recoWithoutFakes = self.mig.project('y')           # reco dist(j) = P(r=j and it is in truth)
    self.prior = "uniform"
    self.priorAttributes = {}
    # for systematic uncertainties
    self.bkg_syst = {}
    self.reco_syst = {}
    self.systematics = []


  '''
  Use a Gaussian prior with bin-by-bin width given by widths and mean given by means.
  If either widths or means is set to None, the truth distribution is used with
  widths given by the square root of the bin contents.
  '''
  def setGaussianPrior(self, widths = None, means = None):
    if means == None:
      self.priorAttributes['mean'] = copy.deepcopy(self.truth.val)
    else:
      self.priorAttributes['mean'] = copy.deepcopy(means)
    if widths == None:
      self.priorAttributes['sd'] = copy.deepcopy(self.truth.err)**0.5
    else:
      self.priorAttributes['sd'] = copy.deepcopy(widths)
    self.prior = "gaussian"

  '''
  Use an entropy-based prior.
  '''
  def setEntropyPrior(self):
    self.prior = "entropy"

  '''
  Use a curvature-based prior.
  '''
  def setCurvaturePrior(self):
    self.prior = "curvature"

  '''
  Use a first derivative-based prior.
  '''
  def setFirstDerivativePrior(self):
    self.prior = "first derivative"

  '''
  Add systematic uncertainty.
  '''
  def addUncertainty(self, name, bkg, reco):
    self.bkg_syst[name] = H1D(bkg) - self.bkg
    self.reco_syst[name] = reco - self.recoWithoutFakes
    self.systematics.append(name)

  '''
  Set a uniform prior.
  '''
  def setUniformPrior(self):
    self.prior = "uniform"

  '''
  Transforms an array of doubles into a Theano-type array
  So that it can be used in the model
  '''
  def asMat(self, x):
    return np.asarray(x,dtype=theano.config.floatX)

  '''
  Create the model
  and store it in self.model
  the prior is the unfolded distribution
  The idea is that the prior is sampled with toy experiments and the
  reconstructed variable R_j = \sum_i truth_i * response_ij + background_j
  is constrained to be the observed data
  under such conditions, the posterior converges to the unfolded distribution
  '''
  def run(self, data):
    self.data = H1D(data)                    # copy data
    self.datasubbkg = self.data - self.bkg   # For monitoring: data - bkg

    self.model = pm.Model()                  # create the model
    with self.model:                         # all in this scope is in the model's context
      self.var_alpha = theano.shared(value = 1.0, borrow = False)
      self.var_data = theano.shared(value = self.data.val, borrow = False) # in case one wants to change data

      # Define the prior
      if self.prior == "gaussian":
        #self.T = pm.Normal('Truth', mu = self.priorAttributes['mean'], sd = self.priorAttributes['sd'], shape = (self.Nt))
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*0.5*theano.tensor.sqr((val - self.priorAttributes['mean'])/self.priorAttributes['sd']).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "entropy":
        self.T = pm.DensityDist('Truth', logp = lambda val: self.var_alpha*((val/val.sum())*theano.tensor.log(val/val.sum())).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "curvature":
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*theano.tensor.sqr(theano.tensor.extra_ops.diff(theano.tensor.extra_ops.diff(val))).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "first derivative":
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*theano.tensor.abs_(theano.tensor.extra_ops.diff(theano.tensor.extra_ops.diff(val/(self.truth.x_err*2))/np.diff(self.truth.x))/(2*theano.tensor.mean(theano.tensor.extra_ops.diff(val/(self.truth.x_err*2))/np.diff(self.truth.x)))).sum(), shape = (self.Nt), testval = self.truth.val)
      else: # if none of the names above matched, assume it is uniform
        self.T = pm.Uniform('Truth', 0.0, 10*max(self.truth.val), shape = (self.Nt))

      self.var_bkg = theano.shared(value = self.asMat(self.bkg.val))
      #self.var_bkg.reshape((self.Nr, 1))
      self.var_response = theano.shared(value = self.asMat(self.response.val))
      self.R = theano.tensor.dot(self.T, self.var_response) + self.var_bkg

      # reco result including systematics:
      self.R_full = self.R

      self.theta = {}
      self.var_bkg_syst = {}
      self.var_reco_syst = {}
      self.R_syst = {}
      for name in self.systematics:
        self.theta[name] = pm.Normal('t_'+name, mu = 0, sd = 1) # nuisance parameter
        # get background constribution
        self.var_bkg_syst[name] = theano.shared(value = self.asMat(self.bkg_syst[name].val))
        # calculate differential impact in the reco result
        self.var_reco_syst[name] = theano.shared(value = self.asMat(self.reco_syst[name].val))
        self.R_syst[name] = self.var_reco_syst[name] + self.var_bkg_syst[name]
        # add it to the total reco result
        self.R_full += self.theta[name]*self.R_syst[name]
      self.U = pm.Poisson('U', mu = self.R_full, observed = self.var_data, shape = (self.Nr, 1))

  '''
  Calculate the sum of the bias using only the expected values.
  '''
  def getBiasFromMAP(self, N):
    fitted = np.zeros((N, len(self.truth.val)))
    bias = np.zeros(len(self.truth.val))
    bias_variance = np.zeros(len(self.truth.val))
    for k in range(0, N):
      pseudo_data = getDataFromModel(self.bkg, self.mig, self.eff, self.truth)
      self.setData(pseudo_data)
      #self.run(pseudo_data)
      with self.model:
        res = pm.find_MAP()
        fitted[k, :] = res['Truth'] - self.truth.val
    bias = np.mean(fitted, axis = 0)
    bias_std = np.std(fitted, axis = 0)
    #print "getBiasFromMAP with alpha = ", self.var_alpha.get_value(), " N = ", N, ", mean, std = ", bias, bias_std
    bias_binsum = np.sum(np.abs(bias)/np.sqrt(self.truth.err))/len(self.truth.val)
    bias_std_binsum = np.sum(bias_std/np.sqrt(self.truth.err))/len(self.truth.val)
    bias_chi2 = np.sum(np.power(bias/bias_std, 2))
    return [bias_binsum, bias_std_binsum, bias_chi2]

  '''
  Scan alpha values to minimise bias^2 over variance.
  '''
  def scanAlpha(self, N = 1000, rangeAlpha = np.arange(0.0, 10.0, 1.0), fname = "scanAlpha.png", fname_chi2 = "scanAlpha_chi2.png"):
    bkp_alpha = self.var_alpha.get_value()
    bias = np.zeros(len(rangeAlpha))
    bias_std = np.zeros(len(rangeAlpha))
    bias_chi2 = np.zeros(len(rangeAlpha))
    minBias = 1e10
    bestAlpha = 0
    bestChi2 = 0
    bestI = 0
    for i in range(0, len(rangeAlpha)):
      self.setAlpha(rangeAlpha[i])
      bias[i], bias_std[i], bias_chi2[i] = self.getBiasFromMAP(N) # only take mean values for speed
      if np.abs(bias_chi2[i] - len(self.truth.val)) < minBias:
        minBias = np.abs(bias_chi2[i] - len(self.truth.val))
        bestChi2 = bias_chi2[i]
        bestAlpha = rangeAlpha[i]
        bestI = i
    fig = plt.figure(figsize=(10, 10))
    plt_bias = H1D(bias)
    plt_bias.val = bias
    plt_bias.err = np.power(bias_std, 2)
    plt_bias.x = rangeAlpha
    plt_bias.x_err = np.zeros(len(rangeAlpha))
    plotH1DLines(plt_bias, "alpha", "sum(bias/truth error) per bin", "Effect of alpha in the bias - Y errors are sum(sqrt(var)/truth errors)", fname)
    plt_bias_chi2 = H1D(bias_chi2)
    plt_bias_chi2.val = bias_chi2
    plt_bias_chi2.err = np.zeros(len(rangeAlpha))
    plt_bias_chi2.x = rangeAlpha
    plt_bias_chi2.x_err = np.zeros(len(rangeAlpha))
    plotH1DLines(plt_bias_chi2, "alpha", "sum(bias^2/Var(bias)) per bin", "Effect of alpha in the bias", fname_chi2)
    self.setAlpha(bkp_alpha)
    return [bestAlpha, bestChi2, bias[bestI], bias_std[bestI]]
    
  '''
  Set value of alpha.
  '''
  def setAlpha(self, alpha):
    with self.model:
      self.var_alpha.set_value(float(alpha), borrow = False)

  '''
  Update input data.
  '''
  def setData(self, data):
    self.data = H1D(data)
    with self.model:
      self.var_data.set_value(self.data.val, borrow = False)
  

  '''
  Samples the prior with N toy experiments
  Saves the toys in self.trace, the unfolded
  distribution mean and mode in self.hunf and self.hunf_mode respectively
  the sqrt of the variance in self.hunf_err
  '''
  def sample(self, N = 100000):
    self.N = N
    with self.model:
      start = pm.find_MAP()
      step = pm.NUTS(state = start)
      self.trace = pm.sample(N, step, start = start)
      pm.summary(self.trace)

      self.hnp = H1D(np.zeros(len(self.systematics)))
      self.hunf = H1D(self.truth)
      self.hunf_mode = H1D(self.truth)
      for i in range(0, self.Nt):
        self.hunf.val[i] = np.mean(self.trace.Truth[:, i])
        self.hunf.err[i] = np.std(self.trace.Truth[:, i])**2
        self.hunf_mode.val[i] = stats.mode(self.trace.Truth[:, i])[0][0]
        self.hunf_mode.err[i] = np.std(self.trace.Truth[:, i])**2

      for k in range(0, len(self.systematics)):
        self.hnp.val[k] = np.mean(self.trace['t_'+self.systematics[k]])
        self.hnp.err[k] = np.std(self.trace['t_'+self.systematics[k]])**2
        self.hnp.x[k] = self.systematics[k]
        self.hnp.x_err[k] = 1

  '''
  Plot the distributions for each bin regardless of the other bins
  Marginalizing each bin's PDF
  '''
  def plotMarginal(self, fname):
    fig = plt.figure(figsize=(10, 20))
    m = np.mean(self.trace.Truth) + 3*np.std(self.trace.Truth)
    for i in range(0, self.Nt):
      ax = fig.add_subplot(self.Nt, 1, i+1, title='Truth')
      sns.distplot(self.trace.Truth[:, i], kde = True, hist = True, label = "Truth bin %d" % i, ax = ax)
      ax.set_title("Bin %d value" % i)
      ax.set_ylabel("Probability")
      ax.set_xlim([0, m])
    plt.xlabel("Truth bin value")
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

  '''
  Plot the distributions for each nuisance parameter regardless of the other bins
  '''
  def plotNPMarginal(self, syst, fname):
    i = self.systematics.index(syst)
    fig = plt.figure(figsize=(10, 10))
    sns.distplot(self.trace['t_'+self.systematics[i]], kde = True, hist = True, label = self.systematics[i])
    plt.title(self.systematics[i])
    plt.ylabel("Probability")
    plt.xlim([-5, 5])
    plt.xlabel("Nuisance parameter value")
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

  '''
  Plot the marginal distributions as well as the scatter plots of bins pairwise.
  '''
  def plotPairs(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sns.pairplot(pm.trace_to_dataframe(self.trace), kind="reg")
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

  '''
  Plot the covariance matrix.
  '''
  def plotCov(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH2D(np.cov(self.trace.Truth, rowvar = False), "Unfolded bin", "Unfolded bin", "Covariance matrix of unfolded bins", fname)

  '''
  Plot the Pearson correlation coefficients.
  '''
  def plotCorr(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH2D(np.corrcoef(self.trace.Truth, rowvar = 0), "Unfolded bin", "Unfolded bin", "Pearson correlation coefficients of unfolded bins", fname)

  '''
  Plot the Pearson correlation coefficients including the NPs.
  '''
  def plotCorrWithNP(self, fname):
    tmp = np.zeros((self.Nt+len(self.systematics), self.N))
    for i in range(0, self.Nt):
      tmp[i, :] = self.trace.Truth[:, i]
    for i in range(0, len(self.systematics)):
      tmp[self.Nt+i, :] = self.trace['t_'+self.systematics[i]]
    tmplabel = ["Unfolded bin %d" % i for i in range(0, self.Nt)] + self.systematics
    plotH2DWithText(np.corrcoef(tmp, rowvar = 1), tmplabel, "Variable", "Variable", "Pearson correlation coefficients of posterior", fname)

  '''
  Plot skewness.
  '''
  def plotSkewness(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.skew(self.trace.Truth, axis = 0, bias = False)
    sk.err = np.zeros(len(sk.val))
    plotH1D(sk, "Particle-level observable", "Skewness", "Skewness of the distribution after unfolding", fname)

  '''
  Plot nuisance parameter means and spread.
  '''
  def plotNP(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH1DWithText(self.hnp, "Nuisance parameter", "Nuisance parameter posteriors mean and width", fname)

  '''
  Plot kurtosis.
  '''
  def plotKurtosis(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.kurtosis(self.trace.Truth, axis = 0, fisher = True, bias = False)
    sk.err = np.zeros(len(sk.val))
    plotH1D(sk, "Particle-level observable", "Fisher kurtosis", "Fisher kurtosis of the distribution after unfolding", fname)

  '''
  Plot data, truth, reco and unfolded result
  '''
  def plotUnfolded(self, fname = "plotUnfolded.png"):
    fig = plt.figure(figsize=(10, 10))
    plt.errorbar(self.data.x, self.data.val, self.data.err**0.5, self.data.x_err, fmt = 'bs', linewidth=2, label = "Pseudo-data", markersize=10)
    plt.errorbar(self.datasubbkg.x, self.datasubbkg.val, self.datasubbkg.err**0.5, self.datasubbkg.x_err, fmt = 'co', linewidth=2, label = "Background subtracted", markersize=10)
    plt.errorbar(self.recoWithoutFakes.x, self.recoWithoutFakes.val, self.recoWithoutFakes.err**0.5, self.recoWithoutFakes.x_err, fmt = 'mv', linewidth=2, label = "Expected signal (no fakes) distribution", markersize=5)
    plt.errorbar(self.hunf_mode.x, self.hunf_mode.val, self.hunf_mode.err**0.5, self.hunf_mode.x_err, fmt = 'r^', linewidth=2, label = "Unfolded mode", markersize = 5)
    plt.errorbar(self.truth.x, self.truth.val, self.truth.err**0.5, self.truth.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(self.hunf.x, self.hunf.val, self.hunf.err**0.5, self.hunf.x_err, fmt = 'rv', linewidth=2, label = "Unfolded mean", markersize=5)
    plt.legend()
    plt.ylabel("Events")
    plt.xlabel("Observable")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


  '''
  Plots only the unfolded distribution and expected after some normalisation factor f.
  '''
  def plotOnlyUnfolded(self, f = 1.0, normaliseByBinWidth = True, units = "fb", fname = "plotOnlyUnfolded.png"):
    fig = plt.figure(figsize=(10, 10))
    expectedCs = f*self.truth
    observedCs = f*self.hunf
    if normaliseByBinWidth:
      expectedCs = expectedCs.overBinWidth()
      observedCs = observedCs.overBinWidth()
    plt.errorbar(expectedCs.x, expectedCs.val, expectedCs.err**0.5, expectedCs.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(observedCs.x, observedCs.val, observedCs.err**0.5, observedCs.x_err, fmt = 'rv', linewidth=2, label = "Unfolded mean", markersize=5)
    plt.legend()
    plt.ylabel("Differential cross section ["+units+"]")
    plt.xlabel("Observable")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

