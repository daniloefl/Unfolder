
import copy
import pymc3 as pm
import numpy as np
import seaborn as sns 
import theano
import theano.tensor
import matplotlib.pyplot as plt
from Histogram import H1D, H2D

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
  '''
  def __init__(self, bkg, mig, eff):
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
    self.truth = self.mig.project('x').divideBinomial(self.eff) # truth dist(i)*eff(i) = P(t=i and it is reconstructed)
    self.reco = self.mig.project('y')           # reco dist(j) = P(r=j and it is in truth)
    self.prior = "uniform"
    self.priorAttributes = {}

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
      self.priorAttributes['sd'] = copy.deepcopy(self.truth.err)
    else:
      self.priorAttributes['sd'] = copy.deepcopy(widths)
    self.prior = "gaussian"

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
    self.data = H1D(data)    # copy data
    self.model = pm.Model()  # create the model
    with self.model:         # all in this scope is in the model's context
      # Define the prior
      if self.prior == "gaussian":
        self.T = pm.Normal('Truth', mu = self.priorAttributes['mean'], sd = self.priorAttributes['sd'], shape = (self.Nt))
      else: # if none of the names above matched, assume it is uniform
        self.T = pm.Uniform('Truth', 0.0, 10*max(self.truth.val), shape = (self.Nt))
      self.var_bkg = theano.shared(value = self.asMat(self.bkg.val))
      self.var_bkg.reshape((self.Nr, 1))
      self.var_response = theano.shared(value = self.asMat(self.response.val))
      self.R = theano.tensor.dot(self.T, self.var_response) + self.var_bkg
      self.U = pm.Poisson('U', mu = self.R, observed = self.data.val, shape = (self.Nr, 1))

  '''
  Samples the prior with N toy experiments
  Saves the toys in self.trace, the unfolded
  distribution mean and mode in self.hunf and self.hunf_mode respectively
  the sqrt of the variance in self.hunf_err
  '''
  def sample(self, N = 100000):
    with self.model:
      start = pm.find_MAP()
      step = pm.NUTS(state = start)
      self.trace = pm.sample(N, step, start = start)
      pm.summary(self.trace)

      from scipy import stats
      self.hunf = H1D(self.truth)
      self.hunf_mode = H1D(self.truth)
      for i in range(0, self.Nt):
        self.hunf.val[i] = np.mean(self.trace.Truth[:, i])
        self.hunf.err[i] = np.std(self.trace.Truth[:, i])
        self.hunf_mode.val[i] = stats.mode(self.trace.Truth[:, i])[0][0]
        self.hunf_mode.err[i] = np.std(self.trace.Truth[:, i])

  '''
  Plot the distributions for each bin regardless of the other bins
  Marginalizing each bin's PDF
  '''
  def plotMarginal(self, fname):
    fig = plt.figure(figsize=(10, 20))
    for i in range(0, self.Nt):
      ax = fig.add_subplot(self.Nt, 1, i+1, title='Truth')
      sns.distplot(self.trace.Truth[:, i], kde = True, hist = True, label = "Truth bin %d" % i, ax = ax)
      ax.set_title("Bin %d value" % i)
      ax.set_ylabel("Probability")
    plt.xlabel("Truth bin value")
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

  '''
  Plot data, truth, reco and unfolded result
  '''
  def plotUnfolded(self, fname = "plotUnfolded.png"):
    fig = plt.figure(figsize=(10, 10))
    plt.errorbar(self.data.x, self.data.val, self.data.err, self.data.x_err, fmt = 'bs', linewidth=2, label = "Pseudo-data", markersize=5)
    plt.errorbar(self.reco.x, self.reco.val, self.reco.err, self.reco.x_err, fmt = 'co', linewidth=2, label = "Background subtracted", markersize=5)
    plt.errorbar(self.hunf_mode.x, self.hunf_mode.val, self.hunf_mode.err, self.hunf_mode.x_err, fmt = 'r^', linewidth=2, label = "Unfolded mode")
    plt.errorbar(self.truth.x, self.truth.val, self.truth.err, self.truth.x_err, fmt = 'gv', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(self.hunf.x, self.hunf.val, self.hunf.err, self.hunf.x_err, fmt = 'rv', linewidth=2, label = "Unfolded mean", markersize=5)
    plt.legend()
    plt.ylabel("Events")
    plt.xlabel("Observable")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


