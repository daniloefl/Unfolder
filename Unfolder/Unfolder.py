
import copy
import pymc3 as pm
import numpy as np
import seaborn as sns 
import theano
import theano.tensor
import matplotlib.pyplot as plt
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D, plotH1DWithText, plotH2DWithText, plotH1DLines
from Unfolder.ComparisonHelpers import getDataFromModel
from scipy import stats
from scipy import optimize

# FIXME
# Change this to estimate the mode using Sympy
# Useful as a cross check
doSymbolicMode = False
try:
  import sympy
except:
  doSymbolicMode = False

theano.config.compute_test_value = 'warn'

'''
Shows reference for Full Bayesian Unfolding to encourage people to give credit for the original
work (which is not mine!).
'''
def printReferences():
  print("This software has been produced using the ideas put forth in:")
  print("Choudalakis, G., ``Fully Bayesian Unfolding'', physics.data-an:1201.4612, https://arxiv.org/abs/1201.4612")
  print("Please cite it if you plan to publish this.")
  print("More information on the software itself can be found in https://github.com/daniloefl/Unfolder")
  print("")

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
    self.nll = 0

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
    self.response_unfsyst = {}
    self.bkg_unfsyst = {}
    self.unf_systematics = []
    self.prior_unfsyst = {}
    self.fb = 0
    self.constrainArea = False
    self.tot_bkg = self.bkg.integral()[0]
    self.ave_eff = self.recoWithoutFakes.integral()[0]/self.truth.integral()[0]


  '''
  Use a Gaussian prior with bin-by-bin width given by widths and mean given by means.
  If either widths or means is set to None, the truth distribution is used with
  widths given by the square root of the bin contents.
  '''
  def setGaussianPrior(self, widths = None, means = None):
    if means == None:
      self.priorAttributes['bias'] = copy.deepcopy(self.truth.val)
    else:
      self.priorAttributes['bias'] = copy.deepcopy(means)
    if widths == None:
      self.priorAttributes['sd'] = copy.deepcopy(self.truth.err)**0.5
    else:
      self.priorAttributes['sd'] = copy.deepcopy(widths)
    self.prior = "gaussian"
    self.fb = 0

  '''
  Use an entropy-based prior.
  '''
  def setEntropyPrior(self):
    self.prior = "entropy"

  '''
  Use a curvature-based prior.
  '''
  def setCurvaturePrior(self, fb = 1, means = None):
    self.prior = "curvature"
    if means == None:
      self.priorAttributes['bias'] = copy.deepcopy(self.truth.val)
    else:
      self.priorAttributes['bias'] = copy.deepcopy(means)
    self.fb = fb

  '''
  Use a first derivative-based prior.
  '''
  def setFirstDerivativePrior(self, fb = 1, means = None):
    self.prior = "first derivative"
    if means == None:
      self.priorAttributes['bias'] = copy.deepcopy(self.truth.val)
    else:
      self.priorAttributes['bias'] = copy.deepcopy(means)
    self.fb = fb

  '''
  Whether to constrain normalisation.
  '''
  def setConstrainArea(self, value = True):
    self.constrainArea = value

  '''
  Add systematic uncertainty at reconstruction level.
  To add uncertainty due to the unfolding factors, one can proceed as follows, taking the difference between
  the reconstructed-level distributions using the alternative migration matrices or the nominal.

  from Unfolder.Histogram import getNormResponse
  addUncertainty(name, nominalBackground, np.dot(nominalTruth.val, getNormResponse(alternativeMigration, alternativeEfficiency).val))

  Where Nr is the number of reconstruction-level bins. The first term keeps the effect in the background at zero so that the background
  is not correlated with this uncertainty.
  The second term is the reconstruction-level histogram one would get if the nominal truth is folded with an alternative unfolding factor.
  '''
  def addUncertainty(self, name, bkg, reco):
    self.bkg_syst[name] = H1D(bkg) - self.bkg
    self.reco_syst[name] = H1D(reco) - self.recoWithoutFakes
    self.systematics.append(name)

  '''
  Add uncertainty in the core of the unfolding factors.
  Note that this creates a non-linearity in the model.
  Analytical calculations show that if one has a difference
  in efficiencies between the nominal and other models, this creates
  an extra inflection point in the likelihood at nuisance parameter = - (efficiency difference)/(nominal efficiency).
  At this point, the likelihood is exactly zero for a one bin analytical calculation.
  If this point is in the range sampled for the nuisance parameter, it will cause an asymmetry in the
  posterior of the nuisance parameter, which will shift the mean of the posterior of the
  unfolded distributions.
  It is recommended to add a linear uncertainty by calling:
  from Unfolder.Histogram import getNormResponse
  addUncertainty(name, nominalBackground, np.dot(nominalTruth.val, getNormResponse(alternativeMigration, alternativeEfficiency).val))

  Where Nr is the number of reconstruction-level bins. The first term keeps the effect in the background at zero so that the background
  is not correlated with this uncertainty.
  The second term is the reconstruction-level histogram one would get if the nominal truth is folded with an alternative unfolding factor.
  The prior can be "lognormal" or "normal" to choose what the prior on the nuisance parameter would be.
  A lognormal prior generates a posterior with a similar mean and mode, leading to a more natural interpretation of the marginal mean.
  '''
  def addUnfoldingUncertainty(self, name, bkg, mig, eff, prior = "categorical"):
    # calculate response matrix, defined as response[i, j] = P(r = j|t = i) = P(t = i, r = j)/P(t = i)
    self.response_unfsyst[name] = H2D(mig)
    self.bkg_unfsyst[name] = H1D(bkg)
    for i in range(0, self.Nt): # for each truth bin
      rsum = 0.0
      for j in range(0, self.Nr): # for each reco bin
        rsum += mig.val[i, j]    # calculate the sum of all reco bins in the same truth bin
      # rsum is now the total sum of events that has that particular truth bin
      # now, for each reco bin in truth bin i, divide that row by the total number of events in it
      # and multiply the response matrix by the efficiency
      for j in range(0, self.Nr):
        self.response_unfsyst[name].val[i, j] = mig.val[i, j]/rsum*eff.val[i]  # P(r|t) = P(t, r)/P(t) = Mtr*eff(t)/sum_k=1^Nr Mtk
        # keep the same efficiency (FIXME -- only uncomment for tests)
        #self.response_unfsyst[name].val[i, j] = mig.val[i, j]/rsum*self.eff.val[i]  # P(r|t) = P(t, r)/P(t) = Mtr*eff(t)/sum_k=1^Nr Mtk
        self.response_unfsyst[name].err[i, j] = 0 # FIXME
    self.unf_systematics.append(name)
    self.prior_unfsyst[name] = prior

  '''
  Set a uniform prior.
  '''
  def setUniformPrior(self, capAtZero = True):
    self.prior = "flat"
    if capAtZero:
      self.prior = "uniform"
    self.fb = 0

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

    # for the uniform prior, capping at zero
    self.minT = 0
    self.maxT = 10*np.amax(self.truth.val)


    self.model = pm.Model()                  # create the model
    with self.model:                         # all in this scope is in the model's context
      self.var_alpha = theano.shared(value = 1.0, borrow = False)
      self.var_data = theano.shared(value = self.data.val, borrow = False) # in case one wants to change data

      # Define the prior
      if self.prior == "gaussian":
        #self.T = pm.Normal('Truth', mu = self.priorAttributes['mean'], sd = self.priorAttributes['sd'], shape = (self.Nt))
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*0.5*theano.tensor.sqr((val - self.priorAttributes['bias'])/self.priorAttributes['sd']).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "entropy":
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*((val/val.sum())*theano.tensor.log(val/val.sum())).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "curvature":
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*theano.tensor.sqr(theano.tensor.extra_ops.diff(theano.tensor.extra_ops.diff((val - self.fb*self.priorAttributes['bias'])))).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "first derivative":
        self.T = pm.DensityDist('Truth', logp = lambda val: -self.var_alpha*theano.tensor.abs_(theano.tensor.extra_ops.diff(theano.tensor.extra_ops.diff((val - self.fb*self.priorAttributes['bias'])/(self.truth.x_err*2))/np.diff(self.truth.x))/(2*theano.tensor.mean(theano.tensor.extra_ops.diff(val/(self.truth.x_err*2))/np.diff(self.truth.x)))).sum(), shape = (self.Nt), testval = self.truth.val)
      elif self.prior == "uniform": # if using uniform, cap at zero
        self.T = pm.Uniform('Truth', self.minT, self.maxT, shape = (self.Nt), testval = self.truth.val)
      else: # if none of the names above matched, assume it is uniform
        self.T = pm.Flat('Truth', shape = (self.Nt), testval = self.truth.val)

      self.theta = {}
      self.unf_theta = {}
      for name in self.unf_systematics:
        if self.prior_unfsyst[name] == "lognormal":
          self.unf_theta[name] = pm.Lognormal('tu_'+name, mu = 0, sd = 0.69, testval = 1)
        elif self.prior_unfsyst[name] == 'categorical':
          self.unf_theta[name] = pm.Categorical('tu_'+name, p = [0.67, 0.33], testval = 0)
        else:
          self.unf_theta[name] = pm.Normal('tu_'+name, mu = 0, sd = 1, testval = 0)
        #self.unf_theta[name] = pm.Laplace('tu_'+name, mu = 0, b = (2.0)**(-0.5), testval = 0)
      for name in self.systematics:
        self.theta[name] = pm.Normal('t_'+name, mu = 0, sd = 1, testval = 0)

      self.var_bkg = theano.shared(value = self.asMat(self.bkg.val))
      self.var_response = theano.shared(value = self.asMat(self.response.val))
      self.R = theano.tensor.dot(self.T, self.var_response)

      self.var_response_unfsyst = {}
      self.var_response_alt_unfsyst = {}
      self.var_bkg_unfsyst = {}
      for name in self.unf_systematics:
        self.var_response_unfsyst[name] = theano.shared(value = self.asMat(self.response_unfsyst[name].val - self.response.val))
        self.var_response_alt_unfsyst[name] = theano.shared(value = self.asMat(self.response_unfsyst[name].val))
        self.var_bkg_unfsyst[name] = theano.shared(value = self.asMat(self.bkg_unfsyst[name].val - self.bkg.val))

      self.var_bkg_syst = {}
      self.var_reco_syst = {}
      self.R_syst = {}
      for name in self.systematics:
        # get background constribution
        self.var_bkg_syst[name] = theano.shared(value = self.asMat(self.bkg_syst[name].val))
        # calculate differential impact in the reco result
        self.var_reco_syst[name] = theano.shared(value = self.asMat(self.reco_syst[name].val))
        self.R_syst[name] = self.var_reco_syst[name] + self.var_bkg_syst[name]

      # reco result including systematics:
      self.R_full = 0

      for name in self.unf_systematics:
        # add it to the total reco result
        if self.prior_unfsyst[name] == 'lognormal':
          self.R_full += theano.tensor.dot((1 - self.unf_theta[name])*self.T, self.var_response_unfsyst[name])
          self.R_full += (1 - self.unf_theta[name])*self.var_bkg_unfsyst[name]
        elif self.prior_unfsyst[name] == 'categorical':
          self.R_full += theano.tensor.dot(1.0*self.unf_theta[name]*self.T, self.var_response_alt_unfsyst[name])
          self.R_full += self.unf_theta[name]*self.var_bkg_unfsyst[name]
          self.R = theano.tensor.dot(self.R, (1.0 - self.unf_theta[name]))
        else:
          self.R_full += theano.tensor.dot(self.unf_theta[name]*self.T, self.var_response_unfsyst[name])
          self.R_full += self.unf_theta[name]*self.var_bkg_unfsyst[name]

      for name in self.systematics:
        # add it to the total reco result
        self.R_full += self.theta[name]*self.R_syst[name]

      self.R_full += self.R

      # cut-off at 0 to create efficiency boundaries
      self.R_full = theano.tensor.maximum(self.R_full, 0)

      self.R_full += self.var_bkg

      self.U = pm.Poisson('U', mu = self.R_full, observed = self.var_data, shape = (self.Nr, 1))
      #self.U = pm.Normal('U', mu = self.R_full, sd = theano.tensor.sqrt(self.R_full), observed = self.var_data, shape = (self.Nr, 1))
      if self.constrainArea:
        self.Norm = pm.Poisson('Norm', mu = self.T.sum()*self.ave_eff + self.tot_bkg, observed = self.var_data.sum(), shape = (1))

      # define symbolic negative log-likelihood
      if doSymbolicMode:
        self.symb_data = []
        self.symb_T = []
        self.symb_theta = {}
        self.symb_R = [0 for k in range(self.Nr)]
        for i in range(self.Nr):
          self.symb_data.append(sympy.symbols('data_%d' % i))
        for i in range(self.Nt):
          self.symb_T.append(sympy.symbols('T_%d' % i))
        for name in self.systematics:
          self.symb_theta[name] = sympy.symbols('theta_%s' % name)
        for name in self.unf_systematics:
          self.symb_theta[name] = sympy.symbols('theta_%s' % name)
        for i in range(self.Nr):
          for k in range(self.Nt):
            self.symb_R[i] += self.symb_T[k] * self.response.val[k, i]
          self.symb_R[i] += self.bkg.val[i]
          for name in self.systematics:
            self.symb_R[i] += self.symb_theta[name]*(self.reco_syst[name].val[i] + self.bkg_syst[name].val[i])
          for name in self.unf_systematics:
            if self.prior_unfsyst[name] == 'lognormal':
              self.symb_R[i] += (1 - self.symb_theta[name])*(self.bkg_unfsyst[name].val[i] - self.bkg.val[i])
            else:
              self.symb_R[i] += self.symb_theta[name]*(self.bkg_unfsyst[name].val[i] - self.bkg.val[i])
            for k in range(self.Nt):
              if self.prior_unfsyst[name] == 'lognormal':
                self.symb_R[i] += (1 - self.symb_theta[name])*self.symb_T[k]*(self.response_unfsyst[name].val[k, i] - self.response.val[k, i])
              else:
                self.symb_R[i] += self.symb_theta[name]*self.symb_T[k]*(self.response_unfsyst[name].val[k, i] - self.response.val[k, i])
        self.dummy = sympy.symbols('dummy')
        for i in range(self.Nr):
          self.nll += -self.symb_data[i]*sympy.log(self.symb_R[i]) + self.symb_R[i] + sympy.summation(sympy.log(self.dummy), (self.dummy, 1, self.symb_data[i]))
        for name in self.systematics:
          self.nll += 0.5*self.symb_theta[name]**2 + 0.5*np.log(2*np.pi)
        for name in self.unf_systematics:
          if self.prior_unfsyst[name] == 'lognormal':
            mu = 0
            sd = 0.69
            tau = sd**(-2.0)
            self.nll += 0.5*tau*sympy.log(self.symb_theta[name] - mu)**2 - 0.5*np.log(tau/np.sqrt(2*np.pi)) + sympy.log(self.symb_theta[name])
          else:
            self.nll += 0.5*self.symb_theta[name]**2 + 0.5*np.log(2*np.pi)
        print("Defined -ln(L) symbolically as:")
        print(self.nll)

  '''
  Make theano graph used for calculation.
  '''
  def graph(self, fname = "graph.png"):
    from theano.printing import pydotprint
    pydotprint(self.model.logpt, outfile=fname, var_with_name_simple=True) 
    '''
    import matplotlib.pyplot as plt
    import networkx as nx
    fig = plt.figure(figsize=(10,10))
    graph = nx.DiGraph()
    variables = self.model.named_vars.values()
    for var in variables:
      graph.add_node(var)
      dist = var.distribution
      for param_name in getattr(dist, "vars", []):
        param_value = getattr(dist, param_name)
        owner = getattr(param_value, "owner", None)
        if param_value in variables:
          graph.add_edge(param_value, var)
        elif owner is not None:
          parents = _get_all_parents(param_value)
          for parent in parents:
            if parent in variables:
              graph.add_edge(parent, var)
    pos = nx.fruchterman_reingold_layout(graph)
    nx.draw_networkx(graph, pos, arrows=True, node_size=1000, node_color='w', font_size=8)
    plt.axis('off')
    plt.savefig(fname)
    plt.close()
    '''

  '''
  Calculate bias in each bin.
  '''
  def getBiasPerBinFromMAP(self, N, bkg = None, mig = None, eff = None):
    if bkg == None: bkg = self.bkg
    if mig == None: mig = self.mig
    if eff == None: eff = self.eff

    truth = mig.project('x')/eff
    fitted = np.zeros((N, len(self.truth.val)))
    bias = np.zeros(len(self.truth.val))
    import sys
    for k in range(0, N):
      if k % 100 == 0:
        print("getBiasFromMAP: Throwing toy experiment {0}/{1}\r".format(k, N))
        sys.stdout.flush()
      pseudo_data = getDataFromModel(bkg, mig, eff)
      self.setData(pseudo_data)
      res = pm.find_MAP(model = self.model, disp = False)
      if not 'Truth' in res and self.prior == "uniform":
        res["Truth"] = (self.maxT - self.minT) * np.exp(res["Truth_interval_"])/(1.0 + np.exp(res["Truth_interval_"])) + self.minT
      fitted[k, :] = res["Truth"]
    bias = np.mean(fitted, axis = 0)
    bias_std = np.std(fitted - bias, axis = 0, ddof = 1)
    plt_bias = H1D(truth)
    plt_bias.val = bias
    plt_bias.err = np.power(bias_std, 2)
    plt_bias.err_up = np.power(bias_std, 2)
    plt_bias.err_dw = np.power(bias_std, 2)
    return plt_bias

  '''
  Calculate the sum of the bias using only the expected values.
  '''
  def getBiasFromMAP(self, N, bkg = None, mig = None, eff = None):
    if bkg == None: bkg = self.bkg
    if mig == None: mig = self.mig
    if eff == None: eff = self.eff

    truth = mig.project('x')/eff
    fitted = np.zeros((N, len(self.truth.val)))
    bias = np.zeros(len(self.truth.val))
    bias_norm = np.zeros(N)
    import sys
    for k in range(0, N):
      if k % 100 == 0:
        print("getBiasFromMAP: Throwing toy experiment {0}/{1}\r".format(k, N))
        sys.stdout.flush()
      pseudo_data = getDataFromModel(bkg, mig, eff)
      self.setData(pseudo_data)
      #self.run(pseudo_data)
      with self.model:
        res = pm.find_MAP(disp = False)
        if not 'Truth' in res and (self.prior == "uniform" or self.prior == "flat"):
          res["Truth"] = (self.maxT - self.minT) * np.exp(res["Truth_interval_"])/(1.0 + np.exp(res["Truth_interval_"])) + self.minT
        fitted[k, :] = res["Truth"] - truth.val
        bias_norm[k] = np.sum(res['Truth'] - truth.val)
    print
    # systematic bias
    self.setData(mig.project('y') + bkg)
    with self.model:
      res = pm.find_MAP(disp = False)
      bias_syst = np.mean(np.abs(res["Truth"] - truth.val))
    bias = np.mean(fitted, axis = 0)
    bias_std = np.std(fitted, axis = 0, ddof = 1)
    bias_norm_mean = np.mean(bias_norm)
    bias_norm_std = np.std(bias_norm, ddof = 1)
    #print("getBiasFromMAP with alpha = ", self.var_alpha.get_value(), " N = ", N, ", mean, std = ", bias, bias_std)
    bias_binsum = np.mean(np.abs(bias))
    bias_std_binsum = np.mean(bias_std)
    bias_chi2 = np.mean(np.power(bias/bias_std, 2))
    return [bias_binsum, bias_std_binsum, bias_chi2, bias_norm_mean, bias_norm_std, bias_syst]

  '''
  Scan alpha values to minimise bias^2 over variance.
  '''
  def scanAlpha(self, bkg, mig, eff, N = 1000, rangeAlpha = np.arange(0.0, 10.0, 1.0), fname = "scanAlpha.png", fname_chi2 = "scanAlpha_chi2.png", fname_norm = "scanAlpha_norm.png"):
    bkp_alpha = self.var_alpha.get_value()
    bias = np.zeros(len(rangeAlpha))
    bias_std = np.zeros(len(rangeAlpha))
    bias_chi2 = np.zeros(len(rangeAlpha))
    bias_norm = np.zeros(len(rangeAlpha))
    bias_norm_std = np.zeros(len(rangeAlpha))
    bias_syst = np.zeros(len(rangeAlpha))
    minBias = 1e10
    bestAlpha = 0
    bestChi2 = 0
    bestI = 0
    import sys
    for i in range(0, len(rangeAlpha)):
      print("scanAlpha: parameter = ", rangeAlpha[i], " / ", rangeAlpha[-1])
      sys.stdout.flush()
      self.setAlpha(rangeAlpha[i])
      bias[i], bias_std[i], bias_chi2[i], bias_norm[i], bias_norm_std[i], bias_syst[i] = self.getBiasFromMAP(N, bkg, mig, eff) # only take mean values for speed
      print(" -- --> scanAlpha: parameter = ", rangeAlpha[i], " / ", rangeAlpha[-1], " with chi2 = ", bias_chi2[i], ", mean and std = ", bias[i], bias_std[i])
      if np.abs(bias_chi2[i] - 0.5) < minBias:
        minBias = np.abs(bias_chi2[i] - 0.5)
        bestChi2 = bias_chi2[i]
        bestAlpha = rangeAlpha[i]
        bestI = i
    fig = plt.figure(figsize=(10, 10))
    plt_bias = H1D(bias)
    plt_bias.val = bias
    plt_bias.err = np.zeros(len(rangeAlpha))
    plt_bias.err_up = np.zeros(len(rangeAlpha))
    plt_bias.err_dw = np.zeros(len(rangeAlpha))
    plt_bias.x = rangeAlpha
    plt_bias.x_err = np.zeros(len(rangeAlpha))
    plt_bias_e = H1D(bias)
    plt_bias_e.val = bias_std
    plt_bias_e.err = np.zeros(len(rangeAlpha))
    plt_bias_e.err_up = np.zeros(len(rangeAlpha))
    plt_bias_e.err_dw = np.zeros(len(rangeAlpha))
    plt_bias_e.x = rangeAlpha
    plt_bias_e.x_err = np.zeros(len(rangeAlpha))
    plt_bias_syst = H1D(bias)
    plt_bias_syst.val = bias_syst
    plt_bias_syst.err = np.zeros(len(rangeAlpha))
    plt_bias_syst.err_up = np.zeros(len(rangeAlpha))
    plt_bias_syst.err_dw = np.zeros(len(rangeAlpha))
    plt_bias_syst.x = rangeAlpha
    plt_bias_syst.x_err = np.zeros(len(rangeAlpha))
    #plotH1DLines({r"$E_{\mathrm{bins}}[|E_{\mathrm{toys}}[\mathrm{bias}]|]$": plt_bias, r"$E_{\mathrm{bins}}[\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]}]$": plt_bias_e, r"$E_{\mathrm{bins}}[|\mathrm{only \;\; syst. \;\; bias}|]$": plt_bias_syst}, r"$\alpha$", "Bias", "", fname)
    plotH1DLines({r"$E_{\mathrm{bins}}[|E_{\mathrm{toys}}[\mathrm{bias}]|]$": plt_bias, r"$E_{\mathrm{bins}}[\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]}]$": plt_bias_e}, r"$\alpha$", "Bias", "", fname)
    plt_bias_norm = H1D(bias)
    plt_bias_norm.val = bias_norm
    plt_bias_norm.err = np.zeros(len(rangeAlpha))
    plt_bias_norm.err_up = np.zeros(len(rangeAlpha))
    plt_bias_norm.err_dw = np.zeros(len(rangeAlpha))
    plt_bias_norm.x = rangeAlpha
    plt_bias_norm.x_err = np.zeros(len(rangeAlpha))
    plt_bias_norm_e = H1D(bias)
    plt_bias_norm_e.val = bias_norm_std
    plt_bias_norm_e.err = np.zeros(len(rangeAlpha))
    plt_bias_norm_e.err_up = np.zeros(len(rangeAlpha))
    plt_bias_norm_e.err_dw = np.zeros(len(rangeAlpha))
    plt_bias_norm_e.x = rangeAlpha
    plt_bias_norm_e.x_err = np.zeros(len(rangeAlpha))
    plotH1DLines({r"$E_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]$": plt_bias_norm, r"$\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]}$": plt_bias_norm_e}, r"$\alpha$", "Normalisation bias", "", fname_norm)
    plt_bias_chi2 = H1D(bias_chi2)
    plt_bias_chi2.val = bias_chi2
    plt_bias_chi2.err = np.ones(len(rangeAlpha))*np.sqrt(float(len(self.truth.val))/float(N)) # error in chi^2 considering errors in the mean of std/sqrt(N)
    plt_bias_chi2.err_up = np.ones(len(rangeAlpha))*np.sqrt(float(len(self.truth.val))/float(N)) # error in chi^2 considering errors in the mean of std/sqrt(N)
    plt_bias_chi2.err_dw = np.ones(len(rangeAlpha))*np.sqrt(float(len(self.truth.val))/float(N)) # error in chi^2 considering errors in the mean of std/sqrt(N)
    plt_bias_chi2.x = rangeAlpha
    plt_bias_chi2.x_err = np.zeros(len(rangeAlpha))
    plt_cte = H1D(plt_bias_chi2)
    plt_cte.val = 0.5*np.ones(len(rangeAlpha))
    plt_cte.err = np.zeros(len(rangeAlpha))
    plt_cte.err_up = np.zeros(len(rangeAlpha))
    plt_cte.err_dw = np.zeros(len(rangeAlpha))
    plotH1DLines({r"$E_{\mathrm{bins}}[E_{\mathrm{toys}}[\mathrm{bias}]^2/\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]]$": plt_bias_chi2, "0.5": plt_cte}, r"$\alpha$", r"Bias $\mathrm{mean}^2/\mathrm{variance}$", "", fname_chi2)
    self.setAlpha(bkp_alpha)
    return [bestAlpha, bestChi2, bias[bestI], bias_std[bestI], bias_norm[bestI], bias_norm_std[bestI]]
    
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
  def sample(self, N = 50000):
    self.N = N
    with self.model:
      #start = pm.find_MAP()
      #step = pm.NUTS(state = start)
      self.trace = pm.sample(N) #, step, start = start)
      self.N = self.trace.Truth.shape[0]
      #print("Number of truth samples:", self.N)

      #pm.summary(self.trace)

      self.hnp = H1D(np.zeros(len(self.systematics)))
      self.hnpu = H1D(np.zeros(len(self.unf_systematics)))
      self.hunf = H1D(self.truth)

      self.hunf_median = H1D(self.truth)
      self.hnp_median = H1D(np.zeros(len(self.systematics)))
      self.hnpu_median = H1D(np.zeros(len(self.unf_systematics)))
      self.hnp_median.x = [""]*len(self.systematics)
      self.hnpu_median.x = [""]*len(self.unf_systematics)

      self.hunf_mode = H1D(self.truth)
      self.hnp_mode = H1D(np.zeros(len(self.systematics)))
      self.hnpu_mode = H1D(np.zeros(len(self.unf_systematics)))
      self.hnp_mode.x = [""]*len(self.systematics)
      self.hnpu_mode.x = [""]*len(self.unf_systematics)

      self.hunf_smode = H1D(self.truth)
      self.hnp_smode = H1D(np.zeros(len(self.systematics)))
      self.hnpu_smode = H1D(np.zeros(len(self.unf_systematics)))
      self.hnp_smode.x = [""]*len(self.systematics)
      self.hnpu_smode.x = [""]*len(self.unf_systematics)

      x0 = np.zeros(self.Nt+len(self.systematics)+len(self.unf_systematics))
      for i in range(0, self.Nt):
        m = np.mean(self.trace.Truth[:, i])
        s = np.std(self.trace.Truth[:, i], ddof = 1)
        x0[i] = m
        self.hunf.val[i] = m
        self.hunf.err[i] = s**2
        self.hunf.err_up[i] = s**2
        self.hunf.err_dw[i] = s**2
        self.hunf_median.val[i] = np.median(self.trace.Truth[:, i])
        self.hunf_median.err[i] = s**2
        self.hunf_median.err_up[i] = s**2
        self.hunf_median.err_dw[i] = s**2
      

      self.hnp.x = [""]*len(self.systematics)
      self.hnpu.x = [""]*len(self.unf_systematics)
      for k in range(0, len(self.systematics)):
        m = np.mean(self.trace['t_'+self.systematics[k]])
        s = np.std(self.trace['t_'+self.systematics[k]], ddof = 1)
        x0[self.Nt+k] = m
        self.hnp.val[k] = m
        self.hnp.err[k] = s**2
        self.hnp.err_up[k] = s**2
        self.hnp.err_dw[k] = s**2
        self.hnp.x[k] = self.systematics[k]
        self.hnp.x_err[k] = 1
        self.hnp_median.val[k] = np.median(self.trace['t_'+self.systematics[k]])
        self.hnp_median.err[k] = s**2
        self.hnp_median.err_up[k] = s**2
        self.hnp_median.err_dw[k] = s**2
        self.hnp_median.x[k] = self.systematics[k]
        self.hnp_median.x_err[k] = 1
      for k in range(0, len(self.unf_systematics)):
        m = np.mean(self.trace['tu_'+self.unf_systematics[k]])
        s = np.std(self.trace['tu_'+self.unf_systematics[k]], ddof = 1)
        x0[self.Nt+len(self.systematics)+k] = m
        self.hnpu.val[k] = m
        self.hnpu.err[k] = s**2
        self.hnpu.err_up[k] = s**2
        self.hnpu.err_dw[k] = s**2
        self.hnpu.x[k] = self.unf_systematics[k]
        self.hnpu.x_err[k] = 1
        self.hnpu_median.val[k] = np.median(self.trace['tu_'+self.unf_systematics[k]])
        self.hnpu_median.err[k] = s**2
        self.hnpu_median.err_up[k] = s**2
        self.hnpu_median.err_dw[k] = s**2
        self.hnpu_median.x[k] = self.unf_systematics[k]
        self.hnpu_median.x_err[k] = 1

      # get mode
      mode = self.getPosteriorMode()
      for k in range(0, self.Nt):
        self.hunf_mode.val[k] = mode.x[k]
        #self.hunf_mode.err[k] = mode.hess_inv.todense()[k,k]
        self.hunf_mode.err[k] = mode.errMode[k]**2
        self.hunf_mode.err_up[k] = mode.errModeUp[k]**2
        self.hunf_mode.err_dw[k] = mode.errModeDw[k]**2
        self.hunf_smode.val[k] = mode.symb_x[k]
        self.hunf_smode.err[k] = mode.symb_errMode[k]**2
        self.hunf_smode.err_up[k] = mode.symb_errModeUp[k]**2
        self.hunf_smode.err_dw[k] = mode.symb_errModeDw[k]**2
      for k in range(0, len(self.systematics)):
        self.hnp_mode.val[k] = mode.x[self.Nt+k]
        #self.hnp_mode.err[k] = mode.hess_inv.todense()[self.Nt+k, self.Nt+k]
        self.hnp_mode.err[k] = mode.errMode[self.Nt+k]**2
        self.hnp_mode.err_up[k] = mode.errModeUp[self.Nt+k]**2
        self.hnp_mode.err_dw[k] = mode.errModeDw[self.Nt+k]**2
        self.hnp_mode.x[k] = self.systematics[k]
        self.hnp_mode.x_err[k] = 1
        self.hnp_smode.val[k] = mode.symb_x[self.Nt+k]
        self.hnp_smode.err[k] = mode.symb_errMode[self.Nt+k]**2
        self.hnp_smode.err_up[k] = mode.symb_errModeUp[self.Nt+k]**2
        self.hnp_smode.err_dw[k] = mode.symb_errModeDw[self.Nt+k]**2
        self.hnp_smode.x[k] = self.systematics[k]
        self.hnp_smode.x_err[k] = 1
      for k in range(0, len(self.unf_systematics)):
        self.hnpu_mode.val[k] = mode.x[self.Nt+len(self.systematics)+k]
        #self.hnpu_mode.err[k] = mode.hess_inv.todense()[self.Nt+len(self.systematics)+k, self.Nt+len(self.systematics)+k]
        self.hnpu_mode.err[k] = mode.errMode[self.Nt+len(self.systematics)+k]**2
        self.hnpu_mode.err_up[k] = mode.errModeUp[self.Nt+len(self.systematics)+k]**2
        self.hnpu_mode.err_dw[k] = mode.errModeDw[self.Nt+len(self.systematics)+k]**2
        self.hnpu_mode.x[k] = self.unf_systematics[k]
        self.hnpu_mode.x_err[k] = 1
        self.hnpu_smode.val[k] = mode.symb_x[self.Nt+len(self.systematics)+k]
        self.hnpu_smode.err[k] = mode.symb_errMode[self.Nt+len(self.systematics)+k]**2
        self.hnpu_smode.err_up[k] = mode.symb_errModeUp[self.Nt+len(self.systematics)+k]**2
        self.hnpu_smode.err_dw[k] = mode.symb_errModeDw[self.Nt+len(self.systematics)+k]**2
        self.hnpu_smode.x[k] = self.unf_systematics[k]
        self.hnpu_smode.x_err[k] = 1

  '''
  Estimate modes from GKE.
  '''
  def getPosteriorMode(self):
    d = self.Nt + len(self.systematics) + len(self.unf_systematics)
    r = self.N
    value = np.zeros( (d, r) )
    S = np.zeros( (d, 1) )
    dS = np.zeros( (d, 1) )
    for i in range(0, self.Nt):
      value[i, :] = copy.deepcopy(self.trace.Truth[:, i])
      m = np.mean(self.trace.Truth[:, i])
      s = np.std(self.trace.Truth[:, i])
      S[i, 0] = m
      dS[i, 0] = s
    for i in range(0, len(self.systematics)):
      value[self.Nt + i, :] = copy.deepcopy(self.trace['t_'+self.systematics[i]][:])
      m = np.mean(self.trace['t_'+self.systematics[i]])
      s = np.std(self.trace['t_'+self.systematics[i]])
      S[self.Nt + i, 0] = m
      dS[self.Nt + i, 0] = s
    for i in range(0, len(self.unf_systematics)):
      value[self.Nt +len(self.systematics) + i, :] = copy.deepcopy(self.trace['tu_'+self.unf_systematics[i]][:])
      m = np.mean(self.trace['tu_'+self.unf_systematics[i]])
      s = np.std(self.trace['tu_'+self.unf_systematics[i]])
      S[self.Nt + len(self.systematics) + i, 0] = m
      dS[self.Nt +len(self.systematics) + i, 0] = s
    pdf = stats.gaussian_kde(value)
    def mpdf(x, args):
      pdf = args['pdf']
      p = pdf(x)
      p = np.asarray(p)
      p[p == 0] = 1e20
      p[p > 0] = -np.log(p)
      #if p > 0:
      #  p = -np.log(p)
      #elif p == 0:
      #  p = 1e20
      #else:
      #  print("Negative PDF:", p)
      return p

    bounds = []
    for i in range(0, self.Nt+len(self.systematics)+len(self.unf_systematics)):
      bounds.append((S[i, 0]-5*dS[i,0], S[i,0]+5*dS[i,0]))
    for i in range(len(self.unf_systematics)):
      if self.prior_unfsyst[self.unf_systematics[i]] == 'lognormal':
        bounds[self.Nt+len(self.systematics)+i] = (1e-6, S[i,0]+3*dS[i,0])
    args = {'pdf': pdf}
    print("Start minimization with %s = %f" % (str(S), mpdf(S, args)))
    res = optimize.minimize(mpdf, S, args = args, bounds = bounds, method='L-BFGS-B', options={'maxiter': 100, 'disp': True})
    print(res)

    # get 68% interval
    # T is a list of linear spaces around the mode +/- 5 sigma
    # ie: T = [np.linspace(mode1 - 5*sigma1, mode1 + 5*sigma1, Ngrid), np.linspace(mode2 - 5*sigma2, mode2 + 5*sigma2, Ngrid), ...]
    T = []
    Ndims = self.Nt+len(self.systematics)+len(self.unf_systematics)
    Ngrid = 1000*Ndims
    for i in range(0, self.Nt+len(self.systematics)+len(self.unf_systematics)):
      if i >= self.Nt+len(self.systematics) and self.prior_unfsyst[self.unf_systematics[i - self.Nt+len(self.systematics)]] == 'lognormal':
        T.append(np.linspace(1e-6, res.x[i]+5*dS[i,0], Ngrid))
      else:
        T.append(np.linspace(res.x[i]-5*dS[i,0], res.x[i]+5*dS[i,0], Ngrid))

    # build mesh with coordinates
    meshT = np.meshgrid(T)
    # meshTpos[i, k] has the i-th coordinate for mesh point k
    meshTpos = np.reshape(np.vstack(map(np.ravel, meshT)), (Ndims, -1))

    # get the -ln(pdf) value for each mesh point k
    meshPdf = mpdf(meshTpos, args)
    # get index permutation that would order the pdf values list in increasing order of -ln(pdf)
    pdfPerm = meshPdf.argsort()
    # get pdf values under decreasing order (since indices above are in increasing order of -ln(pdf) )
    orderedPdf = np.exp(-meshPdf[pdfPerm])
    # get pdf normalisation to normalise it to 1
    pdfNorm = np.sum(orderedPdf)
    # get decreasingly ordered pdf values so that total integral is 1
    orderedPdfNorm = np.asarray([x/pdfNorm for x in orderedPdf])
    # perform cumulative sum to get cdf values, summing up most probable values first
    orderedCdf = np.cumsum(orderedPdfNorm)
    # get first index of the cdf, where the cumulative probability is larger than 68%
    boundaryIdx = np.argwhere(orderedCdf > 0.68)[0][0]
    # transpose mesh (so now i-th coordinate for mesh point k is in index (k, i) ) and get points ordered by most probable mesh point to least probable
    orderedMesh = (meshTpos.T)[pdfPerm]
    # get only mesh points up to 68% boundary under the ordering principle above
    confidenceSurface = orderedMesh[0:boundaryIdx, :]
    # get maximum and minimum values for each coordinate i
    errMin = [np.amin(confidenceSurface[:, i]) for i in range(0, Ndims)]
    errMax = [np.amax(confidenceSurface[:, i]) for i in range(0, Ndims)]
    # get mode errors by taking differences between values
    modeErr = [0.5*(errMax[k] - errMin[k]) for k in range(0, Ndims)]
    modeErrUp = [errMax[k] - res.x[k] for k in range(0, Ndims)]
    modeErrDw = [res.x[k] - errMin[k] for k in range(0, Ndims)]
    # add this to result structure
    res.errMode = modeErr
    res.errModeUp = modeErrUp
    res.errModeDw = modeErrDw

    res.symb_x = res.x
    res.symb_errMode = modeErr
    res.symb_errModeUp = modeErrUp
    res.symb_errModeDw = modeErrDw
    print(res)

    # now doing it symbolically
    if doSymbolicMode:
      dataTuples = []
      for i in range(self.Nr):
        dataTuples.append( (self.symb_data[i], int(self.data.val[i])) )
      nllWithData = self.nll.subs(dataTuples)
      print("-ln(L) with data substituted in:")
      print(nllWithData)
      xx = [self.symb_T[k] for k in range(self.Nt)] + [self.symb_theta[k] for k in self.systematics] + [self.symb_theta[k] for k in self.unf_systematics]
      nllWithDataHandle = sympy.utilities.lambdify(xx, nllWithData, modules = 'numpy')
      def nllWithDataProxy(zz):
        return nllWithDataHandle(*zz)
      jac = [nllWithData.diff(s) for s in xx]
      jacHandle = [sympy.utilities.lambdify(xx, jf, modules = 'numpy') for jf in jac]
      def jacProxy(zz):
        return np.array([jfn(*zz) for jfn in jacHandle])
      symb_res = optimize.minimize(nllWithDataProxy, S, bounds = bounds, jac = jacProxy, method='L-BFGS-B', options={'maxiter': 100, 'disp': True})
      print("Symbolic minimisation:")
      print(symb_res)
      res.symb_x = symb_res.x

      # get 68% interval
      # T is a list of linear spaces around the mode +/- 2 sigma
      # ie: T = [np.linspace(mode1 - 2*sigma1, mode1 + 2*sigma1, Ngrid), np.linspace(mode2 - 2*sigma2, mode2 + 2*sigma2, Ngrid), ...]
      symb_T = []
      for i in range(0, self.Nt+len(self.systematics)+len(self.unf_systematics)):
        if i >= self.Nt+len(self.systematics) and self.prior_unfsyst[self.unf_systematics[i - self.Nt+len(self.systematics)]] == 'lognormal':
          symb_T.append(np.linspace(1e-6, res.x[i]+2*dS[i,0], Ngrid))
        else:
          symb_T.append(np.linspace(res.symb_x[i]-2*dS[i,0], res.symb_x[i]+2*dS[i,0], Ngrid))
  
      # build mesh with coordinates
      symb_meshT = np.meshgrid(symb_T)
      # meshTpos[i, k] has the i-th coordinate for mesh point k
      symb_meshTpos = np.reshape(np.vstack(map(np.ravel, symb_meshT)), (Ndims, -1))

      # get the -ln(pdf) value for each mesh point k
      symb_meshPdf = nllWithDataProxy(symb_meshTpos)
      # get index permutation that would order the pdf values list in increasing order of -ln(pdf)
      symb_pdfPerm = symb_meshPdf.argsort()
      # get pdf values under decreasing order (since indices above are in increasing order of -ln(pdf) )
      symb_orderedPdf = np.exp(-symb_meshPdf[symb_pdfPerm])
      # get pdf normalisation to normalise it to 1
      symb_pdfNorm = np.sum(symb_orderedPdf)
      # get decreasingly ordered pdf values so that total integral is 1
      symb_orderedPdfNorm = np.asarray([x/symb_pdfNorm for x in symb_orderedPdf])
      # perform cumulative sum to get cdf values, summing up most probable values first
      symb_orderedCdf = np.cumsum(symb_orderedPdfNorm)
      # get first index of the cdf, where the cumulative probability is larger than 68%
      symb_boundaryIdx = np.argwhere(symb_orderedCdf > 0.68)[0][0]
      # transpose mesh (so now i-th coordinate for mesh point k is in index (k, i) ) and get points ordered by most probable mesh point to least probable
      symb_orderedMesh = (symb_meshTpos.T)[symb_pdfPerm]
      # get only mesh points up to 68% boundary under the ordering principle above
      symb_confidenceSurface = symb_orderedMesh[0:symb_boundaryIdx, :]
      # get maximum and minimum values for each coordinate i
      symb_errMin = [np.amin(symb_confidenceSurface[:, i]) for i in range(0, Ndims)]
      symb_errMax = [np.amax(symb_confidenceSurface[:, i]) for i in range(0, Ndims)]
      # get mode errors by taking differences between values
      symb_modeErr = [0.5*(symb_errMax[k] - symb_errMin[k]) for k in range(0, Ndims)]
      symb_modeErrUp = [symb_errMax[k] - res.symb_x[k] for k in range(0, Ndims)]
      symb_modeErrDw = [res.symb_x[k] - symb_errMin[k] for k in range(0, Ndims)]
      # add this to result structure
      res.symb_errMode = symb_modeErr
      res.symb_errModeUp = symb_modeErrUp
      res.symb_errModeDw = symb_modeErrDw

    print(res)
    
    return res

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
      ax.axvline(self.hunf_mode.val[i], linestyle = '--', linewidth = 1.5, color = 'r', label = 'Mode')
      if doSymbolicMode:
        ax.axvline(self.hunf_smode.val[i], linestyle = '-.', linewidth = 1.5, color = 'b', label = 'Mode (symbolic)')
      ax.axvline(self.hunf.val[i], linestyle = ':', linewidth = 1.5, color = 'm', label = 'Marginal mean')
      ax.axvline(self.hunf_median.val[i], linestyle = '-', linewidth = 1.5, color = 'k', label = 'Marginal median')
      ax.legend()
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
    plt.axvline(self.hnp_mode.val[i], linestyle = '--', linewidth = 1.5, color = 'r', label = 'Mode')
    if doSymbolicMode:
      plt.axvline(self.hnp_smode.val[i], linestyle = '-.', linewidth = 1.5, color = 'b', label = 'Mode (symbolic)')
    plt.axvline(self.hnp.val[i], linestyle = ':', linewidth = 1.5, color = 'm', label = 'Marginal mean')
    plt.axvline(self.hnp_median.val[i], linestyle = '-', linewidth = 1.5, color = 'k', label = 'Marginal median')
    plt.title(self.systematics[i])
    plt.ylabel("Probability")
    plt.xlim([-5, 5])
    plt.xlabel("Nuisance parameter value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s"%fname)
    plt.close()

  '''
  Plot the distributions for each unfolding nuisance parameter regardless of the other bins
  '''
  def plotNPUMarginal(self, syst, fname):
    i = self.unf_systematics.index(syst)
    fig = plt.figure(figsize=(10, 10))
    sns.distplot(self.trace['tu_'+self.unf_systematics[i]], kde = True, hist = True, label = self.unf_systematics[i])
    plt.axvline(self.hnpu_mode.val[i], linestyle = '--', linewidth = 1.5, color = 'r', label = 'Mode')
    if doSymbolicMode:
      plt.axvline(self.hnpu_smode.val[i], linestyle = '-.', linewidth = 1.5, color = 'b', label = 'Mode (symbolic)')
    plt.axvline(self.hnpu.val[i], linestyle = ':', linewidth = 1.5, color = 'm', label = 'Marginal mean')
    plt.axvline(self.hnpu_median.val[i], linestyle = '-', linewidth = 1.5, color = 'k', label = 'Marginal median')
    plt.title(self.unf_systematics[i])
    plt.ylabel("Probability")
    if self.prior_unfsyst[syst] == 'lognormal':
      plt.xlim([0, 8])
    else:
      plt.xlim([-5, 5])
    plt.xlabel("Nuisance parameter value")
    plt.legend()
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
    plotH2D(np.cov(self.trace.Truth, rowvar = 0), "", "", "Covariance of unfolded bins", logz=False, fname = fname, fmt = "")

  '''
  Plot the Pearson correlation coefficients.
  '''
  def plotCorr(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH2D(np.corrcoef(self.trace.Truth, rowvar = 0), "", "", "Correlation of unfolded bins", logz = False, fname = fname)

  '''
  Plot the Pearson correlation coefficients including the NPs.
  '''
  def plotCorrWithNP(self, fname):
    tmp = np.zeros((self.Nt+len(self.systematics)+len(self.unf_systematics), self.N))
    for i in range(0, self.Nt):
      tmp[i, :] = self.trace.Truth[:, i]
    for i in range(0, len(self.systematics)):
      tmp[self.Nt+i, :] = self.trace['t_'+self.systematics[i]]
    for i in range(0, len(self.unf_systematics)):
      tmp[self.Nt+len(self.systematics)+i, :] = self.trace['tu_'+self.unf_systematics[i]]
    tmplabel = ["%d" % i for i in range(0, self.Nt)] + self.systematics + self.unf_systematics
    plotH2DWithText(np.corrcoef(tmp, rowvar = 1), tmplabel, "", "", "Correlation of posterior", fname)

  '''
  Plot skewness.
  '''
  def plotSkewness(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.skew(self.trace.Truth, axis = 0, bias = False)
    sk.err = np.zeros(len(sk.val))
    sk.err_up = np.zeros(len(sk.val))
    sk.err_dw = np.zeros(len(sk.val))
    plotH1D(sk, "Unfolded bins", "Skewness", "Skewness of unfolded bins", logy = False, fname = fname)

  '''
  Plot nuisance parameter mode.
  '''
  def plotNP(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH1DWithText(self.hnp_mode, "Nuisance parameter", "Nuisance parameter mode", fname)

  '''
  Plot unfolding nuisance parameter means and spread.
  '''
  def plotNPU(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH1DWithText(self.hnpu_mode, "Nuisance parameter", "Nuisance parameter mode", fname)

  '''
  Plot kurtosis.
  '''
  def plotKurtosis(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.kurtosis(self.trace.Truth, axis = 0, fisher = True, bias = False)
    sk.err = np.zeros(len(sk.val))
    sk.err_up = np.zeros(len(sk.val))
    sk.err_dw = np.zeros(len(sk.val))
    plotH1D(sk, "Unfolded bins", "Fisher kurtosis", "Kurtosis of unfolded bins", logy = False, fname = fname)

  '''
  Plot data, truth, reco and unfolded result
  '''
  def plotUnfolded(self, fname = "plotUnfolded.png"):
    fig = plt.figure(figsize=(10, 10))
    ymax = 0
    for item in [self.truth, self.hunf_mode, self.hunf_smode, self.hunf]:
      ma = np.amax(item.val)
      ymax = np.amax([ymax, ma])
    #plt.errorbar(self.data.x, self.data.val, self.data.err**0.5, self.data.x_err, fmt = 'bs', linewidth=2, label = "Pseudo-data", markersize=10)
    #plt.errorbar(self.datasubbkg.x, self.datasubbkg.val, self.datasubbkg.err**0.5, self.datasubbkg.x_err, fmt = 'co', linewidth=2, label = "Background subtracted", markersize=10)
    #plt.errorbar(self.recoWithoutFakes.x, self.recoWithoutFakes.val, self.recoWithoutFakes.err**0.5, self.recoWithoutFakes.x_err, fmt = 'mv', linewidth=2, label = "Expected signal (no fakes) distribution", markersize=5)
    plt.errorbar(self.truth.x, self.truth.val, [self.truth.err_dw**0.5, self.truth.err_up**0.5], self.truth.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(self.hunf_median.x, self.hunf_median.val, [self.hunf_median.err_dw**0.5, self.hunf_median.err_up**0.5], self.hunf_median.x_err, fmt = 'k^', linewidth=2, label = "Marginal median", markersize = 5)
    if doSymbolicMode:
      plt.errorbar(self.hunf_smode.x, self.hunf_smode.val, [self.hunf_smode.err_dw**0.5, self.hunf_smode.err_up**0.5], self.hunf_smode.x_err, fmt = 'b^', linewidth=2, label = "Unfolded mode (symbolic)", markersize = 5)
    plt.errorbar(self.hunf.x, self.hunf.val, [self.hunf.err_dw**0.5, self.hunf.err_up**0.5], self.hunf.x_err, fmt = 'rv', linewidth=2, label = "Marginal mean", markersize=5)
    plt.errorbar(self.hunf_mode.x, self.hunf_mode.val, [self.hunf_mode.err_dw**0.5, self.hunf_mode.err_up**0.5], self.hunf_mode.x_err, fmt = 'm^', linewidth=2, label = "Unfolded mode", markersize = 5)
    plt.ylim([0, ymax*1.2])
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
    observedCs_mode = f*self.hunf_mode
    if normaliseByBinWidth:
      expectedCs = expectedCs.overBinWidth()
      observedCs = observedCs.overBinWidth()
      observedCs_mode = observedCs_mode.overBinWidth()
    plt.errorbar(expectedCs.x, expectedCs.val, [expectedCs.err_dw**0.5, expectedCs.err_up**0.5], expectedCs.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(observedCs.x, observedCs.val, [observedCs.err_dw**0.5, observedCs.err_up**0.5], observedCs.x_err, fmt = 'rv', linewidth=2, label = "Marginal mean", markersize=5)
    plt.errorbar(observedCs_mode.x, observedCs_mode.val, [observedCs_mode.err_dw**0.5, observedCs_mode.err_up**0.5], observedCs_mode.x_err, fmt = 'bv', linewidth=2, label = "Unfolded mode", markersize=5)
    ymax = 0
    for item in [expectedCs, observedCs, observedCs_mode]:
      ma = np.amax(item.val)
      ymax = np.amax([ymax, ma])
    plt.legend()
    if units != "":
      plt.ylabel("Differential cross section ["+units+"]")
    else:
      plt.ylabel("Events")
    plt.xlabel("Observable")
    plt.ylim([0, ymax*1.2])
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

