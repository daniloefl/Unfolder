
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
  '''
  def addUnfoldingUncertainty(self, name, bkg, mig, eff):
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
        self.response_unfsyst[name].err[i, j] = 0 # FIXME
    self.unf_systematics.append(name)

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
        self.unf_theta[name] = pm.Normal('tu_'+name, mu = 0, sd = 1, testval = 0)
      for name in self.systematics:
        self.theta[name] = pm.Normal('t_'+name, mu = 0, sd = 1, testval = 0)

      self.var_bkg = theano.shared(value = self.asMat(self.bkg.val))
      self.var_response = theano.shared(value = self.asMat(self.response.val))
      self.R = theano.tensor.dot(self.T, self.var_response) + self.var_bkg

      self.var_response_unfsyst = {}
      self.var_bkg_unfsyst = {}
      for name in self.unf_systematics:
        self.var_response_unfsyst[name] = theano.shared(value = self.asMat(self.response_unfsyst[name].val - self.response.val))
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
      self.R_full = self.R

      for name in self.systematics:
        # add it to the total reco result
        self.R_full += self.theta[name]*self.R_syst[name]

      for name in self.unf_systematics:
        # add it to the total reco result
        #self.R_full += theano.tensor.dot(self.unf_theta[name]*self.asMat(self.truth.val), self.var_response_unfsyst[name])
        self.R_full += theano.tensor.dot(self.unf_theta[name], self.var_bkg_unfsyst[name] + theano.tensor.dot(self.T, self.var_response_unfsyst[name]))

      self.U = pm.Poisson('U', mu = self.R_full, observed = self.var_data, shape = (self.Nr, 1))
      if self.constrainArea:
        self.Norm = pm.Poisson('Norm', mu = self.T.sum()*self.ave_eff + self.tot_bkg, observed = self.var_data.sum(), shape = (1))

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
    plt_bias.x = rangeAlpha
    plt_bias.x_err = np.zeros(len(rangeAlpha))
    plt_bias_e = H1D(bias)
    plt_bias_e.val = bias_std
    plt_bias_e.err = np.zeros(len(rangeAlpha))
    plt_bias_e.x = rangeAlpha
    plt_bias_e.x_err = np.zeros(len(rangeAlpha))
    plt_bias_syst = H1D(bias)
    plt_bias_syst.val = bias_syst
    plt_bias_syst.err = np.zeros(len(rangeAlpha))
    plt_bias_syst.x = rangeAlpha
    plt_bias_syst.x_err = np.zeros(len(rangeAlpha))
    #plotH1DLines({r"$E_{\mathrm{bins}}[|E_{\mathrm{toys}}[\mathrm{bias}]|]$": plt_bias, r"$E_{\mathrm{bins}}[\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]}]$": plt_bias_e, r"$E_{\mathrm{bins}}[|\mathrm{only \;\; syst. \;\; bias}|]$": plt_bias_syst}, r"$\alpha$", "Bias", "", fname)
    plotH1DLines({r"$E_{\mathrm{bins}}[|E_{\mathrm{toys}}[\mathrm{bias}]|]$": plt_bias, r"$E_{\mathrm{bins}}[\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]}]$": plt_bias_e}, r"$\alpha$", "Bias", "", fname)
    plt_bias_norm = H1D(bias)
    plt_bias_norm.val = bias_norm
    plt_bias_norm.err = np.zeros(len(rangeAlpha))
    plt_bias_norm.x = rangeAlpha
    plt_bias_norm.x_err = np.zeros(len(rangeAlpha))
    plt_bias_norm_e = H1D(bias)
    plt_bias_norm_e.val = bias_norm_std
    plt_bias_norm_e.err = np.zeros(len(rangeAlpha))
    plt_bias_norm_e.x = rangeAlpha
    plt_bias_norm_e.x_err = np.zeros(len(rangeAlpha))
    plotH1DLines({r"$E_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]$": plt_bias_norm, r"$\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]}$": plt_bias_norm_e}, r"$\alpha$", "Normalisation bias", "", fname_norm)
    plt_bias_chi2 = H1D(bias_chi2)
    plt_bias_chi2.val = bias_chi2
    plt_bias_chi2.err = np.ones(len(rangeAlpha))*np.sqrt(float(len(self.truth.val))/float(N)) # error in chi^2 considering errors in the mean of std/sqrt(N)
    plt_bias_chi2.x = rangeAlpha
    plt_bias_chi2.x_err = np.zeros(len(rangeAlpha))
    plt_cte = H1D(plt_bias_chi2)
    plt_cte.val = 0.5*np.ones(len(rangeAlpha))
    plt_cte.err = np.zeros(len(rangeAlpha))
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
      self.trace = pm.sample(N, tune = N/2) #, step, start = start)
      self.N = len(self.trace)

      pm.summary(self.trace)

      self.hnp = H1D(np.zeros(len(self.systematics)))
      self.hnpu = H1D(np.zeros(len(self.unf_systematics)))
      self.hunf = H1D(self.truth)
      self.hunf_mode = H1D(self.truth)
      for i in range(0, self.Nt):
        self.hunf.val[i] = np.mean(self.trace.Truth[:, i])
        self.hunf.err[i] = np.std(self.trace.Truth[:, i], ddof = 1)**2
        m = self.hunf.val[i]
        s = np.sqrt(self.hunf.err[i])
        pdf = stats.gaussian_kde(self.trace.Truth[:, i])
        g = np.linspace(m-2*s, m+2*s, 1000)
        mode = g[np.argmax(pdf(g))]
        self.hunf_mode.val[i] = mode
        self.hunf_mode.err[i] = self.hunf.err[i]

      self.hunf_np0 = H1D(self.truth)
      self.hunf_np1p = H1D(self.truth)
      self.hunf_np1m = H1D(self.truth)
      self.hnp.x = [""]*len(self.systematics)
      self.hnpu.x = [""]*len(self.unf_systematics)
      for i in range(0, self.Nt):
        L = range(0, len(self.trace.Truth[:, 0]))
        for k in range(0, len(self.systematics)):
          L = [x for x in L if x in np.where(np.fabs(self.trace['t_'+self.systematics[k]][:]) < 0.1)[0]]
        for k in range(0, len(self.unf_systematics)):
          L = [x for x in L if x in np.where(np.fabs(self.trace['tu_'+self.unf_systematics[k]][:]) < 0.1)[0]]
        self.hunf_np0.val[i] = np.mean(self.trace.Truth[L, i])
        self.hunf_np0.err[i] = np.std(self.trace.Truth[L, i], ddof = 1)**2

        L = range(0, len(self.trace.Truth[:, 0]))
        for k in range(0, len(self.systematics)):
          L = [x for x in L if x in np.where(self.trace['t_'+self.systematics[k]][:] > 0.9)[0]]
        for k in range(0, len(self.unf_systematics)):
          L = [x for x in L if x in np.where(self.trace['tu_'+self.unf_systematics[k]][:] > 0.9)[0]]
        self.hunf_np1p.val[i] = np.mean(self.trace.Truth[L, i])
        self.hunf_np1p.err[i] = np.std(self.trace.Truth[L, i], ddof = 1)**2

        L = range(0, len(self.trace.Truth[:, 0]))
        for k in range(0, len(self.systematics)):
          L = [x for x in L if x in np.where(self.trace['t_'+self.systematics[k]][:] < -0.9)[0]]
        for k in range(0, len(self.unf_systematics)):
          L = [x for x in L if x in np.where(self.trace['tu_'+self.unf_systematics[k]][:] < -0.9)[0]]
        self.hunf_np1m.val[i] = np.mean(self.trace.Truth[L, i])
        self.hunf_np1m.err[i] = np.std(self.trace.Truth[L, i], ddof = 1)**2
      for k in range(0, len(self.systematics)):
        self.hnp.val[k] = np.mean(self.trace['t_'+self.systematics[k]])
        self.hnp.err[k] = np.std(self.trace['t_'+self.systematics[k]], ddof = 1)**2
        self.hnp.x[k] = self.systematics[k]
        self.hnp.x_err[k] = 1
      for k in range(0, len(self.unf_systematics)):
        self.hnpu.val[k] = np.mean(self.trace['tu_'+self.unf_systematics[k]])
        self.hnpu.err[k] = np.std(self.trace['tu_'+self.unf_systematics[k]], ddof = 1)**2
        self.hnpu.x[k] = self.unf_systematics[k]
        self.hnpu.x_err[k] = 1

  '''
  Estimate modes from GKE.
  '''
  def getPosteriorMode(self):
    dt = self.trace.Truth.shape[1]
    du = len(self.unf_systematics)
    d = dt + du
    r = self.trace.Truth.shape[0]
    value = np.zeros( (d, r) )
    S = np.zeros( (d, 1) )
    dS = np.zeros( (d, 1) )
    for i in range(0, dt):
      value[i, :] = copy.deepcopy(self.trace.Truth[:, i])
      m = np.mean(self.trace.Truth[:, i])
      s = np.std(self.trace.Truth[:, i])
      S[i, 0] = m
      dS[i, 0] = s
    for i in range(0, du):
      value[dt + i, :] = copy.deepcopy(self.trace['tu_'+self.unf_systematics[i]][:])
      m = np.mean(self.trace['tu_'+self.unf_systematics[i]])
      s = np.std(self.trace['tu_'+self.unf_systematics[i]])
      S[dt + i, 0] = m
      dS[dt + i, 0] = s
    #pdf = stats.kde.gaussian_kde(value)
    H = np.cov(value)
    Hi = np.linalg.inv(H)
    detH = np.linalg.det(H)
    def mpdf(x):
      #scale = np.power(2*np.pi, -r*0.5) *np.power(detH, -0.5)  # this is constant ...
      scale = 1.0
      p = 0
      for i in range(0, r):
        xs = x.reshape(d, 1) - value[:, i].reshape(d, 1)
        xs = xs.reshape(d, 1)
        p += scale*np.exp(-0.5*float(np.matmul(np.transpose(xs), np.matmul(Hi, xs))))
      if p > 0:
        p = -2*np.log(p)
      elif p == 0:
        p = 1e20
      else:
        print("Negative PDF:", p)
      return p
    print("Start minimization with %s = %f" % (str(S), mpdf(S)))
    res = optimize.minimize(mpdf, S, method='L-BFGS-B', options={'ftol': 1e-6, 'gtol': 0, 'maxiter': 100, 'eps': 1e-3, 'disp': True})
    print(res)

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
  Plot the distributions for each unfolding nuisance parameter regardless of the other bins
  '''
  def plotNPUMarginal(self, syst, fname):
    i = self.unf_systematics.index(syst)
    fig = plt.figure(figsize=(10, 10))
    sns.distplot(self.trace['tu_'+self.unf_systematics[i]], kde = True, hist = True, label = self.unf_systematics[i])
    plt.title(self.unf_systematics[i])
    plt.ylabel("Probability")
    plt.xlim([-3, 3])
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
    plotH2D(np.cov(self.trace.Truth, rowvar = 0), "Unfolded bin", "Unfolded bin", "Covariance matrix of unfolded bins", logz=False, fname = fname, fmt = "")

  '''
  Plot the Pearson correlation coefficients.
  '''
  def plotCorr(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH2D(np.corrcoef(self.trace.Truth, rowvar = 0), "Unfolded bin", "Unfolded bin", "Pearson correlation of unfolded bins", logz = False, fname = fname)

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
    tmplabel = ["Unf. bin %d" % i for i in range(0, self.Nt)] + self.systematics + self.unf_systematics
    plotH2DWithText(np.corrcoef(tmp, rowvar = 1), tmplabel, "Variable", "Variable", "Pearson correlation of posterior", fname)

  '''
  Plot skewness.
  '''
  def plotSkewness(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.skew(self.trace.Truth, axis = 0, bias = False)
    sk.err = np.zeros(len(sk.val))
    plotH1D(sk, "Particle-level observable", "Skewness", "Skewness of the distribution after unfolding", logy = False, fname = fname)

  '''
  Plot nuisance parameter means and spread.
  '''
  def plotNP(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH1DWithText(self.hnp, "Nuisance parameter", "Nuisance parameter posteriors mean and width", fname)

  '''
  Plot unfolding nuisance parameter means and spread.
  '''
  def plotNPU(self, fname):
    fig = plt.figure(figsize=(10, 10))
    plotH1DWithText(self.hnpu, "Nuisance parameter", "Nuisance parameter posteriors mean and width", fname)

  '''
  Plot kurtosis.
  '''
  def plotKurtosis(self, fname):
    fig = plt.figure(figsize=(10, 10))
    sk = H1D(self.truth)
    sk.val = stats.kurtosis(self.trace.Truth, axis = 0, fisher = True, bias = False)
    sk.err = np.zeros(len(sk.val))
    plotH1D(sk, "Particle-level observable", "Fisher kurtosis", "Fisher kurtosis of the distribution after unfolding", logy = False, fname = fname)

  '''
  Plot data, truth, reco and unfolded result
  '''
  def plotUnfolded(self, fname = "plotUnfolded.png"):
    fig = plt.figure(figsize=(10, 10))
    #plt.errorbar(self.data.x, self.data.val, self.data.err**0.5, self.data.x_err, fmt = 'bs', linewidth=2, label = "Pseudo-data", markersize=10)
    #plt.errorbar(self.datasubbkg.x, self.datasubbkg.val, self.datasubbkg.err**0.5, self.datasubbkg.x_err, fmt = 'co', linewidth=2, label = "Background subtracted", markersize=10)
    #plt.errorbar(self.recoWithoutFakes.x, self.recoWithoutFakes.val, self.recoWithoutFakes.err**0.5, self.recoWithoutFakes.x_err, fmt = 'mv', linewidth=2, label = "Expected signal (no fakes) distribution", markersize=5)
    plt.errorbar(self.truth.x, self.truth.val, self.truth.err**0.5, self.truth.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(self.hunf_mode.x, self.hunf_mode.val, self.hunf_mode.err**0.5, self.hunf_mode.x_err, fmt = 'm^', linewidth=2, label = "Unfolded mode", markersize = 5)
    plt.errorbar(self.hunf.x, self.hunf.val, self.hunf.err**0.5, self.hunf.x_err, fmt = 'rv', linewidth=2, label = "Unfolded mean", markersize=5)
    plt.errorbar(self.hunf_np0.x, self.hunf_np0.val, self.hunf_np0.err**0.5, self.hunf_np0.x_err, fmt = 'b*', linewidth=1, label = "Unfolded mean for abs(NP) < 0.1", markersize=5)
    plt.errorbar(self.hunf_np1p.x, self.hunf_np1p.val, self.hunf_np1p.err**0.5, self.hunf_np1p.x_err, fmt = 'b^', linewidth=1, label = "Unfolded mean for NP > 0.9", markersize=5)
    plt.errorbar(self.hunf_np1m.x, self.hunf_np1m.val, self.hunf_np1m.err**0.5, self.hunf_np1m.x_err, fmt = 'bv', linewidth=1, label = "Unfolded mean for NP < -0.9", markersize=5)
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
    plt.errorbar(expectedCs.x, expectedCs.val, expectedCs.err**0.5, expectedCs.x_err, fmt = 'g^', linewidth=2, label = "Truth", markersize=10)
    plt.errorbar(observedCs.x, observedCs.val, observedCs.err**0.5, observedCs.x_err, fmt = 'rv', linewidth=2, label = "Unfolded mean", markersize=5)
    plt.errorbar(observedCs_mode.x, observedCs_mode.val, observedCs_mode.err**0.5, observedCs_mode.x_err, fmt = 'bv', linewidth=2, label = "Unfolded mode", markersize=5)
    plt.legend()
    if units != "":
      plt.ylabel("Differential cross section ["+units+"]")
    else:
      plt.ylabel("Events")
    plt.xlabel("Observable")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

