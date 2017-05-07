
try:
  import ROOT
except:
  pass
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from Histogram import H1D, H2D, plotH1D, plotH2D, plotH1DWithText, plotH2DWithText, plotH1DLines

'''
To be used to scan TUnfold's object to find the optimal tau value for
the regularisation.
'''
def printLcurve(tunfolder, fname):
  nScan   = 30
  tauMin  = 0
  tauMax  = 0
  iBest   = -1
  logTauX = ROOT.TSpline3()
  logTauY = ROOT.TSpline3()
  lCurve  = ROOT.TGraph()
  iBest = tunfolder.ScanLcurve(nScan,tauMin,tauMax,lCurve, logTauX, logTauY)
  tau   = tunfolder.GetTau()
  print "With TUnfold: Lcurve scan chose tau =", tau

  c = ROOT.TCanvas()
  t = ROOT.Double()
  x = ROOT.Double()
  y = ROOT.Double()
  logTauX.GetKnot(iBest,t,x)
  logTauY.GetKnot(iBest,t,y)
  bestLcurve = ROOT.TGraph(1, array.array('d',[x]), array.array('d', [y]))
  lCurve.Draw("AC")
  bestLcurve.SetMarkerColor(ROOT.kRed+2)
  bestLcurve.Draw("*")
  c.Print(fname)
  return tau

'''
Use this to run the TUnfold unfolding method.
For example:
tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
#printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(0)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff

comparePlot([f_data, f_data - f_bkg, truth, tunfold_result], ["Data", "Data - bkg", "Particle-level", "TUnfold"], luminosity*1e-3, True, "fb/GeV", "plotTUnfold.png")
'''
def getTUnfolder(bkg, mig, eff, data, regMode = None, normMode = None):
  if regMode == None: regMode = ROOT.TUnfold.kRegModeDerivative
  if normMode == None: normMode = ROOT.TUnfold.kEConstraintArea
  tunfolder = ROOT.TUnfoldDensity(mig.T().toROOT("tunfold_mig"), ROOT.TUnfold.kHistMapOutputVert, regMode, normMode, ROOT.TUnfoldDensity.kDensityModeeNone)
  bkg_noerr = H1D(bkg)
  for k in range(0, len(bkg.err)):
    bkg_noerr.err[k] = 0
  dataBkgSub = data - bkg_noerr
  tunfolder.SetInput(dataBkgSub.toROOT("data_minus_bkg"), 0)
  return tunfolder

'''
Use this to plot the histograms in listHist with the legends in listLegend.
'''
def comparePlot(listHist, listLegend, f = 1.0, normaliseByBinWidth = True, units = "fb", fname = "comparePlot.png"):
  fig = plt.figure(figsize=(10, 10))
  newListHist = []
  for item in listHist:
    newListHist.append(f*item)
    if normaliseByBinWidth:
      newListHist[-1].overBinWidth()
  sty = ['rv', 'bo', 'g^', 'mo', 'cv', 'ks']
  siz = [14, 12, 10, 8, 6, 4]
  c = 0
  for item in newListHist:
    plt.errorbar(item.x, item.val, item.err**0.5, item.x_err, fmt = sty[c], linewidth=2, label = listLegend[c], markersize=siz[c])
    c += 1
  plt.legend()
  if units != "":
    plt.ylabel("Differential cross section ["+units+"]")
  else:
    plt.ylabel("Events")
  plt.xlabel("Observable")
  plt.tight_layout()
  plt.savefig(fname)
  plt.close()

'''
This returns a migration-corrected object from RooUnfold using the D'Agostini unfolding procedure.
Here is an example of how to use it:
dagostini_mig = getDAgostini(bkg, mig, eff, data)
comparePlot([data, data - bkg, truth, dagostini_result], ["Data", "Data - bkg", "Particle-level", "D'Agostini"], luminosity*1e-3, True, "fb/GeV", "plotDAgostini.png")
'''
def getDAgostini(bkg, mig, eff, data, nIter = 1):
  reco = (mig.project('y') + bkg).toROOT("reco_rp")
  reco.SetDirectory(0)
  truth = (mig.project('x')/eff).toROOT("truth_p")
  truth.SetDirectory(0)
  m = mig.T().toROOT("m")
  m.SetDirectory(0)
  unf_response = ROOT.RooUnfoldResponse(reco, truth, m)
  dataBkgSub = data # - bkg
  dd = dataBkgSub.toROOT("dataBkgSub_dagostini")
  dd.SetDirectory(0)
  dagostini = ROOT.RooUnfoldBayes(unf_response, dd, int(nIter))
  dagostini.SetVerbose(-1)
  dagostini_hreco = dagostini.Hreco()
  dagostini_hreco.SetDirectory(0)
  del dagostini
  del unf_response
  del m
  r = H1D(dagostini_hreco)
  del dagostini_hreco
  return r

'''
Use model to get pseudo-data from toy experiments.
'''
def getDataFromModel(bkg, mig, eff):
  truth = mig.project('x')/eff
  response_noeff = H2D(mig) # = P(r|t) = Mtr/sum_k=1^Nr Mtk
  for i in range(0, mig.shape[0]): # for each truth bin
    rsum = 0.0
    for j in range(0, mig.shape[1]): # for each reco bin
      rsum += mig.val[i, j]    # calculate the sum of all reco bins in the same truth bin
    for j in range(0, mig.shape[1]): # for each reco bin
      response_noeff.val[i, j] = mig.val[i, j]/rsum

  data = H1D(bkg) # original bkg histogram is ignored: only used to clone X axis
  # simulate background
  for j in range(0, len(bkg.val)): # j is the reco bin
    bv = bkg.val[j]
    if bv < 0: bv = 0
    bkgCount = np.random.poisson(bv) # this simulates a counting experiment for the bkg
    data.val[j] = bkgCount # overwrite background so that we use a Poisson
    data.err[j] = bkgCount

  # for each truth bin
  for i in range(0, len(truth.val)): # i is the truth bin
    trueCount = np.random.poisson(truth.val[i]) # this simulates a counting experiment for the truth
    #trueCount = int(truth.val[i]) # dirac delta pdf for the truth distribution
    # calculate cumulative response for bin i
    # C(k|i) = sum_l=0^k P(r=l|t=i)
    C = np.zeros(len(bkg.val))
    for k in range(0, len(bkg.val)):
      for l in range(0, k+1):
        C[k] += response_noeff.val[i, l]
    # a uniform random number is between 0 and C[0] with prob. response_noeff.val[i, 0]
    # it is between C[0] and C[1] with prob. response_noeff.val[i, 1], etc.
    for n in range(0, trueCount): # number of experiments is the count in the truth bin
      # simulate efficiency by rejecting events with efficiency eff.val[i]
      if np.random.uniform(0, 1) > eff.val[i]:
        continue

      # find the reco bin using the migration matrix mig
      # we know that the probability of getting reco bin j given that we are in truth bin i is:
      # P(r=j|t=i) = response_noeff.val[i, j]
      # first throw a random number between 0 and 1
      rn = np.random.uniform(0, 1)
      recoBin = len(bkg.val) - 1 # set it to the last bin
      for k in range(0, len(bkg.val)): # loop over reco bins and get where the random number is in the cum. distribution
        if rn >= C[k]: # if the random number is bigger than the cum. distribution boundary
          # keep going as we are not yet at the boundary
          continue
        # if the random number is smaller than the cum. dist., we have already crossed the boundary
        # stop and set the reco bin
        recoBin = k
        break
      data.val[recoBin] += 1
      data.err[recoBin] += 1
  return data

'''
Calculate the sum of the bias using only the expected values.
'''
def getBiasFromToys(unfoldFunction, alpha, N, bkg, mig, eff, truth):
  fitted = np.zeros((N, len(truth.val)))
  #fitted2 = np.zeros((N, len(truth.val)))
  bias = np.zeros(len(truth.val))
  bias_norm = np.zeros(N)
  import sys
  for k in range(0, N):
    if k % 100 == 0:
      print "getBiasFromToys: Throwing toy experiment {0}/{1}\r".format(k, N),
      sys.stdout.flush()
    pseudo_data = getDataFromModel(bkg, mig, eff)
    unfolded = unfoldFunction(alpha, pseudo_data)
    fitted[k, :] = (unfolded.val - truth.val)
    bias_norm[k] = np.sum(unfolded.val - truth.val)
    #fitted2[k, :] = unfolded.val
    #if k % 500 == 0 and k > 0:
    #  f = plt.figure()
    #  plt.errorbar(truth.x, truth.val, np.sqrt(truth.val), truth.x_err, fmt = "ro", markersize = 10, label = "Truth")
    #  for l in range(k-10, k):
    #    plt.errorbar(unfolded.x, fitted2[l, :], np.sqrt(fitted2[l, :]),  truth.x_err, fmt = "bv", markersize = 10, label = "Unfolded test %d" % l)
    #  plt.legend(loc = "upper right")
    #  plt.show()
    #  plt.close()
  print
  # systematic bias
  bias_syst = np.mean(unfoldFunction(alpha, mig.project('y') + bkg).val - truth.val)
  bias = np.mean(fitted, axis = 0)
  bias_std = np.std(fitted, axis = 0, ddof = 1)
  bias_norm_mean = np.mean(bias_norm)
  bias_norm_std = np.std(bias_norm, ddof = 1)
  bias_binsum = np.mean(bias)
  bias_std_binsum = np.mean(bias_std)
  bias_chi2 = np.mean(np.power(bias/bias_std, 2))
  #print "bias mean = ", np.mean(fitted, axis = 0), ", bias std = ", np.std(fitted, axis = 0)
  return [bias_binsum, bias_std_binsum, bias_chi2, bias_norm_mean, bias_norm_std, bias_syst]

'''
Scan general regularization parameter to minimize bias^2 over variance.
unfoldFunction receives the reg. parameter and a data vector to unfold and returns the unfolded spectrum.
'''
def scanRegParameter(unfoldFunction, bkg, mig, eff, truth, N = 1000, rangeAlpha = np.arange(0.0, 1.0, 1e-3), fname = "scanRegParameter.png", fname_chi2 = "scanRegParameter_chi2.png", fname_norm = "scanRegParameter_norm.png"):
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
    #if i % 100 == 0:
    print "scanRegParameter: parameter = ", rangeAlpha[i], " / ", rangeAlpha[-1]
    sys.stdout.flush()
    bias[i], bias_std[i], bias_chi2[i], bias_norm[i], bias_norm_std[i], bias_syst[i] = getBiasFromToys(unfoldFunction, rangeAlpha[i], N, bkg, mig, eff, truth)
    print " -- --> scanRegParameter: parameter = ", rangeAlpha[i], " / ", rangeAlpha[-1], " with chi2 = ", bias_chi2[i], ", mean and std = ", bias[i], bias_std[i]
    if np.abs(bias_chi2[i] - 0.5) < minBias:
      minBias = np.abs(bias_chi2[i] - 0.5)
      bestAlpha = rangeAlpha[i]
      bestChi2 = bias_chi2[i]
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
  plotH1DLines({plt_bias: r"$E_{\mathrm{bins}}[E_{\mathrm{toys}}[\mathrm{bias}]]$", plt_bias_e: r"$E_{\mathrm{bins}}[\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]}]$", plt_bias_syst: r"$E_{\mathrm{bins}}[\mathrm{only \;\; syst. \;\; bias}]$"}, "Regularization parameter", "Bias", "", fname)
  plt_bias_norm = H1D(bias)
  plt_bias_norm.val = bias_norm
  plt_bias_norm.err = np.power(bias_norm_std, 2)
  plt_bias_norm.x = rangeAlpha
  plt_bias_norm.x_err = np.zeros(len(rangeAlpha))
  plt_bias_norm_e = H1D(bias)
  plt_bias_norm_e.val = bias_norm_std
  plt_bias_norm_e.err = np.zeros(len(rangeAlpha))
  plt_bias_norm_e.x = rangeAlpha
  plt_bias_norm_e.x_err = np.zeros(len(rangeAlpha))
  plotH1DLines({plt_bias_norm: r"$E_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]$", plt_bias_norm_e: r"$\sqrt{\mathrm{Var}_{\mathrm{toys}}[\mathrm{norm. \;\; bias}]}$"}, "Regularization parameter", "Normalisation bias", "", fname_norm)
  plt_bias_chi2 = H1D(bias_chi2)
  plt_bias_chi2.val = bias_chi2
  plt_bias_chi2.err = np.ones(len(rangeAlpha))*np.sqrt(float(len(truth.val))/float(N)) # error in chi^2 considering errors in the mean of std/sqrt(N)
  plt_bias_chi2.x = rangeAlpha
  plt_bias_chi2.x_err = np.zeros(len(rangeAlpha))
  plt_cte = H1D(plt_bias_chi2)
  plt_cte.val = 0.5*np.ones(len(rangeAlpha))
  plt_cte.err = np.zeros(len(rangeAlpha))
  plotH1DLines({plt_bias_chi2: r"$E_{\mathrm{bins}}[E_{\mathrm{toys}}[\mathrm{bias}]^2/\mathrm{Var}_{\mathrm{toys}}[\mathrm{bias}]]$", plt_cte: "0.5"}, "Regularisation parameter", r"Bias $\mathrm{mean}^2/\mathrm{variance}$", "", fname_chi2)
  return [bestAlpha, bestChi2, bias[bestI], bias_std[bestI], bias_norm[bestI], bias_norm_std[bestI]]

