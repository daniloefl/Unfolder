#!/usr/bin/env python

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
import pymc3 as pm
import matplotlib.cm as cm
import scipy

from Unfolder.ComparisonHelpers import *
from Unfolder.Unfolder import Unfolder
from Unfolder.Histogram import H1D, H2D, plotH1D, plotH2D
from readHistograms import *

sns.set(color_codes=True)
sns.set(font_scale=1.3)

varname = "m_{tt}"
extension = "png"

normMode = ROOT.TUnfold.kEConstraintArea

# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
bkg_noerr = {}
mig = {}
eff = {}
eff_noerr = {}
nrt = {}

truth[""], recoWithFakes[""], bkg[""], mig[""], eff[""], nrt[""] = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
truth["me"], recoWithFakes["me"], bkg["me"], mig["me"], eff["me"], nrt["me"] = getHistograms("out_ttallhad_psrw_Syst.root", "aMcAtNloHerwigppEvtGen", "mttAsymm")
truth["ps"], recoWithFakes["ps"], bkg["ps"], mig["ps"], eff["ps"], nrt["ps"] = getHistograms("out_ttallhad_psrw_Syst.root", "PowhegHerwigppEvtGen", "mttAsymm")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i].T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig_%s.%s" % (i,extension))

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", "bkg_%s.%s" % (i, extension))
  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff_%s.%s" % (i,extension))

  bkg_noerr[i] = H1D(bkg[i])
  for k in range(0, len(bkg_noerr[i].err)):
    bkg_noerr[i].err[k] = 0
  eff_noerr = H1D(eff_noerr[i].err)
  for k in range(0, len(eff_noerr[i].err)):
    eff_noerr[i].err[k] = 0
  

# generate perfect fake data
data = recoWithFakes[""]

# generate fake data from model
pseudo_data = getDataFromModel(bkg[""], mig[""], eff[""])

comparePlot([data, pseudo_data, data - bkg[""], pseudo_data - bkg[""]],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments",
             "Reco. projected from unfolding factors - bkg", "Reco. simulated with toy experiments - bkg"],
            luminosity*1e-3, True, "fb/GeV", "pseudoData.png")

# functor to unfold
class TUnfoldForRegularizationTest:
  def __init__(self, f_bkg, f_mig, f_eff, f_data, fb = 1.0, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = 0):
    self.f_bkg = f_bkg
    self.tunfolder_reg = getTUnfolder(f_bkg, f_mig, f_eff, f_data, regMode = regMode, normMode = normMode)
    self.f_eff = f_eff
    self.fb = fb
    self.f_bkg_noerr = H1D(self.f_bkg)
    for k in range(0, len(self.f_bkg_noerr.err)):
      self.f_bkg_noerr.err[k] = 0
    self.f_eff_noerr = H1D(self.f_eff)
    for k in range(0, len(self.f_eff_noerr.err)):
      self.f_eff_noerr.err[k] = 0

  def __call__(self, tau, data):
    #f_truth, f_recoWithFakes, f_bkg, f_mig, f_eff, f_nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
    #tunfolder_reg = getTUnfolder(f_bkg, f_mig, data, regMode = ROOT.TUnfold.kRegModeDerivative)
    dataMinusBkg = (data - self.f_bkg_noerr).toROOT("data_minus_bkg_tmp")
    dataMinusBkg.SetDirectory(0)
    self.tunfolder_reg.SetInput(dataMinusBkg, self.fb)
    self.tunfolder_reg.DoUnfold(tau)
    tmp = self.tunfolder_reg.GetOutput("tunfold_result_tmp")
    tmp.SetDirectory(0)
    tunfold_mig = H1D(tmp)
    tunfold_result = tunfold_mig/self.f_eff_noerr
    del tmp
    del dataMinusBkg
    return tunfold_result

alpha = {}
alphaChi2 = {}
bestAlphaBias = {}
bestAlphaBiasStd = {}
bestAlphaNormBias = {}
bestAlphaNormBiasStd = {}

for i in ["", "me", "ps"]:
  print "Checking bias due to configuration '%s'" % i
  alpha[i] = -1
  alphaChi2[i] = -1
  bestAlphaBias[i] = -1
  bestAlphaBiasStd[i] = -1
  bestAlphaNormBias[i] = -1
  bestAlphaNormBiasStd[i] = -1
  # use this range for fb = 0
  alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = scanRegParameter(TUnfoldForRegularizationTest(bkg[""], mig[""], eff[""], data, 0.0, ROOT.TUnfold.kRegModeDerivative, normMode), bkg[i], mig[i], eff[i], truth[i], 1000, np.arange(0.0, 10e-3, 0.25e-3), "scanTau_%s_TUnfold.png" % i, "scanTau_%s_chi2_TUnfold.png" % i, "scanTau_%s_norm_TUnfold.png" % i)
  #alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = scanRegParameter(TUnfoldForRegularizationTest(bkg[""], mig[""], eff[""], data, 1.0, ROOT.TUnfold.kRegModeDerivative, normMode), bkg[i], mig[i], eff[i], truth[i], 1000, np.arange(0.0, 20e-3, 1e-3), "scanTau_%s_TUnfold.png" % i, "scanTau_%s_chi2_TUnfold.png" % i, "scanTau_%s_norm_TUnfold.png" % i)
  print "For configuration '%s': Found tau = %f with bias chi2 = %f, bias mean = %f, bias std = %f, norm bias = %f, norm bias std = %f" % (i, alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i])

pseudo_tunfolder = getTUnfolder(bkg[""], mig[""], eff[""], pseudo_data, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = normMode)

#tau_pseudo = printLcurve(pseudo_tunfolder, "tunfold_lcurve_pseudo.png")
pseudo_tunfolder.DoUnfold(alpha[""])
pseudo_tunfold_mig = H1D(pseudo_tunfolder.GetOutput("tunfold_pseudo_result"))
pseudo_tunfold_result = pseudo_tunfold_mig/eff_noerr['']

tunfolder = getTUnfolder(bkg[""], mig[""], eff[""], data, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = normMode)
# no regularization
#tau = printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(alpha[""])
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff_noerr['']

comparePlot([data, pseudo_data, truth[""],
             tunfold_result,
             pseudo_tunfold_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (TUnfold) from projected reco.",
             "Unfolded (TUnfold) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_TUnfold.png")

print "TUnfold -- tau   = ",   alpha, " with bias = ", bestAlphaBias, ", std = ", bestAlphaBiasStd, ", norm bias = ", bestAlphaNormBias, ", norm bias std = ", bestAlphaNormBiasStd


