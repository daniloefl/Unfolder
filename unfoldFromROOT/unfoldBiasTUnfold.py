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
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")

recoWithoutFakes = mig.project("y")

# plot migration matrix as it will be used next for unfolding
plotH2D(mig.T(), "Particle-level bin", "Reconstructed-level bin", "Number of events for each (reco, truth) configuration", "mig.%s" % extension)

# plot 1D histograms for cross checks
plotH1D(bkg, "Reconstructed "+varname, "Events", "Background", "bkg.%s" % extension)
plotH1D(truth, "Particle-level "+varname, "Events", "Particle-level distribution", "truth.%s" % extension)
plotH1D(nrt, "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt.%s" % extension)
plotH1D(recoWithFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes.%s" % extension)
plotH1D(recoWithoutFakes, "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes.%s" % extension)
plotH1D(eff, "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff.%s" % extension)

# generate perfect fake data
data = recoWithFakes

# generate fake data from model
pseudo_data = getDataFromModel(bkg, mig, eff, truth)

comparePlot([data, pseudo_data, data - bkg, pseudo_data - bkg],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments",
             "Reco. projected from unfolding factors - bkg", "Reco. simulated with toy experiments - bkg"],
            luminosity*1e-3, True, "fb/GeV", "pseudoData.png")

# Try alternative
# Create alternative method for unfolding
f_truth, f_recoWithFakes, f_bkg, f_mig, f_eff, f_nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
f_data = f_recoWithFakes
pseudo_f_data = getDataFromModel(f_bkg, f_mig, f_eff, f_truth)

# functor to unfold
class TUnfoldForRegularizationTest:
  def __init__(self, f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = 0):
    self.f_bkg = f_bkg
    self.tunfolder_reg = getTUnfolder(f_bkg, f_mig, f_data, regMode = regMode, normMode = normMode)

  def __call__(self, tau, data):
    #f_truth, f_recoWithFakes, f_bkg, f_mig, f_eff, f_nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
    #tunfolder_reg = getTUnfolder(f_bkg, f_mig, data, regMode = ROOT.TUnfold.kRegModeDerivative)
    dataMinusBkg = (data - self.f_bkg).toROOT("data_minus_bkg_tmp")
    dataMinusBkg.SetDirectory(0)
    self.tunfolder_reg.SetInput(dataMinusBkg)
    self.tunfolder_reg.DoUnfold(tau)
    tmp = self.tunfolder_reg.GetOutput("tunfold_result_tmp")
    tmp.SetDirectory(0)
    tunfold_mig = H1D(tmp)
    tunfold_result = tunfold_mig/eff
    del tmp
    del dataMinusBkg
    return tunfold_result

bestTau, bestTauChi2, bestTauBias, bestTauStd = scanRegParameter(TUnfoldForRegularizationTest(f_bkg, f_mig, f_data, ROOT.TUnfold.kRegModeCurvature, normMode), f_bkg, f_mig, f_eff, f_truth, 1000, np.arange(0.0, 10e-3, 0.125e-3), "scanTau_TUnfold.png", "scanTau_chi2_TUnfold.png")
print "Found optimal tau:", bestTau, bestTauChi2, bestTauBias, bestTauStd

pseudo_tunfolder = getTUnfolder(f_bkg, f_mig, pseudo_f_data, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = normMode)
#pseudo_tunfolder = getTUnfolder(f_bkg, f_mig, pseudo_f_data, regMode = ROOT.TUnfold.kRegModeCurvature)
#pseudo_tunfolder = getTUnfolder(f_bkg, f_mig, pseudo_f_data, regMode = ROOT.TUnfold.kRegModeNone)

#tau_pseudo = printLcurve(pseudo_tunfolder, "tunfold_lcurve_pseudo.png")
pseudo_tunfolder.DoUnfold(bestTau)
pseudo_tunfold_mig = H1D(pseudo_tunfolder.GetOutput("tunfold_pseudo_result"))
pseudo_tunfold_result = pseudo_tunfold_mig/eff

tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeDerivative, normMode = normMode)
#tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeCurvature)
#tunfolder = getTUnfolder(f_bkg, f_mig, f_data, regMode = ROOT.TUnfold.kRegModeNone)
# no regularization
#tau = printLcurve(tunfolder, "tunfold_lcurve.png")
tunfolder.DoUnfold(bestTau)
tunfold_mig = H1D(tunfolder.GetOutput("tunfold_result"))
tunfold_result = tunfold_mig/eff

comparePlot([f_data, pseudo_f_data, f_truth,
             tunfold_result,
             pseudo_tunfold_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (TUnfold) from projected reco.",
             "Unfolded (TUnfold) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_TUnfold.png")

print "TUnfold -- tau   = ",   bestTau, " with bias = ", bestTauBias, ", std = ", bestTauStd

