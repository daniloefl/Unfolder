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

# get histograms from file
truth, recoWithFakes, bkg, mig, eff, nrt = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttCoarse")

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

# functor to unfold
class DAgostiniForRegularizationTest:
  def __init__(self, f_bkg, f_mig, f_eff):
    self.f_bkg = f_bkg
    self.f_mig = f_mig
    self.f_eff = f_eff

  def __call__(self, i, data):
    dagostini_mig = getDAgostini(self.f_bkg, self.f_mig, self.f_eff, data, nIter = i)
    return dagostini_mig

bestI, bestIChi2, bestIBias, bestIStd = scanRegParameter(DAgostiniForRegularizationTest(bkg, mig, eff), bkg, mig, eff, truth, 1000, np.arange(1.0, 21.0, 1.0), "scanIter_DAgostini.png", "scanIter_chi2_DAgostini.png")
print "Found optimal iter:", bestI, bestIChi2, bestIBias, bestIStd

pseudo_dagostini_mig = getDAgostini(bkg, mig, eff, pseudo_data, nIter = bestI)
pseudo_dagostini_result = pseudo_dagostini_mig

dagostini_mig = getDAgostini(bkg, mig, eff, data, nIter = bestI)
dagostini_result = dagostini_mig

comparePlot([data, pseudo_data, truth,
             dagostini_result,
             pseudo_dagostini_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (d'Agostini) from projected reco.",
             "Unfolded (d'Agostini) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_DAgostini.png")

print "DAgostini -- it.  = ",   bestI, " with bias = ", bestIBias, ", std = ", bestIStd

