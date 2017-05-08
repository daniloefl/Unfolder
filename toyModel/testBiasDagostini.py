#!/usr/bin/env python

# Unfolds using TUnfold for many values of tau with first derivative regularisation
# fb can be changed to use or not a bias

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

sns.set(context = "paper", style = "whitegrid", font_scale=2)

varname = "observable"
extension = "eps"

truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

# get histograms from file
truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms("histograms.pkl", "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms("histograms.pkl", "B")
truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i], "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig_%s.%s" % (i,extension), fmt = "")

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", "bkg_%s.%s" % (i, extension))
  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff_%s.%s" % (i,extension))

# generate perfect fake data
data = recoWithFakes["A"]

# generate fake data from model
pseudo_data = getDataFromModel(bkg["A"], mig["A"], eff["A"])

# functor to unfold
class DAgostiniForRegularizationTest:
  def __init__(self, f_bkg, f_mig, f_eff):
    self.f_bkg = f_bkg
    self.f_mig = f_mig
    self.f_eff = f_eff

  def __call__(self, i, data):
    dagostini_mig = getDAgostini(self.f_bkg, self.f_mig, self.f_eff, data, nIter = i)
    return dagostini_mig

alpha = {}
alphaChi2 = {}
bestAlphaBias = {}
bestAlphaBiasStd = {}
bestAlphaNormBias = {}
bestAlphaNormBiasStd = {}

for i in ["A", "B", "C"]:
  print "Checking bias due to configuration '%s'" % i
  alpha[i] = -1
  alphaChi2[i] = -1
  bestAlphaBias[i] = -1
  bestAlphaBiasStd[i] = -1
  bestAlphaNormBias[i] = -1
  bestAlphaNormBiasStd[i] = -1
  alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = scanRegParameter(DAgostiniForRegularizationTest(bkg[""], mig[""], eff[""]), bkg[i], mig[i], eff[i], truth[i], 1000, np.arange(1.0, 21.0, 1.0), "scanIter_%s_DAgostini.%s" % (i, extension), "scanIter_%s_chi2_DAgostini.%s" % (i, extension), "scanIter_%s_norm_DAgostini.%s" % (i, extension))
  print "For configuration '%s': Found iter = %d with bias chi2 = %f, bias mean = %f, bias std = %f, norm bias = %f, norm bias std = %f" % (i, alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i])

pseudo_dagostini_mig = getDAgostini(bkg[i], mig[i], eff[i], pseudo_data, nIter = alpha["A"])
pseudo_dagostini_result = pseudo_dagostini_mig

dagostini_mig = getDAgostini(bkg["A"], mig["A"], eff["A"], data, nIter = alpha["A"])
dagostini_result = dagostini_mig

comparePlot([truth["A"],
             dagostini_result,
             pseudo_dagostini_result],
            ["Particle-level",
             "Unfolded (d'Agostini) from projected reco.",
             "Unfolded (d'Agostini) from independently simulated reco."],
            1.0, False, "fb/GeV", "biasTest_DAgostini.%s" % extension)

print "DAgostini -- it.  = ",   alpha, " with bias = ", bestAlphaBias, ", std = ", bestAlphaBiasStd, ", norm bias = ", bestAlphaNormBias, ", norm bias std = ", bestAlphaNormBiasStd

