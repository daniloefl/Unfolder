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

truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

# get histograms from file
truth[""], recoWithFakes[""], bkg[""], mig[""], eff[""], nrt[""] = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttCoarse")
truth["me"], recoWithFakes["me"], bkg["me"], mig["me"], eff["me"], nrt["me"] = getHistograms("out_ttallhad_psrw_Syst.root", "aMcAtNloHerwigppEvtGen", "mttCoarse")
truth["ps"], recoWithFakes["ps"], bkg["ps"], mig["ps"], eff["ps"], nrt["ps"] = getHistograms("out_ttallhad_psrw_Syst.root", "PowhegHerwigppEvtGen", "mttCoarse")

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

# generate perfect fake data
data = recoWithFakes[""]

# generate fake data from model
pseudo_data = getDataFromModel(bkg[""], mig[""], eff[""], truth[""])

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

alpha = {}
alphaChi2 = {}
bestAlphaBias = {}
bestAlphaBiasStd = {}

for i in ["", "me", "ps"]:
  alpha[i] = -1
  alphaChi2[i] = -1
  bestAlphaBias[i] = -1
  bestAlphaBiasStd[i] = -1
  alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i] = scanRegParameter(DAgostiniForRegularizationTest(bkg[""], mig[""], eff[""]), bkg[i], mig[i], eff[i], truth[i], 1000, np.arange(1.0, 21.0, 1.0), "scanIter_DAgostini.png", "scanIter_chi2_DAgostini.png")
  print "For configuration '%s': Found iter = %d with bias chi2 = %f, bias mean = %f, bias std = %f" % (alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i])

pseudo_dagostini_mig = getDAgostini(bkg[i], mig[i], eff[i], pseudo_data, nIter = bestI[""])
pseudo_dagostini_result = pseudo_dagostini_mig

dagostini_mig = getDAgostini(bkg[""], mig[""], eff[""], data, nIter = bestI[""])
dagostini_result = dagostini_mig

comparePlot([data, pseudo_data, truth[""],
             dagostini_result,
             pseudo_dagostini_result],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments", "Particle-level",
             "Unfolded (d'Agostini) from projected reco.",
             "Unfolded (d'Agostini) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_DAgostini.png")

print "DAgostini -- it.  = ",   alpha, " with bias = ", bestAlphaBias, ", std = ", bestAlphaBiasStd

