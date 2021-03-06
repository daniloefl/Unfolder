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
truth[""], recoWithFakes[""], bkg[""], mig[""], eff[""], nrt[""] = getHistograms("out_ttallhad_psrw_Syst.root", "nominal", "mttAsymm")
truth["me"], recoWithFakes["me"], bkg["me"], mig["me"], eff["me"], nrt["me"] = getHistograms("out_ttallhad_psrw_Syst.root", "aMcAtNloHerwigppEvtGen", "mttAsymm")
truth["ps"], recoWithFakes["ps"], bkg["ps"], mig["ps"], eff["ps"], nrt["ps"] = getHistograms("out_ttallhad_psrw_Syst.root", "PowhegHerwigppEvtGen", "mttAsymm")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

# generate perfect fake data
data = recoWithFakes[""]

# generate fake data from model
pseudo_data = getDataFromModel(bkg[""], mig[""], eff[""])

comparePlot([data, pseudo_data, data - bkg[""], pseudo_data - bkg[""]],
            ["Reco. projected from unfolding factors", "Reco. simulated with toy experiments",
             "Reco. projected from unfolding factors - bkg", "Reco. simulated with toy experiments - bkg"],
            luminosity*1e-3, True, "fb/GeV", "pseudoData.%s" % extension)

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

for i in ["", "me", "ps"]:
  print "Checking bias due to configuration '%s'" % i
  alpha[i] = -1
  alphaChi2[i] = -1
  bestAlphaBias[i] = -1
  bestAlphaBiasStd[i] = -1
  bestAlphaNormBias[i] = -1
  bestAlphaNormBiasStd[i] = -1
  alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = scanRegParameter(DAgostiniForRegularizationTest(bkg[""], mig[""], eff[""]), bkg[i], mig[i], eff[i], truth[i], 1000, np.arange(1.0, 21.0, 1.0), "scanIter_%s_DAgostini.%s" % (i, extension), "scanIter_%s_chi2_DAgostini.%s" % (i, extension), "scanIter_%s_norm_DAgostini.%s" % (i, extension))
  print "For configuration '%s': Found iter = %d with bias chi2 = %f, bias mean = %f, bias std = %f, norm bias = %f, norm bias std = %f" % (i, alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i])

pseudo_dagostini_mig = getDAgostini(bkg[i], mig[i], eff[i], pseudo_data, nIter = alpha[""])
pseudo_dagostini_result = pseudo_dagostini_mig

dagostini_mig = getDAgostini(bkg[""], mig[""], eff[""], data, nIter = alpha[""])
dagostini_result = dagostini_mig

comparePlot([truth[""],
             dagostini_result,
             pseudo_dagostini_result],
            ["Particle-level",
             "Unfolded (d'Agostini) from projected reco.",
             "Unfolded (d'Agostini) from independently simulated reco."],
            luminosity*1e-3, True, "fb/GeV", "biasTest_DAgostini.%s" % extension)

print "DAgostini -- it.  = ",   alpha, " with bias = ", bestAlphaBias, ", std = ", bestAlphaBiasStd, ", norm bias = ", bestAlphaNormBias, ", norm bias std = ", bestAlphaNormBiasStd

