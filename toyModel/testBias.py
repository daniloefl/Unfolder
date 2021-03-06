#!/usr/bin/env python3

# Unfolds using or not uncertaintis in FBU (see uncUnfList) for many values of
# alpha with first derivative regulatisation and plots the bias as a function of alpha
# The fb setting can be changed to use or not a biasa histogram for the regularisation

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

# bias in regularisation?
fb = 0.0

# uncertainties
uncUnfList = []
import sys
if len(sys.argv) > 1:
  if "withSyst" in sys.argv:
    uncUnfList = ["B", "C"]
  if "bias" in sys.argv:
    fb = 1.0

print uncUnfList
print fb 

# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms(direc = "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms(direc = "B")
#truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

# generate perfect fake data
data = recoWithFakes["A"]

# generate fake data from model
pseudo_data = getDataFromModel(bkg["A"], mig["A"], eff["A"])

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
#m.setEntropyPrior()
#m.setFirstDerivativePrior()

# add migration uncertainty
for k in uncUnfList:
  m.addUnfoldingUncertainty(k, bkg[k], mig[k], eff[k])

# try a curvature-based prior
# first choose alpha using only a MAP estimate
#m.setEntropyPrior()
#m.setCurvaturePrior()
m.setFirstDerivativePrior(fb)
#m.setGaussianPrior()

#m.setConstrainArea(True)
m.run(data)
# does the same for the pseudo-data

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

  # for first deriv:
  t_bkg = bkg[i]
  t_mig = mig[i]
  t_eff = eff[i]
  alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = m.scanAlpha(t_bkg, t_mig, t_eff, 1000, np.arange(0.0, 10, 0.5), "scanAlpha_%s.%s" % (i, extension), "scanAlpha_%s_chi2.%s" % (i, extension), "scanAlpha_%s_norm.%s" % (i, extension))
  # for curvature
  #alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = m.scanAlpha(t_bkg, t_mig, t_eff, 1000, np.arange(0.0, 4e-8, 2e-9), "scanAlpha_%s.%s" % (i, extension), "scanAlpha_%s_chi2.%s" % (i, extension), "scanAlpha_%s_norm.%s" % (i, extension))
  # for entropy
  #alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i] = m.scanAlpha(t_bkg, t_mig, t_eff, 1000, np.arange(0.0, 100.0, 4.0), "scanAlpha_%s.%s" % (i, extension), "scanAlpha_%s_chi2.%s" % (i, extension), "scanAlpha_%s_norm.%s" % (i, extension))
  print "For configuration '%s': Found alpha = %f with bias chi2 = %f, bias mean = %f, bias std = %f, norm bias = %f, norm bias std = %f" % (i, alpha[i], alphaChi2[i], bestAlphaBias[i], bestAlphaBiasStd[i], bestAlphaNormBias[i], bestAlphaNormBiasStd[i])

# do the rest with the best alpha from stat. test only
m.setAlpha(alpha["A"])

m.run(pseudo_data)
m.setData(pseudo_data)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal_pseudo.%s" % extension)
for i in uncUnfList:
  m.plotNPUMarginal(i, "plotNPUMarginal_pseudo_%s.%s" % (i, extension))

# plot unfolded spectrum
m.plotUnfolded("plotUnfolded_pseudo.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded_pseudo.%s" % extension)

# plot correlations graphically
# it takes forever and it is just pretty
# do it only if you really want to see it
# I just get annoyed waiting for it ...
#m.plotPairs("pairPlot.%s" % extension) # takes forever!

suf = "_pseudo"
m.plotCov("covPlot%s.%s" % (suf, extension))
m.plotCorr("corrPlot%s.%s" % (suf, extension))
m.plotCorrWithNP("corrPlotWithNP%s.%s" % (suf, extension))
m.plotSkewness("skewPlot%s.%s" % (suf, extension))
m.plotKurtosis("kurtosisPlot%s.%s" % (suf, extension))
m.plotNP("plotNP%s.%s" % (suf, extension))

m.plotUnfolded("plotUnfolded_pseudo.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded_pseudo.%s" % extension)

pseudo_fbu_result = m.hunf

# keep alpha and prior, but now unfold the original distribution
m.run(data)
m.setData(data)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)
for i in uncUnfList:
  m.plotNPUMarginal(i, "plotNPUMarginal_%s.%s" % (i, extension))

m.plotUnfolded("plotUnfolded.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

suf = ""
m.plotCov("covPlot%s.%s" % (suf, extension))
m.plotCorr("corrPlot%s.%s" % (suf, extension))
m.plotCorrWithNP("corrPlotWithNP%s.%s" % (suf, extension))
m.plotSkewness("skewPlot%s.%s" % (suf, extension))
m.plotKurtosis("kurtosisPlot%s.%s" % (suf, extension))
m.plotNP("plotNP%s.%s" % (suf, extension))

fbu_result = m.hunf

comparePlot([truth["A"],
             fbu_result,
             pseudo_fbu_result],
            ["Particle-level",
             "Unfolded (FBU) from projected reco.",
             "Unfolded (FBU) from independently simulated reco.",
            ],
            1.0, False, "", logy = True, fname = "biasTest.%s" % extension)

print "FBU     -- alpha = ",     alpha, " with bias = ", bestAlphaBias, ", std = ", bestAlphaBiasStd, ", norm bias = ", bestAlphaNormBias, ", norm bias std = ", bestAlphaNormBiasStd

