#!/usr/bin/env python

# Unfolds using uncertainties in FBU, but with no regularisation


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

# uncertainties
uncUnfList = []

# bias in regularisation?
fb = 0.0

data_input = "A"

import sys
if len(sys.argv) > 1:
  if "withSyst" in sys.argv:
    uncUnfList = ["B", "C"]
  if "bias" in sys.argv:
    fb = 1.0

  if "inputA" in sys.argv:
    data_input = "A"
  if "inputB" in sys.argv:
    data_input = "B"
  if "inputC" in sys.argv:
    data_input = "C"


# get histograms from file
truth = {}
recoWithFakes = {}
recoWithoutFakes = {}
bkg = {}
mig = {}
eff = {}
nrt = {}

truth["A"], recoWithFakes["A"], bkg["A"], mig["A"], eff["A"], nrt["A"] = getHistograms("histograms.pkl", "A")
truth["B"], recoWithFakes["B"], bkg["B"], mig["B"], eff["B"], nrt["B"] = getHistograms("histograms.pkl", "B")
truth["C"], recoWithFakes["C"], bkg["C"], mig["C"], eff["C"], nrt["C"] = getHistograms("histograms.pkl", "C")

for i in recoWithFakes:
  recoWithoutFakes[i] = mig[i].project("y")

  # plot migration matrix as it will be used next for unfolding
  plotH2D(mig[i], "Reconstructed-level bin", "Particle-level bin", "Number of events for each (reco, truth) configuration", "mig_%s.%s" % (i,extension), fmt ="")

  # plot 1D histograms for cross checks
  plotH1D(bkg[i], "Reconstructed "+varname, "Events", "Background", "bkg_%s.%s" % (i, extension))
  plotH1D(truth[i], "Particle-level "+varname, "Events", "Particle-level distribution", "truth_%s.%s" % (i, extension))
  plotH1D(nrt[i], "Particle-level "+varname, "Events", "Events in particle-level selection but not reconstructed", "nrt_%s.%s" % (i,extension))
  plotH1D(recoWithFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution with fakes", "recoWithFakes_%s.%s" % (i,extension))
  plotH1D(recoWithoutFakes[i], "Reconstructed "+varname, "Events", "Reconstructed-level distribution without fakes", "recoWithoutFakes_%s.%s" % (i,extension))
  plotH1D(eff[i], "Particle-level "+varname, "Efficiency", "Efficiency of particle-level selection", "eff_%s.%s" % (i,extension))

# generate perfect fake data
import sys
data = recoWithFakes[data_input]
#data = recoWithFakes["B"]
#data = recoWithFakes["C"]

# Create unfolding class
m = Unfolder(bkg["A"], mig["A"], eff["A"], truth["A"])
m.setUniformPrior()
#m.setGaussianPrior()
#m.setCurvaturePrior()
#m.setFirstDerivativePrior()

m.run(data)
m.setAlpha(0.0)
m.sample(100000)

unf_orig = m.hunf
  
for k in uncUnfList:
  m.addUnfoldingUncertainty(k, bkg[k], mig[k], eff[k])

# plot response matrix P(r|t)*eff(r)
plotH2D(m.response, "Reconstructed-level bin", "Particle-level bin", "Transpose of response matrix P(r|t)*eff(t)", "responseMatrix.%s" % extension, vmin = 0, vmax = 1.2*np.amax(eff["A"].val))
# and also the migration probabilities matrix
plotH2D(m.response_noeff, "Reconstructed-level bin", "Particle-level bin", "Transpose of migration probabilities P(r|t)", "migrationMatrix.%s" % extension, vmin = 0, vmax = 1)

m.run(data)
m.setAlpha(0.0)
m.sample(100000)

# plot marginal distributions
m.plotMarginal("plotMarginal.%s" % extension)

for i in uncUnfList:
  m.plotNPUMarginal(i, "plotNPUMarginal_%s.%s" % (i, extension))

# plot correlations
#m.plotPairs("pairPlot.%s" % extension) # takes forever
m.plotCov("covPlot.%s" % extension)
m.plotCorr("corrPlot.%s" % extension)
m.plotCorrWithNP("corrPlotWithNP.%s" % extension)
m.plotSkewness("skewPlot.%s" % extension)
m.plotKurtosis("kurtosisPlot.%s" % extension)

m.plotNP("plotNP.%s" % extension)
m.plotNPU("plotNPU.%s" % extension)

# plot unfolded spectrum
m.plotUnfolded("plotUnfolded.%s" % extension)
m.plotOnlyUnfolded(1.0, False, "", "plotOnlyUnfolded.%s" % extension)

comparePlot([truth["A"], truth["B"], truth["C"], m.hunf, unf_orig], ["Truth A", "Truth B", "Truth C", "FBU with systematic uncertainties", "FBU no systematic uncertainties"], 1.0, False, "", "compareMethods.%s" % extension)

